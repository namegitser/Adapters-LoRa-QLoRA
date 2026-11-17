#!/usr/bin/env python
"""H3: LoRA rank sweep on DistilBERT + SST-2 (plus a full fine-tuning reference).

One invocation = one (method, rank, seed) run = results/h3/<run>.json
LoRA targets q_lin + v_lin (as in the LoRA paper); the classification head
(pre_classifier + classifier, randomly initialised) is trained in every run.
By default alpha = 2*r so the LoRA scale alpha/r stays constant across the sweep.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT_ROOT, SST2, count_params, dtype_name, env_info, lora_only_params,  # noqa: E402
                    peak_gb, pick_amp_dtype, reset_peak, set_seed, weights_size_mb, write_result)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["full", "lora"], default="lora")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=None, help="default 2*r (constant scale)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--targets", default="q_lin,v_lin")
    p.add_argument("--model", default="distilbert/distilbert-base-uncased")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="default 2e-5 full, 5e-4 LoRA")
    p.add_argument("--train-subset", type=int, default=0, help="0 = full 67k train set")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--exp", default="h3")
    return p.parse_args()


@torch.no_grad()
def accuracy(model, loader, amp):
    model.eval()
    correct = n = 0
    for b in loader:
        b = {k: v.cuda() for k, v in b.items()}
        with torch.autocast("cuda", dtype=amp):
            logits = model(**{k: v for k, v in b.items() if k != "labels"}).logits
        correct += int((logits.argmax(-1) == b["labels"]).sum())
        n += len(b["labels"])
    model.train()
    return correct / n


def main():
    args = parse_args()
    set_seed(args.seed)
    amp = pick_amp_dtype(args.amp)
    is_lora = args.method == "lora"
    alpha = (args.alpha or 2 * args.r) if is_lora else None
    lr = args.lr if args.lr is not None else (5e-4 if is_lora else 2e-5)
    run = f"lora_r{args.r}_seed{args.seed}" if is_lora else f"full_seed{args.seed}"
    rec = {"exp": args.exp, "run": run, "method": args.method, "model": args.model,
           "lora_r": args.r if is_lora else None, "lora_alpha": alpha,
           "lora_targets": args.targets if is_lora else None, "seed": args.seed, "epochs": args.epochs,
           "batch": args.batch, "max_len": args.max_len, "lr": lr, "amp_dtype": dtype_name(amp),
           "status": "started", "env": env_info()}
    print(f"=== h3/{run} | GPU={rec['env'].get('gpu_name')}", flush=True)
    t_wall = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        ds = load_dataset(*SST2)
        train = ds["train"]
        if args.train_subset:
            train = train.shuffle(seed=1234).select(range(args.train_subset))
        val = ds["validation"]  # 872 labelled examples (test labels are hidden)

        def enc(batch):
            return tok(batch["sentence"], truncation=True, max_length=args.max_len)
        cols = ["sentence", "idx"]
        train = train.map(enc, batched=True, remove_columns=cols).rename_column("label", "labels")
        val = val.map(enc, batched=True, remove_columns=cols).rename_column("label", "labels")
        coll = DataCollatorWithPadding(tok)
        g = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, generator=g,
                                  collate_fn=coll, num_workers=2)
        val_loader = DataLoader(val, batch_size=128, collate_fn=coll)
        rec.update(n_train=len(train), n_val=len(val))

        reset_peak()
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2, dtype=torch.float32).cuda()  # fp32 master weights, bf16 autocast
        if is_lora:
            from peft import LoraConfig, get_peft_model
            from peft.tuners.lora import LoraLayer
            cfg = LoraConfig(task_type="SEQ_CLS", r=args.r, lora_alpha=alpha, lora_dropout=args.dropout,
                             target_modules=args.targets.split(","),
                             modules_to_save=["pre_classifier", "classifier"])
            model = get_peft_model(model, cfg)
            model.print_trainable_parameters()
            # Paper formula check: r*(d_in + d_out) per adapted matrix
            analytic = sum(args.r * (m.get_base_layer().in_features + m.get_base_layer().out_features)
                           for m in model.modules() if isinstance(m, LoraLayer))
            n_mat = sum(1 for m in model.modules() if isinstance(m, LoraLayer))
            measured = lora_only_params(model)
            rec.update(lora_params=measured, lora_params_analytic=analytic, lora_matrices=n_mat,
                       analytic_match=(measured == analytic))
            print(f"LoRA params measured={measured:,} analytic r*(d+k) over {n_mat} matrices={analytic:,} "
                  f"match={measured == analytic}", flush=True)
        trainable, total = count_params(model)
        rec.update(trainable_params=trainable, total_params=total,
                   trainable_pct=round(100 * trainable / total, 4),
                   head_params=(trainable - rec.get("lora_params", 0)) if is_lora else None)

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        total_steps = args.epochs * len(train_loader)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))

        reset_peak()
        epoch_times, epoch_acc = [], []
        model.train()
        for ep in range(args.epochs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i, b in enumerate(train_loader):
                b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
                with torch.autocast("cuda", dtype=amp):
                    loss = model(**b).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if i % 500 == 0:
                    print(f"epoch {ep + 1} step {i}/{len(train_loader)} loss {float(loss):.4f}", flush=True)
            torch.cuda.synchronize()
            epoch_times.append(round(time.perf_counter() - t0, 2))
            epoch_acc.append(round(accuracy(model, val_loader, amp), 4))
            print(f"epoch {ep + 1}: {epoch_times[-1]}s  val_acc={epoch_acc[-1]}", flush=True)

        pa, pr = peak_gb()
        out_dir = CKPT_ROOT / args.exp / run
        model.save_pretrained(out_dir)
        rec.update(val_acc=epoch_acc[-1], val_acc_per_epoch=epoch_acc,
                   sec_per_epoch=round(sum(epoch_times) / len(epoch_times), 2),
                   peak_mem_gb=round(pa, 3), peak_reserved_gb=round(pr, 3),
                   ckpt_size_mb=round(weights_size_mb(out_dir), 2), status="ok")
        code = 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        rec.update(status="error", error=str(e).splitlines()[0][:300], error_trace=traceback.format_exc())
        code = 1
    rec["wall_time_s"] = round(time.time() - t_wall, 1)
    write_result(args.exp, run, rec)
    sys.exit(code)


if __name__ == "__main__":
    main()
