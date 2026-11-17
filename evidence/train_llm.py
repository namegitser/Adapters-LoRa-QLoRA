#!/usr/bin/env python
"""H1 / H2: full fine-tuning vs LoRA vs QLoRA on a causal LM (Alpaca-cleaned).

One invocation = one run = one JSON file in results/<exp>/<run>.json
(+ a per-step loss curve CSV next to it).

Out-of-memory is treated as a RESULT, not a crash: it is caught, recorded with the
stage where it happened, and the script exits 0 so the next method still runs.

Precision per method (same for H1 and H2):
  full  : fp32 master weights + bf16 autocast + AdamW  (classic mixed precision, ~16 bytes/param)
  lora  : frozen base in bf16, LoRA A/B in fp32, bf16 autocast
  qlora : frozen base in 4-bit NF4 (+ double quant), LoRA A/B in fp32, bf16 compute
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          get_cosine_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ALPACA, CKPT_ROOT, RESULTS_DIR, count_params, dtype_name, env_info,  # noqa: E402
                    is_oom, lora_only_params, mem_now_gb, peak_gb, pick_amp_dtype,
                    reset_peak, set_seed, weights_size_mb, write_result)

DATA_SEED = 1234  # fixed: every method/seed sees the same train/held-out split

PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further "
    "context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["full", "lora", "qlora"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--exp", default="h1", help="results subfolder (h1 / h2)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--steps", type=int, default=300, help="optimizer steps")
    p.add_argument("--n-examples", type=int, default=2000, help="total examples drawn from Alpaca")
    p.add_argument("--n-eval", type=int, default=200, help="held out from n-examples")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-5 full, 2e-4 LoRA/QLoRA")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false",
                   help="disable gradient checkpointing (on by default for ALL methods)")
    p.add_argument("--no-save", dest="save", action="store_false")
    p.add_argument("--skip-eval-before", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--timing-skip", type=int, default=5, help="first N steps excluded from speed")
    return p.parse_args()


# ----------------------------------------------------------------------------- data
def build_data(tok, args):
    ds = load_dataset(ALPACA, split="train").shuffle(seed=DATA_SEED).select(range(args.n_examples))

    def encode(ex):
        tmpl = PROMPT_INPUT if ex["input"].strip() else PROMPT_NO_INPUT
        p_ids = tok(tmpl.format(**ex), add_special_tokens=False)["input_ids"]
        r_ids = tok(ex["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + r_ids)[: args.max_len]
        labels = ([-100] * len(p_ids) + r_ids)[: args.max_len]   # loss on the response only
        return {"input_ids": ids, "labels": labels}

    ds = ds.map(encode, remove_columns=ds.column_names)
    ds = ds.filter(lambda ex: any(l != -100 for l in ex["labels"]))  # drop prompt-only truncations
    n_eval = min(args.n_eval, len(ds) // 5)
    train = ds.select(range(len(ds) - n_eval))
    held = ds.select(range(len(ds) - n_eval, len(ds)))
    return train, held


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        attn[i, :n] = 1
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def forever(loader):
    while True:
        yield from loader


# ----------------------------------------------------------------------------- model
def load_model(args, amp):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    gpu = {"": 0}
    if args.method == "full":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, device_map=gpu)
    elif args.method == "lora":
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=amp, device_map=gpu)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=amp)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, dtype=amp, device_map=gpu)
    model.config.use_cache = False

    gc_kwargs = {"use_reentrant": False}
    if args.method == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.grad_ckpt, gradient_checkpointing_kwargs=gc_kwargs)
    elif args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

    if args.method in ("lora", "qlora"):
        if args.grad_ckpt:
            model.enable_input_require_grads()
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                         target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    return model


@torch.no_grad()
def evaluate(model, loader, amp):
    """Token-weighted mean cross-entropy on the held-out responses."""
    model.eval()
    total, n = 0.0, 0
    for b in loader:
        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", dtype=amp):
            out = model(**b)
        k = int((b["labels"][:, 1:] != -100).sum())
        total += float(out.loss) * k
        n += k
    model.train()
    return total / max(n, 1)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    set_seed(args.seed)
    amp = pick_amp_dtype(args.amp)
    lr = args.lr if args.lr is not None else (1e-5 if args.method == "full" else 2e-4)
    run = args.run_name or f"{args.method}_seed{args.seed}"
    is_peft = args.method != "full"

    rec = {
        "exp": args.exp, "run": run, "method": args.method, "model": args.model, "status": "started",
        "seed": args.seed, "steps": args.steps, "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum, "effective_batch": args.micro_batch * args.grad_accum,
        "max_len": args.max_len, "lr": lr, "amp_dtype": dtype_name(amp), "grad_ckpt": args.grad_ckpt,
        "lora_r": args.lora_r if is_peft else None, "lora_alpha": args.lora_alpha if is_peft else None,
        "lora_targets": "all-linear" if is_peft else None, "env": env_info(),
    }
    print(f"=== {args.exp}/{run}: {args.method} on {args.model} | amp={rec['amp_dtype']} "
          f"| GPU={rec['env'].get('gpu_name')} ({rec['env'].get('gpu_mem_gb')} GB)", flush=True)

    t_wall = time.time()
    stage, steps_done, curve = "load_data", 0, []
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        train_ds, held_ds = build_data(tok, args)
        rec.update(n_train=len(train_ds), n_eval=len(held_ds))

        stage = "load_model"
        reset_peak()
        model = load_model(args, amp)
        rec["mem_after_load_gb"] = round(mem_now_gb(), 3)
        trainable, total = count_params(model)
        rec.update(trainable_params=trainable, total_params=total,
                   trainable_pct=round(100 * trainable / total, 4),
                   lora_params=lora_only_params(model) if is_peft else None)
        if is_peft:
            model.print_trainable_parameters()
        print(f"trainable={trainable:,} total={total:,} ({rec['trainable_pct']}%) "
              f"| weights on GPU={rec['mem_after_load_gb']} GB", flush=True)

        g = torch.Generator().manual_seed(args.seed)
        coll = lambda b: collate(b, tok.pad_token_id)  # noqa: E731
        train_loader = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True,
                                  generator=g, collate_fn=coll, num_workers=2, pin_memory=True)
        held_loader = DataLoader(held_ds, batch_size=args.micro_batch, shuffle=False,
                                 collate_fn=coll, num_workers=2)

        if not args.skip_eval_before:
            stage = "eval_before"
            rec["eval_loss_before"] = round(evaluate(model, held_loader, amp), 4)
            print(f"held-out loss before training: {rec['eval_loss_before']}", flush=True)

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))

        model.train()
        it = forever(train_loader)
        reset_peak()  # peak from here on = weights + grads + optimizer states + activations
        timed_s, timed_tok = 0.0, 0
        for step in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            stage = f"train_step{step}_forward_backward"
            step_loss, step_tok = 0.0, 0
            for _ in range(args.grad_accum):
                b = {k: v.cuda(non_blocking=True) for k, v in next(it).items()}
                with torch.autocast("cuda", dtype=amp):
                    loss = model(**b).loss / args.grad_accum
                scaler.scale(loss).backward()
                step_loss += float(loss)
                step_tok += int(b["attention_mask"].sum())
            stage = f"train_step{step}_optimizer"
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            steps_done = step + 1
            if step >= args.timing_skip:
                timed_s += dt
                timed_tok += step_tok
            curve.append((steps_done, round(step_loss, 5), round(dt, 4)))
            if steps_done % args.log_every == 0 or steps_done == 1:
                pa, _ = peak_gb()
                print(f"step {steps_done:4d}/{args.steps} loss {step_loss:.4f} {dt:.3f}s/step "
                      f"peak {pa:.2f} GB", flush=True)

        pa, pr = peak_gb()
        n_timed = max(steps_done - args.timing_skip, 1)
        rec.update(peak_mem_gb=round(pa, 3), peak_reserved_gb=round(pr, 3),
                   sec_per_step=round(timed_s / n_timed, 4),
                   tokens_per_sec=round(timed_tok / timed_s, 1) if timed_s else None,
                   train_loss_last20=round(sum(c[1] for c in curve[-20:]) / len(curve[-20:]), 4),
                   steps_done=steps_done)

        stage = "eval_after"
        rec["eval_loss_after"] = round(evaluate(model, held_loader, amp), 4)
        rec["eval_ppl_after"] = round(math.exp(rec["eval_loss_after"]), 3)
        print(f"held-out loss after training: {rec['eval_loss_after']}", flush=True)

        if args.save:
            stage = "save"
            out_dir = CKPT_ROOT / args.exp / run
            out_dir.mkdir(parents=True, exist_ok=True)
            del opt
            if args.method == "full":
                model = model.to(torch.bfloat16)  # ship in 16-bit, like any released model
                model.save_pretrained(out_dir, safe_serialization=True)
                tok.save_pretrained(out_dir)
            else:
                model.save_pretrained(out_dir)  # adapter weights only
            rec["ckpt_dir"] = str(out_dir)
            rec["ckpt_size_mb"] = round(weights_size_mb(out_dir), 1)
        rec["status"] = "ok"
        exit_code = 0
    except Exception as e:  # noqa: BLE001
        pa, pr = peak_gb()
        rec.update(peak_mem_gb=round(pa, 3), peak_reserved_gb=round(pr, 3), steps_done=steps_done,
                   failed_stage=stage, error=str(e).splitlines()[0][:300],
                   error_trace=traceback.format_exc())
        traceback.print_exc()
        if is_oom(e):
            rec["status"] = "OOM"
            print(f"\n*** OUT OF MEMORY at stage '{stage}' after {steps_done} steps; "
                  f"peak allocated {pa:.2f} GB of {rec['env'].get('gpu_mem_gb')} GB ***\n", flush=True)
            exit_code = 0  # OOM is a measured outcome (H2)
        else:
            rec["status"] = "error"
            exit_code = 1

    rec["wall_time_s"] = round(time.time() - t_wall, 1)
    if curve:
        cpath = RESULTS_DIR / args.exp / f"{run}_curve.csv"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with open(cpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "sec"])
            w.writerows(curve)
    write_result(args.exp, run, rec)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
