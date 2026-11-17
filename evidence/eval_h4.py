#!/usr/bin/env python
"""H4: "forgets less?" Perplexity on WikiText-2 (test) for the base model and each H1 model.

Variants (each -> results/h4/<variant>.json):
  base      H1 base model in bf16                 (reference for full + lora)
  base4bit  H1 base model in 4-bit NF4            (reference for qlora: isolates the quantization effect)
  full      H1 full fine-tune checkpoint
  lora      base bf16 + H1 LoRA adapter (merged)
  qlora     base 4-bit + H1 QLoRA adapter
Sliding-window perplexity (window 1024, stride 512), the standard Hugging Face recipe.
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
import time
import traceback
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT_ROOT, WIKITEXT, dtype_name, env_info, peak_gb, pick_amp_dtype,  # noqa: E402
                    reset_peak, write_result)

ALL = ["base", "base4bit", "full", "lora", "qlora"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B", help="must be the model H1 trained")
    p.add_argument("--variants", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--h1-dir", default=str(CKPT_ROOT / "h1"), help="where H1 saved its checkpoints")
    p.add_argument("--seed", type=int, default=42, help="which H1 seed's checkpoints to evaluate")
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=0, help="0 = whole test set")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--exp", default="h4")
    return p.parse_args()


def load_variant(v, args, amp):
    from peft import PeftModel
    gpu = {"": 0}
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=amp)
    ck = Path(args.h1_dir)
    if v == "base":
        return AutoModelForCausalLM.from_pretrained(args.base, dtype=amp, device_map=gpu), args.base
    if v == "base4bit":
        return AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, dtype=amp,
                                                    device_map=gpu), args.base + " (nf4)"
    path = ck / f"{v}_seed{args.seed}"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - did H1 finish with saving enabled?")
    if v == "full":
        return AutoModelForCausalLM.from_pretrained(path, dtype=amp, device_map=gpu), str(path)
    if v == "lora":
        m = AutoModelForCausalLM.from_pretrained(args.base, dtype=amp, device_map=gpu)
        return PeftModel.from_pretrained(m, path).merge_and_unload(), str(path)
    m = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, dtype=amp, device_map=gpu)
    return PeftModel.from_pretrained(m, path), str(path)


@torch.no_grad()
def perplexity(model, ids, window, stride, amp):
    model.eval()
    seq_len = ids.size(1)
    nll_sum, n_tok, prev_end = 0.0, 0, 0
    for begin in range(0, seq_len, stride):
        end = min(begin + window, seq_len)
        trg = end - prev_end  # only score tokens not scored by the previous window
        x = ids[:, begin:end].cuda()
        y = x.clone()
        y[:, :-trg] = -100
        with torch.autocast("cuda", dtype=amp):
            loss = model(input_ids=x, labels=y).loss
        k = int((y[:, 1:] != -100).sum())
        nll_sum += float(loss) * k
        n_tok += k
        prev_end = end
        if end == seq_len:
            break
    return math.exp(nll_sum / n_tok), nll_sum / n_tok, n_tok


def main():
    args = parse_args()
    amp = pick_amp_dtype(args.amp)
    tok = AutoTokenizer.from_pretrained(args.base)
    test = load_dataset(*WIKITEXT, split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    if args.max_tokens:
        ids = ids[:, : args.max_tokens]
    print(f"WikiText-2 test: {ids.size(1):,} tokens", flush=True)

    failures = 0
    for v in args.variants:
        rec = {"exp": args.exp, "run": v, "variant": v, "base_model": args.base, "window": args.window,
               "stride": args.stride, "amp_dtype": dtype_name(amp), "h1_seed": args.seed, "env": env_info()}
        t0 = time.time()
        model = None
        try:
            reset_peak()
            model, src = load_variant(v, args, amp)
            ppl, nll, n = perplexity(model, ids, args.window, args.stride, amp)
            pa, _ = peak_gb()
            rec.update(source=src, wikitext2_ppl=round(ppl, 4), wikitext2_nll=round(nll, 5),
                       n_tokens_scored=n, peak_mem_gb=round(pa, 3), status="ok")
            print(f"[{v}] WikiText-2 perplexity = {ppl:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            failures += 1
            rec.update(status="error", error=str(e).splitlines()[0][:300])
        rec["wall_time_s"] = round(time.time() - t0, 1)
        write_result(args.exp, v, rec)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
