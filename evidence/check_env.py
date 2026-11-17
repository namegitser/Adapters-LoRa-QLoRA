#!/usr/bin/env python
"""Sanity check on a GPU node: versions, GPU, bf16, bitsandbytes 4-bit, cached assets."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ALPACA, SST2, WIKITEXT, env_info  # noqa: E402

ok = True
print(json.dumps(env_info(), indent=2))

if not torch.cuda.is_available():
    print("FAIL: no CUDA device visible (did the job get a GPU? check --gres)")
    sys.exit(1)
print("bf16 supported:", torch.cuda.is_bf16_supported(),
      "(if False, scripts fall back to fp16 automatically)")

try:
    import bitsandbytes as bnb
    lin = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.float16, quant_type="nf4").cuda()
    y = lin(torch.randn(2, 64, device="cuda", dtype=torch.float16))
    print("bitsandbytes 4-bit NF4 matmul: OK", tuple(y.shape))
except Exception as e:  # noqa: BLE001
    ok = False
    print("FAIL: bitsandbytes 4-bit test:", repr(e))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

for m in ["Qwen/Qwen2.5-0.5B", "distilbert/distilbert-base-uncased", "Qwen/Qwen2.5-3B"]:
    try:
        AutoTokenizer.from_pretrained(m)
        print(f"cached model   OK   {m}")
    except Exception as e:  # noqa: BLE001
        tag = "(only needed for H2)" if "3B" in m else ""
        ok = ok and "3B" in m
        print(f"cached model   MISSING {m} {tag}: {type(e).__name__}")

for name, cfg in [(ALPACA, None), SST2, WIKITEXT]:
    try:
        load_dataset(name, cfg)
        print(f"cached dataset OK   {name} {cfg or ''}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"cached dataset MISSING {name} {cfg or ''}: {type(e).__name__}")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED (see above)")
sys.exit(0 if ok else 1)
