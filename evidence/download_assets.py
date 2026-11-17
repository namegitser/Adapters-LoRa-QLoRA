#!/usr/bin/env python
"""Pre-download every model and dataset H1-H4 need into $HF_HOME (run on the login node)."""
import argparse
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

MODELS = ["Qwen/Qwen2.5-0.5B", "distilbert/distilbert-base-uncased"]   # H1/H4, H3
H2_MODELS = ["Qwen/Qwen2.5-3B"]                              # H2
DATASETS = [
    ("yahma/alpaca-cleaned", None),          # H1, H2
    ("nyu-mll/glue", "sst2"),                # H3
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),  # H4
]
IGNORE = ["*.bin", "*.h5", "*.msgpack", "*.ot", "*.onnx", "flax_*", "tf_*", "*.tflite"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-h2", action="store_true", help="also fetch Qwen2.5-3B (~6 GB) for H2")
    ap.add_argument("--extra", nargs="*", default=[], help="more model ids, e.g. Qwen/Qwen2.5-1.5B")
    args = ap.parse_args()

    print("HF_HOME =", os.environ.get("HF_HOME", "~/.cache/huggingface"))
    for m in MODELS + (H2_MODELS if args.with_h2 else []) + args.extra:
        print(f"[model]   {m}")
        snapshot_download(m, ignore_patterns=IGNORE)
    for name, cfg in DATASETS:
        print(f"[dataset] {name} {cfg or ''}")
        load_dataset(name, cfg)
    print("All assets cached.")


if __name__ == "__main__":
    main()
