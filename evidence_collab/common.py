"""Shared helpers for the HPC evidence scripts (H1-H4)."""
from __future__ import annotations

import json
import os
import platform
import random
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

EVIDENCE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", EVIDENCE_DIR / "results"))
CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", EVIDENCE_DIR / "checkpoints"))

ALPACA = "yahma/alpaca-cleaned"
SST2 = ("nyu-mll/glue", "sst2")
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1")
GIB = 1024 ** 3  # GPU memory is reported in GiB ("GB" on slides), matching how GPUs are labelled


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_amp_dtype(name: str) -> torch.dtype:
    """bf16 on Ampere+ (A100/A40/L40/H100), fp16 on older GPUs (V100/T4)."""
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def env_info() -> dict:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    for pkg in ("transformers", "peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = p.name
        info["gpu_mem_gb"] = round(p.total_memory / GIB, 1)
    return info


def count_params(model) -> tuple[int, int]:
    """(trainable, total). PEFT's helper counts 4-bit packed weights correctly."""
    if hasattr(model, "get_nb_trainable_parameters"):
        return model.get_nb_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_only_params(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def mem_now_gb() -> float:
    return torch.cuda.memory_allocated() / GIB if torch.cuda.is_available() else 0.0


def peak_gb() -> tuple[float, float]:
    """(max allocated by tensors, max reserved by the caching allocator), in GB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.max_memory_allocated() / GIB, torch.cuda.max_memory_reserved() / GIB


def weights_size_mb(path: Path) -> float:
    """Size of the weight files only (*.safetensors / *.bin), not tokenizer/config."""
    files = [f for f in Path(path).rglob("*") if f.is_file() and f.suffix in (".safetensors", ".bin")]
    return sum(f.stat().st_size for f in files) / 1e6


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def write_result(exp: str, name: str, record: dict) -> Path:
    out = RESULTS_DIR / exp / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str))
    short = {k: v for k, v in record.items() if k not in ("env", "error_trace")}
    print("RESULT " + json.dumps(short, default=str), flush=True)
    print(f"[saved] {out}", flush=True)
    return out
