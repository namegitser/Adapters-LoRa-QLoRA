#!/usr/bin/env python
"""Collect results/h*/ JSON files into slide-ready tables (Markdown + CSV) and PNG charts.

Runs on CPU (login node or laptop):  python make_report.py
Output: results/summary/  (h1_table.md, h1_memory.png, ..., footer.txt)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "summary"
# Fixed identity colours (same method = same colour on every chart)
COLOR = {"full": "#2a78d6", "lora": "#eb6834", "qlora": "#1baf7a"}
LABEL = {"full": "Full FT", "lora": "LoRA", "qlora": "QLoRA"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
ORDER = ["full", "lora", "qlora"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 13,
})


def load(exp: str) -> pd.DataFrame:
    rows = []
    for f in sorted((RESULTS_DIR / exp).glob("*.json")):
        r = json.loads(f.read_text())
        env = r.pop("env", {}) or {}
        r.pop("error_trace", None)
        rows.append({**r, **{f"env_{k}": v for k, v in env.items()}})
    return pd.DataFrame(rows)


def fmt_params(n):
    if pd.isna(n):
        return "-"
    n = float(n)
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}K"


def fmt_size(mb):
    if pd.isna(mb):
        return "-"
    return f"{mb / 1e3:.2f} GB" if mb >= 1000 else f"{mb:.1f} MB"


def num(x, d=2):
    return "-" if x is None or pd.isna(x) else f"{x:.{d}f}"


def save_table(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(df.to_markdown(index=False))
    print(f"\n## {name}\n{df.to_markdown(index=False)}")


def bar(ax, labels, values, colors, texts):
    xs = range(len(labels))
    ax.bar(xs, values, color=colors, width=0.55, edgecolor=SURFACE, linewidth=2)
    for x, v, t in zip(xs, values, texts):
        ax.annotate(t, (x, v), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    color=INK, fontsize=10)
    ax.set_xticks(list(xs), labels)
    ax.grid(axis="x", visible=False)


def gpu_line(df):
    return f"1x {df['env_gpu_name'].dropna().iloc[0]} ({df['env_gpu_mem_gb'].dropna().iloc[0]} GB)" \
        if "env_gpu_name" in df and df["env_gpu_name"].notna().any() else "GPU ?"


# ----------------------------------------------------------------------------- H1
def report_h1():
    df = load("h1")
    if df.empty:
        return print("H1: no results yet")
    df = df[df.status == "ok"]
    agg = df.groupby("method").agg(
        trainable=("trainable_params", "first"), total=("total_params", "first"),
        pct=("trainable_pct", "first"), mem=("peak_mem_gb", "mean"), sps=("sec_per_step", "mean"),
        tps=("tokens_per_sec", "mean"), before=("eval_loss_before", "mean"),
        after=("eval_loss_after", "mean"), size=("ckpt_size_mb", "mean"), n=("seed", "count"),
    ).reindex([m for m in ORDER if m in set(df.method)])
    table = pd.DataFrame({
        "Method": [LABEL[m] for m in agg.index],
        "Trainable params": [fmt_params(v) for v in agg.trainable],
        "% of total": [f"{v:.2f}%" for v in agg.pct],
        "Peak GPU mem (GB)": [num(v) for v in agg.mem],
        "Sec / step": [num(v, 3) for v in agg.sps],
        "Tokens / s": [num(v, 0) for v in agg.tps],
        "Held-out loss (before -> after)": [f"{num(b, 3)} -> {num(a, 3)}" for b, a in zip(agg.before, agg.after)],
        "Checkpoint size": [fmt_size(v) for v in agg["size"]],
        "Seeds": agg.n.tolist(),
    })
    save_table(table, "h1_table")

    # derived ratios for the talk track
    if {"full", "lora"} <= set(agg.index):
        lines = [f"LoRA trains {agg.trainable['full'] / agg.trainable['lora']:.0f}x fewer params than full FT",
                 f"LoRA peak memory = {agg.mem['lora'] / agg.mem['full']:.2f}x of full FT "
                 f"({agg.mem['full'] / agg.mem['lora']:.1f}x less)",
                 f"LoRA checkpoint = {agg['size']['full'] / agg['size']['lora']:.0f}x smaller"]
        if "qlora" in agg.index:
            lines += [f"QLoRA peak memory = {agg.mem['full'] / agg.mem['qlora']:.1f}x less than full FT, "
                      f"{agg.mem['lora'] / agg.mem['qlora']:.2f}x less than LoRA",
                      f"QLoRA sec/step = {agg.sps['qlora'] / agg.sps['lora']:.2f}x LoRA's (>1 means slower)"]
        (OUT / "h1_ratios.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))

    ms = list(agg.index)
    cols = [COLOR[m] for m in ms]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bar(ax, [LABEL[m] for m in ms], agg.mem, cols, [f"{v:.1f} GB\nloss {a:.3f}" for v, a in zip(agg.mem, agg.after)])
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title(f"H1 - {df.model.iloc[0].split('/')[-1]}, {gpu_line(df)}", loc="left")
    ax.set_ylim(0, agg.mem.max() * 1.25)
    fig.tight_layout(); fig.savefig(OUT / "h1_memory.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    bar(ax, [LABEL[m] for m in ms], agg.sps, cols, [f"{v:.2f} s" for v in agg.sps])
    ax.set_ylabel("Seconds per optimizer step (lower is faster)")
    ax.set_title("H1 - training speed", loc="left")
    ax.set_ylim(0, agg.sps.max() * 1.2)
    fig.tight_layout(); fig.savefig(OUT / "h1_speed.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m in ms:
        seed = df[df.method == m].seed.iloc[0]
        c = RESULTS_DIR / "h1" / f"{m}_seed{seed}_curve.csv"
        if c.exists():
            cur = pd.read_csv(c)
            ax.plot(cur.step, cur.train_loss.rolling(10, min_periods=1).mean(), color=COLOR[m], lw=2, label=LABEL[m])
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Train loss (10-step mean)")
    ax.set_title("H1 - training loss", loc="left"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "h1_loss_curves.png", dpi=200); plt.close(fig)
    return df


# ----------------------------------------------------------------------------- H2
def report_h2():
    df = load("h2")
    if df.empty:
        return print("H2: no results yet")
    model_name = df.model.iloc[0].split("/")[-1]
    df = df.set_index("method").reindex([m for m in ORDER if m in set(df.method)])
    gpu_gb = df["env_gpu_mem_gb"].dropna().iloc[0]
    def outcome(r):
        if r.status == "OOM":
            return f"OOM at {str(r.failed_stage).replace('_', ' ')}"
        return "trains" if r.status == "ok" else r.status
    table = pd.DataFrame({
        f"Method ({model_name})": [LABEL[m] for m in df.index],
        "Outcome": [outcome(r) for _, r in df.iterrows()],
        "Trainable params": [fmt_params(v) for v in df.get("trainable_params", pd.Series(index=df.index))],
        "Weights on GPU (GB)": [num(v) for v in df.get("mem_after_load_gb", pd.Series(index=df.index))],
        "Peak GPU mem (GB)": [num(v) if r.status == "ok" else f"OOM (> {gpu_gb:.0f} GB)" for v, (_, r)
                              in zip(df.peak_mem_gb, df.iterrows())],
        "Sec / step": [num(v, 2) for v in df.get("sec_per_step", pd.Series(index=df.index))],
    })
    save_table(table, "h2_table")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ms = list(df.index)
    vals = [gpu_gb if df.loc[m, "status"] == "OOM" else df.loc[m, "peak_mem_gb"] for m in ms]
    bars = ax.bar(range(len(ms)), vals, color=[COLOR[m] for m in ms], width=0.55, edgecolor=SURFACE, linewidth=2)
    for b, m, v in zip(bars, ms, vals):
        oom = df.loc[m, "status"] == "OOM"
        if oom:
            b.set_hatch("//"); b.set_alpha(0.45)
        ax.annotate("OOM" if oom else f"{v:.1f} GB", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK, fontsize=10)
    ax.axhline(gpu_gb, color=MUTED, ls="--", lw=1)
    ax.annotate(f"GPU capacity {gpu_gb:.0f} GB", (len(ms) - 1, gpu_gb), xytext=(0, 4),
                textcoords="offset points", ha="center", color=MUTED, fontsize=9)
    ax.set_xticks(range(len(ms)), [LABEL[m] for m in ms]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Peak GPU memory (GB)"); ax.set_ylim(0, gpu_gb * 1.15)
    ax.set_title(f"H2 - {model_name} on {gpu_line(df.reset_index())}", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "h2_memory.png", dpi=200); plt.close(fig)
    return df.reset_index()


# ----------------------------------------------------------------------------- H3
def report_h3():
    df = load("h3")
    if df.empty:
        return print("H3: no results yet")
    df = df[df.status == "ok"]
    lo = df[df.method == "lora"].groupby("lora_r").agg(
        acc_mean=("val_acc", "mean"), acc_min=("val_acc", "min"), acc_max=("val_acc", "max"),
        lora=("lora_params", "first"), analytic=("lora_params_analytic", "first"),
        match=("analytic_match", "all"), trainable=("trainable_params", "first"),
        pct=("trainable_pct", "first"), size=("ckpt_size_mb", "first"), n=("seed", "count")).sort_index()
    full = df[df.method == "full"]
    table = pd.DataFrame({
        "Rank r": [int(r) for r in lo.index],
        "LoRA params (measured)": [f"{int(v):,}" for v in lo.lora],
        "r(d+k) analytic": [f"{int(v):,}" for v in lo.analytic],
        "Match": ["yes" if m else "NO" for m in lo.match],
        "Trainable incl. head": [f"{int(v):,} ({p:.2f}%)" for v, p in zip(lo.trainable, lo.pct)],
        "Val acc (mean, min-max)": [f"{a * 100:.2f}% ({lo_ * 100:.2f}-{hi * 100:.2f})"
                                    for a, lo_, hi in zip(lo.acc_mean, lo.acc_min, lo.acc_max)],
        "Seeds": lo.n.tolist(),
    })
    if not full.empty:
        table.loc[len(table)] = ["full FT", "-", "-", "-",
                                 f"{int(full.trainable_params.iloc[0]):,} (100%)",
                                 f"{full.val_acc.mean() * 100:.2f}% ({full.val_acc.min() * 100:.2f}-"
                                 f"{full.val_acc.max() * 100:.2f})", len(full)]
    save_table(table, "h3_table")

    # two separate single-axis charts (never dual-axis)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = list(lo.index)
    ax.plot(x, lo.acc_mean * 100, color=COLOR["lora"], lw=2, marker="o", ms=7, label="LoRA (mean of seeds)")
    ax.fill_between(x, lo.acc_min * 100, lo.acc_max * 100, color=COLOR["lora"], alpha=0.15, lw=0)
    if not full.empty:
        ax.axhline(full.val_acc.mean() * 100, color=COLOR["full"], ls="--", lw=2, label="Full fine-tune")
    ax.set_xscale("log", base=2); ax.set_xticks(x, [str(int(r)) for r in x])
    ax.set_xlabel("LoRA rank r"); ax.set_ylabel("SST-2 validation accuracy (%)")
    ax.set_title("H3 - accuracy vs rank (DistilBERT, SST-2)", loc="left"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "h3_acc_vs_rank.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(x, lo.lora, color=COLOR["lora"], lw=2, marker="o", ms=7)
    for r, v in zip(x, lo.lora):
        ax.annotate(fmt_params(v), (r, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(x, [str(int(r)) for r in x])
    ax.set_xlabel("LoRA rank r"); ax.set_ylabel("Trainable LoRA params (log)")
    ax.set_title("H3 - LoRA params grow linearly with r", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "h3_params_vs_rank.png", dpi=200); plt.close(fig)
    return df


# ----------------------------------------------------------------------------- H4
def report_h4(h1: pd.DataFrame | None):
    df = load("h4")
    if df.empty:
        return print("H4: no results yet")
    ppl = df[df.status == "ok"].set_index("variant").wikitext2_ppl
    ref = {"full": "base", "lora": "base", "qlora": "base4bit"}
    held = {}
    if h1 is not None and not h1.empty:
        h = h1[h1.seed == df.h1_seed.iloc[0]].set_index("method")
        held = h.eval_loss_after.to_dict()
        # untrained references: the loss each base had before training (bf16 for full, 4-bit for qlora)
        if "full" in h.index:
            held["base"] = h.loc["full", "eval_loss_before"]
        if "qlora" in h.index:
            held["base4bit"] = h.loc["qlora", "eval_loss_before"]
    rows = []
    for v in ["base", "base4bit", "full", "lora", "qlora"]:
        if v not in ppl:
            continue
        r = ref.get(v)
        delta = f"{ppl[v] - ppl[r]:+.3f} ({100 * (ppl[v] / ppl[r] - 1):+.1f}%)" if r in ppl else "-"
        rows.append({"Model": {"base": "Base (bf16)", "base4bit": "Base (4-bit NF4)"}.get(v, LABEL.get(v, v)),
                     "WikiText-2 ppl (forgetting)": f"{ppl[v]:.3f}",
                     "Change vs its base": delta if r else "reference",
                     "Alpaca held-out loss (learning)": num(held.get(v), 3)})
    save_table(pd.DataFrame(rows), "h4_table")

    ms = [m for m in ORDER if m in ppl and ref[m] in ppl]
    if ms:
        rel = [100 * (ppl[m] / ppl[ref[m]] - 1) for m in ms]
        fig, ax = plt.subplots(figsize=(6.4, 4))
        bar(ax, [LABEL[m] for m in ms], rel, [COLOR[m] for m in ms], [f"{v:+.1f}%" for v in rel])
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_ylabel("WikiText-2 perplexity change vs base (%)")
        ax.set_title("H4 - how much general text ability moved", loc="left")
        lo_, hi = min(0, min(rel)), max(0, max(rel))
        pad = (hi - lo_) * 0.2 or 1
        ax.set_ylim(lo_ - pad, hi + pad)
        fig.tight_layout(); fig.savefig(OUT / "h4_forgetting.png", dpi=200); plt.close(fig)


def footer(frames):
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return
    d = pd.concat(frames, ignore_index=True)
    first = lambda c: d[c].dropna().iloc[0] if c in d and d[c].notna().any() else "?"  # noqa: E731
    jobs = ", ".join(sorted({str(j) for j in d.get("env_slurm_job_id", pd.Series()).dropna()}))
    txt = (f"Measured on [cluster], 1x {first('env_gpu_name')} ({first('env_gpu_mem_gb')} GB), "
           f"PyTorch {first('env_torch')}, Transformers {first('env_transformers')}, PEFT {first('env_peft')}, "
           f"bitsandbytes {first('env_bitsandbytes')}, seed 42, SLURM job IDs {jobs}.")
    (OUT / "footer.txt").write_text(txt + "\n")
    print("\nSlide footer:\n" + txt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h1 = report_h1()
    h2 = report_h2()
    h3 = report_h3()
    report_h4(h1)
    footer([h1, h2, h3])
    print(f"\nEverything written to {OUT}")


if __name__ == "__main__":
    main()
