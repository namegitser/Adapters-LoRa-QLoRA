# Evidence flow: H1–H4 on the HPC (SLURM)

How to produce the slide 4 / slide 5 evidence from `plan.md` §4b on the cluster.
Every number that goes on a slide must come from a file in `results/` that these jobs write.

| Exp | Question it answers | Feeds | GPU time | Priority |
|---|---|---|---|---|
| **H1** | Full FT vs LoRA vs QLoRA on the same small LLM (Qwen2.5-0.5B): params, memory, speed, quality | Slide 4 (main table + chart) | ~20–40 min | ⭐ must |
| **H2** | Does a 3B model fit one 24 GB GPU? (full FT OOM vs QLoRA trains) | Slide 4 "wow" line | ~20–30 min | extra |
| **H3** | How small can the rank r be? (accuracy vs r, DistilBERT/SST-2) | Slide 4 or Q&A on "how to choose r" | ~1–1.5 h (14 short runs) | extra |
| **H4** | Does LoRA forget less than full FT? (WikiText-2 perplexity) | Slide 5 "honest weakness" | ~15–25 min | optional, needs H1 |

### Our cluster: Plaksha HPC (`plakshahpc`), general access
From the [slurm-builder guide](https://kestrel-lab.github.io/slurm-builder/) (snapshot 2026-08-19); live `sacctmgr`/`sinfo` output wins if they differ.

| Limit (account `general`, QoS `general`) | Value | What it means for us |
|---|---|---|
| Partition | `rtx` only (`h200` is faculty/paid only) | `PARTITION=rtx` |
| GPUs | slices: `gpu:2g.48gb` (2 on cn02), `gpu:1g.24gb` (4 on cn03) | everything runs on **24 GB** slices (4 of them → shorter queue); 48 GB only if you scale H1 up to 1.5B |
| GPUs per user | **1** | only one of our jobs runs at a time; the rest wait in the queue (`QOSMaxGRESPerUser`) |
| Running / submitted jobs per user | 2 / 10, DenyOnLimit | never submit more than 10 jobs; no job arrays (each task counts) |
| Max walltime | 12 h | all our jobs ask for ≤ 4 h |
| Preemption | course/faculty/rfphd/paid jobs can preempt general jobs after 10 min; the job is **requeued** | jobs restart from the top and **skip runs that already finished** (`already_done` in `job_env.sh`); logs are appended |

Ordering with 1 GPU: run **H1 first** (it is the slide 4 table), then H4, H2, H3. One person's account runs one job at a time, so if teammates also have general access, split jobs across accounts (e.g. one person runs H1 → H4, another runs H2, a third H3) and copy the `results/` folders together before `make_report.py`.

---

## 0. Folder map

```
evidence/
├── flow.md                 ← this file
├── slurm/
│   ├── cluster.env         ← THE ONLY FILE TO EDIT (partition, GPU type, modules, paths)
│   ├── job_env.sh          ← shared job preamble (modules, venv, offline mode, nvidia-smi header)
│   ├── submit.sh           ← submits jobs with cluster.env settings
│   ├── check.sbatch        ← 15-min sanity job
│   ├── h1.sbatch  h2.sbatch  h3.sbatch  h4.sbatch
├── setup_env.sh            ← one-time: venv + packages + downloads (login node)
├── download_assets.py      ← pre-downloads models/datasets into $HF_HOME
├── check_env.py            ← GPU / bf16 / bitsandbytes / cache check
├── common.py               ← shared helpers (memory, param counting, result JSON)
├── train_llm.py            ← H1 + H2 trainer (full | lora | qlora)
├── train_h3.py             ← H3 rank-sweep trainer
├── eval_h4.py              ← H4 perplexity evaluator
├── make_report.py          ← results/*.json → tables (.md/.csv) + charts (.png) + slide footer
├── logs/                   ← SLURM stdout (screenshots come from here)
└── results/
    ├── h1/ h2/ h3/ h4/     ← one JSON per run (+ loss-curve CSVs for H1/H2)
    └── summary/            ← what goes on the slides
```

Checkpoints go to `$CKPT_ROOT` (outside this folder, since they are GBs).

---

## 1. One-time setup (do this before any H)

### 1.1 Copy the folder to the cluster
```bash
# from the laptop (Git Bash / WSL), adjust user@host
scp -r evidence  user@hpc-login:~/peft-evidence
ssh user@hpc-login
cd ~/peft-evidence
```
If you edited files on Windows and a job fails with `$'\r': command not found`, run
`dos2unix slurm/* *.sh` (or `sed -i 's/\r$//' slurm/* *.sh`).

### 1.2 Learn the cluster (5 minutes), then edit `slurm/cluster.env`
`cluster.env` is already filled in for Plaksha general access (partition, account, QoS, GPU slices).
Still unknown from the guide, so check these on the login node: **module names**, **storage/scratch paths and quotas**, **internet on compute nodes**, and the **driver version** (`cu128` torch needs ≥ 570).
```bash
sacctmgr show user "$USER" withassoc format=User,Account,Partition,QOS,DefaultQOS   # confirm general access
sinfo -p rtx -o "%P %N %G %T"         # GPU slices and whether they are free
sinfo -s                              # partition names            → PARTITION
sinfo -o "%P %G" | grep -i gpu        # GPU types per partition    → GRES / GRES_BIG
module avail cuda; module avail python   # module names            → MODULES
nvidia-smi                            # driver version (login node may have none; check job will print it)
curl -sI https://huggingface.co | head -1   # internet on login node? (needed for downloads)
squeue -p <partition> | wc -l         # how busy the queue is
```
Set in `slurm/cluster.env`:
- `PARTITION`, `GRES` (e.g. `gpu:a100:1`), `GRES_BIG` (largest GPU, e.g. `gpu:a100_80g:1`, used by H2), `ACCOUNT`/`QOS` if required.
- `MODULES` (empty string if you don't use modules), `TORCH_INDEX` (`cu121` needs driver ≥ 530, else `cu118`).
- `HF_HOME`, `CKPT_ROOT`: point at scratch/work storage if `$HOME` is small (need ~20 GB + ~4 GB).
- `OFFLINE=1` if compute nodes have no internet (the usual case). Jobs then read only the pre-downloaded cache.

### 1.3 Build the environment and download everything (login node)
```bash
bash setup_env.sh --with-h2     # drop --with-h2 if you will skip H2 (saves ~6 GB)
```
This creates the venv, installs `torch` + `transformers peft datasets accelerate bitsandbytes`, and downloads
`Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-3B` (H2), `distilbert/distilbert-base-uncased`, `yahma/alpaca-cleaned`, `nyu-mll/glue:sst2`, `Salesforce/wikitext:wikitext-2-raw-v1`.

### 1.4 Run the sanity job
```bash
bash slurm/submit.sh check
squeue -u $USER                       # wait for it to finish (PD = pending, R = running)
cat logs/peft-check_<jobid>.out
```
Must see: the GPU name, `bf16 supported: True` (False is OK, scripts fall back to fp16), `bitsandbytes 4-bit NF4 matmul: OK`, all `cached ... OK`, and `ALL CHECKS PASSED`. **Write down the GPU name and memory**: they go in every slide footer.

### 1.5 Submit everything at once (recommended when the deadline is tight)
```bash
bash slurm/submit.sh all      # 4 jobs: H1, H2, H3 queued; H4 automatically starts after H1 succeeds
```
With general access only one runs at a time (1 GPU/user); the others show `PD (QOSMaxGRESPerUser)`, which is normal.
Or submit each one separately as described below. Useful commands while waiting:
```bash
squeue -u $USER                               # state of your jobs
tail -f logs/peft-h1_<jobid>.out              # live log
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it ends
scancel <jobid>                               # kill a job
```

---

## 2. H1: Full FT vs LoRA vs QLoRA on Qwen2.5-0.5B ⭐

### 2.1 What it tests
Two claims, measured by us on one model, one dataset, one GPU:
1. LoRA and QLoRA cut trainable parameters, memory and checkpoint size by a lot while reaching similar quality.
2. QLoRA uses even less memory than LoRA **but trains slower** (4-bit weights are de-quantized on every forward/backward pass). This is the honest trade-off.

### 2.2 Design: what is held fixed and what changes
| Fixed for all three runs | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` (base, not instruct; open, not gated; 494 M params). Smallest model that is still a real LLM *and* where weights/optimizer memory, not activations, dominate. Below ~0.5 B (e.g. 135 M models) activations and the 150 k-vocab output layer eat most of the memory and the LoRA savings shrink to almost nothing, which would not prove the point |
| Data | 2,000 examples of `yahma/alpaca-cleaned` (shuffled with fixed data seed 1234); last 200 held out → ~1,800 train / 200 eval |
| Format | Alpaca prompt template; loss **only on the response tokens** (prompt tokens masked with -100) |
| Sequence length | 512 (truncate), dynamic padding per batch |
| Batch | micro-batch 4 × grad-accum 4 = **16 sequences / optimizer step** |
| Steps | 300 optimizer steps (≈ 2.7 epochs), cosine LR, 10 warm-up steps, grad-clip 1.0 |
| Optimizer | `torch.optim.AdamW`, weight decay 0 |
| Gradient checkpointing | **on for all three** (so the memory comparison is fair) |
| Seed | 42 |

| Varies | Full FT | LoRA | QLoRA |
|---|---|---|---|
| Base weights | fp32 master weights, bf16 autocast (classic mixed precision, ~16 bytes/param) | bf16, frozen | **4-bit NF4 + double quant**, frozen |
| Trained weights | all ~494 M | LoRA A,B (fp32) on **all linear layers** (q,k,v,o,gate,up,down) | same as LoRA |
| r / α / dropout | – | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| Learning rate | 1e-5 | 2e-4 | 2e-4 |
| Saved checkpoint | full model in bf16 | adapter only | adapter only |

QLoRA uses the standard PEFT recipe `prepare_model_for_kbit_training` (it up-casts layer norms/embeddings to fp32 for stability; on Qwen2.5-0.5B the 151 k-token embedding is ~27 % of all parameters and is never quantized, so expect QLoRA to save only a little memory over LoRA here while still being slower. That is an honest, explainable result: "4-bit pays off when the linear layers dominate, i.e. on bigger models". H2 on 3B shows the bigger gap).

### 2.3 Run it
```bash
bash slurm/submit.sh h1
# optional second seed for error bars (≈ doubles the run time):
SEEDS="42 43" bash slurm/submit.sh h1
```
`slurm/h1.sbatch` runs, one after another on the same GPU:
```bash
python train_llm.py --exp h1 --method full  --model Qwen/Qwen2.5-0.5B --steps 300 --seed 42 --max-len 512 --micro-batch 4 --grad-accum 4
python train_llm.py --exp h1 --method lora  ...same...
python train_llm.py --exp h1 --method qlora ...same...
```
A failure in one method does not stop the others.

### 2.4 What each run measures (and how)
| Metric | How it is measured |
|---|---|
| Trainable / total params | PEFT `get_nb_trainable_parameters()` (counts 4-bit packed weights correctly); plain count for full FT. Also printed by `print_trainable_parameters()` in the log |
| Weights on GPU | `torch.cuda.memory_allocated()` right after loading |
| **Peak GPU memory** | `torch.cuda.reset_peak_memory_stats()` before training, `torch.cuda.max_memory_allocated()` after → weights + grads + optimizer states + activations. In GiB (same unit as the "24 GB" on the slice). `nvidia-smi` shows ~0.5–1 GB more (CUDA context + allocator cache); `peak_reserved_gb` is also stored |
| **Sec/step, tokens/sec** | `time.perf_counter()` around each optimizer step with `torch.cuda.synchronize()`; first 5 steps excluded (warm-up). Tokens = non-padding tokens |
| Held-out loss | token-weighted cross-entropy on the 200 held-out responses, **before** and **after** training |
| Checkpoint size | size of the `*.safetensors` files written by `save_pretrained()` |
| Loss curve | per-step train loss → `results/h1/<run>_curve.csv` |

### 2.5 Watch the log
```bash
tail -f logs/peft-h1_<jobid>.out
```
You should see per run: the header with GPU name, `trainable params: ...` (LoRA/QLoRA), `held-out loss before training`, then `step  10/300 loss ... s/step peak ... GB` every 10 steps, then `held-out loss after training`, then a `RESULT {...}` line and `[saved] results/h1/<run>.json`.

### 2.6 Sanity checks before trusting the numbers
- **LoRA trainable params ≈ 8.80 M (~1.8 %)**. Hand check for r=16 on Qwen2.5-0.5B (hidden 896, MLP 4864, k/v out 128, 24 layers): per layer `16·[(896+896)·2 + (896+128)·2 + (896+4864)·3] = 366,592`, × 24 = **8,798,208**. LoRA and QLoRA must print the same number.
- Full FT trainable = total ≈ 494 M.
- Peak memory order must be **Full ≫ LoRA > QLoRA**. Full FT should be roughly 494 M × 16 bytes ≈ 7.4 GiB plus activations (~10 GiB total); LoRA ~2–4 GiB.
- Held-out loss **after < before** for all three. If full FT got worse, the LR is too high (try `--lr 5e-6`).
- Adapter checkpoints ≈ 8.8 M × 4 bytes ≈ 35 MB; full checkpoint (bf16) ≈ 1 GB.
- Every JSON has `"status": "ok"`. If full FT ever says `"OOM"`, that is itself a result: put "OOM on X GB" in the full-FT row. 0.5B fits a 24 GB slice comfortably. To scale H1 up to 1.5B: `MODEL=Qwen/Qwen2.5-1.5B bash slurm/submit.sh h1` with `GRES=gpu:2g.48gb:1` (full FT needs ~30 GB) and run H4 with the same `MODEL`.

### 2.7 Turn it into slide content
After the job ends (see §6 for `make_report.py`):
- `results/summary/h1_table.md`: the slide 4 table (Method · Trainable params · % · Peak mem · Sec/step · Tokens/s · Held-out loss · Checkpoint size).
- `results/summary/h1_memory.png`: the one chart (peak memory bars, held-out loss as labels).
- `results/summary/h1_speed.png`: proves "QLoRA is slower than LoRA".
- `results/summary/h1_ratios.txt`: ready-made sentences ("LoRA trains N× fewer params", "QLoRA sec/step = 1.xx× LoRA's").
- Compare to the paper on the slide: LoRA paper reports ~10,000× fewer trainable params and ~3× less memory on GPT-3 175B; our ratio will be much smaller (1.5 B model, all-linear r=16, activations matter). **Say the difference and why.**

---

## 3. H2: a 3B model on one 24 GB GPU (the "wow" line)

### 3.1 What it tests
QLoRA's headline claim at small scale: full fine-tuning a model too big for the GPU fails, QLoRA trains it on the same GPU. A 3B model on a 24 GB slice proves this as well as 7B on 48 GB would, with a smaller download and shorter queue.
Slide line: *"Full FT of Qwen2.5-3B: OOM on a 24 GB GPU. QLoRA: __ GB peak, trains."*

### 3.2 Design
Same trainer and settings as H1 (`train_llm.py`), with:
| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B` (~3.1 B params) |
| Methods | `full` (expected OOM), `lora` (bf16 base), `qlora` (4-bit base), in that order |
| Steps | 50 (enough to show it trains and to get a stable sec/step) |
| Eval | held-out loss after training only (`--skip-eval-before`), no checkpoint saving (`--no-save`) |
| GPU | `GRES_BIG` = a 24 GB slice |

Expected arithmetic (for interpreting, not for the slide): full FT needs ~3.1 B × 16 bytes ≈ 46 GiB → OOM on 24 GB (fp32 weights alone are ~11.5 GiB, so it most likely fails in the first backward pass when the gradients are allocated). LoRA needs ~5.8 GiB of bf16 weights + activations; QLoRA ~2–3 GiB of weights + activations. Both fit.

### 3.3 Run it
```bash
bash slurm/submit.sh h2
```
Needs `--with-h2` to have been used in setup (`check` job shows `cached model OK Qwen/Qwen2.5-3B`).

### 3.4 How the OOM is captured (this is the measured result)
`train_llm.py` catches `torch.cuda.OutOfMemoryError`, prints the traceback and a banner:
```
*** OUT OF MEMORY at stage 'train_step0_optimizer' after 0 steps; peak allocated 7x.xx GB of 79.x GB ***
```
and writes `results/h2/full_seed42.json` with `"status": "OOM"`, `failed_stage`, the weights-on-GPU number and the peak at failure, then exits 0 so LoRA and QLoRA still run.
`failed_stage` tells the story precisely:
- `load_model` → even the fp32 weights don't fit (small GPU);
- `train_step0_forward_backward` → weights fit, weights + gradients + activations don't;
- `train_step0_optimizer` → weights + grads fit, the AdamW states (2 × fp32 per param) don't.

**Screenshot the banner + traceback from `logs/peft-h2_<jobid>.out`**: that screenshot is slide-worthy evidence.

### 3.5 Sanity checks
- `full`: status `OOM`. (If it does not OOM, you have a > 120 GB GPU; report the peak instead, still a valid result.)
- `lora` and `qlora`: status `ok`, `steps_done = 50`, peak memory QLoRA < LoRA, sec/step QLoRA > LoRA.
- The `nvidia-smi` line printed between runs should show ~0 MiB used (each method is a fresh process).

### 3.6 Slide content
- `results/summary/h2_table.md` (Outcome, weights on GPU, peak memory, sec/step).
- `results/summary/h2_memory.png`: OOM bar hatched up to the GPU capacity line, LoRA/QLoRA real peaks.

---

## 4. H3: rank sweep, accuracy vs r

### 4.1 What it tests
LoRA paper finding: very small ranks already work. We measure SST-2 accuracy for r ∈ {1, 2, 4, 8, 16, 64}, 2 seeds each, plus a full fine-tune reference line. Also checks the paper's parameter formula `r·(d + k)` per adapted matrix against what PEFT actually creates.

### 4.2 Design
| Setting | Value |
|---|---|
| Model / data | `distilbert/distilbert-base-uncased` / GLUE SST-2: full train set (67,349), validation set (872) for accuracy |
| LoRA targets | `q_lin`, `v_lin` (as in the LoRA paper), dropout 0.1 |
| α | **α = 2r** (keeps the scale α/r constant, so only the rank changes). Use `--alpha 16` to fix α instead and say so |
| Head | `pre_classifier` + `classifier` (newly initialised) are trained in every run, LoRA and full |
| Training | 2 epochs, batch 32, max length 128, AdamW, linear schedule with 6 % warm-up, bf16 autocast |
| LR | 5e-4 LoRA, 2e-5 full FT |
| Seeds | 42, 43 |

### 4.3 Run it (one sequential job on a 24 GB slice)
```bash
bash slurm/submit.sh h3
# short on time? fewer ranks / one seed:
RANKS="1 4 16 64" SEEDS="42" bash slurm/submit.sh h3
```
`slurm/h3.sbatch` loops over all 14 runs in one job (LoRA r ∈ {1,2,4,8,16,64} × seeds {42,43}, then full FT × 2 seeds), roughly 1–1.5 h in total.
It is **not** an array job on purpose: general QoS allows 1 GPU per user and 10 submitted jobs with DenyOnLimit, and each array task counts as a job, so a 14-task array would be rejected.

Each run: `python train_h3.py --method lora --r <r> --seed <seed>` → `results/h3/lora_r<r>_seed<seed>.json`. Log: `logs/peft-h3_<jobid>.out`.
If the job is preempted or hits the time limit, just resubmit: finished runs are skipped.

### 4.4 Sanity checks
- Each log prints `LoRA params measured=... analytic r*(d+k) over 12 matrices=... match=True`. DistilBERT: 6 layers × 2 matrices × r × (768 + 768) = **18,432 · r** (r=8 → 147,456). `match=False` means something is wrong; do not use that run.
- Trainable params from `print_trainable_parameters()` = LoRA params + head (≈ 592 K). At small r **the head dominates**; say so if asked "why isn't it 0.02 %?".
- Accuracy for any r should be well above 50 % (chance) and in the high 80s; full FT is the ceiling line.
- The two seeds should differ by well under 1 point; a large gap means noisy runs, so report the range.

### 4.5 Slide content
- `results/summary/h3_table.md`: r · LoRA params (measured vs analytic, match) · trainable incl. head · accuracy (mean, min–max).
- `results/summary/h3_acc_vs_rank.png`: accuracy vs r (log scale) with seed range band and the full-FT dashed line.
- `results/summary/h3_params_vs_rank.png`: the parameter cost of each rank (separate chart on purpose: two different units never share one chart).
- Talking point: if accuracy plateaus from r ≈ 4–8, it confirms the paper; if it keeps rising with r, we say so.

---

## 5. H4: "LoRA forgets less" (needs H1 checkpoints)

### 5.1 What it tests
Biderman et al. 2024: LoRA learns less but **forgets less** of the base model's general ability than full FT. We measure "forgetting" as the change in perplexity on general text (WikiText-2 test) after Alpaca fine-tuning, and "learning" as the Alpaca held-out loss from H1.

### 5.2 Design
| Variant | What is loaded | Compared against |
|---|---|---|
| `base` | H1's base model (Qwen2.5-0.5B), bf16 | reference |
| `base4bit` | same model, 4-bit NF4 | reference for QLoRA (separates "4-bit damage" from "fine-tuning damage") |
| `full` | `$CKPT_ROOT/h1/full_seed42` | `base` |
| `lora` | base bf16 + `$CKPT_ROOT/h1/lora_seed42` adapter, merged | `base` |
| `qlora` | base 4-bit + `$CKPT_ROOT/h1/qlora_seed42` adapter | `base4bit` |

Perplexity: whole WikiText-2-raw test set joined with `\n\n`, sliding window 1024 tokens with stride 512 (each token scored once, with ≥ 512 tokens of context), token-weighted NLL → `exp`.

### 5.3 Run it
Automatically chained after H1 if you used `submit.sh all`. Otherwise, after H1 has finished successfully:
```bash
bash slurm/submit.sh h4               # H1 already done
bash slurm/submit.sh h4 <H1_JOB_ID>   # or: queue now, start only if H1 ends OK (afterok)
```
If H1 never succeeds, an `afterok` H4 stays pending forever with reason `DependencyNeverSatisfied`: `scancel` it.

### 5.4 Sanity checks
- Log shows `WikiText-2 test: ~3xx,xxx tokens` and one `[variant] WikiText-2 perplexity = ...` per variant.
- `base` perplexity should be roughly in the low-to-mid teens for a 0.5B model; `base4bit` somewhat higher than `base` (4-bit hurts small models more).
- A `FileNotFoundError` for a variant means H1 did not save that checkpoint (check `results/h1/*.json` → `ckpt_dir`).

### 5.5 Reading the result
`results/summary/h4_table.md` puts both sides next to each other:

| Model | WikiText-2 ppl (forgetting) | Change vs its base | Alpaca held-out loss (learning) |
|---|---|---|---|

- If `full` has the largest perplexity increase and LoRA/QLoRA smaller ones **and** full FT has the lowest Alpaca loss → mini-reproduction of "learns less, forgets less".
- If not (likely possible with only 300 steps of a light instruction dataset), **report it honestly**: "with 300 steps on Alpaca we saw no clear difference". That still earns the marks.
- Chart: `results/summary/h4_forgetting.png` (% perplexity change vs the matching base).

---

## 6. After the jobs: build the tables, charts and footer

On the login node (CPU only, a few seconds):
```bash
source slurm/cluster.env && source "$VENV/bin/activate"
python make_report.py
ls results/summary/
```
It skips any H that has no results yet, so run it again whenever more jobs finish.

Copy back to the laptop for the slides:
```bash
# from the laptop
scp -r user@hpc-login:~/peft-evidence/results  ./evidence/
scp    user@hpc-login:~/peft-evidence/logs/*.out ./evidence/logs/
```

Slide footer: `results/summary/footer.txt` is pre-filled from the run metadata:
*"Measured on [cluster], 1× [GPU, memory], PyTorch x.y, Transformers x.y, PEFT x.y, bitsandbytes x.y, seed 42, SLURM job IDs …"*. Replace `[cluster]` with the cluster's name.

Evidence to keep for Q&A: `results/h*/*.json` (every number with its settings and versions), `logs/*.out` (screenshots: `print_trainable_parameters()` lines, the H2 OOM banner, the `RESULT` lines).

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Job stays `PD` with reason `Resources`/`Priority` | Queue is busy. Ask for a smaller GPU type in `GRES`, use a shorter `--time`, or fall back to Colab (plan §4) for slide 4 |
| `sbatch: error: invalid partition` / `invalid gres` | Fix `PARTITION` / `GRES` in `cluster.env` (`sinfo -o "%P %G"`) |
| `OSError: ... offline mode` / `couldn't connect to huggingface.co` | Asset not pre-downloaded. Re-run `python download_assets.py` on the login node (with `HF_HOME` exported), same `HF_HOME` as the jobs |
| `$'\r': command not found` | Windows line endings: `sed -i 's/\r$//' slurm/* *.sh` |
| bitsandbytes `CUDA Setup failed` / no GPU kernels | Make sure `MODULES` loads a CUDA module compatible with the torch wheel; `pip install -U bitsandbytes` |
| `bf16 supported: False` (V100/T4) | Scripts automatically use fp16 + GradScaler. Note "fp16" in the footer. QLoRA still works |
| H1 full FT OOM on a 24–32 GB GPU | Record it as a result, or reduce `--micro-batch 2 --grad-accum 8` for **all three** methods (same effective batch) so they stay comparable |
| `TIME LIMIT` in the log | Raise `#SBATCH --time` in that sbatch file or lower `STEPS="150" bash slurm/submit.sh h1` (then say 150 steps on the slide) |
| H4 pending forever | H1 failed; `scancel` H4, fix H1, then `bash slurm/submit.sh h4` |
| `PD (QOSMaxGRESPerUser)` | Normal on general access: your other GPU job is running. Wait, or have a teammate submit from their account |
| `sbatch: error: ... QOSMaxSubmitJobPerUserLimit` | You have 10 jobs queued/running; `scancel` old ones |
| Log shows the job restarting (`restart count: 1`) | It was preempted and requeued; finished runs are skipped automatically, the interrupted run starts over |
| `CUDA error: no kernel image is available` | The torch wheel doesn't support this GPU. Keep `TORCH_INDEX=.../cu128` (needs driver ≥ 570) and reinstall torch |
