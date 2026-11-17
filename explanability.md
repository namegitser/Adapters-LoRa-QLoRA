# Explainability log

What was done at each step, and every problem hit along the way.

## Background (before this log existed): HPC route, `evidence/`

| Step | What was done | Problem encountered → resolution |
|---|---|---|
| 1 | Built `evidence/`: scripts for H1–H4, SLURM jobs, `flow.md` | – |
| 2 | Adapted to Plaksha general access (rtx partition, 1 GPU/user, 10-job cap, preemption) | A 14-task H3 array would break the 10-job cap → H3 made one sequential job; jobs skip finished runs after requeue |
| 3 | Shrunk models: H1 Qwen2.5-0.5B, H2 Qwen2.5-3B, all on 24 GB slices | – |
| 4 | Environment setup on the login node | pip/conda failed with `OSError: [Errno 14] Bad address` when streaming large files into `/home`; plain `dd`/`cp` worked. Wasted attempts on PyPI torch/conda before reading the HPC guide |
| 5 | Read https://hpc.plaksha.edu.in | Guide says: `module load python` (Python 3.13 + torch 2.13 cu130), venv with `--system-site-packages`, never pip-install torch, big data in `/scratch`. Setup rewritten to that; pip installs via `--root` into `/tmp` then `cp` |
| 6 | Setup finished: packages OK, 7.1 GB cache in `/scratch/$USER/hf_cache` | `distilbert-base-uncased` 404 on the new download path → canonical id `distilbert/distilbert-base-uncased`. transformers 5.23 removed `torch_dtype`/`low_cpu_mem_usage` → switched to `dtype=` |
| 7 | Check + smoke-test job submitted (job 14358) | Old `check.sbatch` on HPC (scp of that file had not happened) → rewritten in place; duplicate job 14357 cancelled |
| 8 | Queue estimate | Job estimated to start 18 Nov (03:25, then 12:40–16:06); `scavenger` QoS is not allowed on `rtx` or `h200`. Slides are due 17 Nov EOD → HPC cannot deliver today |

## 2026-11-17: Colab fallback, `evidence_collab/`

**Decision (less-code check):** no new training code. Reuse the HPC scripts unchanged and wrap them in one notebook; the alternative (a separate Colab-specific trainer) would duplicate ~400 lines.

| Step | What was done | Problem encountered → resolution |
|---|---|---|
| 1 | Reverted my footer edit in `evidence/make_report.py` so `evidence/` stays the exact HPC version | User asked to keep the HPC version untouched; nothing had been deleted |
| 2 | Created `evidence_collab/` with copies of `common.py`, `train_llm.py`, `train_h3.py`, `eval_h4.py`, `make_report.py` | – |
| 3 | One-line change in the copied `make_report.py`: footer says "Measured on Google Colab" instead of "[cluster] … SLURM job IDs" | Colab runs have no SLURM job ID, so the HPC footer would print an empty list |
| 4 | Generated `evidence_collab/peft_evidence_colab.ipynb` (17 cells): install packages → `%%writefile` the 5 scripts → H1 → report → optional H2/H3/H4 → zip + download | T4 has no fast bf16, and `torch.cuda.is_bf16_supported()` can report True via slow emulation → every command passes `--amp fp16` explicitly. H3 reduced for Colab time (r ∈ {1,4,16,64}, 1 seed, 20k examples, 1 epoch) |
| – | Not yet run: the notebook is untested until executed on Colab | Risk: Qwen in fp16 can produce `nan` loss; the notebook says to switch `AMP` to `bf16` if that happens |

| 5 | First Colab run (T4): H1 full FT trained normally to step 270 (loss 1.35 → ~0.7, peak 11.45 GB, ~2 s/step, weights 1.84 GB = 494 M × 4 B as expected) | User reports the cell "failed" after step 270; the error itself was not in the pasted log → asked for the cell's last lines and `results/h1/` contents to locate it (end of training, final eval, bf16 save, or a Colab crash) |

| 6 | Cause found: Colab popup "Failed to assign a backend / No backend with GPU available" | Free Colab GPU quota used up, so the runtime was taken away mid-run (around step 270) and `/content` was lost. Advised not to accept the CPU-only runtime (too slow; 4-bit QLoRA needs a GPU) |
| 7 | Notebook edited in place (less-code: a few cell edits, no new scripts) so it runs on Colab **or Kaggle** (free 30 GPU h/week, same T4): work dir auto-detected (`/kaggle/working/evidence` vs `/content/evidence`), `%%writefile` paths made relative, last cell falls back to `/kaggle/working/results.zip`; H1 loop skips methods whose result JSON already exists | Colab quota resets at an unknown time; Kaggle needs a phone-verified account for GPU + internet |

## 2026-11-17: Kaggle version, `evidence-kaggle/`

**Decision (less-code check):** the scripts are already embedded in the notebook, so the folder holds only one notebook: a copy of the Colab notebook with 4 cell edits, no new code or script copies.

| Step | What was done | Problem encountered → resolution |
|---|---|---|
| 1 | Judged Kaggle a good fit: same T4, published 30 GPU h/week, and "Save & Run All (Commit)" runs in the background, so a closed laptop or disconnect does not kill the run (the exact failure we hit on Colab) | Needs phone verification and Internet switched on |
| 2 | Created `evidence-kaggle/peft_evidence_kaggle.ipynb` with edits: Kaggle instructions (incl. background commit run), work dir `/kaggle/working/evidence`, `CKPT_ROOT=/tmp/peft_ckpts` (keeps the saved output small; checkpoints are only needed within the run for H4), footer "Measured on Kaggle", `results.zip` written to `/kaggle/working` | Verified: 17 cells, no Colab-only code or `/content` paths left. Not yet run on Kaggle |

## 2026-11-17: Slide deck (while Kaggle runs)

**Decision (less-code check):** built as a Slides artifact (HTML/CSS slides, no code to maintain) rather than generating a .pptx with a script.

| Step | What was done | Problem encountered → resolution |
|---|---|---|
| 1 | Created deck "Parameter-Efficient Fine-Tuning": https://claude.ai/artifact/6ZXppdcq6XQjS2RGy7spxf, 8 main slides + 2 backup, following plan.md §3 and the rubric (understanding: slides 3–4; evidence: slide 6; trade-offs: slide 7; clarity: one idea per slide, visuals; questions: backup Q&A slide; sources: slide 8) | – |
| 2 | Visual system: navy/warm-paper palette; method colours match the result charts (full FT blue, LoRA orange, QLoRA green); Space Grotesk / Inter / JetBrains Mono; speaker notes with speaker and timing on every slide | – |
| 3 | Every result left as an orange `[__]` gap or a dashed chart box naming the file that fills it (`results/summary/h1_table.md`, `h1_memory.png`, …). Only known facts pre-filled: 494 M total params (measured), 8.8 M LoRA params (computed), paper numbers with citations | – |
| 4 | Checked text height against the 824 px content area | Trade-offs slide estimated ~960 px and evidence slide borderline → shortened table/card wording, smaller chart box |
| 5 | Visual preview | Not possible here: no Playwright/Chromium installed on this machine (rule: install nothing) → user should page through the deck and report anything that overflows |

## 2026-11-17: First real results (Kaggle, Tesla T4 14.6 GB, fp16)

| Step | What was done | Problem encountered → resolution |
|---|---|---|
| 1 | H1 full FT finished (`status: ok`): 494,032,768 trainable (100 %), weights 1.84 GB, **peak 11.45 GB** (13.07 reserved), **1.95 s/step**, 1,493 tok/s, checkpoint **988 MB**, wall 619 s | – |
| 2 | Checked quality numbers | **Held-out loss rose 1.349 → 1.426** while train loss fell to 0.73: full FT overfit (300 steps × 16 ≈ 2.7 epochs over 1,800 examples, lr 1e-5). Kept as an honest result, no re-run; to be compared with LoRA/QLoRA before deciding how to present it |

| 3 | Kaggle Version 1 finished: full FT ok (above); **QLoRA ok**: 8,798,208 trainable (1.7497 %, matches the hand calculation exactly), weights 0.713 GB, **peak 4.74 GB**, **2.56 s/step** (slower than full FT's 1.95), held-out **1.448 → 1.402**, adapter **35.2 MB**. Footer: Kaggle, Tesla T4 14.6 GB, PyTorch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, bitsandbytes 0.50.2 | **LoRA failed at load**: `ImportError: Found an incompatible version of torchao (0.10.0), only > 0.16.0 supported`. PEFT 0.21 probes torchao when wrapping plain linear layers; Kaggle preinstalls an old torchao. QLoRA was unaffected (bnb layers dispatch first) |
| 4 | Report cell crashed: `h1_ratios.txt` missing | `make_report.py` writes ratios only when full + LoRA both exist; the notebook's uncaught error stopped papermill, so H2/H3/H4 never ran |
| 5 | Fixed both notebooks (Kaggle + Colab), 2 cell edits each, scripts unchanged: install cell adds `%pip uninstall -y -q torchao`; report cell prints `h1_ratios.txt`/`footer.txt` only if they exist | Re-run needed: a new Kaggle version starts from scratch (~25 min H1, ~1.5 h with extras) |

| 6 | **Overfitting addressed** (user asked): cause = 300 steps × 16 = 4,800 samples over only 1,800 training examples (≈2.7 epochs) → full FT memorised. Less-code fix: H1 cell passes `--n-examples 5000` (4,800 train + 200 held out) so the same 300 steps are exactly one epoch, no repeats; steps, batch, LR, memory/speed comparability unchanged. Applied to both notebooks; slide 6 setup line updated ("4,800 Alpaca examples, 1 epoch"). `evidence/` (HPC) left unchanged as requested | The 200 held-out examples are now a different set, so Version 2 losses are not comparable with Version 1; Version 2 is used for everything |

| 7 | Careful review of the Version 1 log + a trace of every cell that never ran (H2, H3, H4, zip) against Kaggle's versions | Found 3 more ways the commit could stop or mislead, fixed in both notebooks (cell edits only, scripts unchanged): (a) report cell displayed files without checking they exist → any missing H1 output raises, and papermill stops Run All at the first Python error → now every display/print is guarded; (b) H1 "skip finished" skipped runs whose JSON exists even with `status: error` → now skips only `status == "ok"`; (c) final cell used the `zip` CLI → now `shutil.make_archive`. Verified: all code cells parse as Python. Checked OK: H2 full-FT OOM is caught in-script; 3B LoRA/QLoRA fit the T4; H3/H4 code paths and `/tmp/peft_ckpts` handoff; `!python` cells cannot stop the notebook; total ≈ 70–80 min |
| – | Remaining known risk (not fixable offline) | fp16 on Qwen can give `nan` loss (most likely on the 3B in H2): not a crash, the run records `nan`; fallback is `AMP = "bf16"` (slower on T4) |

Observations worth presenting: 4-bit quantization alone raised the base model's held-out loss 1.349 → 1.448 (QLoRA "before"); full FT overfit (loss up), QLoRA improved.

## 2026-11-17: Kaggle Version 2 results

| Step | What was done | Problem encountered → resolution |
|---|---|---|
| 1 | H1 all three methods `ok`. Full FT 11.43 GB, 2.02 s/step, loss 1.318 → 1.285, 988 MB; LoRA 8.8 M (1.75 %), 4.64 GB, 2.10 s/step, 1.318 → 1.272, 35.2 MB; QLoRA 4.74 GB, 2.69 s/step, 1.412 → 1.322, 35.2 MB. Ratios: 56× fewer params, 2.5× less memory, 28× smaller checkpoint | Overfitting fix worked: full FT loss now falls (1.318 → 1.285). QLoRA memory ≈ LoRA at 0.5B (activations dominate), so presented honestly in the speaker notes |
| 2 | H2 (Qwen2.5-3B): full FT `OOM` at step 0 (14.33 of 14.6 GB), recorded as the result; LoRA finished; QLoRA trained at peak 7.61 GB | The user read the OOM traceback as a failure; it is the intended H2 outcome, caught in-script |
| 3 | Filled slides 6 (evidence table, ratios, footer, notes), 8 (takeaways 01, 02), 9 (H2 card) and republished the deck (version 7). Less-code: text edits only | Still open: H1 memory chart, H2 LoRA GB, H3, H4, notebook link: these are filled from `results.zip` once the run finishes |

| 4 | Copied the deck source (`deck.json`, `deck.css`, `slides/*.html`, version 7) into `slides/` in the project folder | Less-code: plain copy. A .pptx is exported from the artifact page's download, not generated with a script; the copy has to be refreshed after later deck edits |

## 2026-11-17: HPC run, hardened with the Kaggle lessons

| Step | What was done | Problem encountered → resolution |
|---|---|---|
| 1 | Check job 14358 ran 16:44–16:46, `COMPLETED 0:0` (the script exits 1 on any smoke failure, so all smoke tests passed) | It started much earlier than the queue estimate |
| 2 | Reviewed every SLURM file against the Kaggle lessons. Already safe: OOM recorded, not fatal (H2); skip-on-requeue only for `ok`/`OOM`; eval_h4 handles each variant separately; HPC uses bf16 (RTX PRO 6000), so the fp16 `nan` risk does not apply; torchao/LoRA covered by the passing smoke test | – |
| 3 | Two one-line fixes in `evidence/slurm` (local and on the HPC via `sed`): `h1.sbatch` adds `--n-examples 5000` (the same overfitting fix as Kaggle, one epoch); `submit.sh` chains H4 with `afterany` instead of `afterok` | `afterok` would leave H4 stuck forever (DependencyNeverSatisfied, still counting toward the 10-job cap) if any single H1 method failed |
| 4 | Pre-flight checks given to the user before `submit.sh all`: the `/scratch` cache still exists (it is purged without warning, and the jobs run offline), no stale `results/h1` files, and the edits are present on the HPC (the stale-file lesson from `check.sbatch`) | – |

| 5 | Pre-flight passed (smoke ALL PASSED, 7.1 GB cache, only `results/smoke`); submitted `submit.sh all`: H1 14447, H2 14448, H3 14449, H4 14450 (waits on H1, `afterany`) | – |

## 2026-11-17: Final Kaggle results into the deck

| Step | What was done | Problem encountered → resolution |
|---|---|---|
| 1 | Kaggle Version 2 finished: 15 runs `ok` + the expected H2 full-FT `OOM`. Results copied to `evidence-kaggle/results/` (from the user's Downloads) | Chat does not accept .zip → read the extracted folder from disk instead |
| 2 | Key numbers: H2 (3B) full FT OOM, LoRA 10.0 GB, QLoRA 7.6 GB · H3 SST-2: r=1 85.7 %, r=4 87.6 %, r=16 87.5 %, r=64 88.1 % = full FT 88.1 % · H4 WikiText-2 ppl change: full +2.3 %, LoRA +1.0 %, QLoRA −1.4 % vs its 4-bit base | QLoRA's −1.4 % is relative to the 4-bit base, which already starts worse (13.87 vs 12.50); explained in the notes, so it is not read as "negative forgetting". H3 is one seed: sub-0.5 pt differences are noise |
| 3 | Checked all 4 charts visually, uploaded them as deck assets, filled every remaining number (slides 6, 7, 8) | A newer page-saved version existed (the user's own layout tweaks) → re-read that version and merged onto it, so the tweaks are kept |
| 4 | User asked for clean, readable slides (more pages allowed): split the crowded 3-card backup slide into 3 slides (H2, H3, H4), each a large chart + 3 stat cards; shortened takeaway 01 back to one line | Less-code: generated the 3 slides from one template function. No visual preview possible here (no browser automation); sizes checked against the 824 px content height |
| 5 | Synced `slides/` in the project with the published deck (version 14, 12 slides) | – |

| 6 | Wrote `deep-explanation.md`: a from-scratch study guide covering foundations (loss, perplexity, Adam, fp16/bf16 mixed precision, the 16 B/param rule, checkpointing, 6N FLOPs), Adapters/LoRA/QLoRA theory (gradients, init, α/r, NF4 with a worked quantization example, double quantization), every experiment as prediction → result → hand derivation → caveats, the engineering lessons, a trade-off guide, 16 drills with solutions, viva Q&A, a cheat sheet and references | Before writing, re-read `train_llm.py`/`train_h3.py`/`common.py` and all result JSONs. Hand derivations match the measurements: LoRA counts exact (8,798,208; 29,933,568; 18,432·r); weights after load 1.840 / 0.953 / 0.713 GiB (0.5B) and 11.50 / 5.86 / 2.62 GiB (3B) within ~1 %. Found and documented: QLoRA ≈ LoRA memory at 0.5B because 152K-vocab logits dominate activations; checkpoint ratio is 28× (not 56×) because the adapter is saved in fp32 |

| 7 | Added the Kaggle notebook link to slide 6 ("How we measured": short link text, because the user narrowed that card) and slide 8 (sources item 10, full URL). Merged onto the user's latest page edits; synced `slides/` | The first publish was refused: the user had re-arranged slide 6 in the browser → re-read and applied the link to their version |

Still for the user: team names (cover), `.pptx` download from the deck page.

HPC job 14358 stays queued; if it runs early, its results are used alongside or instead of Colab's.
