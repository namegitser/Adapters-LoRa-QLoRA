# Plan: Parameter-Efficient Fine-Tuning (LoRA, QLoRA, Adapters)

**Format:** 5–6 min talk + 1–2 min Q&A · every member presents · slides due on Moodle **Tue 17 Nov, EOD**

---

## 1. The story in one line

> "Fine-tuning a whole model is expensive. Most of that update turns out to be low-rank, so we can freeze the model and train a tiny add-on instead. We tested how tiny, how cheap, and what it costs."

Every slide moves this one story forward: **problem → idea → where it's used → our evidence → costs and limits → takeaway**.

---

## 2. How the rubric maps to slides

| Rubric item (weight) | Where we cover it | What proves it |
|---|---|---|
| Understanding & accuracy (25%) | Slides 2–3 | LoRA explained from first principles (W = W₀ + BA), with parameter math worked live; each claim tied to a paper |
| Evidence we produced (25%) | Slide 4 | Our own table: full FT vs LoRA (vs QLoRA), with parameters, memory, time and accuracy, plus a "how we measured" line |
| Trade-offs & limits (20%) | Slide 5 | When to use, when not to, what it costs, and one honest weakness |
| Clarity & delivery (15%) | All | One idea per slide, a visual instead of bullets where possible, rehearsed timing |
| Questions (15%) | Section 7 below | Prepared answers to likely questions; "I don't know, here's how I'd find out" |
| Sources (required) | Slide 6 | Full citations, including the code and library we ran |

---

## 3. Presentation flow (≈ 6 min)

Speaker labels assume **4 members (A–D)**. With 3 or 5 members, merge or split slides 2 and 3.

### Slide 1: The problem (A · 0:00–0:50)
- **Hook:** "Fully fine-tuning a 7B model in fp16 with Adam needs roughly 16 bytes per parameter, or about 112 GB of GPU memory. That's more than one A100 holds." (Our own arithmetic: 2 bytes for weights + 2 for gradients + 12 for fp32 Adam states and master weights; cite the ZeRO paper for the 16-byte figure.)
- Two more costs: one full copy of the model per task, and catastrophic forgetting.
- **Question for the audience:** "Do we really need to update all 7 billion numbers?"
- *Visual:* a big model next to a stack of per-task copies.

### Slide 2: The idea from first principles (B · 0:50–2:00)
- **Adapters** (Houlsby 2019): insert small bottleneck MLPs (down-projection → nonlinearity → up-projection) inside each layer and train only those.
- **LoRA** (Hu 2021): freeze W₀ and learn ΔW = B·A, where A is r×k, B is d×r, and r ≪ d.
  - Why it works: the weight change during fine-tuning has low "intrinsic rank" (Aghajanyan 2020; Hu 2021).
  - **Work the numbers on the slide:** for a 4096×4096 projection, full = 16.8M parameters. LoRA with r = 8: 8 × (4096 + 4096) = 65,536 parameters, or **0.39%**.
  - B starts at zero, so training begins at exactly the pretrained model. After training, BA can be **merged into W₀**, so inference latency is unchanged. (Adapters add latency because they are extra layers.)
- **QLoRA** (Dettmers 2023): store the frozen base in **4-bit NF4**, keep the LoRA weights in 16-bit, and add double quantization and paged optimizers. This let them fine-tune a 65B model on one 48 GB GPU.
- *Visual:* one diagram with a frozen W₀ (grey) and a trainable B·A side path (colour), with input x feeding both and the outputs added.

### Slide 3: Where it's used today (B or C · 2:00–2:40)
- **Hugging Face PEFT**: the standard open-source library (LoRA, QLoRA, adapters, prompt tuning).
- **Apple Intelligence** (2024): one on-device base model plus small task-specific LoRA adapters swapped in at runtime.
- **Multi-LoRA serving**: S-LoRA and vLLM serve thousands of adapters on one base model.
- **Image generation**: Stable Diffusion style and character LoRAs (files of a few MB vs a ~2–7 GB base).
- Keep it short: 4 logos or bullets, each with a source.

### Slide 4: What we measured ⭐ (C · 2:40–4:20) is the most important slide
- **Setup line:** model, dataset, GPU, epochs, seed.
- **One table** (fill with our real numbers only):

| Method | Trainable params | % of total | Peak GPU mem | Time / epoch | Val. accuracy | Saved checkpoint size |
|---|---|---|---|---|---|---|
| Full fine-tune | | 100% | | | | |
| LoRA r=8 | | | | | | |
| QLoRA (optional) | | | | | | |

- **Data source:** Colab results (Section 4), or HPC results (Section 4b) if they finish in time. The HPC runs are the stronger option because they use a real LLM and test QLoRA.
- **One chart:** bar chart of memory or trainable params, with accuracy as labels.
- **"How we got it"** footer: link to the Colab notebook plus the exact APIs used (`print_trainable_parameters()`, `torch.cuda.max_memory_allocated()`).
- **Compare to the paper:** "LoRA paper claims ~10,000× fewer trainable params and 3× less memory on GPT-3. On our small model we saw __×." If our result differs, **say so and explain why** (e.g. on a small model, activations dominate memory, so the savings are smaller). An honest contradiction gets full marks.

### Slide 5: Costs, limits, when NOT to use it (D · 4:20–5:30)
| Use PEFT when… | Don't (or think twice) when… |
|---|---|
| Limited GPU memory | The task needs a lot of new knowledge (e.g. a new language or domain), where LoRA can underperform full FT |
| Many tasks or users, one base model | You have abundant compute and need top accuracy |
| Style, format or instruction adaptation | Prompting or RAG would already solve it with zero training |

- **Costs:** new hyperparameters to tune (rank r, α, target modules, learning rate). QLoRA trains **slower** than 16-bit LoRA because of dequantization overhead. 4-bit quantization can lose a little quality. Unmerged adapters and classic adapter layers add inference latency.
- **Honest weakness:** Biderman et al. 2024, *"LoRA Learns Less and Forgets Less"*. On code and math continued pre-training, LoRA lags full fine-tuning; the upside is that it forgets less of the base model's knowledge.
- Optional, if our experiment shows it: our own weakness, e.g. "LoRA was __ points lower" or "LoRA wasn't faster per epoch".

### Slide 6: Takeaways + Sources (A · 5:30–6:00)
- Three takeaways:
  1. Train <1% of params and keep most of the quality.
  2. QLoRA puts fine-tuning on a single GPU.
  3. Not free: rank choice, knowledge-heavy tasks, speed.
- Full citation list (Section 6) in small font. The slide stays up during Q&A.

---

## 4. The experiment (evidence slide)

**Goal:** produce our own numbers for full FT vs LoRA (and optionally QLoRA) on the same task.

**Environment:** Google Colab, free **T4 GPU** (laptop has no CUDA). Record the GPU name in the slide footer.

**Experiment A (core, ~30–45 min run time)**
- Model: `distilbert-base-uncased` (~66M params) · Dataset: GLUE **SST-2** (sentiment). Use a 10k-example train subset if time is short, and the full validation set (872).
- Runs: (1) full fine-tuning, (2) LoRA r=8, α=16, targets `q_lin`, `v_lin`, dropout 0.1. Same seed, 1–2 epochs, batch 32, max length 128. Typical learning rates: 2e-5 for full FT, 5e-4 for LoRA.
- Measure for each run:
  - trainable vs total params: `model.print_trainable_parameters()`
  - peak memory: `torch.cuda.reset_peak_memory_stats()` before, then `torch.cuda.max_memory_allocated()/1e9` after
  - wall-clock time per epoch: `time.time()`
  - validation accuracy
  - saved size on disk: `model.save_pretrained()`, then folder size (LoRA saves only the adapter, about a few MB)

**Experiment B (optional, reproduces the QLoRA idea, ~10 min)**
- Load a ~1B causal LM (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) in fp16 vs 4-bit NF4 (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`), and record `model.get_memory_footprint()` for each.
- Expected ballpark is roughly 2.2 GB vs about 0.7–0.8 GB, but **report only what we actually measure**.

**Paper number to reproduce (cheap, analytic + measured):** LoRA's parameter-reduction ratio. Compute r(d+k) per adapted matrix by hand, then check that it matches `print_trainable_parameters()` (the classifier head adds a little; explain the difference).

**Rules:** run each config at least once, and ideally 2 seeds. Screenshot the outputs. Put the notebook link on the slide. No number goes on a slide unless it came from our run or a cited paper.

---

## 4b. HPC option (alternative or add-on to Section 4)

Section 4 (Colab) stays as the **safe baseline**. If HPC jobs start in time, use these results on slide 4 instead, or alongside. A bigger GPU lets us test the claims at a scale where PEFT actually matters (1–7B LLMs, not a 66M classifier).

**First, check:** GPU type and memory (`nvidia-smi`), scheduler (SLURM assumed below), queue wait time, internet access on compute nodes (if there is none, pre-download models and datasets on the login node with `huggingface-cli download`), and available CUDA/Python modules.

Pick **one** of the experiments below. H1 is the recommended one. H2 and H3 are extras if time allows.

### H1: Full FT vs LoRA vs QLoRA on the same 1.5B LLM ⭐ (recommended, ~1–1.5 h GPU)
- Model: `Qwen/Qwen2.5-1.5B` (open, not gated). Data: 2,000 examples of `yahma/alpaca-cleaned`, 200 held out.
- Runs (same data, seed, steps of about 300, sequence length 512):
  1. Full FT, bf16, AdamW
  2. LoRA r=16, α=32, all linear layers, bf16 base
  3. QLoRA: same LoRA settings with a 4-bit NF4 base
- Measure: trainable params, **peak GPU memory**, **tokens/sec (or seconds/step)**, final held-out loss, saved checkpoint size.
- **Why it's presentable:** one clean 3-row table plus one bar chart of memory. It directly tests two claims:
  - LoRA and QLoRA cut memory a lot.
  - QLoRA is *slower* than LoRA. This is an honest trade-off, and we measure it ourselves.

### H2: The "wow" demo: a 7B model on one GPU (~20 min)
- Try **full FT** of `Qwen/Qwen2.5-7B` on one GPU. Expect an **OOM (out-of-memory) error**, and screenshot it. That is a real measured result.
- Then run **QLoRA** on the same model for 50 steps and record peak memory.
- Slide line: "Full FT: OOM on a __ GB GPU. QLoRA: __ GB peak, trains." This reproduces QLoRA's core claim at a smaller scale.

### H3: Rank sweep (accuracy vs r) (~30 min, very cheap)
- DistilBERT + SST-2 (Section 4 setup), LoRA with r ∈ {1, 2, 4, 8, 16, 64}, 2 seeds each.
- Plot validation accuracy vs r, with trainable params on a second axis.
- Tests the LoRA paper's finding that very small ranks already work. If accuracy keeps rising with r, we say so.

### H4 (optional, links to slide 5): "Forgets less"
- After H1, measure each model's perplexity on general text (WikiText-2 test set) and compare with the base model.
- If the LoRA models' perplexity rises less than full FT's, that is a mini reproduction of Biderman 2024. If not, we report that honestly.

### SLURM job template (edit the partition, modules and GPU type to match your cluster)
```bash
#!/bin/bash
#SBATCH --job-name=peft-h1
#SBATCH --partition=gpu           # ← your GPU partition
#SBATCH --gres=gpu:1              # e.g. gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

module load cuda python           # ← your cluster's module names
source ~/venvs/peft/bin/activate  # pip install torch transformers peft datasets accelerate bitsandbytes
export HF_HOME=$HOME/hf_cache     # where pre-downloaded models live

nvidia-smi --query-gpu=name,memory.total --format=csv   # goes on the slide footer
for METHOD in full lora qlora; do
  python train_h1.py --method $METHOD --model Qwen/Qwen2.5-1.5B --steps 300 --seed 42
done
```
Each run should print a single results line (method, trainable params, peak memory in GB, seconds/step, eval loss, adapter size) and append it to `results.csv`. That CSV feeds the slide 4 table directly.

**HPC slide footer:** "Measured on [cluster], 1× [GPU name, memory], PyTorch x.y, PEFT x.y, seed 42, job ID ____."

---

## 5. Task split & timeline (today: 17 Nov)

| Time | Task | Owner |
|---|---|---|
| Now → +1 h | Set up Colab, run Experiment A (both configs) | C |
| Now | Submit HPC job H1 (and H2 if the queue is short); Colab is the fallback if it doesn't finish | D |
| In parallel | Read LoRA / QLoRA / Houlsby abstracts and method sections; draft slides 1–3 | A, B |
| In parallel | Read Biderman 2024; draft slide 5 | D |
| +1.5 h | Fill the slide 4 table and chart with real numbers | C |
| +2 h | Full run-through with timer (target 5:45), cut anything over | All |
| +2.5 h | Q&A drill (Section 7), final fixes, export PDF/PPTX | All |
| Before EOD | **Upload to Moodle** | A |

---

## 6. Sources (starting list for the final slide)

1. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022. arXiv:2106.09685
2. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023. arXiv:2305.14314
3. Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019. arXiv:1902.00751
4. Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, 2020. arXiv:2012.13255
5. Biderman et al., *LoRA Learns Less and Forgets Less*, TMLR 2024. arXiv:2405.09673
6. Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, 2020. arXiv:1910.02054 (16-bytes-per-parameter accounting)
7. Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*, 2023. arXiv:2311.03285
8. Apple, *Introducing Apple's On-Device and Server Foundation Models*, Apple ML Research blog, June 2024
9. Hugging Face PEFT library documentation: https://huggingface.co/docs/peft
10. Our Colab notebook (link)

Optional extras: Liu et al. 2024 *DoRA* (arXiv:2402.09353); Li & Liang 2021 *Prefix-Tuning*; Lester et al. 2021 *Prompt Tuning*.

> Before submitting, open each paper and check that every number we quote on a slide appears in it.

---

## 7. Likely questions with prepared answers

| Question | Short answer |
|---|---|
| How do you choose rank r? | Start small (4–16). The LoRA paper found very low ranks already work well for attention weights. Raise r for harder or knowledge-heavy tasks, and tune it on a validation set. |
| Why initialise B to zero? | So that BA = 0 at the start, and training begins exactly at the pretrained model. |
| Does LoRA slow inference? | Not if merged (W = W₀ + BA). Kept separate (to hot-swap many adapters) it adds a small cost. Adapters always add layers. |
| LoRA vs adapters? | Both freeze the base. Adapters add sequential layers, which adds latency. LoRA adds a parallel low-rank update that can be merged. |
| What does α do? | It scales the update as α/r, so changing r doesn't force re-tuning of the learning rate. |
| Is QLoRA lower quality? | The paper reports matching 16-bit full fine-tuning on their benchmarks thanks to NF4 + double quantization. It is slower to train, though. |
| Why didn't you see the paper's 10,000×? | That ratio is for a 175B model with only some matrices adapted. Our model is about 2,600× smaller, so the ratio is smaller. |
| Which layers do you apply LoRA to? | The original paper used the attention q and v projections. QLoRA found that applying it to all linear layers was needed to match full FT. |
