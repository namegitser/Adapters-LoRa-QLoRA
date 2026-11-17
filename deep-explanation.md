# Parameter-Efficient Fine-Tuning, from scratch

This guide starts from first principles and ends at our measured results. It covers full fine-tuning, Adapters, LoRA and QLoRA, the theory and math behind each, the four experiments we ran (H1–H4), what broke along the way and why, and how to defend every number in a viva.

Every number in this file is either:
- **measured** in our Kaggle run (files in `evidence-kaggle/results/`), or
- **derived by hand** below, with the working shown, or
- **quoted from a paper**, with the paper named.

If you can redo every derivation marked ✍️ on paper, you have mastered the material.

---

## Contents

1. [The story of this project, in one page](#1-the-story-of-this-project-in-one-page)
2. [Foundations you need first](#2-foundations-you-need-first)
3. [The problem: what full fine-tuning really costs](#3-the-problem-what-full-fine-tuning-really-costs)
4. [The PEFT family: Adapters, LoRA, QLoRA](#4-the-peft-family-adapters-lora-qlora)
5. [Our experiments: design, prediction, result, explanation](#5-our-experiments-design-prediction-result-explanation)
6. [Engineering lessons: what broke and why](#6-engineering-lessons-what-broke-and-why)
7. [Trade-offs: when to use what](#7-trade-offs-when-to-use-what)
8. [Hand-calculation drills (with solutions)](#8-hand-calculation-drills-with-solutions)
9. [Viva questions and model answers](#9-viva-questions-and-model-answers)
10. [Cheat sheet of our numbers](#10-cheat-sheet-of-our-numbers)
11. [Glossary](#11-glossary)
12. [References](#12-references)

---

## 1. The story of this project, in one page

**Question.** Can you adapt a pretrained language model to a new task without updating all of its weights, and what do you give up?

**Claim we set out to test** (from the LoRA and QLoRA papers): training a tiny low-rank add-on gives most of the quality of full fine-tuning, for a fraction of the memory and storage, and forgets less of what the base model knew.

**Four hypotheses, each a small, controlled experiment:**

| ID | Hypothesis | Model / data | What we measure |
|---|---|---|---|
| H1 | LoRA and QLoRA match full fine-tuning (FT) quality with far fewer trainable params and less memory | Qwen2.5-0.5B, Alpaca-cleaned | trainable params, peak GPU memory, sec/step, held-out loss, checkpoint size |
| H2 | A bigger model that cannot be fully fine-tuned on one GPU *can* be trained with LoRA/QLoRA | Qwen2.5-3B, Alpaca | does it fit? peak memory |
| H3 | A very small rank r is enough (accuracy plateaus) | DistilBERT, SST-2 sentiment | validation accuracy vs r ∈ {1, 4, 16, 64}, vs full FT |
| H4 | LoRA "forgets less" than full FT | the H1 models, WikiText-2 | perplexity on general text before vs after fine-tuning |

**Principle:** smallest models that still prove the point. We are testing *mechanisms* (memory scaling, rank, forgetting), not trying to build a great chatbot.

**Infrastructure journey** (details in §6):
1. Plaksha HPC with SLURM. This was the primary plan, and the setup and smoke tests passed, but the queue was too slow for the deadline.
2. Google Colab. The free GPU quota ran out mid-run.
3. Kaggle (Tesla T4, 14.6 GB, fp16). This is where all the reported results come from.
4. The HPC jobs were later re-submitted as a bonus (job IDs 14447–14450).

**Headline results** (Kaggle T4, seed 42):

| | Full FT | LoRA r=16 | QLoRA r=16 |
|---|---|---|---|
| H1 trainable params | 494.0 M (100 %) | 8.8 M (1.75 %) | 8.8 M (1.75 %) |
| H1 peak memory | 11.43 GB | 4.64 GB | 4.74 GB |
| H1 sec/step | 2.02 | 2.10 | 2.69 |
| H1 held-out loss (before → after) | 1.318 → 1.285 | 1.318 → 1.272 | 1.412 → 1.322 |
| H1 checkpoint | 988 MB | 35.2 MB | 35.2 MB |
| H2 (3B) peak memory | **OOM** at step 0 | 9.99 GB | 7.61 GB |
| H4 WikiText-2 perplexity change | +2.3 % | +1.0 % | −1.4 % (vs its 4-bit base) |

H3: r=1 85.7 %, r=4 87.6 %, r=16 87.5 %, r=64 88.1 %, full FT 88.1 % (SST-2 validation accuracy).

---

## 2. Foundations you need first

### 2.1 What a "parameter" is, and where they live in a transformer

A transformer layer has two blocks:

- **Attention**: four linear maps (projections), $W_Q, W_K, W_V, W_O$.
- **MLP** (feed-forward): in Qwen2.5 (a "SwiGLU" MLP) three linear maps, $W_{gate}, W_{up}, W_{down}$.

A linear map $y = Wx$ with $x \in \mathbb{R}^{k}$, $y \in \mathbb{R}^{d}$ has a weight matrix $W \in \mathbb{R}^{d \times k}$, i.e. $d \cdot k$ parameters (plus $d$ if it has a bias).

Plus: the **token embedding** (vocab × hidden), small **normalization** vectors, and the **LM head** (hidden → vocab), which in Qwen2.5-0.5B/3B is *tied* to (the same matrix as) the embedding.

**Qwen2.5-0.5B shapes** (hidden $h=896$, MLP width $4864$, 24 layers, 14 query heads and 2 key/value heads of size 64, vocab 151,936):

| Matrix | in → out | Note |
|---|---|---|
| q_proj | 896 → 896 | 14 heads × 64 |
| k_proj | 896 → 128 | only 2 KV heads × 64 (grouped-query attention) |
| v_proj | 896 → 128 | same |
| o_proj | 896 → 896 | |
| gate_proj, up_proj | 896 → 4864 | |
| down_proj | 4864 → 896 | |
| embedding | 151,936 × 896 = **136.1 M** | 28 % of the whole model! |

Total measured: **494,032,768** parameters.

**Qwen2.5-3B**: hidden 2048, MLP 11,008, 36 layers, 2 KV heads of size 128 (so k/v map 2048 → 256), same vocab. Total measured: **3,085,938,688**.

**Grouped-query attention (GQA)** is why k_proj and v_proj are tiny: many query heads share a few key/value heads. This matters for LoRA counts below.

### 2.2 The training objective: next-token cross-entropy

A causal LM outputs, at every position $t$, a probability distribution $p_\theta(\cdot \mid x_{<t})$ over the vocabulary. The loss over the tokens we train on is

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{t \in \text{target}} \log p_\theta(x_t \mid x_{<t})
$$

measured in **nats** (natural log).

- **Response-only loss.** In our Alpaca setup the prompt ("### Instruction … ### Response:") gets label −100, which PyTorch ignores, so the model is graded only on the answer it writes. This is standard instruction tuning (`train_llm.py: build_data`).
- **Held-out loss.** This is the same loss on 200 examples never trained on. It is token-weighted: summed over tokens, then divided by the token count (`evaluate()`).

**Perplexity** is the exponentiated loss:

$$
\text{PPL} = e^{\mathcal{L}}
$$

✍️ *Intuition:* a perplexity of $P$ means the model is, on average, as uncertain as if choosing uniformly among $P$ tokens. Our LoRA held-out loss 1.272 → $e^{1.272} = 3.57$: "as if choosing among ~3.6 tokens".

✍️ *Loss differences are ratios of perplexity:* $\Delta\mathcal{L} = 0.094$ nats means PPL is multiplied by $e^{0.094} = 1.099$, i.e. 10 % worse. (Our 4-bit base: 1.412 − 1.318 = 0.094.)

### 2.3 The optimizer: Adam/AdamW, and why it is expensive

For each parameter $\theta$ with gradient $g_t$ at step $t$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}, \quad
\theta_t = \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\right)
$$

- $m$ (first moment, momentum) and $v$ (second moment) are **two extra numbers per trainable parameter**, kept in fp32 → **8 bytes per trainable parameter** just for the optimizer.
- AdamW = decoupled weight decay ($\lambda$); we used $\lambda=0$ for the LLM runs and 0.01 for DistilBERT.
- **Key consequence:** optimizer memory scales with the number of **trainable** parameters. Freeze 98 % of the weights and you delete 98 % of this cost. This is the heart of PEFT's memory saving.

Other training pieces we used (all in `train_llm.py`):
- **Gradient accumulation**: 4 micro-batches of 4 sequences → effective batch 16 per optimizer step.
- **Learning-rate schedule**: cosine decay with 10 warm-up steps.
- **Gradient clipping**: rescale gradients if their global norm exceeds 1.0.

### 2.4 Number formats: fp32, fp16, bf16, and mixed precision

| Format | Bits (sign/exponent/mantissa) | Largest value | Relative step near 1 | Notes |
|---|---|---|---|---|
| fp32 | 1 / 8 / 23 | 3.4e38 | $2^{-23} \approx 1.2\times10^{-7}$ | "full precision" |
| fp16 | 1 / 5 / 10 | **65,504** | $2^{-10} \approx 9.8\times10^{-4}$ | narrow range: overflows and underflows |
| bf16 | 1 / 8 / 7 | 3.4e38 | $2^{-7} \approx 7.8\times10^{-3}$ | fp32's range, less precision; needs Ampere or newer GPUs |
| NF4 | 4-bit code | n/a | n/a | QLoRA storage only, never computed in (§4.3) |

**Mixed-precision training** (Micikevicius et al. 2018), which we used for full FT:
1. Keep an **fp32 master copy** of the weights.
2. Run forward/backward in fp16/bf16 under `torch.autocast` (fast tensor cores, half the activation memory).
3. Apply the update to the fp32 master weights.

✍️ **Why the fp32 master copy is necessary:** our full-FT learning rate was $\eta = 10^{-5}$. With a typical gradient size of 0.01, an update is $\approx 10^{-7}$. A weight of value 0.5 in fp16 can only change in steps of $2^{-11} \approx 4.9\times10^{-4}$ (the gap between adjacent fp16 numbers near 0.5). So $0.5 + 10^{-7}$ rounds back to $0.5$ in fp16: **the update vanishes**. In fp32 the gap near 0.5 is $\approx 6\times10^{-8}$, so the update survives.

**Loss scaling (fp16 only).** Small gradients (< ~$6\times10^{-8}$) underflow to zero in fp16. `GradScaler` multiplies the loss by a large factor $S$ before backward, so gradients are $S\times$ larger, then divides them by $S$ before the optimizer step. If any gradient overflows to inf/NaN, the step is skipped and $S$ is reduced. bf16 does not need this because it has fp32's exponent range.

**Why fp16 on Kaggle:** the T4 (Turing architecture) has fp16 tensor cores but no fast bf16. The risk is overflow above 65,504 (Qwen activations can be large), which shows up as a NaN loss; it did not happen in our runs. The HPC's RTX PRO 6000 supports bf16, which removes that risk.

### 2.5 Memory accounting: the four buckets

During training, GPU memory holds:

1. **Weights**
2. **Gradients**, one per *trainable* weight
3. **Optimizer states**, Adam's $m$ and $v$ per *trainable* weight
4. **Activations**: intermediate tensors saved in the forward pass for use in backward, plus temporary buffers

✍️ **The 16-bytes-per-parameter rule** (ZeRO paper, Rajbhandari et al. 2020) for mixed-precision Adam:

| Item | Bytes per parameter |
|---|---|
| fp16 working weights | 2 |
| fp16 gradients | 2 |
| fp32 master weights | 4 |
| fp32 Adam $m$ | 4 |
| fp32 Adam $v$ | 4 |
| **Total** | **16** |

Our implementation stores fp32 weights and fp32 gradients and casts on the fly with autocast: 4 + 4 + 8 = **16 bytes** as well, just split differently.

**Activations** grow with batch × sequence length × hidden × layers, and, for LMs, with **batch × sequence × vocabulary** for the output logits.

### 2.6 Gradient checkpointing

Normally every layer's intermediate tensors are kept until backward. **Gradient checkpointing** (Chen et al. 2016) keeps only each layer's *input*, and recomputes the layer's forward during backward.

- Memory: activations drop from "everything in every layer" to "one tensor per layer plus one layer's worth at a time".
- Cost: one extra forward pass, about **+33 % compute** (a forward is ~1/3 of forward + backward).

We turned it on for **all** methods (fair comparison), so the only variable between methods is how weights are stored and trained.

### 2.7 Compute: the 6N rule

✍️ For a model with $N$ parameters, training costs about **$6N$ FLOPs per token**:
- forward: $2N$ (one multiply and one add per weight),
- backward: $4N$ ($2N$ for gradients with respect to activations, and $2N$ for gradients with respect to weights).

With gradient checkpointing, add one more forward: **$8N$**.

**Why LoRA saves less compute than memory:** with frozen weights you skip the $2N$ "weight-gradient" part, but you **still need the activation gradients flowing back through every frozen layer** to reach the LoRA matrices in earlier layers. So LoRA's ideal saving is $8N \to 6N$ (25 %), not 98 %. We will see that, at 0.5B, even this saving is eaten by overhead (§5.2).

---

## 3. The problem: what full fine-tuning really costs

### 3.1 Memory

✍️ **A 7B model, fully fine-tuned with Adam in mixed precision:**
$7\times10^9 \times 16 \text{ B} = 112 \text{ GB}$, *before* activations. One A100 has 80 GB. This is the hook on our slide 2.

✍️ **Our Qwen2.5-3B (H2):** $3.086\times10^9 \times 16 = 49.4 \text{ GB}$ (46 GiB) needed vs 14.6 GB available. Even just fp32 weights plus fp32 gradients ($8 \times 3.086\text{B} = 23$ GiB) is more than the T4 has. We measured exactly this: the weights alone took 11.50 GiB after loading, and it ran out of memory during the very first backward pass.

### 3.2 Storage and serving

Every fully fine-tuned task is a **complete copy** of the model. Ten tasks on a 7B model = ten 14 GB files (fp16). Our 0.5B full-FT checkpoint was 988 MB; the LoRA adapter was 35.2 MB.

### 3.3 Catastrophic forgetting

Updating every weight to fit a narrow task can damage general abilities the model already had (McCloskey & Cohen 1989; for LLMs, Biderman et al. 2024). H4 measures this directly.

### 3.4 The key question

> Do we really need to change all $N$ numbers to adapt a model?

The answer, from intrinsic-dimension research: **no**.

---

## 4. The PEFT family: Adapters, LoRA, QLoRA

### 4.1 Why a tiny update can work: intrinsic dimensionality

**Aghajanyan et al. 2020** fine-tuned models while restricting the update to a random low-dimensional subspace: $\theta = \theta_0 + P z$, with $z \in \mathbb{R}^{d}$ small and $P$ a fixed random projection. A few hundred to a few thousand dimensions reached 90 % of full fine-tuning performance on tasks like MRPC with RoBERTa. They also found that **larger pretrained models have smaller intrinsic dimension**: pretraining compresses the problem.

**Hu et al. 2021 (LoRA)** made the hypothesis concrete: the **weight update** $\Delta W$ needed for adaptation has **low rank**.

### 4.2 Adapters (Houlsby et al. 2019)

Insert a small **bottleneck** module after the attention and MLP sub-blocks of every layer:

$$
\text{Adapter}(h) = h + W_{up}\, \sigma(W_{down}\, h), \qquad W_{down} \in \mathbb{R}^{m \times d},\; W_{up} \in \mathbb{R}^{d \times m},\; m \ll d
$$

- Params per adapter ≈ $2md + m + d$; two adapters per layer.
- The residual connection plus near-zero initialization means the adapter starts close to the identity function, so training starts at the pretrained model.
- **Result (paper):** within 0.4 % of full FT on GLUE, training 3.6 % extra parameters per task.
- **Cost:** the adapter is **in series**, an extra sequential step in every layer, so it adds **inference latency** that cannot be removed.

### 4.3 LoRA (Hu et al. 2021): the core of this project

#### 4.3.1 The equation

Freeze the pretrained $W_0 \in \mathbb{R}^{d\times k}$. Learn a low-rank update:

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x,
\qquad A \in \mathbb{R}^{r\times k},\; B \in \mathbb{R}^{d \times r},\; r \ll \min(d,k)
$$

- $A$ compresses the input to $r$ numbers; $B$ expands back to $d$.
- $\text{rank}(BA) \le r$.
- $\alpha/r$ is a fixed **scaling** factor (§4.3.5).

#### 4.3.2 Parameter count

✍️ Full matrix: $d \cdot k$. LoRA: $r(d + k)$. **Compression ratio**: $\frac{dk}{r(d+k)}$.

Example (slide 3): a 4096 × 4096 projection, $r = 8$:
$dk = 16{,}777{,}216$ ; $r(d+k) = 8 \times 8192 = 65{,}536$ ; ratio $= 256\times$; fraction $= 0.39\%$.

#### 4.3.3 ✍️ Our exact LoRA count for Qwen2.5-0.5B (r = 16, all linear layers)

Per layer, $r(d_{in} + d_{out})$:

| Matrix | $d_{in}+d_{out}$ | × r=16 |
|---|---|---|
| q_proj 896→896 | 1792 | 28,672 |
| k_proj 896→128 | 1024 | 16,384 |
| v_proj 896→128 | 1024 | 16,384 |
| o_proj 896→896 | 1792 | 28,672 |
| gate_proj 896→4864 | 5760 | 92,160 |
| up_proj 896→4864 | 5760 | 92,160 |
| down_proj 4864→896 | 5760 | 92,160 |
| **per layer** | | **366,592** |

× 24 layers = **8,798,208**, exactly what PEFT printed (`trainable params: 8,798,208`).

- **Percentage:** PEFT reports $8{,}798{,}208 / (494{,}032{,}768 + 8{,}798{,}208) = 1.7497\%$. The denominator includes the LoRA weights themselves.
- **Ratio:** $494.0\text{M} / 8.8\text{M} = 56\times$ fewer trainable parameters.

✍️ **Qwen2.5-3B** (h=2048, MLP 11,008, k/v → 256, 36 layers):
- q, o: $16\times4096 = 65{,}536$ each
- k, v: $16\times2304 = 36{,}864$ each
- gate, up, down: $16\times13{,}056 = 208{,}896$ each

Per layer: 831,488. × 36 = **29,933,568**, again exactly what PEFT printed (0.96 %).

Note that at 3B the fraction is smaller (0.96 % vs 1.75 %): $r(d+k)$ grows linearly in width while $dk$ grows quadratically, so **bigger models get relatively cheaper adapters**. Extrapolate to GPT-3's 12,288-wide matrices with few adapted matrices and you get the paper's 10,000×.

#### 4.3.4 Initialization: why B = 0 and A ≠ 0

- $A$: random (Kaiming-uniform); $B$: zeros. So $BA = 0$ at step 0, and **the model starts exactly at the pretrained model**.

✍️ **Gradients.** Let $G = \partial\mathcal{L}/\partial h$ and $s = \alpha/r$. Then

$$
\frac{\partial\mathcal{L}}{\partial B} = s\, G\,(Ax)^\top, \qquad \frac{\partial\mathcal{L}}{\partial A} = s\, B^\top G\, x^\top
$$

- At step 0, $B=0$, so $\partial\mathcal{L}/\partial A = 0$: only $B$ moves first. After that both learn.
- If both were zero, **both gradients would be zero forever**: training would be stuck.
- If both were random, the model would start at a *perturbed*, worse model.

#### 4.3.5 The α/r scaling

The update is multiplied by $\alpha/r$.

- **Why:** when you change $r$, the size of $BA$ changes. Dividing by $r$ keeps the effective update size roughly constant, so you do not have to re-tune the learning rate for every $r$.
- **Ours:** H1/H2 $\alpha=32, r=16 \Rightarrow s=2$. H3 $\alpha = 2r \Rightarrow s = 2$ at every rank (a deliberate constant scale, so the sweep compares ranks fairly).
- **Further reading:** rsLoRA (Kalajdzievski 2023) argues $\alpha/\sqrt{r}$ is the better scaling for large $r$.

#### 4.3.6 Merging: zero inference latency

After training, compute $W = W_0 + \frac{\alpha}{r} BA$ once. Inference then uses one matrix, exactly as fast as the original model. Unlike adapters, LoRA's path is **parallel**, so it can be folded away.

Or **keep adapters separate** and hot-swap them on one shared base (Apple Intelligence; S-LoRA/vLLM serving thousands of adapters). This costs a small extra matmul per request.

#### 4.3.7 Which matrices to adapt

- The LoRA paper adapted only $W_Q, W_V$ (and found even $r=1$–$4$ good on GPT-3).
- QLoRA found that **all linear layers** are needed to match full FT. We used `target_modules="all-linear"` for H1/H2 and `q_lin, v_lin` (DistilBERT's query/value) for H3, the paper's original setting.

#### 4.3.8 Why LoRA trains with a higher learning rate

We used $2\times10^{-4}$ for LoRA vs $10^{-5}$ for full FT (the community default ratio).
- Full FT moves 494 M weights that each already hold useful values; small steps avoid wrecking them.
- LoRA moves a small set of new weights starting from zero ($B$), which must travel further to have an effect.

### 4.4 QLoRA (Dettmers et al. 2023): LoRA on a 4-bit base

LoRA still stores the frozen base in 16-bit (2 bytes per parameter). QLoRA stores the **frozen base in 4 bits** and keeps LoRA in 16/32-bit. Gradients flow *through* the 4-bit weights into the LoRA matrices, but the 4-bit weights themselves never change.

Three ingredients:

#### 4.4.1 NF4: 4-bit NormalFloat

4 bits = 16 possible values. **Which 16?**
- Pretrained weights are approximately **zero-mean normally distributed**.
- NF4 puts the 16 levels at **quantiles of a standard normal**, so each level is used equally often. For normal data this is **information-theoretically optimal**: no level is wasted on values that rarely occur.
- The levels are normalized to [−1, 1]. They are asymmetric, with 8 positive and 7 negative levels plus an exact 0, so that zero is represented exactly.

The 16 NF4 levels (bitsandbytes, rounded):

```
-1.0000 -0.6962 -0.5251 -0.3949 -0.2844 -0.1848 -0.0911  0.0000
 0.0796  0.1609  0.2461  0.3379  0.4407  0.5626  0.7230  1.0000
```

**Block-wise absmax quantization:** split weights into **blocks of 64**. For each block:
1. Store $c = \max|w|$ (the absmax) in fp32.
2. Normalize $w/c \in [-1,1]$.
3. Store the index (0–15) of the nearest NF4 level: 4 bits.

To use the weight: $\hat w = c \cdot \text{level}[\text{index}]$ (de-quantize, in bf16/fp16, just before each matmul).

✍️ **Worked example** (a 4-weight "block" for simplicity): $w = [0.12, -0.03, 0.45, -0.30]$.
1. $c = 0.45$. Normalized: $[0.267, -0.067, 1.000, -0.667]$.
2. Nearest levels:
   - 0.267 → 0.2461 (distance 0.021, vs 0.071 to 0.3379)
   - −0.067 → −0.0911 (0.024, vs 0.067 to 0)
   - 1.0 → 1.0
   - −0.667 → −0.6962 (0.029)
3. De-quantized: $0.45\times[0.2461, -0.0911, 1, -0.6962] = [0.1107, -0.0410, 0.4500, -0.3133]$.
4. Errors: $[-0.009, -0.011, 0, -0.013]$. Small but non-zero, which is exactly the "4-bit costs a bit" we measured (§5.2).

**Why blocks?** One outlier weight would otherwise set $c$ for the whole matrix, crushing every other weight into the few levels near zero. Small blocks contain the damage.

#### 4.4.2 Double quantization

✍️ Each block of 64 weights stores one fp32 constant: $32/64 = 0.5$ extra bits per parameter → **4.5 bits** total.

Double quantization **quantizes the constants too**: 8-bit constants, with one fp32 constant per 256 of them:

$$
\frac{8}{64} + \frac{32}{64\times256} = 0.125 + 0.00195 = 0.127 \text{ bits/param}
$$

Total ≈ **4.127 bits/param**, a saving of ~0.373 bits/param (≈ 3 GB on a 65B model, per the paper).

#### 4.4.3 Paged optimizers

QLoRA uses NVIDIA unified memory to page optimizer states to CPU RAM during memory spikes, avoiding OOM. **We did not use this** (plain AdamW); our LoRA optimizer states were tiny anyway (8.8 M × 8 B = 70 MB).

#### 4.4.4 Paper result

A 65B model fine-tuned on **one 48 GB GPU**, matching 16-bit full fine-tuning on their benchmarks.

#### 4.4.5 The price of QLoRA

1. **Slower:** every forward *and* backward (and recompute, with checkpointing) must de-quantize the 4-bit weights. We measured **1.28×** slower than LoRA (H1) and 1.25× (H2).
2. **Quantization error:** the frozen base is a slightly different model. We measured held-out loss 1.318 → 1.412 just from quantizing, before any training.
3. **Merging is lossy:** $W_0$ is 4-bit, so merging $BA$ in means de-quantize, add, and (to keep it small) re-quantize, which adds error. In practice you merge into a 16-bit copy of the base.

### 4.5 Other members (for context; not tested)

| Method | What it trains | Note |
|---|---|---|
| Prompt / prefix tuning | learned "virtual tokens" prepended to the input or to each layer | very few params; weaker on small models |
| (IA)³ | per-channel scaling vectors | even fewer params than LoRA |
| DoRA (2024) | splits $W$ into magnitude and direction; LoRA on the direction | closes part of the LoRA–full-FT gap |
| BitFit | only the bias terms | the simplest baseline |

---

## 5. Our experiments: design, prediction, result, explanation

### 5.1 Experimental design principles (why the comparison is fair)

| Held constant across methods | Value (H1) |
|---|---|
| Model | Qwen2.5-0.5B |
| Data | the same 4,800 Alpaca-cleaned training + 200 held-out examples (fixed data seed 1234) |
| Steps, batch | 300 optimizer steps × 16 sequences = exactly **one epoch** |
| Sequence length | ≤ 512 tokens |
| Gradient checkpointing | on for all |
| Seed | 42 |
| Hardware | one Tesla T4, fp16 autocast |

Only the **method** changes. Measurements:
- `torch.cuda.max_memory_allocated()` (reset after loading, so it captures training's peak),
- wall-clock seconds per step (first 5 steps excluded as warm-up),
- token-weighted held-out loss,
- parameter counts from PEFT,
- the size of the saved files.

**Out-of-memory is a result, not a crash:** the script catches it, records the stage, and exits normally, so the next method still runs (that is how H2's OOM became data).

**Honesty about limits:** one seed (no error bars), small models, short training. These measure *mechanisms*, not leaderboard quality.

### 5.2 H1: full FT vs LoRA vs QLoRA (Qwen2.5-0.5B)

**Prediction from theory:**
- Trainable params ≈ 56× fewer (computed in §4.3.3).
- Memory: full FT ≈ 16 B × 494 M ≈ 7.4 GiB + activations. LoRA ≈ 2 B × 494 M ≈ 0.92 GiB + tiny LoRA states + activations.
- QLoRA should save a little more on weights and be slower.

**Result:**

| Method | Trainable | Weights after load | Peak | Sec/step | Tokens/s | Held-out loss | Checkpoint |
|---|---|---|---|---|---|---|---|
| Full FT | 494.0 M | 1.840 GiB | 11.43 | 2.019 | 1466 | 1.318 → 1.285 | 988 MB |
| LoRA | 8.8 M | 0.953 GiB | 4.64 | 2.097 | 1411 | 1.318 → 1.272 | 35.2 MB |
| QLoRA | 8.8 M | 0.713 GiB | 4.74 | 2.688 | 1101 | 1.412 → 1.322 | 35.2 MB |

#### ✍️ Explaining every memory number by hand

**Full FT weights after load:** $494{,}032{,}768 \times 4\text{ B} / 2^{30} = 1.840$ GiB ✔ (fp32).

**Full FT peak:**
- Persistent state = weights 4 + grads 4 + Adam 8 = 16 B × 494 M = **7.36 GiB**.
- Peak 11.43 → activations and temporaries ≈ **4.1 GiB**.

**LoRA weights after load:**
- Base in fp16: $494.0\text{M}\times2 = 0.920$ GiB.
- LoRA in fp32: $8.8\text{M}\times4 = 0.033$ GiB.
- Sum **0.953** ✔.

**LoRA peak:**
- Persistent ≈ 0.953 + LoRA grads and Adam ($8.8\text{M}\times12\text{ B} = 0.10$ GiB) ≈ 1.05 GiB.
- Peak 4.64 → activations ≈ **3.6 GiB**.

**QLoRA weights after load** (the nicest derivation in the project):
- `prepare_model_for_kbit_training` keeps non-quantized parameters in **fp32**. The biggest of these is the embedding: $151{,}936\times896 = 136.1$ M params × 4 B = **0.507 GiB**.
- The remaining ≈ 357.9 M linear-layer params in NF4: × 0.5 B = **0.167 GiB**, plus quantization constants (≈ 0.127 bits/param ≈ 0.005 GiB).
- LoRA in fp32: **0.033 GiB**.
- Sum ≈ 0.507 + 0.167 + 0.005 + 0.033 = **0.712 GiB** vs measured **0.713** ✔.

#### Why QLoRA barely saved memory at 0.5B (an important insight)

Activations (~3.6–4 GiB) **dominate** at this size, and they are the same for all three methods. Why so large for such a small model? **The vocabulary.**
- Output logits are batch × sequence × 151,936.
- At the worst case of 4 × 512 tokens that is 311 M numbers. In fp32 for the loss, that is **1.16 GiB**, plus the softmax gradient of the same size.
- Gradient checkpointing shrinks the per-layer activations, but not this.

QLoRA only saves on *weights*: 0.953 → 0.713 GiB = 0.24 GiB. De-quantization buffers eat that back, so **4.74 ≥ 4.64**. At 3B, where the weights are 5.75 GiB in fp16, the 4-bit saving finally matters (H2). **Lesson: QLoRA's benefit scales with model size; for small models, activations dominate.**

#### ✍️ Speed: why LoRA was *not* faster

- Theory (§2.7): LoRA ideal ≈ 6N vs full FT 8N FLOPs/token (with checkpointing), up to 25 % less.
- Measured: LoRA 2.10 s vs full 2.02 s, i.e. slightly slower.
- Reasons:
  1. LoRA adds 7 × 24 = 168 extra small matmuls (A then B) plus dropout per forward, each a separate GPU kernel. On a small model these **kernel-launch and memory-bound overheads** rival the saved FLOPs.
  2. Full FT's extra work (the weight gradients) is large, efficient matmuls that run at good GPU utilization.
- The LoRA paper's 25 % training speed-up was on GPT-3 175B, where matmuls are huge.
- **Utilization check:** full FT processed 1466 tok/s × 8 × 494 M FLOPs ≈ 5.8 TFLOP/s, about 9 % of the T4's ~65 TFLOP/s fp16 peak. Small models do not saturate a GPU, so overhead matters.

QLoRA 2.69 s = **1.28× LoRA**: the de-quantization on every pass (×3 with recompute), exactly as theory predicts.

#### Quality

- **Full FT:** 1.318 → 1.285 (−0.033 nats).
- **LoRA:** 1.318 → **1.272** (−0.046), slightly *better* than full FT on this short, one-epoch instruction tuning. Reasons: LoRA's higher LR suits short schedules, and the low-rank constraint acts as a regularizer. One seed, so treat the difference as "comparable" rather than "LoRA wins".
- **QLoRA:** starts at **1.412** (the cost of 4-bit quantization, +0.094 nats ≈ 10 % worse perplexity) and recovers to 1.322 (−0.090). The LoRA update partly learns to compensate for quantization error, but does not fully close the gap at 0.5B. Small models are more sensitive to quantization.

#### Checkpoint sizes ✍️

- **Full FT:** saved in bf16, $494\text{M} \times 2\text{ B} = 988$ MB ✔.
- **LoRA/QLoRA:** only A and B in fp32, $8.8\text{M} \times 4\text{ B} = 35.2$ MB ✔.
- Ratio **28×**. Note: 28 not 56, because the full model was saved in 2 bytes/param and the adapter in 4 bytes/param. Saving the adapter in bf16 would give 17.6 MB, i.e. 56×.

#### The overfitting episode (and what it teaches)

- **Kaggle Version 1** used 2,000 examples (1,800 train): 300 steps × 16 = 4,800 sequences ≈ **2.7 epochs**.
- Full FT's held-out loss went **up** (1.349 → 1.426) while training loss fell to 0.73: classic **memorization**. All 494 M weights have the capacity to memorize 1,800 examples.
- **Fix:** 5,000 examples (4,800 train + 200 held out) so the same 300 steps are exactly **one epoch**. Steps, batch and LR are unchanged, so memory and speed are still comparable.
- After the fix, full FT improves (1.318 → 1.285).
- **Theory link:** capacity vs data. LoRA's rank constraint limits capacity, so it overfits less (QLoRA still improved in V1: 1.448 → 1.402). This is the regularization side of "LoRA learns less".

### 5.3 H2: a model too big to fully fine-tune (Qwen2.5-3B, 50 steps)

**Prediction:**
- Full FT needs 16 × 3.086 B ≈ 46 GiB ≫ 14.6 GB → OOM.
- LoRA: fp16 base 5.75 GiB + 30 M LoRA × 16 B ≈ 0.45 GiB + activations ≈ 10 GiB.
- QLoRA: ≈ 2.6 GiB weights + the same → ≈ 7–8 GiB.

**Result:**

| Method | Outcome | Weights after load | Peak | Sec/step | Loss after 50 steps |
|---|---|---|---|---|---|
| Full FT | **OOM**, stage `train_step0_forward_backward` | 11.50 GiB | 14.33 (limit) | n/a | n/a |
| LoRA | trains | 5.86 GiB | 9.99 | 8.04 | 1.007 |
| QLoRA | trains | 2.62 GiB | 7.61 | 10.01 | 1.040 |

✍️ Explanations:
- **Full FT:** $3{,}085{,}938{,}688 \times 4 = 11.50$ GiB just for fp32 weights ✔. The first backward started creating 11.5 GiB of gradients → OOM after 0 steps. Adam's 23 GiB was never even reached.
- **LoRA weights:** $3.086\text{B}\times2 = 5.75$ GiB + $29.9\text{M}\times4 = 0.11$ GiB = **5.86** ✔.
- **QLoRA weights:**
  - embedding $151{,}936\times2048 = 311.2$ M × 4 B = 1.16 GiB (fp32)
  - remaining ≈ 2.77 B × 0.5 B = 1.29 GiB
  - constants ≈ 0.04 GiB
  - LoRA 0.11 GiB
  - total ≈ **2.60** vs measured **2.62** ✔
- **QLoRA vs LoRA:** 1.25× slower; final loss slightly worse (1.040 vs 1.007), the same quantization cost as in H1.
- **Scale effect:** the 3B model's loss (~1.0) is far below the 0.5B's (~1.27). Bigger pretrained models start better, which is why "fine-tune the biggest model you can fit" matters, and why QLoRA, by making bigger models fit, can beat LoRA on a smaller one.

Note: H2 used the default 2,000 examples (1,800 train), but only 50 × 16 = 800 sequences were seen (less than one epoch), so there was no overfitting concern.

### 5.4 H3: how small can the rank be? (DistilBERT, SST-2)

**Setup:**
- DistilBERT: 6 layers, hidden 768, 66 M params. SST-2: binary movie-review sentiment.
- Kaggle version: 20,000 training sentences, 1 epoch, batch 32, max length 128.
- LoRA on `q_lin, v_lin`, $\alpha=2r$, LR 5e-4. Full FT LR 2e-5.
- Validation: the 872 labelled sentences.

✍️ **LoRA count:**
- Each adapted matrix is 768 × 768, so $r(768+768) = 1536r$.
- 2 matrices × 6 layers = 12 matrices → $12\times1536r = \mathbf{18{,}432\,r}$.
- r=1: 18,432; r=4: 73,728; r=16: 294,912; r=64: 1,179,648. The script compared measured vs $r(d+k)$ for every rank: **exact match** at all four.

✍️ **The classification head** is fully trained too (`modules_to_save`):
- pre_classifier $768\times768+768 = 590{,}592$
- classifier $768\times2+2 = 1{,}538$
- total **592,130**, constant across ranks. Check: r=1 → 18,432 + 592,130 = 610,562 ✔ (matches "Trainable incl. head").

**Result:**

| r | LoRA params | Trainable incl. head | Val. accuracy |
|---|---|---|---|
| 1 | 18,432 | 610,562 (0.90 %) | 85.67 % |
| 4 | 73,728 | 665,858 (0.98 %) | 87.61 % |
| 16 | 294,912 | 887,042 (1.31 %) | 87.50 % |
| 64 | 1,179,648 | 1,771,778 (2.58 %) | 88.07 % |
| full FT | n/a | 66,955,010 (100 %) | 88.07 % |

**Interpretation:**
- **Big jump from r=1 to r=4** (+1.9 points), then a **plateau**. r=64 equals full FT.
- This matches the LoRA paper's finding that small ranks suffice, i.e. the useful update is low-rank.
- Why r=1 is *somewhat* weaker here but fine in the paper: on GPT-3 175B, $W_Q, W_V$ are 12,288-wide and the model's intrinsic dimension is tiny. DistilBERT is small, so its updates need a bit more rank (Aghajanyan: smaller models have larger intrinsic dimension).

**Caveats (say these before anyone asks):**
1. **One seed:** differences under ~0.5 points (r=4 vs r=16) are noise. On 872 examples, one sentence is 0.11 %, and the binomial standard error at 88 % is $\sqrt{0.88\cdot0.12/872} \approx 1.1\%$. So even the r=1 vs r=64 gap of 2.4 points is only about 2 standard errors. Suggestive, not conclusive.
2. **The head dominates the trainable count at low r** (592 K of 610 K at r=1). A head-only baseline (r=0) would isolate the adapter's contribution. We did not run it.
3. Short training (1 epoch on 20 K sentences). The HPC version runs the full sweep (r ∈ {1,2,4,8,16,64} × 2 seeds, full 67 K, 2 epochs).

Other measurements:
- Peak memory: full FT 1.29 GB vs LoRA ~0.67 GB.
- Wall time: 40 s vs ~23 s. In a small model, the optimizer and gradients of all 67 M params are a large share of the work, so here LoRA *is* faster.

### 5.5 H4: does LoRA forget less? (WikiText-2 perplexity)

**Setup:** evaluate each H1 model on the **WikiText-2 test set** (Wikipedia text, unrelated to Alpaca): 299,077 scored tokens. General-text perplexity rising after fine-tuning = forgetting.

✍️ **Sliding-window perplexity** (`eval_h4.py`): window 1024 tokens, stride 512. Each window scores only the tokens not scored by the previous window, so every token is scored once with up to 512–1023 tokens of context:

$$
\text{PPL} = \exp\left(\frac{1}{T}\sum_{t=1}^{T} -\log p(x_t \mid \text{context}_t)\right)
$$

**Result:**

| Model | WikiText-2 PPL | Change vs its base | Alpaca held-out loss |
|---|---|---|---|
| Base (16-bit) | 12.501 | reference | 1.318 |
| Base (4-bit NF4) | 13.872 | reference | 1.412 |
| Full FT | 12.794 | **+2.3 %** | 1.285 |
| LoRA | 12.625 | **+1.0 %** | 1.272 |
| QLoRA | 13.679 | −1.4 % | 1.322 |

**Interpretation:**
- **Full FT forgot ~2.3×** as much as LoRA, while learning the task no better (Alpaca loss 1.285 vs 1.272). This is a small-scale reproduction of **Biderman et al. 2024, "LoRA Learns Less and Forgets Less"**.
- **Why, theoretically:** LoRA constrains $\Delta W$ to rank $r$ and to a small norm (it starts at 0 and moves for one epoch), so the function on unrelated inputs changes little. Biderman et al. found that full FT learns weight perturbations with rank **10–100× higher** than typical LoRA configurations, enough to overwrite general knowledge.
- **The QLoRA −1.4 % needs care:**
  - It is relative to the **4-bit** base, which already starts 11 % worse than 16-bit (13.87 vs 12.50).
  - Fine-tuning partly repaired quantization error that affects all text, not just Alpaca.
  - Relative to the original 16-bit model, QLoRA is +9.4 %.
  - So it is **not** "negative forgetting".
- **Our learning results do not show "LoRA learns less":** Biderman's learning gap appears on hard domains (code, math) with long training. On short instruction tuning, LoRA kept up.

✍️ **Consistency check:** the 4-bit quantization cost appears in two independent measurements. Alpaca loss rose 0.094 nats (PPL ×1.099) and WikiText-2 PPL rose ×1.110. About 10 % worse either way.

---

## 6. Engineering lessons: what broke and why

These count as part of mastery: they are the questions "how did you actually run this?" will lead to.

### 6.1 HPC with SLURM (Plaksha cluster)

**SLURM vocabulary:**

| Term | Meaning |
|---|---|
| job, `sbatch` | a batch script submitted to the queue |
| partition | a group of nodes (`rtx`) |
| QoS / account | policy and limits (`general`: 1 GPU per user, 2 running and 10 submitted jobs, 12 h maximum) |
| `--gres=gpu:1g.24gb:1` | ask for one **MIG slice** (a hardware-partitioned 24 GB piece of an RTX PRO 6000) |
| `--requeue` | if a higher-priority job preempts ours, restart it later; our scripts skip runs already finished (status `ok`/`OOM`) |
| `--dependency=afterany:<id>` | start H4 after H1 ends, whatever its exit status. `afterok` would block forever if H1 partly failed, and the stuck job would still count against the 10-job limit |
| Array jobs | each task counts as a job, so a 14-task array breaks the 10-job limit. H3 therefore runs sequentially in one job |
| `squeue --start` | estimated start time: a worst-case estimate (assumes jobs ahead run to their time limit). Our check job started much earlier than estimated |
| `sacct` | history and exit codes of finished jobs |

**Problems hit:**
1. **`OSError: [Errno 14] Bad address`:** pip/conda writing large files into Lustre `/home` from the login node failed, while plain `cp` worked. Fix: install into `/tmp` (`pip --root`) and copy with `cp -a`. Lesson: read the cluster guide before improvising.
2. **Wrong Python/torch:** the cluster's way is `module load python` (Python 3.13 + torch 2.13 built for the cluster's GPUs), with a venv created with `--system-site-packages`, and **never pip-installing torch**.
3. **Storage:** `/home` has a 20 GB quota, so the 7.1 GB model cache went to `/scratch`, which is huge but purged without warning. That is why we verified it before submitting.
4. **Offline compute nodes:** all models and datasets were pre-downloaded and jobs run with `HF_HUB_OFFLINE=1`.
5. **Library version drift:** transformers 5 renamed `torch_dtype` → `dtype` and removed `low_cpu_mem_usage`. The model ID `distilbert-base-uncased` 404'd and had to become `distilbert/distilbert-base-uncased`.
6. **Stale file on the cluster:** an old `check.sbatch` was run because the new one was never copied. Lesson: verify with `grep` on the cluster after every change.
7. **Smoke test first:** a 3-step run of every code path (`check.sbatch`) found problems in minutes instead of an hour into H1.

### 6.2 Colab

The free GPU quota ran out mid-run ("No backend with GPU available"), and the runtime and its files were lost. Lesson: free notebooks are not durable.

### 6.3 Kaggle (where the results come from)

1. **"Save & Run All (Commit)"** runs the notebook in the background via papermill: it survives closing the laptop. But papermill **stops at the first Python exception**. In Version 1 the report cell raised `FileNotFoundError` (LoRA had failed, so `h1_ratios.txt` did not exist), and H2–H4 never ran. Fix: guard every file read.
2. **torchao incompatibility:** PEFT 0.21 checks torchao when wrapping plain linear layers; Kaggle's preinstalled torchao 0.10 was "incompatible", so **LoRA failed while QLoRA worked** (bitsandbytes layers are dispatched first). Fix: `pip uninstall torchao`. Lesson: preinstalled environments carry hidden version conflicts.
3. **Skip logic:** "skip if the result file exists" also skipped *failed* runs. Fixed to skip only `status == "ok"`.
4. **fp16 NaN risk:** on a T4 without bf16, Qwen can overflow fp16's 65,504 range. It did not happen; the fallback was bf16 (slower, emulated).

### 6.4 General lessons

- Treat expected failures (OOM in H2) as **data**: catch, record, continue.
- Make every job **idempotent** (safe to re-run) and **resumable**.
- **Control variables** ruthlessly: same data split, seed, steps, sequence length, checkpointing.
- Check measurements against **hand calculations**. Every memory number above matches theory to ~1 %, which is how you know the measurement is right.

---

## 7. Trade-offs: when to use what

| Situation | Choose | Why |
|---|---|---|
| Model fits comfortably, max quality needed, lots of new knowledge (new language or domain, code, math) | **Full FT** | highest capacity; Biderman: LoRA learns less on hard domains |
| Limited GPU memory, style/format/instruction adaptation | **LoRA** | ~same quality, 2.5× less memory (ours), tiny checkpoints, mergeable |
| Model only fits in 4-bit | **QLoRA** | the only way to fine-tune it on that GPU (H2); pay ~25–30 % speed and some quantization error |
| Many tasks on one base model | **LoRA adapters, unmerged** | MB-sized per task, hot-swappable (S-LoRA, Apple) |
| Must keep general abilities | **LoRA** (or a lower LR / fewer steps) | forgets less (H4) |
| Prompting or RAG already solves it | **No fine-tuning** | zero training cost |
| Inference latency critical, single task | **Merged LoRA** (not adapters) | zero added latency |

**Hidden costs of LoRA/QLoRA:**
- new hyperparameters: $r$, $\alpha$, target modules, LR
- not faster per step at small scale (ours)
- QLoRA is slower, and its base model is slightly different
- unmerged adapters add a small inference cost

---

## 8. Hand-calculation drills (with solutions)

Try each before reading the answer.

**D1.** LoRA with $r=8$ on a 4096 × 11008 MLP matrix. How many parameters, and what fraction?
> $8\times(4096+11008) = 120{,}832$; full $= 45{,}088{,}768$; fraction 0.27 %.

**D2.** Memory to fully fine-tune a 13B model with mixed-precision Adam, weights + grads + optimizer only?
> $13\times10^9\times16 = 208$ GB.

**D3.** The same 13B with QLoRA (NF4 + double quant), base weights only?
> $13\times10^9 \times 4.127/8 \approx 6.7$ GB.

**D4.** Same 13B with LoRA (fp16 base)?
> $26$ GB, plus LoRA states and activations.

**D5.** Held-out loss 1.272. Perplexity?
> $e^{1.272} = 3.57$.

**D6.** Base loss 1.318, QLoRA base loss 1.412. By what factor did perplexity rise?
> $e^{0.094} = 1.099$, i.e. +9.9 %.

**D7.** Bits per parameter for NF4 with block size 64, (a) without and (b) with double quantization (8-bit constants, second-level blocks of 256)?
> (a) $4 + 32/64 = 4.5$; (b) $4 + 8/64 + 32/16384 = 4.127$.

**D8.** Quantize $[0.2, -0.1, 0.05, -0.4]$ to NF4 (absmax per block). De-quantized values?
> $c = 0.4$; normalized $[0.5, -0.25, 0.125, -1.0]$.
> Nearest levels: 0.5 → 0.4407 or 0.5626? $|0.5-0.4407| = 0.059$, $|0.5-0.5626| = 0.063$ → **0.4407**. −0.25 → −0.2844 (0.034; −0.1848 is 0.065 away). 0.125 → 0.1609 (0.036; 0.0796 is 0.045 away). −1.0 → −1.0.
> De-quantized: $0.4\times[0.4407, -0.2844, 0.1609, -1] = [0.176, -0.114, 0.064, -0.400]$.

**D9.** Why does $\partial\mathcal{L}/\partial A = 0$ at step 0 in LoRA?
> $\partial\mathcal{L}/\partial A = s B^\top G x^\top$ and $B = 0$ at init.

**D10.** A Qwen2.5-0.5B LoRA adapter at r=16 has 8,798,208 params. Size in fp32? In bf16?
> 35.2 MB; 17.6 MB.

**D11.** Training FLOPs for one epoch of H1 full FT: 4,800 examples × ~185 tokens average (from 1466 tok/s × 2.02 s / 16 sequences), with checkpointing?
> Tokens ≈ 888 K; $8 \times 494\text{M} \times 888\text{K} \approx 3.5\times10^{15}$ FLOPs = 3.5 PFLOP. At ~5.8 TFLOP/s achieved → ~600 s. We measured 644 s wall ✔.

**D12.** Logit tensor for batch 4 × 512 tokens, vocab 151,936, fp32?
> $4\times512\times151{,}936\times4 = 1.245\times10^9$ B = 1.16 GiB.

**D13.** Why does 300 steps × batch 16 over 1,800 examples overfit, but over 4,800 not (as much)?
> 4,800 sequences seen: 2.67 epochs vs 1.0 epoch. Repeating examples lets a high-capacity model memorize them.

**D14.** SST-2 validation has 872 examples. Standard error of an 88 % accuracy?
> $\sqrt{0.88\times0.12/872} = 1.1\%$.

**D15.** LoRA count for DistilBERT, r=8, on q_lin and v_lin?
> $12 \times 8 \times 1536 = 147{,}456$.

**D16.** Explain the compression ratio formula, and show it grows with model width at fixed $r$.
> $\frac{dk}{r(d+k)}$; for square $d=k$: $\frac{d}{2r}$, which is linear in $d$. Wider models give bigger savings.

---

## 9. Viva questions and model answers

**Q: What is the core idea of LoRA, in one sentence?**
Freeze the pretrained weights and learn the update as a product of two thin matrices, $\Delta W = BA$ with rank $r \ll d$, because fine-tuning updates are empirically low-rank.

**Q: Why does LoRA save memory?**
Gradients and Adam states exist only for trainable parameters; LoRA trains ~1–2 % of them. The frozen base can also be stored in 16-bit (LoRA) or 4-bit (QLoRA) without fp32 master copies. Activations are *not* reduced; at small scale they dominate (our 0.5B: ~3.6 GiB of 4.64).

**Q: Why wasn't LoRA faster in your H1?**
Backward must still propagate activation gradients through every frozen layer, so the ideal saving is only ~25 % of FLOPs. At 0.5B, the extra small LoRA kernels and launch overhead cancel it. The paper's speed-up is at 175B.

**Q: Why is QLoRA slower?**
4-bit weights must be de-quantized to fp16/bf16 before each matmul, in the forward, backward and recompute passes. We measured 1.28× (H1) and 1.25× (H2).

**Q: Why NF4 and not plain int4?**
Weights are roughly normal; NF4's levels are normal quantiles, so each level is equally likely to be used. That is optimal for normal data. int4's evenly spaced levels waste resolution in the tails and starve the dense center.

**Q: What does double quantization save?**
It quantizes the per-block fp32 scales to 8 bits: 0.5 → 0.127 bits/param overhead, ~0.37 bits/param saved.

**Q: Why initialize B to zero?**
So $BA=0$ and training starts exactly from the pretrained model. A is random so gradients can flow (if both were zero, both gradients would be zero forever).

**Q: What does α do?**
It scales the update by $\alpha/r$, keeping its magnitude roughly stable when you change $r$, so the learning rate need not be re-tuned.

**Q: Your LoRA beat full FT on held-out loss. Does that mean LoRA is better?**
No. One seed, one short epoch on easy instruction data. It shows LoRA is *comparable* here. Biderman et al. show full FT learns more on hard domains given enough training. The low-rank constraint also regularizes, which helps on short runs.

**Q: Why did your QLoRA "improve" WikiText-2 perplexity?**
It is measured against the 4-bit base, which is 11 % worse than 16-bit. Training partly repaired quantization error. Against the 16-bit original it is still +9.4 %.

**Q: Why did your 10,000× become 56×?**
The paper used GPT-3 175B with only $W_Q, W_V$ adapted at small r. Our model is ~350× smaller with all 7 linear layers per block adapted at r=16. The ratio $\frac{dk}{r(d+k)}$ grows with width.

**Q: How do you know your memory numbers are right?**
They match hand calculations to ~1 %: 1.840 GiB fp32 weights, 0.953 GiB fp16 + LoRA, 0.713 GiB NF4 + fp32 embeddings, 11.50 GiB for 3B fp32.

**Q: How did you make the comparison fair?**
Same model, data split, seed, steps, batch, sequence length, gradient checkpointing and hardware. Only the method changes. Peak memory was reset after loading. Speed excludes warm-up steps.

**Q: What would you do with more time?**
- Multiple seeds and error bars.
- A head-only baseline in H3.
- Longer training and a harder domain (math/code) to test "learns less".
- bf16 on newer GPUs.
- DoRA/rsLoRA.
- A bigger model in H2 via QLoRA, to show the "bigger-but-quantized beats smaller-but-16-bit" trade-off.

**Q: When would you *not* use PEFT?**
When the task needs lots of new knowledge (new language or domain), when you have the compute for full FT and need the last point of accuracy, or when prompting or RAG already solves it.

---

## 10. Cheat sheet of our numbers

```
Qwen2.5-0.5B total ............. 494,032,768   (embedding 136.1 M = 28 %)
LoRA r=16 all-linear ........... 8,798,208 = 366,592/layer x 24  (1.75 %)
Qwen2.5-3B total ............... 3,085,938,688
LoRA r=16 on 3B ................ 29,933,568 = 831,488/layer x 36 (0.96 %)

H1 peak GB  full 11.43 | LoRA 4.64 | QLoRA 4.74
H1 s/step   full 2.02  | LoRA 2.10 | QLoRA 2.69   (QLoRA = 1.28x LoRA)
H1 loss     full 1.318->1.285 | LoRA 1.318->1.272 | QLoRA 1.412->1.322
H1 ckpt     full 988 MB | adapters 35.2 MB  (28x)
Ratios      56x fewer params, 2.5x less memory, 28x smaller file

H2 3B       full OOM at step 0 (14.33/14.6 GB) | LoRA 9.99 GB | QLoRA 7.61 GB
H3 SST-2    r1 85.67 | r4 87.61 | r16 87.50 | r64 88.07 | full 88.07  (LoRA = 18,432 r)
H4 PPL      base 12.50 | 4-bit 13.87 | full 12.79 (+2.3 %) | LoRA 12.63 (+1.0 %) | QLoRA 13.68 (-1.4 % vs 4-bit)

Rules: 16 B/param full FT | 6N FLOPs/token (8N with checkpointing) | NF4+DQ = 4.127 bits/param
Setup: Kaggle Tesla T4 14.6 GB, fp16, PyTorch 2.11, Transformers 5.20, PEFT 0.21.2, bitsandbytes 0.50.2, seed 42
```

---

## 11. Glossary

- **Absmax quantization**: scale a block by its largest absolute value so it fits in [−1, 1].
- **Activation**: an intermediate tensor in the forward pass; saved for backward.
- **Adapter**: a small bottleneck MLP inserted in series in each layer (Houlsby).
- **AdamW**: Adam with decoupled weight decay; keeps 2 states per trainable param.
- **α (LoRA alpha)**: the scaling numerator; the update is multiplied by α/r.
- **bf16 / fp16 / fp32**: 16/16/32-bit floating-point formats (§2.4).
- **Catastrophic forgetting**: losing previous abilities when trained on a new task.
- **Checkpoint**: (1) saved model files; (2) in "gradient checkpointing", a stored layer input.
- **Cross-entropy**: $-\log p(\text{correct token})$, averaged.
- **De-quantization**: converting stored 4-bit codes back to 16-bit numbers for compute.
- **Double quantization**: quantizing the quantization constants.
- **Epoch**: one pass over the training data.
- **GQA (grouped-query attention)**: several query heads share one key/value head.
- **GradScaler / loss scaling**: multiply the loss to avoid fp16 gradient underflow.
- **Held-out set**: examples never trained on, used to measure generalization.
- **Intrinsic dimension**: the smallest number of free directions needed to fine-tune well.
- **LoRA**: Low-Rank Adaptation, $W_0 + \frac{\alpha}{r}BA$.
- **Merging**: folding $BA$ into $W_0$ after training.
- **MIG**: Multi-Instance GPU, a hardware partition of one GPU into isolated slices.
- **Mixed precision**: compute in 16-bit, keep master weights in 32-bit.
- **NF4**: 4-bit NormalFloat, with levels at normal-distribution quantiles.
- **OOM**: out of memory.
- **PEFT**: parameter-efficient fine-tuning; also the Hugging Face library.
- **Perplexity**: $e^{\text{loss}}$.
- **QLoRA**: LoRA on a 4-bit NF4 frozen base, with double quantization and paged optimizers.
- **Rank r**: the inner dimension of $B A$; caps the rank of the update.
- **SLURM**: the job scheduler on the HPC (sbatch, squeue, sacct, QoS, gres).
- **Tied embeddings**: the input embedding and output LM head share one matrix.

---

## 12. References

1. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022, arXiv:2106.09685
2. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023, arXiv:2305.14314
3. Houlsby et al., *Parameter-Efficient Transfer Learning for NLP* (Adapters), ICML 2019, arXiv:1902.00751
4. Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, 2020, arXiv:2012.13255
5. Biderman et al., *LoRA Learns Less and Forgets Less*, TMLR 2024, arXiv:2405.09673
6. Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, 2020, arXiv:1910.02054
7. Micikevicius et al., *Mixed Precision Training*, ICLR 2018, arXiv:1710.03740
8. Chen et al., *Training Deep Nets with Sublinear Memory Cost* (gradient checkpointing), 2016, arXiv:1604.06174
9. Kaplan et al., *Scaling Laws for Neural Language Models* (6N FLOPs rule), 2020, arXiv:2001.08361
10. Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*, 2023, arXiv:2311.03285
11. Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (rsLoRA), 2023, arXiv:2312.03732
12. Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, ICML 2024, arXiv:2402.09353
13. Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), ICLR 2019, arXiv:1711.05101
14. Hugging Face PEFT documentation, huggingface.co/docs/peft

**Our files:**
- Code: `evidence/` (HPC/SLURM version), `evidence-kaggle/peft_evidence_kaggle.ipynb` (the run behind the results)
- Results: `evidence-kaggle/results/`
- Slides: `slides/` and the deck artifact
- Step-by-step log: `explanability.md`
