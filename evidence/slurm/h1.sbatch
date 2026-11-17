#!/bin/bash
# H1: Full FT vs LoRA vs QLoRA on Qwen2.5-0.5B (same data, seed, steps, seq len). Fits a 24 GB slice.
# Overrides:  SEEDS="42 43" bash slurm/submit.sh h1
#             MODEL=Qwen/Qwen2.5-1.5B bash slurm/submit.sh h1   (then set GRES=gpu:2g.48gb:1 and run H4 with the same MODEL)
#SBATCH --job-name=peft-h1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

source slurm/job_env.sh
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
STEPS="${STEPS:-300}"
SEEDS="${SEEDS:-42}"

status=0
for SEED in $SEEDS; do
  for METHOD in full lora qlora; do
    already_done h1 "${METHOD}_seed${SEED}" && continue
    echo; echo ">>> H1 ${METHOD} seed=${SEED}  $(date -Is)"
    python train_llm.py --exp h1 --method "$METHOD" --model "$MODEL" \
      --steps "$STEPS" --n-examples 5000 --seed "$SEED" --max-len 512 --micro-batch 4 --grad-accum 4 \
      || status=1
  done
done
echo; echo "H1 finished $(date -Is), exit=${status}"
exit $status
