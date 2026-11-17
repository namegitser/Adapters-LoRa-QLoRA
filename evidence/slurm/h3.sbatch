#!/bin/bash
# H3: LoRA rank sweep on DistilBERT + SST-2, run SEQUENTIALLY in one job:
#   LoRA r in {1,2,4,8,16,64} x seed in {42,43}, then full fine-tuning reference (seeds 42, 43) = 14 runs.
# Not an array job: general QoS allows 1 GPU/user and 10 submitted jobs (an array task counts as a job,
# and DenyOnLimit would reject a 14-task array). A 24 GB slice is plenty for DistilBERT.
# Quick version if time is short:  RANKS="1 4 16 64" SEEDS="42" bash slurm/submit.sh h3
#SBATCH --job-name=peft-h3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

source slurm/job_env.sh
RANKS="${RANKS:-1 2 4 8 16 64}"
SEEDS="${SEEDS:-42 43}"

status=0
for SEED in $SEEDS; do
  for R in $RANKS; do
    already_done h3 "lora_r${R}_seed${SEED}" && continue
    echo; echo ">>> H3 LoRA r=${R} seed=${SEED}  $(date -Is)"
    python train_h3.py --method lora --r "$R" --seed "$SEED" || status=1
  done
done
for SEED in $SEEDS; do
  already_done h3 "full_seed${SEED}" && continue
  echo; echo ">>> H3 full fine-tune seed=${SEED}  $(date -Is)"
  python train_h3.py --method full --seed "$SEED" || status=1
done
echo; echo "H3 finished $(date -Is), exit=${status}"
exit $status
