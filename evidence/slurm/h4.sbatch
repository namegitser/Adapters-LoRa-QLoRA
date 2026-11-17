#!/bin/bash
# H4: WikiText-2 perplexity of base vs H1 models ("LoRA forgets less"?).
# Needs H1's checkpoints. submit.sh chains it with --dependency=afterok:<H1 job id>.
#SBATCH --job-name=peft-h4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

source slurm/job_env.sh
python eval_h4.py --base "${MODEL:-Qwen/Qwen2.5-0.5B}" --seed 42 \
  --variants base base4bit full lora qlora --window 1024 --stride 512
