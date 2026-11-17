#!/bin/bash
# Sanity check + smoke test (~5-10 min on a 24 GB slice):
#   1) check_env.py: GPU, bf16, bitsandbytes 4-bit, cached models/datasets
#   2) a few-step run of EVERY experiment code path (H1/H2 full|lora|qlora, H3, H4) on tiny data,
#      so API/version problems show up now and not one hour into H1. Results go to results/smoke/ (ignored by the report).
#SBATCH --job-name=peft-check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

source slurm/job_env.sh
fails=()

python check_env.py || fails+=("check_env")

for M in full lora qlora; do
  echo; echo ">>> SMOKE train_llm ${M}"
  python train_llm.py --exp smoke --method "$M" --model Qwen/Qwen2.5-0.5B --steps 3 --n-examples 60 \
    --n-eval 12 --no-save --log-every 1 --timing-skip 0 || fails+=("train_llm_${M}")
done

echo; echo ">>> SMOKE train_h3 (LoRA r=4, 256 examples)"
python train_h3.py --exp smoke --method lora --r 4 --train-subset 256 --epochs 1 || fails+=("train_h3")

echo; echo ">>> SMOKE eval_h4 (base + base4bit, first 4096 tokens)"
python eval_h4.py --exp smoke --variants base base4bit --max-tokens 4096 || fails+=("eval_h4")

echo
if (( ${#fails[@]} == 0 )); then
  echo "SMOKE TEST: ALL PASSED. Safe to run: bash slurm/submit.sh all"
else
  echo "SMOKE TEST FAILED: ${fails[*]}  (scroll up for the traceback)"; exit 1
fi
