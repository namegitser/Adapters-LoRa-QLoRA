#!/bin/bash
# H2: Qwen2.5-3B on ONE 24 GB GPU slice. Full FT is expected to OOM (recorded, not a failure);
# LoRA (bf16 base) and QLoRA (4-bit base) run 50 steps each. Submit on your BIGGEST GPU (GRES_BIG).
#SBATCH --job-name=peft-h2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

source slurm/job_env.sh
MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
COMMON=(--exp h2 --model "$MODEL" --steps 50 --seed 42 --max-len 512 --micro-batch 4 --grad-accum 4
        --skip-eval-before --no-save --log-every 5)

status=0
for METHOD in full lora qlora; do
  already_done h2 "${METHOD}_seed42" && continue
  echo; echo ">>> H2 ${METHOD}  $(date -Is)"
  python train_llm.py --method "$METHOD" "${COMMON[@]}" || status=1
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv   # should be ~0 used between runs
done
echo; echo "H2 finished $(date -Is), exit=${status}"
exit $status
