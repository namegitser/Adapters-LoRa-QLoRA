#!/bin/bash
# Sourced at the top of every *.sbatch job. Do not run directly.
# Jobs must be submitted from inside evidence/ (slurm/submit.sh does this for you).
cd "${SLURM_SUBMIT_DIR:?submit from inside evidence/}"
source slurm/cluster.env

if [[ -n "${MODULES}" ]]; then module purge; module load ${MODULES}; fi
source "$VENV/bin/activate"

export HF_HOME CKPT_ROOT
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "${OFFLINE}" == "1" ]]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
mkdir -p logs results "$CKPT_ROOT"

# General-QoS jobs can be preempted and REQUEUED: the whole script restarts from the top.
# already_done <exp> <run> -> true if that run already wrote a final result (ok or OOM), so loops skip it.
already_done() {
  local f="results/$1/$2.json"
  if [[ -f "$f" ]] && grep -Eq '"status": "(ok|OOM)"' "$f"; then
    echo ">>> skip $1/$2 (finished in an earlier attempt: $f)"; return 0
  fi
  return 1
}
echo "restart count: ${SLURM_RESTART_COUNT:-0}"

echo "=================================================================="
echo "job      : ${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-none}"
echo "node     : $(hostname)    started: $(date -Is)"
echo "python   : $(which python)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=================================================================="
