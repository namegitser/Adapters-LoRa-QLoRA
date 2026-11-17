#!/bin/bash
# Submit evidence jobs with the settings from slurm/cluster.env.
#   bash slurm/submit.sh check        # 15-min GPU sanity job (do this first)
#   bash slurm/submit.sh h1           # H1
#   bash slurm/submit.sh h2           # H2 (uses GRES_BIG)
#   bash slurm/submit.sh h3           # H3 (sequential, 24 GB slice)
#   bash slurm/submit.sh h4 [H1_ID]   # H4, optionally waits for H1 job H1_ID to succeed
#   bash slurm/submit.sh all          # H1, H2, H3 now + H4 chained after H1
set -euo pipefail
cd "$(dirname "$0")/.."            # -> evidence/ (jobs read SLURM_SUBMIT_DIR)
source slurm/cluster.env
mkdir -p logs results

base_opts=(--parsable -p "$PARTITION")
[[ -n "$ACCOUNT" ]] && base_opts+=(-A "$ACCOUNT")
[[ -n "$QOS" ]] && base_opts+=(--qos "$QOS")

sub() {  # sub <gres> <sbatch file> [extra sbatch opts...]
  local gres="$1" file="$2"; shift 2
  local id
  id=$(sbatch "${base_opts[@]}" --gres="$gres" --export=ALL "$@" "$file")
  echo "submitted $(basename "$file") -> job $id" >&2
  echo "$id"
}

what="${1:-}"
case "$what" in
  check) sub "$GRES_SMALL" slurm/check.sbatch >/dev/null ;;
  h1)    sub "$GRES" slurm/h1.sbatch >/dev/null ;;
  h2)    sub "$GRES_BIG" slurm/h2.sbatch >/dev/null ;;
  h3)    sub "$GRES_SMALL" slurm/h3.sbatch >/dev/null ;;
  h4)    if [[ -n "${2:-}" ]]; then sub "$GRES_SMALL" slurm/h4.sbatch --dependency=afterany:"$2" >/dev/null
         else sub "$GRES_SMALL" slurm/h4.sbatch >/dev/null; fi ;;
  all)   H1=$(sub "$GRES" slurm/h1.sbatch)
         sub "$GRES_BIG" slurm/h2.sbatch >/dev/null
         sub "$GRES_SMALL" slurm/h3.sbatch >/dev/null
         sub "$GRES_SMALL" slurm/h4.sbatch --dependency=afterany:"$H1" >/dev/null ;;
  *) echo "usage: bash slurm/submit.sh {check|h1|h2|h3|h4 [H1_JOB_ID]|all}"; exit 1 ;;
esac
echo "watch with: squeue -u \$USER    |  tail -f logs/<job>.out" >&2
