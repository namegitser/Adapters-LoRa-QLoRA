#!/bin/bash
# One-time setup on the LOGIN node, from inside ~/peft-evidence (safe to re-run):
#   bash setup_env.sh            # venv + small packages + Qwen2.5-0.5B, DistilBERT, datasets
#   bash setup_env.sh --with-h2  # also Qwen2.5-3B (~6 GB) for H2
#
# Follows https://hpc.plaksha.edu.in "Software": module load python -> venv --system-site-packages -> pip extras.
# Large downloads are staged on the login node's local /tmp and then copied with `cp`, because on master1
# Python/pip/conda streaming large files into /home or /scratch fails with "OSError: [Errno 14] Bad address"
# while plain copies work.
set -eo pipefail
cd "$(dirname "$0")"
source slurm/cluster.env

module purge
module load ${MODULES}
echo ">>> module python: $(python3 --version)"
python3 -c "import torch; print('>>> module torch', torch.__version__, '| cuda', torch.version.cuda)"

# 1) venv that reuses the module's PyTorch
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"

# 2) small extra packages, staged in local /tmp; torch pinned to the module's build so pip never replaces it
STAGE="/tmp/$USER-stage"; mkdir -p "$STAGE/tmp"
export TMPDIR="$STAGE/tmp" PIP_NO_CACHE_DIR=1
python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > "$STAGE/constraints.txt"
pip install --upgrade pip
# --root: pip resolves against this venv (so it sees the module's torch) but writes files under $STAGE/root,
# then a plain cp moves them into the venv on /home.
pip install --root "$STAGE/root" -c "$STAGE/constraints.txt" -r requirements.txt
cp -a "$STAGE/root$VENV/." "$VENV/"
python - <<'EOF'
import torch, transformers, peft, accelerate, datasets, bitsandbytes
assert not torch.__file__.startswith(__import__("sys").prefix), "torch must come from the module, not the venv"
print(">>> torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__,
      "| bitsandbytes", bitsandbytes.__version__, "| datasets", datasets.__version__)
EOF

# 3) models + datasets: download into local /tmp, then copy to $HF_HOME on /scratch
mkdir -p "$HF_HOME" "$CKPT_ROOT" logs results
HF_HOME="$STAGE/hf" python download_assets.py "$@"
cp -a "$STAGE/hf/." "$HF_HOME/"
echo ">>> cache copied to $HF_HOME ($(du -sh "$HF_HOME" | cut -f1))"
rm -rf "$STAGE"

echo
echo "Setup done. Next: bash slurm/submit.sh check"
