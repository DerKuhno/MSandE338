#!/usr/bin/env bash
# ============================================================
# 01_prepare.sbatch  --  Download datasets & build reduced model.
#
# Submit:  sbatch slurm/01_prepare.sbatch
# ============================================================
#SBATCH --job-name=cs338_prepare
#SBATCH --output=slurm/logs/01_prepare_%j.out
#SBATCH --error=slurm/logs/01_prepare_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# No GPU needed for dataset prep; reduce_gemma requires a HF token but no GPU.
#SBATCH --partition=cpu

set -euo pipefail

# SLURM_SUBMIT_DIR is set by Slurm to the directory sbatch was called from.
source "${SLURM_SUBMIT_DIR}/slurm/config.env"
cd "$REPO_DIR"

# Add pip packages installed by 00_install.sh to PYTHONPATH inside the container.
# The glob handles any Python version present in the container.
_SITE="$(ls -d "$PIP_ENV/local/lib/python"*/dist-packages 2>/dev/null | head -1)"
export PYTHONPATH="${_SITE:+$_SITE:}${PYTHONPATH:-}"

mkdir -p slurm/logs

# Convenience wrapper: runs a command inside the container via srun.
container_run() {
    srun \
        --nodes=1 --ntasks=1 \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$CONTAINER_MOUNTS" \
        --container-workdir="$REPO_DIR" \
        --export=ALL,PYTHONPATH="$PYTHONPATH",WORKSPACE_DIR="$WORKSPACE_DIR",CACHE_DIR="$CACHE_DIR",DATASET_DIR="$DATASET_DIR",MODEL_DIR="$MODEL_DIR" \
        "$@"
}

echo "[prepare] WORKSPACE_DIR = $WORKSPACE_DIR"

# 1. Build the reduced Gemma model from the pretrained 2B weights.
#    Reads tokens/hf_token.txt; writes to MODEL_DIR/random_init_models/.
echo "[prepare] Step 1/3: reduce_gemma"
container_run python src/prepare_models/reduce_gemma.py

# 2. Stream only the Korean + English corpora needed for language experiments.
echo "[prepare] Step 2/3: download language datasets"
container_run python slurm/download_language_only.py

# 3. Tokenise and split into train/valid files.
echo "[prepare] Step 3/3: prepare (tokenise) datasets"
container_run python src/prepare_data/prepare.py

echo "[prepare] Done."
