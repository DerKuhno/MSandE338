#!/usr/bin/env bash
# ============================================================
# 00_install.sbatch  --  Install pip dependencies inside the container.
#
# Submit:  sbatch slurm/00_install.sbatch
# ============================================================
#SBATCH --job-name=cs338_install
#SBATCH --output=slurm/logs/00_install_%j.out
#SBATCH --error=slurm/logs/00_install_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu

set -euo pipefail

# SLURM_SUBMIT_DIR is set by Slurm to the directory sbatch was called from.
source "${SLURM_SUBMIT_DIR}/slurm/config.env"
cd "$REPO_DIR"

echo "[install] CONTAINER_IMAGE = $CONTAINER_IMAGE"
echo "[install] PIP_ENV         = $PIP_ENV"
echo "[install] REPO_DIR        = $REPO_DIR"

mkdir -p "$PIP_ENV" slurm/logs

srun \
    --nodes=1 --ntasks=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="${REPO_DIR}:${REPO_DIR},${WORKSPACE_DIR}:${WORKSPACE_DIR}" \
    --container-workdir="$REPO_DIR" \
    pip install --prefix="$PIP_ENV" -e "$REPO_DIR"

echo "[install] Done."
