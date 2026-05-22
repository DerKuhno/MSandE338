#!/usr/bin/env bash
# ============================================================
# jupyter.sbatch  --  Start a Jupyter Lab server on a GPU node.
#
# Submit:  sbatch slurm/jupyter.sbatch
# Then:    bash slurm/connect_jupyter.sh <jobid>
# ============================================================
#SBATCH --job-name=cs338_jupyter
#SBATCH --output=slurm/logs/jupyter_%j.log
#SBATCH --error=slurm/logs/jupyter_%j.log
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=interactive

source "${SLURM_SUBMIT_DIR}/slurm/config.env"
cd "$REPO_DIR"

mkdir -p slurm/logs

PORT=8888
NODE=$(hostname)

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $NODE"
echo "Port:   $PORT"

srun \
    --gpus-per-task=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir="$REPO_DIR" \
    --export=ALL,WORKSPACE_DIR="$WORKSPACE_DIR",CACHE_DIR="$CACHE_DIR",DATASET_DIR="$DATASET_DIR",MODEL_DIR="$MODEL_DIR",WMDP_MODEL_DIR="$WMDP_MODEL_DIR",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    bash -c "
        export PYTHONPATH=\"$(ls -d $PIP_ENV/local/lib/python*/dist-packages 2>/dev/null | head -1):\${PYTHONPATH:-}\"
        jupyter lab \
            --no-browser \
            --port=${PORT} \
            --ip=0.0.0.0 \
            --notebook-dir=${REPO_DIR} \
            --NotebookApp.token='' \
            --NotebookApp.password='' \
            --ServerApp.token='' \
            --ServerApp.password='' \
            --IdentityProvider.token='' \
            --allow-root
    "
