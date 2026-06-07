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
        # Remove stale kernel specs from previous sessions/containers
        jupyter kernelspec remove -f cs221-project 2>/dev/null || true
        python3 -m ipykernel install --sys-prefix --name python3 --display-name 'Python 3'
        # transformers 5.x needs torch>=2.5; downgrade to 4.x if needed
        python3 -c 'import transformers; v=tuple(int(x) for x in transformers.__version__.split(".")[:1]); exit(0 if v[0]<5 else 1)' || \
            pip install -q 'transformers>=4.48.2,<5.0'
        JUPYTER_TOKEN=\$(python3 -c 'import secrets; print(secrets.token_hex(24))')
        echo \"Token: \${JUPYTER_TOKEN}\"
        jupyter lab \
            --no-browser \
            --port=${PORT} \
            --ip=0.0.0.0 \
            --notebook-dir=${REPO_DIR} \
            --ServerApp.allow_origin='*' \
            --ServerApp.disable_check_xsrf=True \
            --IdentityProvider.token=\"\${JUPYTER_TOKEN}\" \
            --allow-root
    "
