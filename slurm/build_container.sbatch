#!/bin/bash
#SBATCH --job-name=build-cs338-container
#SBATCH --output=slurm/logs/build_container_%j.log
#SBATCH --error=slurm/logs/build_container_%j.log
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=interactive

# Pull a clean PyTorch base image from Docker Hub (public, no auth required)
# and bake all project dependencies in.  The resulting sqsh replaces cs338.sqsh.
#
# Base image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
#   - CUDA 12.4, cuDNN 9, PyTorch 2.4.0, Python 3.11
#   - torch already installed; we add the HF stack on top.
#
# Alternative (requires NGC credentials): nvcr.io#nvidia/pytorch:24.05-py3
#
# Submit: sbatch slurm/build_container.sbatch

OUTPUT_SQSH=/lustre/fsw/portfolios/sw/users/hsahota/containers/cs338.sqsh

mkdir -p slurm/logs

echo "Pulling docker://pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel and installing project deps..."
echo "Output: $OUTPUT_SQSH"

srun \
    --container-image="docker://pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel" \
    --container-save="$OUTPUT_SQSH" \
    --container-writable \
    bash -c '
        set -e

        pip install --upgrade pip

        # HF stack — pin together so version checks pass at import time.
        # transformers 5.x requires torch>=2.5 (torch.distributed.tensor.device_mesh);
        # the pytorch:2.4.0 base only has 2.4 so we cap at <5.0.
        pip install \
            "transformers>=4.48.2,<5.0" \
            "datasets>=3.2.0" \
            "accelerate>=1.3.0" \
            "peft>=0.14.0" \
            "tokenizers>=0.21.0" \
            "huggingface-hub>=0.30.0"

        # Evaluation / training utilities
        pip install \
            "bitsandbytes>=0.45.2" \
            "lm-eval>=0.4.7" \
            "tqdm>=4.67.1" \
            "wandb>=0.19.5"

        # JupyterLab and kernel
        pip install \
            "jupyterlab>=4.0.0" \
            "ipykernel>=6.29.5" \
            "ipywidgets>=8.0.0"

        # Other project dependencies (from pyproject.toml)
        pip install \
            "aiolimiter>=1.2.1" \
            "flask>=3.1.0" \
            "google>=3.0.0" \
            "google-genai>=1.12.1" \
            "h5py>=3.12.1" \
            "ipdb>=0.13.13" \
            "matplotlib>=3.10.0" \
            "orjson>=3.10.15" \
            "pydantic>=2.10.6" \
            "python-dotenv>=1.0.1" \
            "seaborn>=0.13.2"

        echo ""
        echo "=== Installed package versions ==="
        pip list | grep -E "^torch |accelerate|datasets|peft|transformers|tokenizers|huggingface|jupyter|lm.eval|bitsandbytes"
        echo "Done."
    '

echo "Container saved to: $OUTPUT_SQSH"
