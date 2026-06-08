#!/bin/bash
#SBATCH --job-name=build-cs338-container
#SBATCH --output=slurm/logs/build_container_%j.log
#SBATCH --error=slurm/logs/build_container_%j.log
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=interactive

# Output container path
OUTPUT_SQSH=/lustre/fsw/portfolios/sw/users/hsahota/containers/cs338.sqsh
BASE_SQSH=/lustre/fsw/portfolios/sw/users/hsahota/containers/pytorch.sqsh

mkdir -p slurm/logs

echo "Building cs338 container from: $BASE_SQSH"
echo "Output: $OUTPUT_SQSH"

srun \
    --container-image="$BASE_SQSH" \
    --container-save="$OUTPUT_SQSH" \
    bash -c '
        set -e

        pip install --upgrade pip

        # Project dependencies (from pyproject.toml)
        pip install \
            "accelerate>=1.3.0" \
            "aiolimiter>=1.2.1" \
            "bitsandbytes>=0.45.2" \
            "datasets>=3.2.0" \
            "flask>=3.1.0" \
            "google>=3.0.0" \
            "google-genai>=1.12.1" \
            "h5py>=3.12.1" \
            "ipdb>=0.13.13" \
            "lm-eval>=0.4.7" \
            "matplotlib>=3.10.0" \
            "orjson>=3.10.15" \
            "peft>=0.14.0" \
            "pydantic>=2.10.6" \
            "python-dotenv>=1.0.1" \
            "seaborn>=0.13.2" \
            "tqdm>=4.67.1" \
            "transformers>=4.48.2" \
            "wandb>=0.19.5"

        # JupyterLab (replaces the old Notebook 6.x in the base image)
        pip install "jupyterlab>=4.0.0" ipykernel ipywidgets

        echo "Installed packages:"
        pip list | grep -E "accelerate|datasets|peft|transformers|jupyter|torch"
        echo "Done."
    '

echo "Container saved to: $OUTPUT_SQSH"
