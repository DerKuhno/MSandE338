#!/usr/bin/env bash
# ============================================================
# submit_pipeline.sh  --  Submit the full language pipeline to Slurm
#                         with proper job dependencies.
#
# Usage (from repo root):
#   bash slurm/submit_pipeline.sh [--start <stage>]
#
#   --start <stage>  Skip earlier stages; pick up from this stage number.
#                    e.g. --start 3  to re-run only unlearn -> distill -> relearn.
#
# Environment overrides (set before calling this script or in ~/.cs338_slurm.env):
#   WORKSPACE_DIR    — where models/datasets are stored  (default in config.env)
#   GPUS_PER_NODE    — GPUs per training job             (default: 1)
#   SBATCH_PARTITION — Slurm partition (picked up automatically by sbatch)
#   SBATCH_ACCOUNT   — Slurm account  (picked up automatically by sbatch)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"

START_STAGE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START_STAGE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$REPO_DIR/slurm/logs"

# sbatch picks up SBATCH_* vars automatically from the environment,
# so no extra flags are needed here.
submit() {
    local script="$1"
    local dep_flag="$2"  # e.g. "--dependency=afterok:123" or ""
    if [[ -n "$dep_flag" ]]; then
        sbatch "$dep_flag" "$script"
    else
        sbatch "$script"
    fi
}

DEP=""   # dependency string, updated after each submission

# Stage 1 — data prep (CPU only, no GPU requested in the sbatch header)
if [[ $START_STAGE -le 1 ]]; then
    echo "[pipeline] Submitting stage 1: prepare"
    JID1=$(submit "$SCRIPT_DIR/01_prepare.sbatch" "$DEP" | awk '{print $NF}')
    echo "[pipeline]   job id = $JID1"
    DEP="--dependency=afterok:$JID1"
fi

# Stage 2 — pretrain
if [[ $START_STAGE -le 2 ]]; then
    echo "[pipeline] Submitting stage 2: pretrain"
    JID2=$(submit "$SCRIPT_DIR/02_pretrain.sbatch" "$DEP" | awk '{print $NF}')
    echo "[pipeline]   job id = $JID2"
    DEP="--dependency=afterok:$JID2"
fi

# Stage 3 — unlearn LR sweep
if [[ $START_STAGE -le 3 ]]; then
    echo "[pipeline] Submitting stage 3: unlearn"
    JID3=$(submit "$SCRIPT_DIR/03_unlearn.sbatch" "$DEP" | awk '{print $NF}')
    echo "[pipeline]   job id = $JID3"
    DEP="--dependency=afterok:$JID3"
fi

# Stage 4 — partial distillation (UNDO)
if [[ $START_STAGE -le 4 ]]; then
    echo "[pipeline] Submitting stage 4: partial_distill"
    JID4=$(submit "$SCRIPT_DIR/04_partial_distill.sbatch" "$DEP" | awk '{print $NF}')
    echo "[pipeline]   job id = $JID4"
    DEP="--dependency=afterok:$JID4"
fi

# Stage 5 — relearn (adversarial evaluation)
if [[ $START_STAGE -le 5 ]]; then
    echo "[pipeline] Submitting stage 5: relearn"
    JID5=$(submit "$SCRIPT_DIR/05_relearn.sbatch" "$DEP" | awk '{print $NF}')
    echo "[pipeline]   job id = $JID5"
fi

echo ""
echo "[pipeline] All stages submitted. Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f slurm/logs/0*_*.out"
