#!/usr/bin/env bash
# ============================================================
# connect_jupyter.sh  --  Tunnel Jupyter from a GPU node to
#                         this machine and print the URL.
#
# Usage:  bash slurm/connect_jupyter.sh [jobid]
# ============================================================
set -euo pipefail

LOCAL_PORT=8888   # port exposed on THIS machine
REMOTE_PORT=8888  # port Jupyter binds inside the container

JOB_ID="${1:-}"
if [[ -z "$JOB_ID" ]]; then
    JOB_ID=$(squeue -u "$USER" -n cs338_jupyter -h -o "%i" | head -1)
    if [[ -z "$JOB_ID" ]]; then
        echo "Usage: $0 <jobid>"
        echo "No running cs338_jupyter job found."
        exit 1
    fi
    echo "[connect_jupyter] Found running job: $JOB_ID"
fi

LOG="slurm/logs/jupyter_${JOB_ID}.log"
if [[ ! -f "$LOG" ]]; then
    echo "Log not found: $LOG"
    echo "Job may still be queued. Check: squeue -j $JOB_ID"
    exit 1
fi

# Wait for the node name to appear (written before srun starts)
echo "Waiting for Jupyter to start..."
NODE=""
for i in $(seq 1 60); do
    NODE=$(grep -m1 "^Node:" "$LOG" 2>/dev/null | awk '{print $2}' || true)
    if [[ -n "$NODE" ]]; then break; fi
    if [[ $i -eq 60 ]]; then
        echo "Timed out. Check: $LOG"
        exit 1
    fi
    sleep 2
done

echo "Job running on node: $NODE"

# Kill any existing tunnel on LOCAL_PORT
existing=$(lsof -ti tcp:${LOCAL_PORT} 2>/dev/null || true)
if [[ -n "$existing" ]]; then
    echo "Killing existing process on port ${LOCAL_PORT}..."
    kill "$existing" 2>/dev/null || true
fi

# Tunnel through the login node's SSH daemon (not directly to compute node)
echo "Setting up SSH tunnel: localhost:${LOCAL_PORT} -> ${NODE}:${REMOTE_PORT}"
ssh -4 -fNL "${LOCAL_PORT}:${NODE}:${REMOTE_PORT}" localhost

# Wait for Jupyter to write its token URL to the log (up to 60s)
echo "Waiting for Jupyter token URL in log..."
TOKEN=""
for i in $(seq 1 30); do
    TOKEN=$(grep -oP '(?<=token=)[a-f0-9]+' "$LOG" 2>/dev/null | tail -1 || true)
    if [[ -n "$TOKEN" ]]; then break; fi
    sleep 2
done

echo ""
echo "====================================================="
echo "  Jupyter is ready!"
if [[ -n "$TOKEN" ]]; then
    echo "  URL (with token): http://localhost:${LOCAL_PORT}/?token=${TOKEN}"
    echo ""
    echo "  In VS Code:"
    echo "  Ctrl+Shift+P -> 'Jupyter: Specify Jupyter Server URL'"
    echo "  Enter: http://localhost:${LOCAL_PORT}/?token=${TOKEN}"
else
    echo "  URL: http://localhost:${LOCAL_PORT}"
    echo ""
    echo "  In VS Code:"
    echo "  Ctrl+Shift+P -> 'Jupyter: Specify Jupyter Server URL'"
    echo "  Enter: http://localhost:${LOCAL_PORT}"
fi
echo ""
echo "  To reach from your LOCAL machine:"
echo "  ssh -N -L ${LOCAL_PORT}:localhost:${LOCAL_PORT} ${USER}@$(hostname -f)"
echo "====================================================="
