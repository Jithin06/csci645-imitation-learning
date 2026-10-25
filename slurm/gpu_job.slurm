#!/bin/bash
# Generic HW2 GPU job: runs  python <script> <args...>  inside the Playground venv.
#   sbatch --job-name=hw2-collect --time=00:45:00 slurm/gpu_job.slurm code/collect_demos.py --teacher scripted --num-demos 100 --out data/scripted_clean
# Override any #SBATCH line on the sbatch command line (e.g. --gpus-per-task=a40:1, --time=...).
#SBATCH --job-name=hw2
#SBATCH --account=seita_1951
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-task=v100:1
#SBATCH --time=02:00:00
#SBATCH --exclude=d14-16
#SBATCH --output=/scratch1/jrajan/645/slurm-logs/%x-%j.out

set -euo pipefail
HW=/scratch1/$USER/645/hw2
PG=/scratch1/$USER/645/mujoco_playground
PY=$PG/.venv/bin/python
cd "$HW"

unset LD_LIBRARY_PATH                       # JAX/PyTorch ship their own CUDA libraries
export MUJOCO_GL=egl                        # headless rendering on the GPU node
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # JAX and PyTorch share one GPU
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4   # match --cpus-per-task
export WANDB_DIR=/scratch1/$USER/645/wandb WANDB_MODE=${WANDB_MODE:-online}

# /scratch1 sometimes fails reads while many jobs start at once (bus error / missing module): retry.
retry() { for i in 1 2 3; do "$@" && return 0; echo "attempt $i failed (exit $?), retrying in 60 s"; sleep 60; done; return 1; }

echo "host $(hostname) | job $SLURM_JOB_ID | commit $(git rev-parse --short HEAD 2>/dev/null || echo none) | $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cmd: python $*"
retry "$PY" "$@"
echo "done $(date)"