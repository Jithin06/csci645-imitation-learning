#!/bin/bash
# Train one ACT student and evaluate it on the 50 held-out seeds, in one job.
#   sbatch slurm/train_eval.slurm <algo> <data.npz> <num_demos> <seed> [extra train args]
#   sbatch slurm/train_eval.slurm act data/scripted_clean.npz 100 0
#SBATCH --job-name=hw2-train
#SBATCH --account=seita_1951
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --exclude=d14-16
#SBATCH --output=/scratch1/jrajan/645/slurm-logs/%x-%j.out

set -euo pipefail
ALGO=$1; DATA=$2; N=$3; SEED=$4; shift 4
TAG=$(basename "$DATA" .npz)_${ALGO}_n${N}_s${SEED}
HW=/scratch1/$USER/645/hw2
PY=/scratch1/$USER/645/mujoco_playground/.venv/bin/python
cd "$HW"
unset LD_LIBRARY_PATH
export MUJOCO_GL=egl JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export WANDB_DIR=/scratch1/$USER/645/wandb WANDB_MODE=${WANDB_MODE:-online}

echo "host $(hostname) | job $SLURM_JOB_ID | commit $(git rev-parse --short HEAD) | $TAG | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" code/train_student.py --algo "$ALGO" --data "$DATA" --num-demos "$N" --seed "$SEED" \
      --out "runs/$TAG" --wandb "$@"
"$PY" code/evaluate.py --policy student --ckpt "runs/$TAG/model.pt" --out "results/eval_$TAG"
echo "done $(date)"