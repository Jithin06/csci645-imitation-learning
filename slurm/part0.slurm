#!/bin/bash
#SBATCH --job-name=hw2-part0
#SBATCH --account=seita_1951
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch1/jrajan/645/slurm-logs/%x-%j.out

cd /scratch1/$USER/645/hw2
unset LD_LIBRARY_PATH
export MUJOCO_GL=egl JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=/scratch1/$USER/645/mujoco_playground/.venv/bin/python

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY -c "import jax; print('jax backend:', jax.default_backend(), jax.devices())"
$PY code/part0_rollout.py
mv panda_pick_cube.mp4 videos/PandaPickCube_random.mp4
uv --no-config pip list --python $PY 2>/dev/null | grep -iE "^(jax|jaxlib|jax-cuda12-plugin|mujoco|mujoco-mjx|warp-lang|brax|flax|playground|imageio-ffmpeg) " > results/part0_versions.txt
nvidia-smi --query-gpu=name,driver_version --format=csv >> results/part0_versions.txt
$PY code/part0_rollout_zero.py
mv panda_pick_cube_zero.mp4 videos/PandaPickCube_zero.mp4
echo "end: $(date)"
