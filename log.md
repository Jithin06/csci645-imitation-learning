# HW2 log

## Setup
- Repo cloned on CARC at /scratch1/jrajan/645/hw2 (own deploy key for CARC).
- MuJoCo Playground at commit d59156e, installed with `uv sync --frozen` (Python 3.12, PyTorch 2.9.1+cu128); not vendored into the repo.
- Task: PandaPickCube, impl='jax', state observations.
- obs: 66-D state; action: 8-D in [-1, 1], joint targets change by 0.04·a per step; control step 0.02 s; 150-step episodes (3 s).
- Env has no success flag. Defined: box within 5 cm of target for 10 consecutive steps.
- Seeds: demos 0–159, evaluation 100000–100049.

## Fixes
- uv: `uv sync --locked` wanted to update the lockfile → used `--frozen` (installs exactly what uv.lock pins).
- No GPU via salloc (queues full) → ran GPU checks as sbatch jobs.
- No ffmpeg on compute nodes, MP4 save failed → `imageio-ffmpeg` in the venv + `use_bundled_ffmpeg()` in common.py.
- Part 0 job 13220880: `uv run` printed nothing on the compute node, no videos → Slurm scripts call `.venv/bin/python` directly.
- Login node: OpenBLAS/JAX `pthread_create failed` (thread limit) → `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` in ~/.bashrc; anything importing JAX runs in a job or salloc; jobs use 4 threads.
- d14-16: CUDA_ERROR_UNKNOWN at init (jobs 13230784, 13231385) → excluded.
- P100 (job 13234704): PyTorch 2.9.1+cu128 has no sm_60 kernels ("no kernel image is available"); JAX is fine on P100 → pinned jobs to V100 (`--gpus-per-task=v100:1`).
- Bus error at startup on d13-02 (13234726) and d14-09 (13407919), numpy ModuleNotFoundError on d14-09 (13407573); files intact, same code runs elsewhere → node-level /scratch1 read failures, both nodes excluded.
- DataLoader: dropped worker processes (`num_workers=0`) to rule out shared-memory issues; same samples, same loss.
- analyze_demos: action stats included zero padding after episode end → masked to valid steps, reran analysis.

## Results (held-out seeds 100000–100049, 50 episodes)
- Collection (job 13230787): teacher 133/160 = 83.1%; kept first 100 successes, 92.9 ± 16.2 steps, 0 out of bounds; 107 s (~103 s JIT).
- Teacher (job 13234627): 43/50 = 86%.
- ACT, 100 demos: 30/50 = 60%; 40k steps, 2211 s on V100, final L1 0.031, KL ~5e-7 (latent collapsed); W&B run aojqzpob.
- ACT, 50 demos: 17/50 = 34%; final L1 0.022.
- ACT, 25 demos: pending (job 13423673).
- ACT, 100 demos, no temporal ensembling: 28/50 = 56%; action jerk 0.099 vs 0.084 with ensembling.
- Edge (x>0.8 or |y|>0.14, 26 eps) vs interior (24 eps): teacher 81/92%, ACT-100 35/88%, ACT-50 19/50%.
- Failures (grasp / near-miss <15 cm): teacher 6/0, ACT-100 11/6, ACT-50 12/11.
- Fisher exact p: ACT-100 edge vs interior 0.0002; ACT-100 vs teacher edge 0.002, interior 1.0; ACT-100 vs ACT-50 overall 0.016, interior 0.011; ensembling on vs off 0.84.