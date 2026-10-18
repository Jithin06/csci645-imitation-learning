# HW2 log

## Oct 17 - Part 0 (env setup)

Made a new repo for HW2 and cloned it on CARC at /scratch1/jrajan/645/hw2.
Added a new deploy key for CARC (the HW1 one can't be reused on another repo).

Cloned MuJoCo Playground next to it (commit d59156e) and installed it with uv.
Not pushing Playground into my repo, just noting the commit.

Using PandaPickCube with impl='jax', state observations.

Errors:
- ssh to GitHub was denied at first. I hadn't added the key under the repo's
  deploy keys yet.
- First push failed because nothing was committed yet (empty folders don't count).
- `uv sync --locked` said the lockfile needs updating. Switched to `--frozen`,
  which installs exactly what uv.lock pins.
- Couldn't get a GPU with salloc (A40, V100, P100 all queued). Submitted the
  GPU check + rollout as an sbatch job instead (slurm/part0.slurm).

Env facts:
- obs: 66-dim state (joint pos/vel, gripper pose, box pose, box-to-target error)
- action: 8-dim in [-1, 1], changes joint targets by 0.04 * action each step
- control step 0.02 s, episode is 150 steps (3 s)
- no success flag in the env, need to define one for Part 1
- seeds: reset 0, random actions 42 (same as README)

## 2026-10-18: fixes
- Part 0 job (13220880): `uv run` printed nothing on the compute node, so no videos were made. Fix: call `.venv/bin/python` directly in Slurm scripts.
- Login node: OpenBLAS `pthread_create failed` (tries 32 threads). Fix: `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` (in ~/.bashrc); jobs use 4.
- common.py pushed (3174ad5); CPU smoke test: zero actions, 2 envs, 20 steps.
- common.py smoke test (salloc CPU node, 2 envs × 20 steps, zero actions): success 0.0, ep_len 20.0, final dist 0.475 m. Pass.