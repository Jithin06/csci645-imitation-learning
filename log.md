# CSCI 645 HW2 — Work Log

## 2026-10-17 — Part 0: environment setup

**Setup**
- Repo: `Jithin06/csci645-imitation-learning`, cloned on CARC at `/scratch1/jrajan/645/hw2`
- CARC access: read/write deploy key `~/.ssh/csci645_hw2_deploy`, SSH alias `github-hw2`
- MuJoCo Playground cloned at `/scratch1/jrajan/645/mujoco_playground`, commit `d59156e`
  (not part of this repo; pinned as a dependency)
- Install command:
  `uv --no-config sync --frozen --python 3.12 --extra cuda --extra learning --extra notebooks`
- Python 3.12.14 (system `/usr/bin/python3.12` on CARC)
- Task: `PandaPickCube`, JAX implementation (`impl='jax'`), state observations

**Snags**
- `uv sync --locked` failed with "lockfile needs to be updated", probably because CARC's uv version
  differs from Playground's → used `--frozen` (installs exactly what `uv.lock` pins, no re-check)
- `numpy==2.4.0 is yanked` warning: harmless, it's the version the lock pins
- `ssh -T github-hw2` was denied until the public key was added under repo → Settings → Deploy keys
- First push failed (`src refspec main does not match any`): nothing had been committed yet
- `salloc` for A40, V100 and P100 all queued (sinfo shows `mix-` = free GPUs reserved for
  waiting jobs) → switched to an sbatch job (`slurm/part0.slurm`) that takes any GPU

**Status**
- [x] Repo created and pushed
- [x] Playground installed
- [ ] Login-node import check (`obs 66 act 8`) + menagerie download
- [ ] GPU check (`jax.default_backend() == gpu`): job submitted, waiting
- [ ] Random-action and zero-action rollout videos
- [ ] Versions and hardware recorded in `results/part0_versions.txt`

**Env facts for the report** (from `pick.py`)
- Observation: 66-D state vector
- Action: 8-D in [-1, 1]; delta joint-position targets
  (ctrl += 0.04 · a, clipped to joint limits)
- ctrl_dt 0.02 s; episode length 150 steps (3 s)
- No built-in success flag; must define one in Part 1
- Seeds used: reset 0, random actions 42 (same as README)
