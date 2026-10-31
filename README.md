# CSCI 645 HW2: Imitation Learning (PandaPickCube)

ACT trained on state observations, using demos from a scripted expert. The experiment compares
100, 50 and 25 demos, with training seeds 0, 1 and 2. The writeup is `report.pdf`.

## What's here
- `code/` - demo collection, training, evaluation and plotting scripts
- `slurm/` - the CARC job scripts I used
- `runs/` - config and training loss for each run
- `results/` - evaluation results, demo stats and figures, package versions
- `videos/` - expert demos and student rollouts
- `log.md` - my notes on setup problems and how I fixed them

The demo dataset (`data/*.npz`) and model checkpoints (`runs/*/model.pt`) are too large to
include. Running the steps below regenerates them.

## Setup
I ran everything on CARC Discovery with one V100 per job. The Slurm scripts assume this layout:

    /scratch1/$USER/645/hw2                 # this repo
    /scratch1/$USER/645/mujoco_playground   # MuJoCo Playground and its .venv
    /scratch1/$USER/645/slurm-logs          # job logs

Install:

    mkdir -p /scratch1/$USER/645/slurm-logs /scratch1/$USER/645/wandb
    cd /scratch1/$USER/645
    git clone https://github.com/Jithin06/csci645-imitation-learning hw2
    git clone https://github.com/google-deepmind/mujoco_playground
    cd mujoco_playground
    git checkout d59156e
    uv sync --frozen --python 3.12 --extra cuda --extra learning --extra notebooks
    uv pip install imageio-ffmpeg     # CARC compute nodes have no ffmpeg, needed for videos

To check the install:

    .venv/bin/python -c "import torch, jax, mujoco; print(torch.__version__, jax.__version__, mujoco.__version__)"

I used torch 2.9.1+cu128, jax 0.6.2 and mujoco 3.6.0. The other package versions are in
`results/part0_versions.txt`.

Before submitting jobs, check these:
- **Slurm account:** the scripts charge `seita_1951`. Use your own with
  `sbatch --account=<yours> ...`, or edit the `#SBATCH --account` line.
- **Log folder:** each script writes its log to the path in its `#SBATCH --output` line. That
  folder has to exist, and the path needs your username if it's different from mine.
- **W&B:** training always logs to W&B. Either run `wandb login` once, or run
  `export WANDB_MODE=offline` before `sbatch`.

## Reproducing the results
Run these from `/scratch1/$USER/645/hw2`. The `--dependency` flags make each job wait for the
one it needs.

    # 1. Collect demos: reset seeds 0-159, keep the first 100 successes
    C=$(sbatch --parsable slurm/gpu_job.slurm code/collect_demos.py --teacher scripted --num-demos 100 --out data/scripted_clean)

    # 2. Demo figures, plus the teacher evaluation on seeds 100000-100049
    sbatch --dependency=afterok:$C slurm/gpu_job.slurm code/analyze_demos.py --data data/scripted_clean.npz --out results/demos_scripted_clean
    sbatch slurm/gpu_job.slurm code/evaluate.py --policy scripted --out results/eval_teacher_scripted

    # 3. Train and evaluate every dataset size and seed (waits for collection)
    for N in 100 50 25; do for SEED in 0 1 2; do
      J=$(sbatch --parsable --dependency=afterok:$C slurm/train_eval.slurm act data/scripted_clean.npz $N $SEED)
      [ $N = 100 ] && [ $SEED = 0 ] && B=$J
    done; done

    # 4. Optional: the baseline without temporal ensembling (waits for the 100-demo seed-0 run)
    sbatch --dependency=afterok:$B slurm/gpu_job.slurm code/evaluate.py --policy student \
      --ckpt runs/scripted_clean_act_n100_s0/model.pt --no-temporal-agg \
      --out results/eval_scripted_clean_act_n100_s0_noTA

When `squeue --me` shows no jobs left, build the summary table:

    ../mujoco_playground/.venv/bin/python code/make_table.py