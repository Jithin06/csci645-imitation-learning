"""Part 1 deliverables: demo video, filmstrip of several demos, dataset statistics plots.

  python code/analyze_demos.py --data data/scripted_clean.npz --out results/demos_scripted_clean
  (add --no-render if EGL rendering is unavailable; plots and stats still get made)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from common import SUCCESS_DIST, make_env, render_episode, save_json

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True)
  p.add_argument("--out", required=True, help="output folder")
  p.add_argument("--no-render", action="store_true")
  p.add_argument("--filmstrip-demos", type=int, default=4)
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  ep = dict(np.load(args.data))
  kept = ep["kept"]
  L, S = ep["ep_len"], ep["success"]
  box0 = ep["qpos"][:, 0, 9:12]       # box free-joint position at reset
  tgt = ep["target_pos"]

  stats = {
      "attempted": int(len(L)), "teacher_success_rate": float(S.mean()),
      "kept": int(len(kept)), "kept_success_fraction": float(S[kept].mean()),
      "kept_len_mean": float(L[kept].mean()), "kept_len_std": float(L[kept].std()),
      "all_len_mean": float(L.mean()), "all_len_std": float(L.std()),
      "action_abs_mean_per_dim": np.abs(ep["actions"][kept]).mean(axis=(0, 1)).round(3).tolist(),
      "action_saturation_frac": float((np.abs(ep["actions"][kept]) > 0.99).mean()),
  }
  save_json(stats, os.path.join(args.out, "dataset_stats.json"))
  print(stats)

  fig, ax = plt.subplots(1, 3, figsize=(15, 4))
  ax[0].hist([L[kept]], bins=20, color="#3b6ea5")
  ax[0].set(title="Episode length (kept demos)", xlabel="control steps (0.02 s)", ylabel="count")
  for i in kept:
    ax[1].plot(ep["box_target_dist"][i, : L[i]], color="#3b6ea5", alpha=0.25, lw=1)
  for i in np.flatnonzero(~S):
    ax[1].plot(ep["box_target_dist"][i, : L[i]], color="#c0392b", alpha=0.6, lw=1)
  ax[1].axhline(SUCCESS_DIST, ls="--", color="k", lw=1)
  ax[1].set(title="Box-to-target distance (blue kept, red failed)", xlabel="step", ylabel="m")
  ax[2].scatter(box0[S, 0], box0[S, 1], s=12, c="#3b6ea5", label="box start (success)")
  ax[2].scatter(box0[~S, 0], box0[~S, 1], s=30, c="#c0392b", marker="x", label="box start (failure)")
  ax[2].scatter(tgt[:, 0], tgt[:, 1], s=8, c="#999", label="target (x, y)")
  ax[2].set(title="Initial-condition coverage (top view)", xlabel="x (m)", ylabel="y (m)", aspect="equal")
  ax[2].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, "dataset_overview.png"), dpi=150)
  print("saved dataset_overview.png")

  if args.no_render:
    return
  env = make_env()
  render_episode(env, ep, int(kept[0]), os.path.join(args.out, "demo_example.mp4"))
  fails = np.flatnonzero(~S)
  if len(fails):
    render_episode(env, ep, int(fails[0]), os.path.join(args.out, "teacher_failure_example.mp4"))
  # Filmstrip: rows = demos, columns = 5 evenly spaced moments.
  n = min(args.filmstrip_demos, len(kept))
  fig, axes = plt.subplots(n, 5, figsize=(15, 2.6 * n))
  for r, i in enumerate(kept[:n]):
    frames = render_episode(env, ep, int(i), os.path.join(args.out, f"_tmp.mp4"), height=240, width=320)
    for c, f in enumerate(np.linspace(0, len(frames) - 1, 5).astype(int)):
      a = axes[r, c] if n > 1 else axes[c]
      a.imshow(frames[f]); a.axis("off")
      a.set_title(f"demo {i}, t={f * 0.02:.2f}s", fontsize=8)
  os.remove(os.path.join(args.out, "_tmp.mp4"))
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, "demo_filmstrip.png"), dpi=120)
  print("saved demo_example.mp4, demo_filmstrip.png")


if __name__ == "__main__":
  main()