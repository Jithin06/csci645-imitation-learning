"""Part 1: collect demonstrations from the scripted teacher and save them with diagnostics.

    python code/collect_demos.py --teacher scripted --num-demos 100 --out data/scripted_clean
    # noisy teacher actions during interaction
    python code/collect_demos.py --teacher scripted --noise-std 0.3 --num-demos 100 --out data/scripted_noise0.3

Writes  <out>.npz         every attempted episode (obs, actions, rewards, ep_len, success, seeds, qpos, ...)
        <out>_stats.json  teacher success rate, length stats, which episodes were kept
Seeds start at --seed-start (default 0). Evaluation uses seeds >= 100000, so they never overlap.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from common import HORIZON, make_env, run_episodes, save_json, summarize
from teachers import NoisyPolicy, make_teacher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", choices=["scripted"], default="scripted")
    p.add_argument("--num-demos", type=int, default=100)
    p.add_argument("--attempts", type=int, default=None,
                   help="episodes to roll out (default: 1.6 x num-demos, enough to keep num-demos successes)")
    p.add_argument("--keep", choices=["success", "all"], default="success",
                   help="'success' keeps the first num-demos successful episodes (in seed order)")
    p.add_argument("--noise-std", type=float, default=0.0, help="Gaussian action noise during collection")
    p.add_argument("--noise-label", choices=["executed", "clean"], default="executed")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--out", required=True, help="path prefix, e.g. data/scripted_clean")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    env = make_env()
    teacher = make_teacher(env, args.teacher, seed=args.seed_start)
    if args.noise_std > 0:
        teacher = NoisyPolicy(teacher, args.noise_std, seed=args.seed_start, label=args.noise_label)

    n_try = args.attempts or int(np.ceil(1.6 * args.num_demos))
    seeds = list(range(args.seed_start, args.seed_start + n_try))
    t0 = time.time()
    ep = run_episodes(env, teacher, seeds, horizon=args.horizon, log_every=25)
    wall = time.time() - t0

    if args.keep == "success":
        keep = np.flatnonzero(ep["success"])[: args.num_demos]
        if len(keep) < args.num_demos:
            print(f"WARNING: only {len(keep)} successes in {n_try} attempts; rerun with a larger --attempts")
    else:
        keep = np.arange(min(args.num_demos, n_try))

    stats = {
        "teacher": args.teacher,
        "noise_std": args.noise_std, "noise_label": args.noise_label,
        "seed_start": args.seed_start, "horizon": args.horizon, "wall_s": round(wall, 1),
        "attempted": summarize(ep),  # teacher success rate lives here
        "kept": summarize({k: ep[k][keep] for k in ("ep_len", "success", "out_of_bounds", "box_target_dist")}),
        "keep_rule": args.keep, "kept_indices": keep.tolist(),
    }
    print("teacher success over all attempts: "
          f"{stats['attempted']['num_success']}/{n_try} = {stats['attempted']['success_rate']:.3f}")
    print(f"kept {len(keep)} episodes, length {stats['kept']['ep_len_mean']:.1f} +- {stats['kept']['ep_len_std']:.1f}")

    np.savez_compressed(args.out + ".npz", kept=keep, **ep)
    save_json(stats, args.out + "_stats.json")
    print("wrote", args.out + ".npz", "and", args.out + "_stats.json")


if __name__ == "__main__":
    main()