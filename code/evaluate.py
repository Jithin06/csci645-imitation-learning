"""Part 4: evaluate a student (or a teacher) on held-out initial conditions.

  python code/evaluate.py --policy student --ckpt runs/scripted_clean_act_n100_s0/model.pt --out results/eval_scripted_clean_act_n100_s0
  python code/evaluate.py --policy scripted --out results/eval_teacher_scripted

Every condition uses the same seeds (--seed-start 100000, --num-episodes 50 by default), which
never appear in the training data (collection uses seeds 0..). Writes summary.json,
episodes.csv, and success/failure videos.
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from common import HORIZON, make_env, render_episode, run_episodes, save_json, summarize
from policies import build_policy
from teachers import make_teacher

class StudentPolicy:
    """Wraps a trained ACT policy for batched closed-loop execution.

    Queries a new 20-step chunk every step and averages every prediction made for the current
    step with weights exp(-m * i), where i = 0 is the oldest prediction (temporal ensembling as
    in the ACT reference code, m = 0.01, so older predictions weigh slightly more).
    --no-temporal-agg executes each chunk open-loop instead.
    """

    def __init__(self, ckpt, device, temporal_agg=True, m=0.01, exec_horizon=None):
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg = blob["cfg"]
        self.algo = self.cfg["algo"]
        self.net = build_policy(self.algo, self.cfg).to(device).eval()
        self.net.load_state_dict(blob["state_dict"])
        self.mean = torch.tensor(blob["obs_mean"], device=device)
        self.std = torch.tensor(blob["obs_std"], device=device)
        self.dev, self.K, self.To = device, self.cfg["chunk"], self.cfg["obs_horizon"]
        self.temporal_agg = temporal_agg
        self.m = m
        self.exec_h = exec_horizon or self.K
        assert 1 <= self.exec_h <= self.K, f"--exec-horizon must be between 1 and the chunk length {self.K}"

    def reset(self, n):
        self.n, self.hist, self.queue = n, None, None
        if self.temporal_agg:
            self.buf = torch.zeros(n, HORIZON + self.K, HORIZON + self.K, 8, device=self.dev)   # [env, query_t, t, a]
            self.filled = torch.zeros(n, HORIZON + self.K, HORIZON + self.K, dtype=torch.bool, device=self.dev)

    def _norm(self, obs):
        o = torch.as_tensor(obs, device=self.dev)
        return ((o - self.mean) / self.std).clamp(-10, 10)

    def act(self, state, obs, t):
        o = self._norm(obs)
        self.hist = o[:, None].repeat(1, self.To, 1) if self.hist is None else torch.cat([self.hist[:, 1:], o[:, None]], 1)
        if self.temporal_agg:
            chunk = self.net.predict_chunk(self.hist)                 # (n, K, 8)
            self.buf[:, t, t:t + self.K] = chunk
            self.filled[:, t, t:t + self.K] = True
            mask = self.filled[:, : t + 1, t]                         # predictions made at steps <= t for step t
            query_idx = torch.arange(0, t + 1, device=self.dev).float()   # 0 = oldest prediction for step t
            w = torch.exp(-self.m * query_idx)[None] * mask
            a = (self.buf[:, : t + 1, t] * w[..., None]).sum(1) / w.sum(1, keepdim=True)
            return a.cpu().numpy()
        if self.queue is None or self.qi >= self.exec_h:
            self.queue, self.qi = self.net.predict_chunk(self.hist), 0
        a = self.queue[:, self.qi]
        self.qi += 1
        return a.cpu().numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["student", "scripted"], required=True)
    p.add_argument("--ckpt", default=None, help="student model.pt (for --policy student)")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=100000)
    p.add_argument("--no-temporal-agg", action="store_true")
    p.add_argument("--exec-horizon", type=int, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = make_env()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.policy == "student":
        torch.manual_seed(0)
        policy = StudentPolicy(args.ckpt, dev, temporal_agg=not args.no_temporal_agg, exec_horizon=args.exec_horizon)
    else:
        policy = make_teacher(env, args.policy)

    seeds = list(range(args.seed_start, args.seed_start + args.num_episodes))
    t0 = time.time()
    ep = run_episodes(env, policy, seeds, log_every=50)
    s = summarize(ep)
    s.update(policy=args.policy, ckpt=args.ckpt, seeds=[seeds[0], seeds[-1]], eval_wall_s=round(time.time() - t0, 1),
             temporal_agg=(not args.no_temporal_agg), device=dev)
    # Action smoothness: mean |a_t - a_{t-1}| over valid steps (useful for failure analysis).
    diffs = [np.abs(np.diff(ep["actions"][i, : ep["ep_len"][i]], axis=0)).mean() for i in range(len(seeds))]
    s["action_jerk_mean"] = float(np.mean(diffs))
    save_json(s, os.path.join(args.out, "summary.json"))
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "success", "ep_len", "final_dist_m", "out_of_bounds", "box_x", "box_y", "target_x", "target_y", "target_z"])
        for i, sd in enumerate(seeds):
            L = ep["ep_len"][i]
            w.writerow([sd, int(ep["success"][i]), L, round(float(ep["box_target_dist"][i, L - 1]), 4),
                        int(ep["out_of_bounds"][i]), *np.round(ep["qpos"][i, 0, 9:11], 3), *np.round(ep["target_pos"][i], 3)])
    np.savez_compressed(os.path.join(args.out, "rollouts.npz"), **ep)
    print(f"SUCCESS RATE {s['success_rate']:.3f} ({s['num_success']}/{s['num_episodes']})")

    if not args.no_video:
        ok, bad = np.flatnonzero(ep["success"]), np.flatnonzero(~ep["success"])
        for name, idx in (("success", ok[:2]), ("failure", bad[:2])):
            for i in idx:
                render_episode(env, ep, int(i), os.path.join(args.out, f"{name}_seed{seeds[i]}.mp4"))
        print("videos written to", args.out)


if __name__ == "__main__":
    main()