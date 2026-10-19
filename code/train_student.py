"""Part 2/3: train an ACT student on scripted-teacher demos.

  python code/train_student.py --algo act --data data/scripted_clean.npz --num-demos 100 --seed 0 --out runs/act_n100_s0
  # smaller dataset: same command with --num-demos 25 (the first 25 kept demos = a subset of the 100)

Writes <out>/model.pt (weights + config + normalisation stats), <out>/loss.csv, <out>/loss.png, <out>/config.json
"""

import argparse
import csv
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from policies import DEFAULTS, build_policy


class DemoDataset(torch.utils.data.Dataset):
    """One sample per (episode, timestep): obs history ending at t, action chunk starting at t.

    Steps past the end of an episode are padded with the episode's last action and flagged
    in `is_pad`, so the loss can ignore them.
    """

    def __init__(self, obs, actions, ep_len, obs_horizon, chunk, obs_mean, obs_std):
        self.obs = ((obs - obs_mean) / obs_std).clip(-10, 10).astype(np.float32)
        self.act, self.L = actions.astype(np.float32), ep_len
        self.To, self.K = obs_horizon, chunk
        self.index = [(i, t) for i in range(len(ep_len)) for t in range(ep_len[i])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        L = self.L[i]
        oi = np.clip(np.arange(t - self.To + 1, t + 1), 0, None)          # repeat first obs
        ai = np.arange(t, t + self.K)
        pad = ai >= L
        act = self.act[i, np.minimum(ai, L - 1)]
        return {"obs": self.obs[i, oi], "actions": act, "is_pad": pad}


def load_demos(path, num_demos):
    d = np.load(path)
    kept = d["kept"][:num_demos]               # first N kept demos -> smaller sets are subsets
    assert len(kept) == num_demos, f"only {len(kept)} kept demos in {path}"
    return d["obs"][kept], d["actions"][kept], d["ep_len"][kept], kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["act"], default="act")
    p.add_argument("--data", required=True)
    p.add_argument("--num-demos", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="csci645-hw2")
    args = p.parse_args()

    cfg = dict(DEFAULTS[args.algo])
    for k in ("steps", "batch_size", "lr", "chunk"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cfg.update(algo=args.algo, data=os.path.abspath(args.data), num_demos=args.num_demos, seed=args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    obs, act, L, kept = load_demos(args.data, args.num_demos)
    valid = np.arange(obs.shape[1])[None] < L[:, None]
    obs_mean = obs[valid].mean(0)
    obs_std = np.maximum(obs[valid].std(0), 1e-2)        # floor: some dims are ~constant
    ds = DemoDataset(obs, act, L, cfg["obs_horizon"], cfg["chunk"], obs_mean, obs_std)
    g = torch.Generator().manual_seed(args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                                     generator=g, num_workers=2, persistent_workers=True)
    print(f"{args.algo}: {len(kept)} demos, {len(ds)} samples, device={dev}")

    policy = build_policy(args.algo, cfg).to(dev)
    ema = copy.deepcopy(policy).eval() if cfg["ema"] > 0 else None
    n_params = sum(p.numel() for p in policy.parameters())
    cfg["num_params"] = n_params
    print(f"parameters: {n_params/1e6:.2f} M")
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, cfg["steps"]) / cfg["steps"])))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=os.path.basename(args.out.rstrip("/")), config=cfg)

    log_f = open(os.path.join(args.out, "loss.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "loss", "lr", "extra", "elapsed_s"])

    step, t0, running = 0, time.time(), []
    policy.train()
    while step < cfg["steps"]:
        for batch in dl:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            loss, extra = policy.compute_loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step(); sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pp in zip(ema.parameters(), policy.parameters()):
                        pe.mul_(cfg["ema"]).add_(pp.detach(), alpha=1 - cfg["ema"])
            running.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == cfg["steps"]:
                m = float(np.mean(running)); running = []
                log.writerow([step, m, sched.get_last_lr()[0], json.dumps(extra), round(time.time() - t0, 1)])
                log_f.flush()
                if run: run.log({"train/loss": m, "lr": sched.get_last_lr()[0], **{f"train/{k}": v for k, v in extra.items()}}, step=step)
                if step % (10 * args.log_every) == 0 or step == cfg["steps"]:
                    print(f"step {step:6d}  loss {m:.5f}  {extra}  ({time.time()-t0:.0f}s)")
            if step >= cfg["steps"]:
                break
    log_f.close()

    final = ema if ema is not None else policy
    torch.save({"state_dict": final.state_dict(), "cfg": cfg, "obs_mean": obs_mean, "obs_std": obs_std,
                "demo_indices": kept}, os.path.join(args.out, "model.pt"))
    print("saved", os.path.join(args.out, "model.pt"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(os.path.join(args.out, "loss.csv"))))
    plt.figure(figsize=(6, 4))
    plt.plot([int(r["step"]) for r in rows], [float(r["loss"]) for r in rows])
    plt.yscale("log"); plt.xlabel("gradient step"); plt.ylabel("training loss")
    plt.title(f"{args.algo}, {args.num_demos} demos, seed {args.seed}")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "loss.png"), dpi=150)
    if run:
        run.log({"loss_curve": wandb.Image(os.path.join(args.out, "loss.png"))}); run.finish()


if __name__ == "__main__":
    main()