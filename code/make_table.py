"""Collect every results/eval_*/summary.json into one table (CSV + markdown) and a bar plot.

  python code/make_table.py

Writes results/summary.csv, results/summary.md, results/summary.png
"""

import csv
import glob
import json
import os
import sys

TEACHER_EVAL = "results/eval_teacher_scripted/summary.json"


def final_l1(run_dir):
    """Final training L1 from the last row of loss.csv (None if missing)."""
    path = os.path.join(run_dir, "loss.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return round(json.loads(rows[-1]["extra"]).get("l1", float("nan")), 4) if rows else None


def main():
    teacher_heldout = json.load(open(TEACHER_EVAL))["success_rate"] if os.path.exists(TEACHER_EVAL) else None

    rows = []
    for f in sorted(glob.glob("results/eval_*/summary.json")):
        s = json.load(open(f))
        name = os.path.basename(os.path.dirname(f))[len("eval_"):]
        cfg, run_dir = {}, None
        if s.get("ckpt") and s["ckpt"].endswith("model.pt"):
            run_dir = os.path.dirname(s["ckpt"])
            cfg = json.load(open(os.path.join(run_dir, "config.json")))
        stats_f = cfg.get("data", "").replace(".npz", "_stats.json")
        teacher_collect = json.load(open(stats_f))["attempted"]["success_rate"] if os.path.exists(stats_f) else None
        is_student = s["policy"] == "student"
        rows.append(dict(
            condition=name,
            policy="ACT" if is_student else "teacher",
            num_demos=cfg.get("num_demos", ""),
            seed=cfg.get("seed", ""),
            temporal_agg=s.get("temporal_agg") if is_student else "",
            teacher_success_collection=teacher_collect if teacher_collect is not None else "",
            teacher_success_heldout=teacher_heldout if teacher_heldout is not None else "",
            student_success=s["success_rate"] if is_student else "",
            successes=f'{s["num_success"]}/{s["num_episodes"]}',
            eval_episodes=s["num_episodes"],
            ep_len_mean=round(s["ep_len_mean"], 1),
            action_jerk=round(s.get("action_jerk_mean", float("nan")), 4),
            final_train_l1=final_l1(run_dir) if run_dir else "",
        ))
    if not rows:
        sys.exit("no results/eval_*/summary.json yet")

    # teacher first, then students by number of demos (largest first), ensembling before no-ensembling
    rows.sort(key=lambda r: (r["policy"] != "teacher", -(r["num_demos"] or 0), r["temporal_agg"] is False))

    keys = list(rows[0])
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    open("results/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r["condition"] for r in rows]
    vals = [r["student_success"] if r["policy"] == "ACT" else r["teacher_success_heldout"] for r in rows]
    colors = ["#999999" if r["policy"] == "teacher" else "#3b6ea5" for r in rows]
    plt.figure(figsize=(max(5, 1.2 * len(rows)), 4))
    bars = plt.bar(labels, vals, color=colors)
    for b, r in zip(bars, rows):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, r["successes"], ha="center", fontsize=8)
    plt.ylabel("success rate (50 held-out seeds)"); plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right"); plt.title("Teacher (grey) and ACT students (blue)")
    plt.tight_layout(); plt.savefig("results/summary.png", dpi=150)
    print("wrote results/summary.csv, results/summary.md, results/summary.png")


if __name__ == "__main__":
    main()