import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs/ffn_lr_probe"
WIDTHS = [4, 10, 22]
LRS = [0.0003, 0.0006]
STEPS = [10000, 20000]
SEEDS = [1, 2]


def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_dir(ff, lr, steps, seed):
    return RUNS / f"FF{ff}_lr{lr:g}_steps{steps}_seed{seed}"


def main():
    grid = []
    curves = {}
    for ff in WIDTHS:
        for lr in LRS:
            for steps in STEPS:
                seed_losses = []
                for seed in SEEDS:
                    rows = read(run_dir(ff, lr, steps, seed) / "C.jsonl")
                    events = {row["event"]: row for row in rows}
                    seed_losses.append(events["last_1000_steps"]["validation_loss_mean"])
                    if seed == SEEDS[0]:
                        curves[(ff, lr, steps)] = (events, rows)
                grid.append({
                    "ffn": ff, "lr": lr, "steps": steps,
                    "seed1_last1000": seed_losses[0], "seed2_last1000": seed_losses[1],
                    "mean": statistics.mean(seed_losses), "sample_std": statistics.stdev(seed_losses),
                })
    with (RUNS / "grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    summary = []
    for ff in WIDTHS:
        candidates = [row for row in grid if row["ffn"] == ff]
        best_grid = min(candidates, key=lambda row: row["mean"])
        seed3_path = run_dir(ff, best_grid["lr"], best_grid["steps"], 3) / "C.jsonl"
        seed3_rows = read(seed3_path)
        seed3_event = {row["event"]: row for row in seed3_rows}
        seed3_loss = seed3_event["last_1000_steps"]["validation_loss_mean"]
        seed_losses = [best_grid["seed1_last1000"], best_grid["seed2_last1000"], seed3_loss]
        start = seed3_event["start"]
        summary.append({
            "ffn": ff, "best_lr": best_grid["lr"], "best_steps": best_grid["steps"],
            "seed1_last1000": seed_losses[0], "seed2_last1000": seed_losses[1], "seed3_last1000": seed3_loss,
            "mean": statistics.mean(seed_losses), "sample_std": statistics.stdev(seed_losses),
            "total_parameters": start["total"], "active_parameters": start["active"],
            "forward_flops_per_token": start["forward_flops_per_token"],
        })
    with (RUNS / "best_by_width.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    d4 = next(row for row in summary if row["ffn"] == 4)["mean"]
    d10 = next(row for row in summary if row["ffn"] == 10)["mean"]
    d = d4 - d10
    verdict = "green" if d > 0.01 else "red" if d < -0.01 else "yellow"
    (RUNS / "verdict.json").write_text(json.dumps({"d": d, "verdict": verdict, "best4": d4, "best10": d10}, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, ff in zip(axes, WIDTHS):
        for lr in LRS:
            for steps in STEPS:
                rows = [row for row in read(run_dir(ff, lr, steps, 1) / "C.jsonl") if row["event"] == "eval"]
                ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], alpha=0.25, label=f"lr={lr:g}, {steps} steps (seed1)")
        ax.set_title(f"FFN-{ff}")
        ax.set_xlabel("Training steps")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Validation loss")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RUNS / "curves.png", dpi=180)
    print(json.dumps({"verdict": verdict, "d": d, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
