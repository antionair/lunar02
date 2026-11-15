import csv
import json
import math
from pathlib import Path
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/scale_probe"
GROUPS = ["dense", "n1m", "n4m"]
LRS = [0.0003, 0.0006]
GRID_SEEDS = [1, 2]
CONFIRM_SEEDS = [1, 2, 3, 4]


def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def events(path):
    rows = read(path)
    by_event = {}
    for row in rows:
        by_event.setdefault(row["event"], []).append(row)
    return by_event, rows


def run_dir(group, lr, seed):
    return OUT / f"{group}_lr{lr:g}_seed{seed}"


def tail_mean(rows):
    evals = [row for row in rows if row["event"] == "eval"]
    tail = [row["val_loss"] for row in evals if row["step"] > max(r["step"] for r in evals) - 1000]
    return sum(tail) / len(tail)


def main():
    grid = []
    for group in GROUPS:
        for lr in LRS:
            losses = []
            for seed in GRID_SEEDS:
                by_event, rows = events(run_dir(group, lr, seed) / "train.jsonl")
                losses.append(tail_mean(rows))
            grid.append({"group": group, "lr": lr, "seed1_loss": losses[0], "seed2_loss": losses[1],
                         "mean_loss": statistics.mean(losses), "sample_std": statistics.stdev(losses)})
    summary = []
    for group in GROUPS:
        candidates = [row for row in grid if row["group"] == group]
        best = min(candidates, key=lambda row: row["mean_loss"])
        losses, ppls, domain, exact, f1, usage, entropy, throughput = [], [], [], [], [], [], [], []
        for seed in CONFIRM_SEEDS:
            by_event, rows = events(run_dir(group, best["lr"], seed) / "train.jsonl")
            loss = tail_mean(rows)
            losses.append(loss)
            ppls.append(math.exp(loss))
            final = by_event["final"][-1]
            domain.append(final["domain_ppl"])
            throughput.append(final["tokens_per_sec"])
            if "fact_qa" in by_event:
                exact.append(by_event["fact_qa"][-1]["exact_match"])
                f1.append(by_event["fact_qa"][-1]["token_f1"])
            if group != "dense":
                final_eval = [row for row in rows if row["event"] == "eval"][-1]
                usage.append(final_eval["cumulative_cell_usage"])
                entropy.append(final_eval["batch_access_entropy_nats"])
        start = events(run_dir(group, best["lr"], 1) / "train.jsonl")[0]["start"][0]
        summary.append({
            "group": group, "best_lr": best["lr"],
            "mean_loss": statistics.mean(losses), "sample_std_loss": statistics.stdev(losses),
            "mean_ppl": statistics.mean(ppls), "sample_std_ppl": statistics.stdev(ppls),
            "mean_domain_ppl": statistics.mean(domain), "sample_std_domain_ppl": statistics.stdev(domain),
            "mean_fact_exact": statistics.mean(exact) if exact else None,
            "mean_fact_f1": statistics.mean(f1) if f1 else None,
            "mean_cumulative_usage": statistics.mean(usage) if usage else None,
            "mean_access_entropy": statistics.mean(entropy) if entropy else None,
            "mean_tokens_per_sec": statistics.mean(throughput),
            "total_parameters": start["total"], "active_parameters": start["active"],
            "value_parameters": start["value_parameters"],
        })
    dense = next(row for row in summary if row["group"] == "dense")
    n1m = next(row for row in summary if row["group"] == "n1m")
    n4m = next(row for row in summary if row["group"] == "n4m")
    g1 = dense["mean_ppl"] - n1m["mean_ppl"]
    g4 = n1m["mean_ppl"] - n4m["mean_ppl"]
    r = g4 / g1 if g1 != 0 else float("nan")
    fact_monotonic = n1m["mean_fact_exact"] >= dense["mean_fact_exact"] and n4m["mean_fact_exact"] >= n1m["mean_fact_exact"]
    generalization_noise = 2 * dense["sample_std_domain_ppl"]
    generalization_ok = n1m["mean_domain_ppl"] <= dense["mean_domain_ppl"] + generalization_noise and n4m["mean_domain_ppl"] <= dense["mean_domain_ppl"] + generalization_noise
    usage_ok = (n1m["mean_cumulative_usage"] or 0) >= 0.01 and (n4m["mean_cumulative_usage"] or 0) >= 0.01
    if r < 0 or not usage_ok or not generalization_ok:
        verdict = "red"
    elif r >= 0.5 and fact_monotonic:
        verdict = "green"
    elif 0 <= r < 0.5:
        verdict = "yellow"
    else:
        verdict = "red"
    with (OUT / "grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUT / "verdict.json").write_text(json.dumps({
        "g1": g1, "g4": g4, "r": r, "fact_monotonic": fact_monotonic,
        "generalization_noise_band": generalization_noise, "generalization_ok": generalization_ok,
        "usage_ok": usage_ok, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, group in zip(axes, GROUPS):
        best = min([row for row in grid if row["group"] == group], key=lambda row: row["mean_loss"])
        for seed in CONFIRM_SEEDS[:2]:
            _, rows = events(run_dir(group, best["lr"], seed) / "train.jsonl")
            evals = [row for row in rows if row["event"] == "eval"]
            ax.plot([row["step"] for row in evals], [row["val_loss"] for row in evals], label=f"lr={best['lr']:g}, seed{seed}")
        ax.set_title(group)
        ax.set_xlabel("Training steps")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out validation loss")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=180)
    print(json.dumps({"verdict": verdict, "summary": summary, "g1": g1, "g4": g4, "r": r}, indent=2))


if __name__ == "__main__":
    main()
