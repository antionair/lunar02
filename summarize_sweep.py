import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
EXPERIMENTS = [("A", RUNS / "long10000", "A"), ("B", RUNS / "B10000", "B")] + [
    (f"C N={n}", RUNS / f"N{n}", "C") for n in [16384, 65536, 262144, 1048576]
]


def events(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {row["event"]: row for row in rows}, rows


def main():
    output = RUNS / "n_sweep"
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for label, directory, model_name in EXPERIMENTS:
        event, rows = events(directory / f"{model_name}.jsonl")
        start, final = event["start"], event["final"]
        row = {
            "model": label,
            "final_validation_loss": final["loss"],
            "final_ppl": final["ppl"],
            "total_parameters": start["total"],
            "active_parameters": start["active"],
            "cell_usage": start["config"]["n_values"] and event["eval"]["observed_cell_fraction"] if model_name == "C" else "",
        }
        if model_name == "C":
            evals = [r for r in rows if r["event"] == "eval"]
            tail = [r for r in evals if r["step"] > 9000]
            row.update({
                "last_1000_validation_loss_mean": event["last_1000_steps"]["validation_loss_mean"],
                "batch_unique_cell_fraction": sum(r["batch_unique_cell_fraction"] for r in tail) / len(tail),
                "batch_access_entropy_nats": sum(r["batch_access_entropy_nats"] for r in tail) / len(tail),
                "zero_memory_loss": event["pkm_ablation"]["loss"],
                "random_replacement_loss": event["random_cell_replacement"]["loss"],
            })
        table.append(row)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]) + [k for k in table[-1] if k not in table[0]])
        writer.writeheader()
        writer.writerows(table)
    c_rows = table[2:]
    sizes = [int(row["model"].split("=")[1]) for row in c_rows]
    losses = [row["final_validation_loss"] for row in c_rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sizes, losses, marker="o", linewidth=2, label="C with product-key memory")
    ax.axhline(table[0]["final_validation_loss"], color="black", linestyle="--", label="A baseline")
    ax.axhline(table[1]["final_validation_loss"], color="darkred", linestyle="--", label="B wide FFN")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.set_xlabel("Number of memory cells N (log scale)")
    ax.set_ylabel("Final validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "n_sweep.png", dpi=180)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
