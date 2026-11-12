import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
GROUPS = [
    ("A (R=1, no PKM)", "multiseed/A_seed{}", "A"),
    ("C N=65536 (R=1)", "multiseed/C65536_seed{}", "C"),
    ("C N=262144 (R=1)", "multiseed/C262144_seed{}", "C"),
    ("R=3, no PKM", "recurrence/R3_A_seed{}", "A"),
    ("R=3, N=262144", "recurrence/R3_C262144_seed{}", "C"),
]


def load_group(pattern, model):
    losses = []
    start = None
    for seed in [1, 2, 3]:
        path = RUNS / pattern.format(seed) / f"{model}.jsonl"
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        events = {row["event"]: row for row in rows}
        losses.append(events["last_1000_steps"]["validation_loss_mean"])
        if start is None:
            start = events["start"]
    return {
        "mean_last_1000_loss": statistics.mean(losses),
        "std_last_1000_loss": statistics.stdev(losses),
        "seed_losses": losses,
        "total_parameters": start["total"],
        "active_parameters": start["active"],
        "forward_flops_per_token": start["forward_flops_per_token"],
    }


def main():
    output = RUNS / "recurrence_summary"
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for label, pattern, model in GROUPS:
        table.append({"model": label, **load_group(pattern, model)})
    fields = ["model", "mean_last_1000_loss", "std_last_1000_loss", "total_parameters",
              "active_parameters", "forward_flops_per_token", "seed_losses"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
