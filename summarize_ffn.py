import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Config, forward_flops_per_token

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    (0.5, 1, ROOT / "runs/ffn_sweep/FF1/C.jsonl"),
    (1.0, 4, ROOT / "runs/N262144/C.jsonl"),
    (2.0, 10, ROOT / "runs/ffn_sweep/FF10/C.jsonl"),
    (4.0, 22, ROOT / "runs/ffn_sweep/FF22/C.jsonl"),
]


def read(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {row["event"]: row for row in rows}


def main():
    output = ROOT / "runs/ffn_sweep"
    table = []
    for scale, ff_mult, path in EXPERIMENTS:
        event = read(path)
        table.append({
            "compute_core_scale": scale,
            "ff_mult": ff_mult,
            "final_validation_loss": event["final"]["loss"],
            "final_ppl": event["final"]["ppl"],
            "last_1000_validation_loss_mean": event["last_1000_steps"]["validation_loss_mean"],
            "total_parameters": event["start"]["total"],
            "active_parameters": event["start"]["active"],
            "forward_flops_per_token": forward_flops_per_token(Config(**event["start"]["config"])),
        })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["compute_core_scale"] for r in table],
            [r["last_1000_validation_loss_mean"] for r in table], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks([r["compute_core_scale"] for r in table], ["0.5x", "1x", "2x", "4x"])
    ax.set_xlabel("Transformer compute-core size")
    ax.set_ylabel("Mean validation loss over last 1000 steps")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "ffn_sweep.png", dpi=180)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
