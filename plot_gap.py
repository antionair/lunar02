import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_evals(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if '"event": "eval"' in line]


def smooth(values, window=20):
    return [sum(values[max(0, i - window + 1):i + 1]) / len(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    root = Path(args.run_dir)
    a = read_evals(root / "A.jsonl")
    c = read_evals(root / "C.jsonl")
    assert [r["step"] for r in a] == [r["step"] for r in c]
    steps = [r["step"] for r in a]
    loss_gap = [cr["loss"] - ar["loss"] for ar, cr in zip(a, c)]
    ppl_gap = [cr["ppl"] - ar["ppl"] for ar, cr in zip(a, c)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, values, label in zip(axes, [loss_gap, ppl_gap], ["Validation loss gap (C - A)", "Validation PPL gap (C - A)"]):
        ax.plot(steps, values, color="steelblue", alpha=0.22, label="Raw")
        ax.plot(steps, smooth(values), color="navy", linewidth=2, label="20-point moving average")
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1].set_xlabel("Training steps")
    fig.suptitle("PKM model C versus baseline A (negative means C is better)")
    fig.tight_layout()
    fig.savefig(root / "c_minus_a_gap.png", dpi=180)
    with (root / "gap.jsonl").open("w", encoding="utf-8") as f:
        for step, loss, ppl in zip(steps, loss_gap, ppl_gap):
            f.write(json.dumps({"step": step, "loss_c_minus_a": loss, "ppl_c_minus_a": ppl}) + "\n")


if __name__ == "__main__":
    main()
