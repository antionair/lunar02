import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
idle_samples = 0
sample = 0
while idle_samples < 5:
    sample += 1
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    utilization = [int(line.split(",")[1]) for line in output.splitlines() if line.strip()]
    idle_samples = idle_samples + 1 if all(value == 0 for value in utilization) else 0
    print(f"sample={sample} utilization={utilization} consecutive_idle={idle_samples}", flush=True)
    if idle_samples < 5:
        time.sleep(30)

env = {**os.environ, "HF_ENDPOINT": "https://hf-mirror.com", "CUDA_VISIBLE_DEVICES": "2"}
common = [
    "--steps", "10000", "--batch-size", "8", "--seq-len", "128",
    "--eval-batches", "8", "--train-tokens", "30000000", "--val-tokens", "200000",
    "--d-model", "128", "--layers", "4", "--heads", "4", "--pkm-heads", "4",
    "--top-k", "8", "--d-query", "64", "--log-every", "25", "--seed", "42",
    "--lr", "0.0003", "--value-lr-mult", "5",
]
experiments = [("B10000", "B", 65536)] + [(f"N{n}", "C", n) for n in [16384, 65536, 262144, 1048576]]
for directory, model, n_values in experiments:
    print(f"Starting {directory}", flush=True)
    command = [
        str(Path.home() / "venv_train/bin/python"), str(ROOT / "train.py"),
        "--output", str(ROOT / "runs" / directory), "--models", model,
        "--n-values", str(n_values), *common,
    ]
    with (ROOT / f"{directory}.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Finished {directory}", flush=True)
subprocess.run([str(Path.home() / "venv_train/bin/python"), str(ROOT / "summarize_sweep.py")], cwd=ROOT, check=True)
print("All experiments and plots complete", flush=True)
