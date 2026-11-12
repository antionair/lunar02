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
    "--top-k", "8", "--d-query", "64", "--log-every", "25",
    "--lr", "0.0003", "--value-lr-mult", "5",
]
experiments = []
for seed in [1, 2, 3]:
    experiments.extend([
        (f"multiseed/A_seed{seed}", "A", 65536, 1, seed),
        (f"multiseed/C65536_seed{seed}", "C", 65536, 1, seed),
        (f"multiseed/C262144_seed{seed}", "C", 262144, 1, seed),
    ])
    experiments.extend([
        (f"recurrence/R3_A_seed{seed}", "A", 65536, 3, seed),
        (f"recurrence/R3_C262144_seed{seed}", "C", 262144, 3, seed),
    ])
for directory, model, n_values, recurrence, seed in experiments:
    print(f"Starting {directory}", flush=True)
    command = [
        str(Path.home() / "venv_train/bin/python"), str(ROOT / "train.py"),
        "--output", str(ROOT / "runs" / directory), "--models", model,
        "--n-values", str(n_values), "--recurrence", str(recurrence),
        "--seed", str(seed), *common,
    ]
    with (ROOT / f"{directory.replace('/', '_')}.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Finished {directory}", flush=True)
subprocess.run([str(Path.home() / "venv_train/bin/python"), str(ROOT / "summarize_recurrence.py")], cwd=ROOT, check=True)
print("All multiseed and recurrence experiments complete", flush=True)
