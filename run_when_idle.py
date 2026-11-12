import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REQUIRED_IDLE_SAMPLES = 5
idle_samples = 0
sample = 0
while idle_samples < REQUIRED_IDLE_SAMPLES:
    sample += 1
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    utilization = [int(line.split(",")[1]) for line in output.splitlines() if line.strip()]
    idle_samples = idle_samples + 1 if all(value == 0 for value in utilization) else 0
    print(f"sample={sample} utilization={utilization} consecutive_idle={idle_samples}", flush=True)
    if idle_samples < REQUIRED_IDLE_SAMPLES:
        time.sleep(30)

print("All GPUs idle for five consecutive samples; starting training", flush=True)
env = {**os.environ, "HF_ENDPOINT": "https://hf-mirror.com", "CUDA_VISIBLE_DEVICES": "2"}
command = [
    str(Path.home() / "venv_train/bin/python"), str(ROOT / "train.py"),
    "--output", str(ROOT / "runs/long10000"), "--models", "A,C",
    "--steps", "10000", "--batch-size", "8", "--seq-len", "128",
    "--eval-batches", "8", "--train-tokens", "30000000", "--val-tokens", "200000",
    "--d-model", "128", "--layers", "4", "--heads", "4", "--n-values", "65536",
    "--pkm-heads", "4", "--top-k", "8", "--d-query", "64", "--log-every", "25",
]
with (ROOT / "long10000.log").open("w") as log:
    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
subprocess.run([
    str(Path.home() / "venv_train/bin/python"), str(ROOT / "plot_gap.py"), str(ROOT / "runs/long10000")
], cwd=ROOT, check=True)
print("Training and plotting complete", flush=True)
