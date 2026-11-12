import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = str(Path.home() / "venv_train/bin/python")


def utilization():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    return [int(line.split(",")[1]) for line in output.splitlines() if line.strip()]


while True:
    values = utilization()
    print(f"30-minute check: utilization={values}", flush=True)
    if all(value == 0 for value in values):
        deadline = time.monotonic() + 300
        continuous = True
        while time.monotonic() < deadline:
            time.sleep(30)
            values = utilization()
            print(f"5-minute confirmation: utilization={values}", flush=True)
            if not all(value == 0 for value in values):
                continuous = False
                break
        if continuous:
            break
    time.sleep(1800)

print("All GPUs continuously idle for five minutes; starting FFN sweep", flush=True)
memory_output = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"
], text=True)
free_memory = [(int(parts[0]), parts[1].strip(), int(parts[2])) for line in memory_output.splitlines() if line.strip() for parts in [line.split(",")]]
gpu, gpu_uuid, free_mb = max(free_memory, key=lambda item: item[2])
print(f"Selected nvidia-smi GPU {gpu} ({gpu_uuid}) with {free_mb} MiB free", flush=True)
env = {**os.environ, "HF_ENDPOINT": "https://hf-mirror.com", "CUDA_VISIBLE_DEVICES": gpu_uuid}
common = [
    "--models", "C", "--steps", "10000", "--batch-size", "8", "--seq-len", "128",
    "--eval-batches", "8", "--train-tokens", "30000000", "--val-tokens", "200000",
    "--d-model", "128", "--layers", "4", "--heads", "4", "--n-values", "262144",
    "--pkm-heads", "4", "--top-k", "8", "--d-query", "64", "--log-every", "25",
    "--seed", "42", "--lr", "0.0003", "--value-lr-mult", "5", "--recurrence", "1",
]
for ff_mult in [1, 10, 22]:
    directory = ROOT / "runs" / "ffn_sweep" / f"FF{ff_mult}"
    result = directory / "C.jsonl"
    if result.exists() and '"event": "random_cell_replacement"' in result.read_text():
        print(f"Skipping completed FF{ff_mult}", flush=True)
        continue
    print(f"Starting FF{ff_mult}", flush=True)
    command = [PYTHON, str(ROOT / "train.py"), "--output", str(directory),
               "--ff-mult", str(ff_mult), *common]
    with (ROOT / f"ffn_sweep_FF{ff_mult}.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Finished FF{ff_mult}", flush=True)
subprocess.run([PYTHON, str(ROOT / "summarize_ffn.py")], cwd=ROOT, check=True)
print("Feasible FFN sweep complete; 32B configuration not runnable on one GPU", flush=True)
