import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = str(Path.home() / "venv_train/bin/python")
ENV_FILE = ROOT / "runs/ffn_lr_probe/environment.json"
PROGRESS_FILE = ROOT / "runs/ffn_lr_probe/progress.jsonl"


def gpu_rows():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"
    ], text=True)
    return [line.split(",") for line in output.splitlines() if line.strip()]


def source_hashes():
    result = {}
    for name in ["pkm.py", "model.py", "train.py"]:
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return result


def write_environment():
    rows = gpu_rows()
    selected = [row for row in rows if int(row[3]) <= 20_480]
    if not selected:
        print("All GPUs exceed 20GB used; waiting 10 minutes", flush=True)
        time.sleep(600)
        return False
    selected.sort(key=lambda row: int(row[4]), reverse=True)
    row = selected[0]
    payload = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "python": subprocess.check_output([PYTHON, "-c", "import sys; print(sys.version.split()[0])"], text=True).strip(),
        "torch": subprocess.check_output([PYTHON, "-c", "import torch; print(torch.__version__)"], text=True).strip(),
        "cuda": subprocess.check_output([PYTHON, "-c", "import torch; print(torch.version.cuda)"], text=True).strip(),
        "gpus": [{"index": int(r[0]), "uuid": r[1].strip(), "name": r[2], "used_mib": int(r[3]), "free_mib": int(r[4]), "utilization": int(r[5])} for r in rows],
        "selected_gpu_uuid": row[1].strip(),
        "source_sha256": source_hashes(),
        "git_commit": None,
    }
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def complete(path):
    return path.exists() and '"event": "random_cell_replacement"' in path.read_text(encoding="utf-8")


def last_mean(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["event"] == "last_1000_steps":
            return row["validation_loss_mean"]
    raise ValueError(path)


def run_one(ff, lr, steps, seed, gpu_uuid):
    directory = ROOT / "runs/ffn_lr_probe" / f"FF{ff}_lr{lr:g}_steps{steps}_seed{seed}"
    result = directory / "C.jsonl"
    if complete(result):
        print(f"Skipping completed {directory.name}", flush=True)
        return
    command = [
        PYTHON, str(ROOT / "train.py"), "--output", str(directory), "--models", "C",
        "--ff-mult", str(ff), "--steps", str(steps), "--batch-size", "8", "--seq-len", "128",
        "--eval-batches", "8", "--train-tokens", "30000000", "--val-tokens", "200000",
        "--d-model", "128", "--layers", "4", "--heads", "4", "--n-values", "262144",
        "--pkm-heads", "4", "--top-k", "8", "--d-query", "64", "--log-every", "25",
        "--seed", str(seed), "--lr", str(lr), "--value-lr-mult", "5", "--recurrence", "1",
    ]
    env = {**os.environ, "HF_ENDPOINT": "https://hf-mirror.com", "CUDA_VISIBLE_DEVICES": gpu_uuid}
    started = time.time()
    with (ROOT / f"probe_{directory.name}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.time() - started
    payload = {"name": directory.name, "ff": ff, "lr": lr, "steps": steps, "seed": seed,
               "elapsed_sec": elapsed, "train_tokens": steps * 8 * 128,
               "tokens_per_sec": steps * 8 * 128 / elapsed, "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with PROGRESS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    print(f"Finished {directory.name}: {payload['tokens_per_sec']:.1f} tok/s", flush=True)


def main():
    while not write_environment():
        pass
    env = json.loads(ENV_FILE.read_text(encoding="utf-8"))
    gpu_uuid = env["selected_gpu_uuid"]
    for steps in [10000, 20000]:
        for ff in [4, 10, 22]:
            for lr in [0.0003, 0.0006]:
                for seed in [1, 2]:
                    run_one(ff, lr, steps, seed, gpu_uuid)
    confirmations = []
    for ff in [4, 10, 22]:
        candidates = []
        for lr in [0.0003, 0.0006]:
            for steps in [10000, 20000]:
                paths = [ROOT / "runs/ffn_lr_probe" / f"FF{ff}_lr{lr:g}_steps{steps}_seed{seed}" / "C.jsonl" for seed in [1, 2]]
                mean = (last_mean(paths[0]) + last_mean(paths[1])) / 2
                candidates.append((mean, lr, steps))
        _, lr, steps = min(candidates)
        run_one(ff, lr, steps, 3, gpu_uuid)
        confirmations.append({"ff": ff, "lr": lr, "steps": steps})
    (ROOT / "runs/ffn_lr_probe/confirmations.json").write_text(json.dumps(confirmations, indent=2), encoding="utf-8")
    subprocess.run([PYTHON, str(ROOT / "summarize_ffn_lr_probe.py")], cwd=ROOT, check=True)
    print("FFN LR probe complete", flush=True)


if __name__ == "__main__":
    main()
