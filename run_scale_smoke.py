import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = str(Path.home() / "venv_train/bin/python")
OUT = ROOT / "runs/scale_probe"
GPU_JOBS = [
    ("dense", 0, "3", 1),
    ("n1m", 1_000_000, "GPU-c24d87e8-5b73-9732-0d10-69aee2a761a2", 1),
    ("n4m", 4_000_000, "GPU-d149ae20-68a4-2f3d-256f-e808281c7830", 1),
]
STEPS = 200
SEQ_LEN = 1000
MICRO_BATCH = 1
GRAD_ACCUM = 8
TOKENS = STEPS * MICRO_BATCH * SEQ_LEN * GRAD_ACCUM


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    processes = []
    for name, n_values, gpu_uuid, micro_batch in GPU_JOBS:
        directory = OUT / f"smoke_{name}"
        command = [
            PYTHON, str(ROOT / "scale_train.py"), "--output", str(directory),
            "--data-dir", str(ROOT / "data/scale"), "--train-tokens", str(TOKENS),
            "--n-values", str(n_values), "--pkm-dv", "512", "--lr", "0.0003",
            "--value-lr-mult", "5", "--seed", "1", "--micro-batch", str(micro_batch),
            "--grad-accum", str(GRAD_ACCUM), "--seq-len", str(SEQ_LEN),
            "--eval-batches", "8", "--log-every", "50", "--device", "cuda",
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_uuid}
        log_path = ROOT / f"scale_smoke_{name}.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((name, process, log))
    results = {}
    for name, process, log in processes:
        code = process.wait()
        log.close()
        result_path = OUT / f"smoke_{name}" / "train.jsonl"
        row = {"name": name, "returncode": code}
        if result_path.exists():
            rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
            final = next((r for r in rows if r["event"] == "final"), None)
            if final:
                row.update(final)
        results[name] = row
        print(json.dumps(row), flush=True)
    payload = {
        "requested_steps": STEPS,
        "tokens": TOKENS,
        "wall_seconds": time.time() - started,
        "jobs": results,
    }
    (OUT / "smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
