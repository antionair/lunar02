"""探针 #4 补跑：dense 基线等训练算力对照（152M tokens）。

任务书规格：与 #3 dense 组逐参数一致（args.json 级 diff 除 train_tokens 外必须为空），
LR {3e-4, 6e-4} × 种子 {1,2,3,4} 共 8 runs，两张 RTX 3080 并行。
"""
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = str(Path.home() / "venv_train/bin/python")
OUT = ROOT / "runs/dense_control"
DATA = ROOT / "data/scale"
QA_FILE = ROOT / "data/scale/qa_frozen.jsonl"
TRAIN_TOKENS = 152_000_000
LRS = [0.0003, 0.0006]
SEEDS = [1, 2, 3, 4]
GPUS = [
    {"name": "gpu0_3080", "uuid": "GPU-d149ae20-68a4-2f3d-256f-e808281c7830"},
    {"name": "gpu2_3080", "uuid": "GPU-c24d87e8-5b73-9732-0d10-69aee2a761a2"},
]


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_environment():
    payload = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "dense equal-training-control for scale probe #3 verdict",
        "train_tokens": TRAIN_TOKENS,
        "grid": {"lr": LRS, "seeds": SEEDS},
        "qa_file_sha256": sha256(QA_FILE),
        "data_bin_sha256": {
            "train_152m.bin": sha256(DATA / "train_152m.bin"),
            "eval_in_domain_4m.bin": sha256(DATA / "eval_in_domain_4m.bin"),
            "eval_domain_4m.bin": sha256(DATA / "eval_domain_4m.bin"),
        },
        "source_sha256": {name: sha256(ROOT / name) for name in [
            "scale_model.py", "scale_train.py", "run_dense_control.py", "summarize_dense_control.py",
        ]},
        "config_reference": "runs/scale_probe/dense_lr0.0003_seed1/args.json (probe #3 dense group)",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def complete(path):
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return '"event": "final"' in text and '"event": "fact_qa"' in text


def run_task(task, gpu, progress):
    name = f"dense152_lr{task['lr']:g}_seed{task['seed']}"
    directory = OUT / name
    result = directory / "train.jsonl"
    if complete(result):
        return {"name": name, **task, "status": "skipped"}
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        PYTHON, str(ROOT / "scale_train.py"), "--output", str(directory),
        "--data-dir", str(DATA), "--train-tokens", str(TRAIN_TOKENS),
        "--n-values", "0", "--pkm-dv", "512",
        "--lr", str(task["lr"]), "--value-lr-mult", "5", "--seed", str(task["seed"]),
        "--micro-batch", "1", "--grad-accum", "8", "--seq-len", "1000",
        "--eval-batches", "8", "--qa-file", str(QA_FILE),
        "--log-every", "25", "--device", "cuda",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu["uuid"],
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    started = time.time()
    with (ROOT / f"dense_control_{name}.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    payload = {"name": name, **task, "gpu": gpu["name"], "returncode": code,
               "elapsed_sec": time.time() - started,
               "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with progress.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    if code != 0:
        with (OUT / "failed_commands.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"command": command, "returncode": code}) + "\n")
    return payload


def worker(task_queue, gpu, progress, results):
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return
        results.append(run_task(task, gpu, progress))
        task_queue.task_done()


def main():
    if not QA_FILE.exists():
        raise FileNotFoundError(f"QA file must be user-audited and frozen: {QA_FILE}")
    write_environment()
    task_queue = queue.Queue()
    for lr in LRS:
        for seed in SEEDS:
            task_queue.put({"lr": lr, "seed": seed})
    progress = OUT / "progress.jsonl"
    results = []
    threads = [threading.Thread(target=worker, args=(task_queue, gpu, progress, results), daemon=True)
               for gpu in GPUS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    (OUT / "worker_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    subprocess.run([PYTHON, str(ROOT / "summarize_dense_control.py")], cwd=ROOT, check=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
