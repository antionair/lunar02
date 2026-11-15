import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
PYTHON = str(Path.home() / "venv_train/bin/python")
OUT = ROOT / "runs/scale_probe"
DATA = ROOT / "data/scale"
QA_FILE = ROOT / "data/scale/qa_frozen.jsonl"
# GPU3 (gpu3_3080, uuid GPU-8e258ee4-017b-5203-eea8-753431ba1110) 自 2026-11-13 起
# 预留给项目所有者使用，不参与 18 runs 调度（n4m 并发信号量本就只有 2，
# 撤下该卡不改变关键路径时长）。
# A4000 自 2026-11-15 起退出调度（用户收回自用）；GPU3 仍为用户预留。
# 确认阶段 dense/n1m 改跑两张 3080（每卡峰值 ~10.3GB < 20GB），
# 与网格阶段（A4000）硬件不同，报告需加脚注。
GPUS = [
    {"name": "gpu0_3080", "uuid": "GPU-d149ae20-68a4-2f3d-256f-e808281c7830", "micro": 2, "accum": 4},
    {"name": "gpu2_3080", "uuid": "GPU-c24d87e8-5b73-9732-0d10-69aee2a761a2", "micro": 2, "accum": 4},
]
GROUPS = [("dense", 0, 38_000_000), ("n1m", 1_000_000, 38_000_000), ("n4m", 4_000_000, 152_000_000)]
LRS = [0.0003, 0.0006]
GRID_SEEDS = [1, 2]
CONFIRM_SEEDS = [3, 4]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_state():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,power.limit",
        "--format=csv,noheader,nounits"
    ], text=True)
    rows = []
    for line in output.splitlines():
        if line.strip():
            parts = line.split(",")
            rows.append({"index": int(parts[0]), "uuid": parts[1].strip(), "name": parts[2],
                         "memory_total_mib": int(parts[3]), "memory_used_mib": int(parts[4]),
                         "memory_free_mib": int(parts[5]), "utilization": int(parts[6]),
                         "power_limit_w": float(parts[7])})
    return rows


def write_environment():
    payload = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "python": subprocess.check_output([PYTHON, "-c", "import sys; print(sys.version.split()[0])"], text=True).strip(),
        "torch": subprocess.check_output([PYTHON, "-c", "import torch; print(torch.__version__)"], text=True).strip(),
        "cuda": subprocess.check_output([PYTHON, "-c", "import torch; print(torch.version.cuda)"], text=True).strip(),
        "gpus": gpu_state(),
        "data_metadata_sha256": sha256(DATA / "metadata.json"),
        "qa_file_sha256": sha256(QA_FILE),
        "source_sha256": {name: sha256(ROOT / name) for name in [
            "pkm.py", "scale_model.py", "scale_train.py", "run_scale_probe.py", "summarize_scale_probe.py"
        ]},
        "git_commit": None,
        "value_memory_plan": {
            "dense": "no value table",
            "n1m": "1,000,000 x 512 FP32 values on CPU; FP32 AdamW states on CPU; selected rows copied to GPU",
            "n4m": "4,000,000 x 512 FP32 values on CPU; FP32 AdamW states on CPU; selected rows copied to GPU",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def grid_tasks():
    return [{"group": group, "n_values": n_values, "tokens": tokens, "lr": lr, "seed": seed}
            for group, n_values, tokens in GROUPS for lr in LRS for seed in GRID_SEEDS]


def confirmation_tasks():
    result = []
    for group, n_values, tokens in GROUPS:
        candidates = []
        for lr in LRS:
            losses = []
            for seed in GRID_SEEDS:
                path = OUT / f"{group}_lr{lr:g}_seed{seed}" / "train.jsonl"
                if not path.exists():
                    raise RuntimeError(f"missing grid run before confirmation: {path}")
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                evals = [row for row in rows if row["event"] == "eval"]
                tail = [row["val_loss"] for row in evals if row["step"] > max(r["step"] for r in evals) - 1000]
                losses.append(sum(tail) / len(tail))
            candidates.append((sum(losses) / len(losses), lr))
        best_lr = min(candidates)[1]
        for seed in CONFIRM_SEEDS:
            result.append({"group": group, "n_values": n_values, "tokens": tokens, "lr": best_lr, "seed": seed, "confirmation": True})
    return sorted(result, key=lambda row: row["tokens"], reverse=True)


def complete(path):
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return '"event": "final"' in text and (not QA_FILE or '"event": "fact_qa"' in text)


def run_task(task, gpu, progress, n4m_semaphore):
    name = f"{task['group']}_lr{task['lr']:g}_seed{task['seed']}"
    directory = OUT / name
    result = directory / "train.jsonl"
    if complete(result):
        return name, "skipped", 0
    directory.mkdir(parents=True, exist_ok=True)
    micro = gpu["micro"] if task["n_values"] < 4_000_000 or gpu["name"] != "gpu1_a4000" else 1
    accum = 8000 // (micro * 1000)
    command = [
        PYTHON, str(ROOT / "scale_train.py"), "--output", str(directory),
        "--data-dir", str(DATA), "--train-tokens", str(task["tokens"]),
        "--n-values", str(task["n_values"]), "--pkm-dv", "512",
        "--lr", str(task["lr"]), "--value-lr-mult", "5", "--seed", str(task["seed"]),
        "--micro-batch", str(micro), "--grad-accum", str(accum), "--seq-len", "1000",
        "--eval-batches", "8", "--qa-file", str(QA_FILE), "--log-every", "25", "--device", "cuda",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu.get("device", gpu["uuid"]),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    started = time.time()
    with (ROOT / f"scale_{name}.log").open("w", encoding="utf-8") as log:
        if task["n_values"] == 4_000_000:
            with n4m_semaphore:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        else:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - started
    payload = {"name": name, **task, "gpu": gpu["name"], "returncode": code,
               "elapsed_sec": elapsed, "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with progress.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    if code != 0:
        with (OUT / "failed_commands.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"command": command, "returncode": code}) + "\n")
    return name, "ok" if code == 0 else f"failed:{code}", elapsed


def run_batch(pending, progress, n4m_semaphore, gpus=GPUS):
    task_queue = queue.Queue()
    for task in pending:
        task_queue.put(task)
    results = []
    def worker(gpu):
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            results.append(run_task(task, gpu, progress, n4m_semaphore))
            task_queue.task_done()
    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def run_split(pending, progress, n4m_semaphore):
    c_tasks = [task for task in pending if task["group"] == "n4m"]
    other_tasks = [task for task in pending if task["group"] != "n4m"]
    results = []
    threads = [
        threading.Thread(target=lambda: results.extend(run_batch(c_tasks, progress, n4m_semaphore, GPUS)), daemon=True),
        threading.Thread(target=lambda: results.extend(run_batch(other_tasks, progress, n4m_semaphore, GPUS)), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def main():
    if not QA_FILE.exists():
        raise FileNotFoundError(f"QA file must be user-audited and frozen before training: {QA_FILE}")
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    if metadata["counts"]["train"] < 152_000_000:
        raise RuntimeError("training split is not frozen at 152M tokens")
    write_environment()
    progress = OUT / "progress.jsonl"
    n4m_semaphore = threading.Semaphore(2)
    for gpu in GPUS:
        if next(row for row in gpu_state() if row["uuid"] == gpu["uuid"])["memory_used_mib"] > 20480:
            raise RuntimeError(f"GPU {gpu['name']} is already over 20GB; refusing to start")
    results = run_split(grid_tasks(), progress, n4m_semaphore)
    results.extend(run_split(confirmation_tasks(), progress, n4m_semaphore))
    (OUT / "worker_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    subprocess.run([PYTHON, str(ROOT / "summarize_scale_probe.py")], cwd=ROOT, check=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
