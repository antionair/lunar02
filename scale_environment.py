import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import torch
from scale_model import ScaleConfig, ScaleLanguageModel, parameter_counts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/scale_probe"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu_text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,power.limit",
        "--format=csv,noheader,nounits"
    ], text=True)
    gpus = []
    for line in gpu_text.splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        gpus.append({"index": int(parts[0]), "uuid": parts[1].strip(), "name": parts[2],
                     "memory_total_mib": int(parts[3]), "memory_used_mib": int(parts[4]),
                     "memory_free_mib": int(parts[5]), "utilization": int(parts[6]),
                     "power_limit_w": float(parts[7])})
    model = ScaleLanguageModel(ScaleConfig())
    counts = parameter_counts(model)
    del model
    data_meta = ROOT / "data/scale/metadata.json"
    payload = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": gpus,
        "gpu_topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        "disk": dict(shutil.disk_usage(ROOT)._asdict()),
        "model": {"config": ScaleConfig().__dict__, **counts},
        "value_memory_plan": {
            "n1m": {"parameters": 1_000_000 * 512, "fp32_parameter_gb": 2.0,
                    "fp32_gradient_gb": 2.0, "fp32_adam_states_gb": 4.0, "location": "CPU"},
            "n4m": {"parameters": 4_000_000 * 512, "fp32_parameter_gb": 8.0,
                    "fp32_gradient_gb": 8.0, "fp32_adam_states_gb": 16.0, "location": "CPU"},
        },
        "data_metadata": json.loads(data_meta.read_text(encoding="utf-8")) if data_meta.exists() else None,
        "data_metadata_sha256": sha256(data_meta) if data_meta.exists() else None,
        "source_sha256": {name: sha256(ROOT / name) for name in [
            "pkm.py", "scale_model.py", "scale_train.py", "prepare_scale_data.py",
            "run_scale_smoke.py", "run_scale_probe.py", "summarize_scale_probe.py"
        ]},
        "git_commit": None,
    }
    (OUT / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
