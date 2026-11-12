import subprocess
import time

REQUIRED_QUIET_SAMPLES = 5
quiet = 0
sample = 0
while quiet < REQUIRED_QUIET_SAMPLES:
    sample += 1
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    utilization = [int(line.split(",")[1]) for line in output.splitlines() if line.strip()]
    quiet = quiet + 1 if all(value == 0 for value in utilization) else 0
    print(f"sample={sample} utilization={utilization} quiet={quiet}", flush=True)
    if quiet < REQUIRED_QUIET_SAMPLES:
        time.sleep(30)

print("All GPUs quiet for five consecutive samples", flush=True)
subprocess.run(["python3", "/home/antionair/pkm_experiment/run_ffn_lr_probe.py"], check=True)
