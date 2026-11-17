"""探针 #4 汇总：dense@152M 对照 + s 值裁决 + 任务书机检与定性交付。

解读规则（任务书先写死）：
  s = (313.5 − PPL_dense152_best) / 254.8
  s ≤ 0.15      → 绿升级：增益字库主导
  0.15 < s < 0.85 → 混合：维持「联合胜利」，报 s 区间
  s ≥ 0.85      → 数据主导：#3 降级为「数据 scaling 胜利」
PPL_dense152_best = best-lr 组 4 种子的 held-out PPL 均值（best-lr 按
与 #3 相同口径：各 run 末 1000 步 eval val_loss 均值的 4 种子均值最小者）。
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/dense_control"
REF = ROOT / "runs/scale_probe"  # 探针 #3 结果
G4_GAIN = 254.83077938369735     # #3 verdict.json 的 g4
DENSE38_PPL = 313.4535584179999  # #3 summary.csv 的 dense mean_ppl
IGNORE_DIFF_KEYS = {"output", "lr", "seed", "train_tokens"}


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def tail_mean_val_loss(events, window=1000):
    evals = [row for row in events if row["event"] == "eval"]
    last_step = max(row["step"] for row in evals)
    tail = [row["val_loss"] for row in evals if row["step"] > last_step - window]
    return sum(tail) / len(tail)


def collect():
    grid = []
    for directory in sorted(OUT.glob("dense152_lr*_seed*")):
        events = read_events(directory / "train.jsonl")
        final = next(row for row in reversed(events) if row["event"] == "final")
        qa = next(row for row in reversed(events) if row["event"] == "fact_qa")
        start = next(row for row in events if row["event"] == "start")
        args = json.loads((directory / "args.json").read_text(encoding="utf-8"))
        grid.append({
            "run": directory.name, "lr": args["lr"], "seed": args["seed"],
            "tail_val_loss": tail_mean_val_loss(events),
            "val_loss": final["val_loss"], "val_ppl": final["val_ppl"],
            "domain_ppl": final["domain_ppl"],
            "fact_exact": qa["exact_match"], "fact_f1": qa["token_f1"],
            "elapsed_sec": final["elapsed_sec"], "tokens_per_sec": final["tokens_per_sec"],
            "peak_gpu_memory_bytes": final["peak_gpu_memory_bytes"],
            "train_tokens": start["train_tokens"], "steps": start["steps"],
        })
    return grid


def mean_std(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, std


def machine_check(grid):
    checks = {}
    ref_args = json.loads((REF / "dense_lr0.0003_seed1" / "args.json").read_text(encoding="utf-8"))
    diffs = {}
    for row in grid:
        args = json.loads((OUT / row["run"] / "args.json").read_text(encoding="utf-8"))
        for key in ref_args:
            if key in IGNORE_DIFF_KEYS:
                continue
            if ref_args[key] != args.get(key):
                diffs.setdefault(key, set()).add(str(args.get(key)))
    checks["config_diff_except_tokens"] = {k: sorted(v) for k, v in diffs.items()}
    checks["config_diff_pass"] = not diffs
    tokens = [row["train_tokens"] for row in grid]
    checks["train_tokens_all"] = tokens
    checks["train_tokens_within_1pct"] = all(abs(t - 152_000_000) / 152_000_000 <= 0.01 for t in tokens)
    checks["overlap_rate_zero_evidence"] = (
        "data/scale/metadata.json (split construction, overlap rate 0); "
        "bin sha256 recorded in runs/dense_control/environment.json and SCALE_SMOKE_REPORT.md"
    )
    checks["qa_provenance"] = (
        "data/scale/qa_frozen.jsonl: 176 questions, user-frozen 2026-11-13 "
        "(rejects recorded in qa_review_rejects.json), sha256 ccb88224c052cd662a39b43b62720bc7aa418b99a26c2e68f5b6bc083ce7dbb3"
    )
    return checks


def throughput_notes():
    rows = []
    summary = {row["group"]: row for row in csv.DictReader((REF / "summary.csv").open())}
    dense152 = collect()
    for group in ["dense", "n1m", "n4m"]:
        row = summary[group]
        tokens = 38_000_000 if group != "n4m" else 152_000_000
        rows.append({
            "group": f"{group}@{tokens/1e6:.0f}M", "tokens": tokens,
            "mean_tokens_per_sec": float(row["mean_tokens_per_sec"]),
            "active_params": int(row["active_parameters"]),
        })
    t_dense152, t_dense152_std = mean_std([r["elapsed_sec"] for r in dense152])
    rows.append({
        "group": "dense@152M (this probe)", "tokens": 152_000_000,
        "mean_tokens_per_sec": mean_std([r["tokens_per_sec"] for r in dense152])[0],
        "mean_wall_hours": t_dense152 / 3600, "active_params": 505_953_280,
    })
    note = (
        "吞吐倒挂解释：dense@38M 在 A4000 上 micro=1/accum=8 约 5.2k tok/s（1.5-2.1h），"
        "N=4M@152M 在 3080 上约 4.5-4.8k tok/s（约 9h）。dense 无字库层、无 CPU 稀疏优化器，"
        "每步纯 GPU 计算且序列短，因此单位 token 最快；N=1M/N=4M 每步有值表命中行拷贝与 "
        "SparseAdamW（CPU）开销，N=4M 额外因 4M 行键表打分（sqrt(N)=2048/子表）比 1M(1024) 贵，"
        "但值表行拷贝只含命中行，故 4M 仅比 1M 慢约 10-15%。"
        "dense@152M 本探针在 3080 上实测见 mean_tokens_per_sec。"
    )
    return rows, note


def usage_entropy_curves():
    curves = {}
    for group in ["n4m"]:
        for directory in sorted(REF.glob(f"{group}_lr0.0003_seed*")):
            events = read_events(directory / "train.jsonl")
            rows = [row for row in events if row["event"] == "eval"]
            curves[directory.name] = [
                {"step": r["step"], "cumulative_cell_usage": r["cumulative_cell_usage"],
                 "batch_access_entropy_nats": r["batch_access_entropy_nats"]}
                for r in rows
            ]
    with (OUT / "n4m_usage_entropy_curves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "step", "cumulative_cell_usage", "batch_access_entropy_nats"])
        writer.writeheader()
        for run, rows in curves.items():
            for row in rows:
                writer.writerow({"run": run, **row})
    return {
        "curves_file": "n4m_usage_entropy_curves.csv",
        "limitation": ("#3 未记录逐格命中计数，无法事后生成真实格使用分布直方图；"
                       "本交付为 4 个 n4m run 的累计使用率/访问熵随步数曲线。"
                       "建议下一轮在 scale_train.py 中增加周期性逐格命中直方图日志。"),
    }


def main():
    grid = collect()
    per_lr = {}
    for row in grid:
        per_lr.setdefault(row["lr"], []).append(row)
    lr_stats = {}
    for lr, rows in per_lr.items():
        tail_mean, _ = mean_std([r["tail_val_loss"] for r in rows])
        lr_stats[lr] = tail_mean
    best_lr = min(lr_stats, key=lr_stats.get)
    best = per_lr[best_lr]
    ppl_mean, ppl_std = mean_std([r["val_ppl"] for r in best])
    dom_mean, dom_std = mean_std([r["domain_ppl"] for r in best])
    fact_f1_mean, _ = mean_std([r["fact_f1"] for r in best])
    s = (DENSE38_PPL - ppl_mean) / G4_GAIN
    if s <= 0.15:
        conclusion = "green_upgrade"
        conclusion_zh = "绿升级：增益字库主导，#3 改口「字库单挑胜利」"
    elif s < 0.85:
        conclusion = "mixed"
        conclusion_zh = "混合：维持「联合胜利」口径，报告 s 区间"
    else:
        conclusion = "data_dominant"
        conclusion_zh = "数据主导：#3 降级为「数据 scaling 胜利」，字库贡献另议"

    checks = machine_check(grid)
    throughput, throughput_note = throughput_notes()
    usage_report = usage_entropy_curves()

    comparison = [
        {"group": "dense@38M", "val_ppl": 313.45, "domain_ppl": 342.29, "tokens": "38M"},
        {"group": "dense@152M", "val_ppl": round(ppl_mean, 2), "domain_ppl": round(dom_mean, 2), "tokens": "152M"},
        {"group": "n1m@38M", "val_ppl": 311.09, "domain_ppl": 336.20, "tokens": "38M"},
        {"group": "n4m@152M", "val_ppl": 56.26, "domain_ppl": 67.42, "tokens": "152M"},
    ]
    report = {
        "s": s, "conclusion": conclusion, "conclusion_zh": conclusion_zh,
        "best_lr": best_lr, "best_lr_tail_val_loss": lr_stats[best_lr],
        "dense152_mean_ppl": ppl_mean, "dense152_std_ppl": ppl_std,
        "dense152_mean_domain_ppl": dom_mean, "dense152_std_domain_ppl": dom_std,
        "dense152_mean_fact_f1": fact_f1_mean,
        "lr_stats_tail_val_loss": lr_stats,
        "machine_checks": checks,
        "throughput_table": throughput,
        "throughput_note_zh": throughput_note,
        "cell_usage_report": usage_report,
        "comparison": comparison,
        "raw_grid": grid,
    }
    (OUT / "dense_control_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "val_ppl", "domain_ppl", "tokens"])
        writer.writeheader()
        writer.writerows(comparison)
    print(json.dumps({k: v for k, v in report.items() if k != "raw_grid"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
