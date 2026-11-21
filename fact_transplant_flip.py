"""事实槽移植·行为翻转演示 v2（零训练）。

臂：residual(γ) / prehead_rel(α·||h||) / logitbias(γ·E·δ, 上界诊断) × topk。
链路同 v1：14B 选题 → 0.5B 原生 layer18 hidden 作键、答案 token embedding 作值 → 注入复测。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F

ROOT = Path("/home/antionair/pkm_experiment")
RUN = ROOT / "runs/dense_control/dense152_lr0.0003_seed1"
QA = ROOT / "data/scale/qa_frozen.jsonl"
DONOR = "/mnt/mark7/10gb/standard/Qwen3-14B-Q4_K_M.gguf"
LLAMA = str(Path.home() / "llama.cpp/build/bin/llama-server")
PORT = 8199


def norm_answer(t):
    import re, unicodedata
    t = unicodedata.normalize("NFKC", t).lower()
    return " ".join(re.findall(r"[a-z0-9]+", t))


def score_answer(pred, row):
    refs = row.get("accepted_aliases") or [row["answer"]]
    np_ = norm_answer(pred)
    return max(float(np_ == norm_answer(r)) for r in refs)


def start_donor():
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
        if r.ok:
            print("复用已运行 donor", flush=True)

            class _P:
                def terminate(self):
                    pass
            return _P()
    except Exception:
        pass
    proc = subprocess.Popen([
        LLAMA, "-m", DONOR, "--port", str(PORT), "--host", "127.0.0.1",
        "-ngl", "99", "-c", "4096", "--embedding",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(200):
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
            if r.ok:
                return proc
        except Exception:
            time.sleep(2)
    proc.terminate()
    raise RuntimeError("donor server 未起")


def donor_complete(prompt, n=24):
    r = requests.post(f"http://127.0.0.1:{PORT}/completion", json={
        "prompt": prompt, "n_predict": n, "temperature": 0.0, "stop": ["\n"]},
        timeout=180)
    return r.json()["content"]


def shots_block(rows):
    return "\n\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/autopsy/fact_transplant_flip2.json"))
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(ROOT))
    from scale_model import ScaleConfig, ScaleLanguageModel
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in QA.read_text().splitlines()]
    shots, test = rows[:4], rows[4:]
    prefix = shots_block(shots) + "\n\n"

    tok = AutoTokenizer.from_pretrained("gpt2")
    device = torch.device("cuda")
    cfg = ScaleConfig(d_model=1024, layers=36, heads=16, ff_mult=4, max_seq_len=1000, n_values=0)
    model = ScaleLanguageModel(cfg)
    model.load_state_dict(torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=True))
    model.to_device(device).eval()

    @torch.no_grad()
    def small_answer(question):
        prompt = prefix + f"Question: {question}\nAnswer:"
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        out = model.generate(ids, max_new_tokens=16)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip()

    print("== 筛题 ==", flush=True)
    donor = start_donor()
    small_scores, donor_scores = [], []
    for r in test:
        p = prefix + f"Question: {r['question']}\nAnswer:"
        d_ans = donor_complete(p).split("\n")[0].strip()
        donor_scores.append(score_answer(d_ans, r))
        small_scores.append(score_answer(small_answer(r["question"]), r))
    donor.terminate()

    target_idx = [i for i in range(len(test)) if donor_scores[i] == 1.0 and small_scores[i] == 0.0]
    control_correct = [i for i in range(len(test)) if small_scores[i] == 1.0]
    print(f"目标集 {len(target_idx)} 题，本对 {len(control_correct)} 题", flush=True)

    print("== 建槽 ==", flush=True)
    keys, vals = [], []
    captured = {}
    block18 = model.blocks[17]
    orig_fwd = block18.forward

    def cap(x):
        captured["h"] = x.detach()
        return orig_fwd(x)
    block18.forward = cap
    emb = model.tokens.weight.data
    for i in target_idx:
        r = test[i]
        prompt = prefix + f"Question: {r['question']}\nAnswer:"
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            model.hidden(ids[:, -1000:])
        h = captured["h"][0, -1].float().cpu()
        keys.append(F.normalize(h, dim=0))
        ans_id = tok(r["answer"].split()[0], add_special_tokens=False)["input_ids"][0]
        vals.append(emb[ans_id].float().cpu())
    block18.forward = orig_fwd
    K = torch.stack(keys)
    V = torch.stack(vals)
    print(f"槽数 {K.shape[0]}", flush=True)

    slot_owner = {ti: pos for pos, ti in enumerate(target_idx)}

    def retrieve(hidden_last):
        hq = F.normalize(hidden_last.float(), dim=0)
        sc = hq.cpu() @ K.T
        tk = sc.topk(min(CUR_TOPK, K.shape[0])).indices
        w = sc[tk].softmax(0)
        return (V[tk] * w.unsqueeze(1)).sum(0).to(device), tk

    @torch.no_grad()
    def forward_with_table(ids, delta_last, mode, scale):
        """逐块前向，在第 17 块边界注入字库输出（与键同空间）。"""
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device)
        x = model.tokens(ids) + model.positions(pos)[None]
        for i, blk in enumerate(model.blocks):
            if i == LAYER_IDX and delta_last is not None and mode == "residual":
                x[:, -1] += scale * delta_last
            x = blk(x)
            if i == LAYER_IDX and delta_last is not None and mode == "prehead_rel":
                hn_add = scale * x[0, -1].norm() * delta_last
                x = x.clone()
                x[0, -1] += hn_add
        return model.lm_head(model.norm(x))[:, -1].float()

    LAYER_IDX = 17

    @torch.no_grad()
    def layer17_in(ids):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device)
        x = model.tokens(ids) + model.positions(pos)[None]
        for blk in model.blocks[:LAYER_IDX]:
            x = blk(x)
        return x[0, -1]

    def run_arm(mode, scale, topk, idx_list, e1=None):
        global CUR_TOPK
        CUR_TOPK = topk
        hits = 0
        for i in idx_list:
            r = test[i]
            prompt = prefix + f"Question: {r['question']}\nAnswer:"
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                h17 = layer17_in(ids[:, -1000:])
                delta, tk1 = retrieve(h17)
                if e1 is not None and len(tk1) > 0:
                    e1.append(int(tk1[0] == slot_owner[i]))
                if mode == "logitbias":
                    logits = forward_with_table(ids[:, -1000:], delta, "none", 0)
                    logits = logits + scale * (model.lm_head.weight.float() @ delta)
                else:
                    logits = forward_with_table(ids[:, -1000:], delta, mode, scale)
            pred = tok.decode([int(logits.argmax())]).strip()
            first_refs = {norm_answer((a or r["answer"]).split()[0]) for a in (r.get("accepted_aliases") or [r["answer"]])}
            if norm_answer(pred) in first_refs:
                hits += 1
        return hits

    print("== 臂评测 ==", flush=True)
    arms = []
    for gamma in [0.5, 1.0, 2.0]:
        for topk in [1, 4]:
            arms.append((f"g{gamma}_k{topk}", "residual", gamma, topk))
    for alpha in [0.05, 0.15, 0.5]:
        for topk in [1, 4]:
            arms.append((f"a{alpha}_k{topk}", "prehead_rel", alpha, topk))
    for gamma in [6.0, 20.0, 60.0]:
        arms.append((f"logitbias_g{gamma}_k1", "logitbias", gamma, 1))
    for alpha in [0.5, 1.5]:
        arms.append((f"a{alpha}_k1", "prehead_rel", alpha, 1))
        arms.append((f"a{alpha}_k4", "prehead_rel", alpha, 4))

    results = {}
    for name, mode, scale, topk in arms:
        e1 = []
        flips = run_arm(mode, scale, topk, target_idx, e1=e1)
        reg = run_arm(mode, scale, topk, control_correct)
        results[name] = {"mode": mode, "scale": scale, "topk": topk,
                         "flips": flips, "target_n": len(target_idx),
                         "regressions": reg, "control_n": len(control_correct),
                         "e1_top1": round(sum(e1) / max(1, len(e1)), 4)}
        print(name, results[name], flush=True)

    report = {
        "n_test": len(test), "n_target": len(target_idx),
        "donor_acc": sum(donor_scores) / len(donor_scores),
        "small_acc_pre": sum(small_scores) / len(small_scores),
        "target_ids": [test[i]["id"] for i in target_idx],
        "grid": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
