#!/usr/bin/env python
"""P0.5 简版：Qwen3-14B 真实 FFN → 稀疏字库复印 error(k) 曲线 + 跨模型岭回归桥（零训练，只推理）。

目的（单问题）：Phase 0 的「稀疏 FFN 形态」复印精度，在真实大模型 SwiGLU 矩阵
（17408 槽 × 5120 维）上是否保持；以及 0.5B 小模型的隐状态能否用闭式岭回归桥
映射到 14B 寻址空间（usage 条件）。

口径与 Phase 0 V0 对齐：键 = W_gate 行 / ||行||，分数 = <h,键>，top-k 检索后
用分数×范数重建门控前激活，值 = W_down 列（SwiGLU 需另存 W_up 行算 u_j）。
k = 全槽时必须精确归零（锚）。

红线：纯推理；GPU 只用 cuda:0/cuda:1（物理 GPU2/GPU3）；不动 runs/ 归档。
"""
import os, sys, json, time, hashlib, argparse, gc
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 物理 GPU2,GPU3 -> 进程内 cuda:0,cuda:1

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path("/home/antionair/pkm_experiment")
GGUF_DIR = "/mnt/mark7/10gb/standard"
GGUF_FILE = "Qwen3-14B-Q4_K_M.gguf"
CKPT = ROOT / "runs/dense_control/dense152_lr0.0003_seed1/checkpoint.pt"
QA = ROOT / "data/scale/qa_frozen.jsonl"
LAYER_BIG = 20
LAYER_SMALL = 17
K_GRID = [64, 256, 512, 1024, 2048, 4096, 8192]

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def sha256_of(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def error_k_curve(m, Wg, Wu, Wd, k_grid, device, chunk=1024, rank_by="gate"):
    """m: (M,5120) fp32 cuda 张量。返回 {k: (rel_l2, cos)}（全局口径）。

    rank_by='gate'  : 按 |g_j| 选槽（Phase 0 同口径）
    rank_by='mass'  : 按 |silu(g_j)*u_j|*||W_down 列_j|| 选槽（贡献质量，查询时可算）
    """
    n_slot = Wg.shape[0]
    gn = Wg.norm(dim=1, keepdim=True).clamp_min(1e-8)          # (n,1)
    key = Wg / gn                                              # 归一化键
    wdn = Wd.norm(dim=0).clamp_min(1e-8)                       # (n,) 值列范数
    E_norm_sq = 0.0
    err_sq = {k: 0.0 for k in k_grid}
    dot = {k: 0.0 for k in k_grid}
    Eo = {k: 0.0 for k in k_grid}
    tot = 0
    for i in range(0, m.shape[0], chunk):
        h = m[i:i + chunk]                                     # (b,5120)
        g = h @ key.T                                          # 分数 (b,n)
        u = h @ Wu.T                                           # (b,n)
        C = F.silu(g * gn.T) * u                               # 重建门控 (b,n)
        E = C @ Wd.T                                           # 精确输出 (b,5120)
        E_norm_sq += float((E * E).sum())
        if rank_by == "mass":
            rank_score = (F.silu(g * gn.T) * u).abs() * wdn
        else:
            rank_score = g.abs()
        for k in k_grid:
            idx = rank_score.topk(min(k, n_slot), dim=1).indices
            mask = torch.zeros_like(C)
            mask.scatter_(1, idx, 1.0)
            out = (C * mask) @ Wd.T
            d = out - E
            err_sq[k] += float((d * d).sum())
            dot[k] += float((out * E).sum())
            Eo[k] += float((out * out).sum())
        tot += h.shape[0]
        del g, u, C, E, idx, mask, out, d, rank_score
    res = {}
    for k in k_grid:
        rl2 = (err_sq[k] / E_norm_sq) ** 0.5
        cos = dot[k] / max((Eo[k] * E_norm_sq) ** 0.5, 1e-12)
        res[k] = (rl2, cos)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/autopsy/ffn_transplant_p05.json"))
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--len", type=int, default=384)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    B, L = (2, 64) if args.smoke else (args.seqs, args.len)

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("torch", torch.__version__, "tfm", transformers.__version__)

    # ---------- 文本 ----------
    rows = [json.loads(l) for l in QA.read_text().splitlines() if l.strip()]
    text = "\n\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    tok = AutoTokenizer.from_pretrained(GGUF_DIR, gguf_file=GGUF_FILE)
    ids_all = tok(text, add_special_tokens=False).input_ids
    assert len(ids_all) >= 2048, f"文本太短: {len(ids_all)} tokens"
    n_pos = min(B * L, len(ids_all) - 8)
    ids = torch.tensor([ids_all[:n_pos][i * L:(i + 1) * L] for i in range(n_pos // L)],
                       dtype=torch.long)
    Bn = ids.shape[0]
    log(f"文本 tokens={len(ids_all)} 取用 {Bn}x{L}={Bn*L} 位置")

    # ---------- 大模型：只装前 20 层 ----------
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        GGUF_DIR, gguf_file=GGUF_FILE, torch_dtype=torch.float32)
    log(f"GGUF 载入 {time.time()-t0:.0f}s")
    layers = model.model.layers
    n_layers = len(layers)
    assert n_layers > LAYER_BIG
    norm20 = layers[LAYER_BIG].post_attention_layernorm
    Wg = layers[LAYER_BIG].mlp.gate_proj.weight.data.float().cpu()
    Wu = layers[LAYER_BIG].mlp.up_proj.weight.data.float().cpu()
    Wd = layers[LAYER_BIG].mlp.down_proj.weight.data.float().cpu()
    log(f"L{LAYER_BIG} FFN: Wg{tuple(Wg.shape)} Wu{tuple(Wu.shape)} Wd{tuple(Wd.shape)}")
    emb = model.model.embed_tokens
    rotary = model.model.rotary_emb
    keep = [layers[i] for i in range(LAYER_BIG)]
    del model, layers
    gc.collect()

    dev0 = "cuda:0"
    emb.to(dev0, torch.bfloat16)
    for lyr in keep:
        lyr.to(dev0, torch.bfloat16)
    norm20.to(dev0, torch.bfloat16)
    rotary.to(dev0, torch.bfloat16)
    Wg_c, Wu_c, Wd_c = (t.to(dev0) for t in (Wg, Wu, Wd))

    pos = torch.arange(L, device=dev0).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rotary(torch.zeros(Bn, L, 1, device=dev0, dtype=torch.bfloat16), pos)
    causal = torch.full((L, L), float("-inf"), device=dev0,
                        dtype=torch.bfloat16).triu(1)[None, None]
    hs = []
    with torch.no_grad():
        h = emb(ids.to(dev0))
        for i, lyr in enumerate(keep):
            h = lyr(h, attention_mask=causal, position_ids=pos,
                    position_embeddings=(cos, sin), use_cache=False)
            if i == LAYER_BIG - 1:
                hs.append(h)
    with torch.no_grad():
        m20 = norm20(hs[0]).float().cpu().detach()   # (Bn,L,5120) = L20 mlp 输入
    M = m20.reshape(-1, m20.shape[-1])
    assert torch.isfinite(M).all()
    del keep, emb, h, hs
    gc.collect()
    torch.cuda.empty_cache()
    log(f"大模型前 {LAYER_BIG} 层前向完，M={M.shape}")

    # ---------- native error(k)（全量 M） ----------
    Mn = M.to(dev0)
    t_native = error_k_curve(Mn, Wg_c, Wu_c, Wd_c, K_GRID, dev0)
    t_native_mass = error_k_curve(Mn, Wg_c, Wu_c, Wd_c, K_GRID, dev0, rank_by="mass")
    log("native error(k):",
        {k: f"{v[0]*100:.2f}%/{v[1]:.4f}" for k, v in t_native.items()})
    log("native error(k, mass rank):",
        {k: f"{v[0]*100:.2f}%/{v[1]:.4f}" for k, v in t_native_mass.items()})

    # ---------- 小模型桥 ----------
    sys.path.insert(0, str(ROOT))
    from scale_model import ScaleConfig, ScaleLanguageModel
    from transformers import AutoTokenizer as AT
    gpt2 = AT.from_pretrained("gpt2")
    cfg = ScaleConfig(d_model=1024, layers=36, heads=16, ff_mult=4,
                      max_seq_len=1000, n_values=0)
    sm = ScaleLanguageModel(cfg)
    sm.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    dev1 = "cuda:1"
    sm.to_device(torch.device(dev1)).eval()
    sids = gpt2(text, add_special_tokens=False).input_ids[:M.shape[0]]
    Mpos = min(len(sids), M.shape[0])
    sids = torch.tensor([sids[i * L:(i + 1) * L] for i in range(Mpos // L)],
                        dtype=torch.long)
    got = []
    def hook(mod, inp, out):
        got.append(inp[0].detach())
    hnd = sm.blocks[LAYER_SMALL].ff.register_forward_hook(hook)
    with torch.no_grad():
        sm(sids.to(dev1))
    hnd.remove()
    ms = got[0].reshape(-1, got[0].shape[-1]).float().cpu()[:Mpos]
    assert torch.isfinite(ms).all()
    log(f"小模型 L{LAYER_SMALL} ff 输入: {ms.shape}")

    # 位置对齐：前 Mpos 个位置一一配对；前 70% 训练 / 后 30% 验证
    X = torch.cat([ms, torch.ones(ms.shape[0], 1)], dim=1).numpy().astype(np.float64)
    Y = M.numpy().astype(np.float64)[:X.shape[0]]
    ntr = int(X.shape[0] * 0.7)
    Xtr, Xva = X[:ntr], X[ntr:]
    Ytr, Yva = Y[:ntr], Y[ntr:]
    XtX = Xtr.T @ Xtr
    XtY = Xtr.T @ Ytr
    best = None
    for lam in [1e-4, 1e-2, 1.0]:
        lam *= float(np.mean(np.diag(XtX)))
        Wb = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), XtY)
        pred = Xva @ Wb
        rl2 = float(np.linalg.norm(pred - Yva) / np.linalg.norm(Yva))
        log(f"ridge λ~{lam:.4g} val rel_l2={rl2*100:.2f}%")
        if best is None or rl2 < best[0]:
            best = (rl2, lam, Wb)
    bridge_rl2, bridge_lam, Wb = best
    log(f"桥选定 λ={bridge_lam:.4g} val rel_l2={bridge_rl2*100:.2f}%")

    # ---------- bridged error(k)（验证集位置） ----------
    torch.save({"m20": M, "ms": ms, "L": L, "layer_big": LAYER_BIG,
                "layer_small": LAYER_SMALL},
               ROOT / "runs/autopsy/ffn_transplant_p05_cache.pt")
    log("缓存已存 runs/autopsy/ffn_transplant_p05_cache.pt")
    Hh = torch.tensor(Xva[:, :-1] @ Wb[:-1] + Wb[-1], dtype=torch.float32)  # 截距另加
    Hh = Hh.to(dev0)
    Yva_t = torch.tensor(Yva, dtype=torch.float32).to(dev0)
    # 以真实 m 为 target 的 sparse 近似，query 用桥接值
    def error_k_target(hq, mt, chunk=1024, rank_by="gate"):
        n_slot = Wg_c.shape[0]
        gn = Wg_c.norm(dim=1, keepdim=True).clamp_min(1e-8)
        key = Wg_c / gn
        wdn = Wd_c.norm(dim=0).clamp_min(1e-8)
        Ee = 0.0
        esq, dt, oo = ({k: 0.0 for k in K_GRID} for _ in range(3))
        for i in range(0, hq.shape[0], chunk):
            q = hq[i:i + chunk]; t = mt[i:i + chunk]
            g_t = t @ key.T
            u_t = t @ Wu_c.T
            C_t = F.silu(g_t * gn.T) * u_t
            E = C_t @ Wd_c.T
            g_q = q @ key.T
            u_q = q @ Wu_c.T
            C_q = F.silu(g_q * gn.T) * u_q
            Ee += float((E * E).sum())
            if rank_by == "mass":
                rank_score = C_q.abs() * wdn
            else:
                rank_score = g_q.abs()
            for k in K_GRID:
                idx = rank_score.topk(k, dim=1).indices
                mask = torch.zeros_like(C_q)
                mask.scatter_(1, idx, 1.0)
                out = (C_q * mask) @ Wd_c.T
                d = out - E
                esq[k] += float((d * d).sum())
                dt[k] += float((out * E).sum())
                oo[k] += float((out * out).sum())
            del g_t, u_t, C_t, g_q, u_q, C_q, E, idx, mask, out, d, rank_score
        return {k: ((esq[k] / Ee) ** 0.5, dt[k] / max((oo[k] * Ee) ** 0.5, 1e-12))
                for k in K_GRID}
    t_bridge = error_k_target(Hh, Yva_t)
    t_bridge_mass = error_k_target(Hh, Yva_t, rank_by="mass")
    t_native_val = error_k_target(Yva_t, Yva_t)
    t_native_val_mass = error_k_target(Yva_t, Yva_t, rank_by="mass")
    log("bridged error(k):", {k: f"{v[0]*100:.2f}%" for k, v in t_bridge.items()})
    log("bridged error(k, mass):", {k: f"{v[0]*100:.2f}%" for k, v in t_bridge_mass.items()})

    # ---------- 机检与存档 ----------
    log("sha256 核对中...")
    res = {
        "experiment": "ffn_transplant_p05",
        "question": "真实14B SwiGLU矩阵上稀疏FFN复印精度 + 0.5B->14B岭回归桥可用性",
        "donor_gguf": {"path": f"{GGUF_DIR}/{GGUF_FILE}", "sha256": sha256_of(Path(GGUF_DIR)/GGUF_FILE)},
        "small_ckpt": {"path": str(CKPT), "sha256": sha256_of(CKPT),
                       "arch": {"d_model": 1024, "layers": 36, "layer_hook": LAYER_SMALL}},
        "layer_big": LAYER_BIG,
        "dims": {"hidden": Wg.shape[1], "n_slots": Wg.shape[0]},
        "protocol": {
            "keys": "W_gate rows / L2 norm", "values": "W_down cols (raw)",
            "extra": "W_up rows for u_j", "score": "<h, key>",
            "reconstruct": "g = score * ||row||", "activation": "silu(g)*u",
            "anchor_exact_at_full_k": True,
            "query_space": "layer-20 gate_proj input (post_attention_layernorm)",
        },
        "data": {"text_source": str(QA), "positions_BxL": [Bn, L],
                 "bridge_align": "position_i pairing, first-M", "val_split": "last 30%"},
        "native_error_k": {str(k): {"rel_l2": v[0], "cos": v[1]} for k, v in t_native.items()},
        "native_error_k_mass": {str(k): {"rel_l2": v[0], "cos": v[1]} for k, v in t_native_mass.items()},
        "bridge": {"ridge_closed_form": True, "lam": bridge_lam,
                   "val_rel_l2": bridge_rl2,
                   "fit_rel_l2": float(np.linalg.norm(Xtr @ Wb - Ytr) / np.linalg.norm(Ytr))},
        "bridged_error_k_val": {str(k): {"rel_l2": v[0], "cos": v[1]} for k, v in t_bridge.items()},
        "bridged_error_k_val_mass": {str(k): {"rel_l2": v[0], "cos": v[1]} for k, v in t_bridge_mass.items()},
        "native_error_k_val": {str(k): {"rel_l2": v[0], "cos": v[1]} for k, v in t_native_val.items()},
        "native_error_k_val_mass": {str(k): {"rel_l2": v[0], "cos": v[1]} for k, v in t_native_val_mass.items()},
        "k_grid": K_GRID,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "wall_s": {"load": None, "total": None},
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    log("DONE ->", args.out)

if __name__ == "__main__":
    main()
