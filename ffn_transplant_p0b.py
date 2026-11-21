"""FFN→字库移植机制验证 Phase 0b（零训练，只读 checkpoint，内存安全重写）

修正原 ffn_transplant_p0.py 的四个问题：
 1. 分块累加，任何 k 下峰值显存 < ~1.2GB（原版 k=4096 要物化 1024x41M = 168GB，两次 OOM）。
 2. 计入偏置：z = mW1^T + b1，pred = sum_j g_j v_j + b2；k=F 时 relL2 精确为 0（地板自检）。
 3. V2 改成真乘积键（PQ）：两个半空间各 64 码本 k-means，两阶段 top-k；
    原版 K1t/K2t 只取块内第 0 行（扔掉 63/64 键），且 topk(512) 对长度 64 的维会报错。
 4. V3 修正值表：余弦检索集 + 真 GELU 权重 + 原始 W2（原版把 ||w1|| 吸进值会重复计量）。
另加：address recall（检索集能找回多少"真正重要"的神经元）；V1 的单一全局标量 alpha 校准。

读数：relative L2 / cosine 随 k 的曲线。V0=oracle 上界，V3=定址损失，V1=定址+加权税。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/antionair/pkm_experiment")
RUN = ROOT / "runs/dense_control/dense152_lr0.0003_seed1"
OUT = ROOT / "runs/autopsy/ffn_transplant_p0b.json"
LAYER = 17          # 第 18 层（字库注入位）
K_LIST = [8, 64, 512, 1024, 4096]
V2_SIDE = 64
V2_KS = [8, 16, 32, 64]
N_TOKENS = 10000
CHUNK = 64
TAU = 8.0
TARGET_TOP = 32


def main():
    device = torch.device("cuda")
    sys.path.insert(0, str(ROOT))
    from scale_model import ScaleConfig, ScaleLanguageModel

    args = json.loads((RUN / "args.json").read_text())
    cfg = ScaleConfig(d_model=args["d_model"], layers=args["layers"], heads=args["heads"],
                      ff_mult=args["ff_mult"], max_seq_len=args["seq_len"], n_values=0)
    model = ScaleLanguageModel(cfg)
    model.load_state_dict(torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=True))
    model.to_device(device).eval()

    block = model.blocks[LAYER]
    stash = {}
    block.ff.register_forward_hook(
        lambda mod, inp, out: stash.update({"m": inp[0].detach(), "y": out.detach()}))

    data = np.memmap(ROOT / "data/scale/eval_in_domain_4m.bin", dtype=np.uint32, mode="r")
    ms, ys = [], []
    with torch.no_grad():
        for b in range(N_TOKENS // 1000):
            chunk = torch.from_numpy(data[b * 1001:(b + 1) * 1001].astype(np.int64)).view(1, 1001)
            model(chunk[:, :-1].to(device))
            ms.append(stash["m"][0].float().cpu())
            ys.append(stash["y"][0].float().cpu())
    m = torch.cat(ms).to(device)
    y = torch.cat(ys).to(device)
    T = m.shape[0]

    sd = model.state_dict()
    W1 = sd[f"blocks.{LAYER}.ff.0.weight"].float()
    b1 = sd[f"blocks.{LAYER}.ff.0.bias"].float()
    W2 = sd[f"blocks.{LAYER}.ff.2.weight"].float()
    b2 = sd[f"blocks.{LAYER}.ff.2.bias"].float()
    F_ = W1.shape[0]

    z = m @ W1.T + b1
    g = F.gelu(z)
    del z
    vnorm = W2.norm(dim=0)                     # (4096,)
    knorm = W1.norm(dim=1)                     # (4096,)
    K = W1 / knorm.unsqueeze(1)                # 单位键
    VrawT = W2.t().contiguous()                # (4096, 1024) 神经元原始值（V0/V3 用）
    VabsT = (W2 * knorm.unsqueeze(0)).t().contiguous()   # 范数吸进值（余弦门控 V1/V2 用）
    y_norm = y.norm(dim=1)

    def rel_l2(pred):
        return ((pred - y).norm(dim=1) / y_norm).mean().item()

    def cos(pred):
        return F.cosine_similarity(pred, y, dim=-1).mean().item()

    def apply(Vmat, idx, w):
        """Vmat: (n_entries, d)。pred_t = sum_{j in idx_t} w_tj * Vmat[j] + b2，分块控内存。"""
        k = idx.shape[1]
        d = Vmat.shape[1]
        out = torch.empty(T, d, device=device)
        for s in range(0, T, CHUNK):
            ii = idx[s:s + CHUNK]
            ww = w[s:s + CHUNK]
            c = ii.shape[0]
            vsel = Vmat[ii.reshape(-1)].view(c, k, d)
            out[s:s + CHUNK] = torch.einsum("ckd,ck->cd", vsel, ww)
        return out + b2

    report = {"layer": LAYER + 1, "tokens": T, "F": F_, "k_list": K_LIST,
              "v2_side": V2_SIDE, "v2_ks": V2_KS, "tau": TAU, "target_top": TARGET_TOP,
              "variants": {}}

    contrib = g.abs() * vnorm.unsqueeze(0)
    target = contrib.topk(TARGET_TOP, dim=1).indices
    target_mask = torch.zeros(T, F_, dtype=torch.bool, device=device)
    target_mask.scatter_(1, target, True)

    def recall(idx):
        return (target_mask.gather(1, idx).sum(dim=1).float() / TARGET_TOP).mean().item()

    # ---- V0a oracle：按 |GELU| 排序（原版口径）----
    order_g = g.abs().argsort(dim=1, descending=True)
    v0a = {}
    for k in K_LIST:
        if k == F_:
            pred = g @ W2.T + b2
        else:
            idx = order_g[:, :k]
            pred = apply(VrawT, idx, g.gather(1, idx))
        v0a[str(k)] = {"rel_l2": rel_l2(pred), "cos": cos(pred)}
    report["variants"]["V0a_oracle_absg"] = v0a

    # ---- V0b oracle：按 |g|*||v|| 排序（贡献贪心）----
    order_c = contrib.argsort(dim=1, descending=True)
    v0b = {}
    for k in K_LIST:
        if k == F_:
            pred = g @ W2.T + b2
        else:
            idx = order_c[:, :k]
            pred = apply(VrawT, idx, g.gather(1, idx))
        v0b[str(k)] = {"rel_l2": rel_l2(pred), "cos": cos(pred)}
    report["variants"]["V0b_oracle_contribution"] = v0b
    report["floor_k4096_rel_l2"] = v0b[str(F_)]["rel_l2"]

    # ---- V1 余弦检索 + softmax 门控（真实可检索形态）----
    cos_all = F.normalize(m, dim=1) @ K.T
    order1 = cos_all.argsort(dim=1, descending=True)
    v1, v3, rec1 = {}, {}, {}
    for k in K_LIST:
        idx = order1[:, :k]
        w = (cos_all.gather(1, idx) * TAU).softmax(1)
        p1 = apply(VabsT, idx, w)
        a = (p1 * y).sum() / (p1 * p1).sum().clamp(min=1e-12)   # 单一全局标量校准
        v1[str(k)] = {"rel_l2": rel_l2(p1), "cos": cos(p1),
                      "rel_l2_alpha": rel_l2(a * p1), "alpha": a.item()}
        del p1
        p3 = apply(VrawT, idx, g.gather(1, idx))
        v3[str(k)] = {"rel_l2": rel_l2(p3), "cos": cos(p3)}
        del p3
        rec1[str(k)] = recall(idx)
    report["variants"]["V1_cosine_softmax"] = v1
    report["variants"]["V3_cosine_set_oracle_gate"] = v3
    report["address_recall"] = {"cosine": rec1}
    del cos_all

    # ---- V2 真·乘积键：64x64 码本 ----
    half = W1.shape[1] // 2
    torch.manual_seed(0)

    def kmeans(X, n_clusters, iters=25):
        C = F.normalize(X[torch.randperm(X.shape[0], device=device)[:n_clusters]].clone(), dim=1)
        for _ in range(iters):
            assign = (X @ C.T).argmax(dim=1)
            for c in range(n_clusters):
                sel = X[assign == c]
                if sel.shape[0]:
                    C[c] = F.normalize(sel.mean(dim=0), dim=0)
        return C, assign

    Ca, a1 = kmeans(F.normalize(K[:, :half], dim=1), V2_SIDE)
    Cb, a2 = kmeans(F.normalize(K[:, half:], dim=1), V2_SIDE)
    cell_of = a1 * V2_SIDE + a2
    counts = torch.bincount(cell_of, minlength=V2_SIDE ** 2).float()
    Vcell = torch.zeros(V2_SIDE ** 2, VabsT.shape[1], device=device)
    Vcell.index_add_(0, cell_of, VabsT)
    Vcell = Vcell / counts.clamp(min=1).unsqueeze(1)
    report["v2_collision"] = {"cells_total": V2_SIDE ** 2,
                              "cells_used": int((counts > 0).sum().item()),
                              "mean_members_nonempty": float(counts[counts > 0].mean().item()),
                              "max_members": int(counts.max().item())}

    s1 = F.normalize(m[:, :half], dim=1) @ Ca.T
    s2 = F.normalize(m[:, half:], dim=1) @ Cb.T
    v2, rec2 = {}, {}
    for kk in V2_KS:
        t1, i1 = s1.topk(kk, dim=-1)
        t2, i2 = s2.topk(kk, dim=-1)
        combo = (t1.unsqueeze(-1) + t2.unsqueeze(-2)).flatten(1)
        cidx = (i1.unsqueeze(-1) * V2_SIDE + i2.unsqueeze(-2)).flatten(1)
        sc, sel = combo.topk(kk, dim=-1)
        idx = cidx.gather(1, sel)
        pred = apply(Vcell, idx, sc.softmax(dim=-1))
        v2[str(kk)] = {"rel_l2": rel_l2(pred), "cos": cos(pred)}
        memb = torch.zeros(T, V2_SIDE ** 2, dtype=torch.bool, device=device)
        memb.scatter_(1, idx, True)
        rec2[str(kk)] = (memb.gather(1, cell_of[target]).float().mean().item())
        del memb
    report["variants"]["V2_product_key_pq64"] = v2
    report["address_recall"]["product_key"] = rec2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
