# lunar02 规模探针 T+2h 环境、显存与冒烟报告

## 环境与资源

- 机器：mark22
- GPU：3×RTX 3080 20GB + 1×RTX A4000 16GB，无 NVLink
- 功耗上限：三张 RTX 3080 已持久化为 150W；A4000 为 137W
- Python：3.12.3
- PyTorch：2.14.1+cu130，CUDA 13.0
- 系统内存：188GB
- 磁盘可用：约 55.8GB
- 代码目录不是 Git 仓库，版本证据使用 SHA-256

## 0.5B 计算组核对

实际 Dense 模型：

- 总参数：505,977,856
- Transformer 计算组参数：453,464,064
- `d_model=1024`
- `layers=36`
- `heads=16`
- `FFN=4×`
- `seq_len=1000`
- 字库注入层：第 18 层后
- PKM：4 头，top-k=8，查询维度 64，`d_v=512`

## 数据冻结

数据：FineWeb-Edu，GPT-2 分词器。

- 训练桶：152,000,000 tokens
- held-out 同域评测：4,000,000 tokens
- held-out 域外评测：4,000,000 tokens
- Dense/N=1M 共用训练前 38,000,000 tokens
- N=4M 使用完整 152,000,000 tokens
- token:N=38:1
- 训练与验证区间不重叠，重叠率 0
- 域外桶使用 `blake2b(domain-or-url) mod 10 == 0` 的文档，且不进入训练桶

数据文件 SHA-256：

```text
train_152m.bin          c724465a1e988c85a016970520f6d76375e321bf2d55e1cc52170449391f9c00
eval_in_domain_4m.bin   048f7ae46dde6a0c767a3419a1b9f111738fdf6985d6b864e47930ffa376ce4e
eval_domain_4m.bin      706115a5f6b671d497cc016bf53d53d61deb53a40c81fb1f516aaeb592ea4f48
```

## 值表显存与内存方案

N=4M、d_v=512：

- 值表参数：2,048,000,000
- FP32 参数：8GB
- FP32 AdamW 状态：16GB
- 命中行梯度临时驻留：随 batch 变化
- 实测进程 CPU RSS：约 27.6GB

实现采用 CPU 值表 + 命中行稀疏反传：

1. 值表本体常驻 CPU；
2. 每步只把实际命中的行复制到 GPU；
3. 被选中行作为临时叶子节点回传梯度；
4. CPU 端 SparseAdamW 只更新命中行；
5. 不生成 8GB 全量值表梯度，也不执行全表 Adam 更新。

N=1M 同方案，实测 CPU RSS 约 9.0GB。

## 200 步冒烟结果

| 配置 | 实测 token/s | 峰值 GPU 显存 | CPU RSS | held-out loss | held-out PPL | 域外 PPL |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 5,261 | 10.14GB | 2.76GB | 6.9765 | 1071.20 | 1091.19 |
| N=1M | 3,005 | 10.32GB | 9.05GB | 6.9868 | 1082.27 | 1107.28 |
| N=4M | 2,898 | 10.32GB | 27.55GB | 6.9685 | 1062.58 | 1063.60 |

另做 N=4M、micro-batch=2 的 10 步预检：

- 吞吐：2,636 token/s
- 峰值 GPU 显存：10.78GB
- CPU RSS：27.57GB

正式运行拟采用：

- RTX 3080：micro-batch=2，梯度累积 4，有效 token batch=8,000
- A4000：micro-batch=1，梯度累积 8，有效 token batch=8,000

## ETA 外推

保守按 N=4M 2,636 token/s：

- 单个 N=4M run：152M / 2,636 ≈ 16.0 小时
- 6 个 N=4M run，三张 3080 并行：约 32.1 小时

Dense/N=1M 任务放在 A4000：

- Dense 6 runs：约 12.1 GPU 小时
- N=1M 6 runs：约 21.1 GPU 小时
- 串行合计约 33.2 小时

因此关键路径预计约 **33–38 小时**，低于 48 小时硬预算，但余量不大。若正式 smoke 后吞吐低于当前值，或任一 run 卡住超过 2 小时，将按任务书停止扩展并上报。

## 当前阻塞项

事实检索桶需要 200 道闭卷事实题，并要求用户审核冻结。当前已生成草稿：

- 远端：`/home/antionair/pkm_experiment/data/scale/qa_draft.jsonl`
- 本机：`C:\Users\asd20\pkm_experiment\data\scale\qa_draft.jsonl`

每题包含问题、短答案、源文档 ID、URL/domain、训练 token 区间和原文摘录。正式训练脚本只接受审核后的：

`data/scale/qa_frozen.jsonl`

在冻结文件出现前不会启动 18 runs。
