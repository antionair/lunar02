# 探针 #4：dense@152M 等训练算力对照报告（#3 归因裁决）

## 裁决

**s = 1.010 ≥ 0.85 → 数据主导：探针 #3 降级为「数据 scaling 胜利」，字库贡献另议。**

## 一行对比表（同口径，held-out PPL / 域外 PPL，best-lr 四种子均值±样本标准差）

| 组 | held-out PPL | 域外 PPL |
|---|---:|---:|
| dense@38M | 313.5 ± 20.1 | 342.3 ± 29.7 |
| **dense@152M** | **56.07 ± 1.38** | **60.57 ± 2.06** |
| n1m@38M | 311.1 ± 26.7 | 336.2 ± 32.4 |
| n4m@152M | 56.26 ± 1.66 | 67.42 ± 2.17 |

dense@152M 与 N=4M@152M 在 held-out PPL 上统计等同（差 0.19，远小于
种子标准差）；域外 PPL dense 低 6.9（约 3σ，倾向性信号，待更多种子复核）。
20.5 亿参数稀疏值表在该规模-数据组合下 PPL 收益为零。

## 设置

- 与 #3 dense 组 `args.json` 逐参数 diff 为空，唯一变化 `train_tokens` 38M→152M
- LR {3e-4, 6e-4} × 种子 {1,2,3,4} 共 8 runs；best-of-grid = 3e-4
  （末 1000 步 val_loss 均值 3.884 vs 5.064），四种子报均值±样本标准差
- 评测三桶与 #3 完全同口径；QA 预期全零，实际全零，仅作记录
- 功耗纪律：日间 250W / 夜间 100W（用户家庭静音要求，属非受控环境变量，
  只影响墙钟不影响 loss）；修正后全程约 46 小时
- 每 run 收尾保存 `checkpoint.pt`（dense 约 2GB），`gen_sample.py` 可复核生成

## 机检证据

1. config diff 除 token 数外为空：`machine_checks.config_diff_except_tokens = {}`。
2. 实际训练 token 数 152M × 8 runs，全部落在 152M ±1%。
3. 重叠率 0：`data/scale/metadata.json` 构造证据 + 三个 bin 的 SHA-256
   记录于 `runs/dense_control/environment.json`。
4. 吞吐实测：dense@152M 4,538 tok/s 均值、单 run 10.8h；全组见
   `throughput_table`（dense_control_summary.json）。
5. 失败命令：无。

## 顺带定性（上轮欠的三件）

1. QA 题集版本：`data/scale/qa_frozen.jsonl`（176 题，2026-11-13 用户终审
   冻结，SHA-256 `ccb88224…`，拒绝题清单 `qa_review_rejects.json`）。
   非草稿直接冻结。
2. 吞吐倒挂解释：dense 无字库层与 CPU 稀疏优化器，单位 token 纯 GPU
   计算最快（38M 档 6.2k tok/s）；N=1M/N=4M 每步多值表命中行拷贝与
   SparseAdamW（CPU）开销（4.1k / 4.6k tok/s）；N=4M 键表按 √N 翻倍
   仅再降约 10%——命中行拷贝只含 top-8 行，不随 N 线性增长。
3. N=4M 格使用分布：4 个 n4m run 的累计使用率/访问熵随步数曲线已导出
   `n4m_usage_entropy_curves.csv`。探针 #3 未记录逐格命中计数，真实
   格使用分布直方图无法事后重建；建议下一轮在 scale_train.py 中增加
   周期性格命中直方图日志。

## 结论与下一步

字库线在 0.5B 计算组 × 152M token 的 PPL 账上收益为零，探针 #3 的
254.8 PPL 增益全部来自 4 倍数据。这把项目问题从「字库有没有用」精确化为
「字库收益的存在条件是什么」：候选方向包括更大 N × 更多数据、多层注入、
预取友好定址、训练配方（值表 LR/路由均衡）。QA 侧因基座模型零样本闭卷
全体为零（探针 #4 checkpoint 实测贪心复读，gen_sample.py 可复核），
「字库存知识」的收益需 few-shot 新口径重测后才可下结论。

## 文件位置

- 汇总：`runs/dense_control/dense_control_summary.json`、`comparison.csv`
- 环境证据：`runs/dense_control/environment.json`
- 逐 run：`runs/dense_control/dense152_lr*_seed*/`（train.jsonl + args.json + checkpoint.pt）
- 调度/汇总脚本：`run_dense_control.py`、`summarize_dense_control.py`、`gen_sample.py`
