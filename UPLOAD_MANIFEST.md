# GitHub 上传清单

## 推荐上传方式

可以直接上传整个 `github_upload/` 目录，也可以拆成两个 GitHub 仓库：

- `github_upload/agentflow`
- `github_upload/lora-gsm8k`

如果你想论文/简历展示更清楚，建议拆成两个仓库；如果只是课程/项目归档，一个总仓库也可以。

## 已排除内容

下面这些内容不建议上传，当前整理包里已经排除：

- `agentflow/experiments/`：完整 tau2 轨迹、smoke run 和中间实验输出
- `agentflow/**/__pycache__/`
- `lora/models/`：Qwen base model 和 LoRA checkpoint
- `lora/packages/`：本地 wheel、依赖缓存
- `lora/benchmarks/gsm8k/*.jsonl`：下载数据、DPO pair 数据
- `lora/results/**/*.jsonl`：逐样本预测、候选答案生成结果
- LoRA smoke-test summary

## AgentFlow 最终上传内容

- `agentflow/agentflow/`：框架核心代码和 planner 实现
- `agentflow/examples/run_tau2_batch.py`：最终 tau2 批量消融实验入口
- `agentflow/examples/run_tau2_single.py`：单任务调试入口
- `agentflow/docs/AgentFlow_airline_experiment_showcase.md`：Airline 消融实验报告
- `agentflow/docs/AgentFlow_airline_experiment_showcase_zh.md`：中文实验报告
- `agentflow/README.md`：项目介绍和运行命令

## LoRA 最终上传内容

- `lora-gsm8k/scripts/train_gsm8k_lora.py`：SFT-LoRA 训练
- `lora-gsm8k/scripts/eval_gsm8k_qwen.py`：GSM8K 评估
- `lora-gsm8k/scripts/generate_gsm8k_candidates_batch.py`：K=5 候选答案生成
- `lora-gsm8k/scripts/build_gsm8k_stable_dpo_pairs.py`：stable DPO 数据构造
- `lora-gsm8k/scripts/build_gsm8k_boundary_dpo_pairs.py`：boundary DPO 数据构造
- `lora-gsm8k/scripts/train_gsm8k_dpo_stable_anchor.py`：stable-anchor DPO 训练
- `lora-gsm8k/scripts/train_gsm8k_dpo_boundary_anchor.py`：boundary-anchor DPO 训练
- `lora-gsm8k/results/gsm8k/all/*.summary.json`：最终评估摘要
- `lora-gsm8k/README.md`：中文复现说明和结果表
