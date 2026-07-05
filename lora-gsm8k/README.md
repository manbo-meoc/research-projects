# Qwen2.5 GSM8K LoRA 消融实验

本项目整理了 `Qwen2.5-0.5B-Instruct` 在 GSM8K 数学推理任务上的 LoRA / SFT / DPO 微调消融实验脚本。

实验对比对象：

- 原始 `Qwen2.5-0.5B-Instruct`
- GSM8K SFT-LoRA
- SFT + DPO wrong-vs-gold
- SFT + DPO K5 replay
- SFT + DPO stable anchor
- SFT + DPO boundary anchor

仓库不包含模型权重、LoRA checkpoint、下载后的 GSM8K JSONL、生成候选答案 JSONL 和 smoke-test 输出。

## 最终结果

所有最终评估都在 GSM8K test set 上运行，共 1,319 道题。

| 方法 | 正确数 / 总数 | Accuracy |
|---|---:|---:|
| Qwen2.5-0.5B-Instruct base | 381 / 1319 | 28.89% |
| SFT-LoRA | 410 / 1319 | 31.08% |
| SFT + DPO wrong-vs-gold train1500 | 380 / 1319 | 28.81% |
| SFT + DPO K5 replay full1500 | 401 / 1319 | 30.40% |
| SFT + DPO stable anchor full1500 | 404 / 1319 | 30.63% |
| SFT + DPO boundary anchor full1500 | 416 / 1319 | 31.54% |

最终最好的变体是 `boundary_anchor_full1500`。这个版本只选择完整但答案错误的候选作为 DPO negative，并在 DPO loss 外加入 SFT anchor loss，减少偏好训练对原有解题能力的破坏。

## 目录结构

```text
scripts/
  download_gsm8k.py
  train_gsm8k_lora.py
  eval_gsm8k_qwen.py
  generate_gsm8k_candidates_batch.py
  build_gsm8k_*_dpo_pairs.py
  train_gsm8k_dpo_*.py
benchmarks/gsm8k/
  *.summary.json
results/gsm8k/all/
  *.summary.json
```

## 环境准备

安装核心依赖：

```bash
pip install torch transformers accelerate datasets peft sentencepiece
```

将基础模型放到：

```text
models/Qwen/Qwen2.5-0.5B-Instruct
```

下载 GSM8K：

```bash
python scripts/download_gsm8k.py
```

下载后会生成：

```text
benchmarks/gsm8k/train.jsonl
benchmarks/gsm8k/test.jsonl
```

## 复现流程

训练 SFT-LoRA：

```bash
python scripts/train_gsm8k_lora.py \
  --model-dir models/Qwen/Qwen2.5-0.5B-Instruct \
  --train-path benchmarks/gsm8k/train.jsonl \
  --output-dir models/qwen25_05b_gsm8k_lora
```

评估 base 和 SFT：

```bash
python scripts/eval_gsm8k_qwen.py \
  --model-dir models/Qwen/Qwen2.5-0.5B-Instruct \
  --data-path benchmarks/gsm8k/test.jsonl \
  --limit 1319 \
  --output results/gsm8k/all/qwen25_05b_base_gsm8k_eval_all.jsonl

python scripts/eval_gsm8k_qwen.py \
  --model-dir models/Qwen/Qwen2.5-0.5B-Instruct \
  --lora-adapter-dir models/qwen25_05b_gsm8k_lora \
  --data-path benchmarks/gsm8k/test.jsonl \
  --limit 1319 \
  --output results/gsm8k/all/qwen25_05b_gsm8k_lora_eval_all.jsonl
```

基于 SFT 模型生成 K=5 候选答案：

```bash
python scripts/generate_gsm8k_candidates_batch.py \
  --model-dir models/Qwen/Qwen2.5-0.5B-Instruct \
  --lora-adapter-dir models/qwen25_05b_gsm8k_lora \
  --data-path benchmarks/gsm8k/train.jsonl \
  --limit 1500 \
  --num-candidates 5 \
  --output results/gsm8k/train/qwen25_05b_gsm8k_sft_lora_candidates_k5_full1500_batch.jsonl
```

构造最终 DPO 数据：

```bash
python scripts/build_gsm8k_stable_dpo_pairs.py \
  --data-path benchmarks/gsm8k/train.jsonl \
  --candidate-path results/gsm8k/train/qwen25_05b_gsm8k_sft_lora_candidates_k5_full1500_batch.jsonl \
  --output benchmarks/gsm8k/dpo_lora_stable_full1500.jsonl

python scripts/build_gsm8k_boundary_dpo_pairs.py \
  --data-path benchmarks/gsm8k/train.jsonl \
  --candidate-path results/gsm8k/train/qwen25_05b_gsm8k_sft_lora_candidates_k5_full1500_batch.jsonl \
  --output benchmarks/gsm8k/dpo_lora_boundary_full1500.jsonl
```

训练最终 DPO 变体：

```bash
python scripts/train_gsm8k_dpo_stable_anchor.py \
  --base-model-dir models/Qwen/Qwen2.5-0.5B-Instruct \
  --sft-adapter-dir models/qwen25_05b_gsm8k_lora \
  --dpo-data-path benchmarks/gsm8k/dpo_lora_stable_full1500.jsonl \
  --output-dir models/qwen25_05b_gsm8k_sft_dpo_lora_stable_anchor_full1500

python scripts/train_gsm8k_dpo_boundary_anchor.py \
  --base-model-dir models/Qwen/Qwen2.5-0.5B-Instruct \
  --sft-adapter-dir models/qwen25_05b_gsm8k_lora \
  --dpo-data-path benchmarks/gsm8k/dpo_lora_boundary_full1500.jsonl \
  --output-dir models/qwen25_05b_gsm8k_sft_dpo_lora_boundary_anchor_full1500
```

评估 DPO adapter：

```bash
python scripts/eval_gsm8k_qwen.py \
  --model-dir models/Qwen/Qwen2.5-0.5B-Instruct \
  --lora-adapter-dir models/qwen25_05b_gsm8k_sft_dpo_lora_boundary_anchor_full1500 \
  --data-path benchmarks/gsm8k/test.jsonl \
  --limit 1319 \
  --output results/gsm8k/all/qwen25_05b_gsm8k_sft_dpo_lora_boundary_anchor_full1500_eval_all.jsonl
```

## 说明

DPO 训练没有依赖 TRL 的 Trainer，而是用 PyTorch 直接实现，便于检查 reference margin 预计算、DPO loss 和 SFT anchor loss。
