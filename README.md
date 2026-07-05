# 研究工程项目上传包

这个目录整理了两个可以上传到 GitHub 的项目：

- `agentflow/`：基于 LangGraph 的轻量级工具调用 Agent 框架，以及 tau2-bench Airline 消融实验。
- `lora-gsm8k/`：基于 `Qwen2.5-0.5B-Instruct` 的 GSM8K LoRA / SFT / DPO 微调消融实验脚本和最终结果摘要。

为了避免仓库过大，以下内容没有放进上传包：

- 模型权重和 LoRA adapter checkpoint
- 下载后的 benchmark JSONL 数据
- 生成的候选答案 JSONL 文件
- 完整逐任务轨迹输出
- smoke test 输出
- Python 缓存文件

每个子目录都有独立 README，包含实验目标、复现命令和最终结果。
