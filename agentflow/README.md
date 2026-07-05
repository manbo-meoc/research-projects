# AgentFlow

AgentFlow 是一个轻量级、基于 LangGraph 思路组织的工具调用 Agent 框架。项目重点不是提出新的大模型算法，而是把 Agent 的状态、Planner、工具注册、图执行和 benchmark adapter 拆成清晰模块，并用 tau2-bench 做可复现实验。

当前最终实验集中在 tau2-bench Airline domain 上，比较 ReAct baseline、Memory/Reflection 消融，以及一个可选的 Airline workflow skill layer。

## 项目特点

- 模块化 Agent 框架：状态、planner、tool registry、graph、外部 benchmark adapter 分离。
- tau2-bench 集成：只替换 assistant-side agent，保留 tau2 官方 user simulator、environment tools 和 evaluator。
- 包含 ReAct baseline、Memory、Reflection、Memory+Reflection、Airline Skill 等消融变体。
- Airline domain 完整 50-task benchmark 已跑通，并整理了最终结果报告。

## 架构概览

```text
User Input / tau2 User Simulator
        ↓
AgentFlow Agent / tau2 Adapter
        ↓
Planner Strategy
        ↓
Tool Call or Final Answer
        ↓
Tool Registry / tau2 Environment Tools
        ↓
Evaluation
```

核心模块：

```text
agentflow/
├── agent.py
├── graph.py
├── planner.py
├── react_planner.py
├── memory.py
├── reflection.py
├── react_memory_reflection_planner.py
├── react_airline_skill_planner.py
├── airline_skill.py
├── state.py
├── tau2_adapter.py
└── tool_registry.py
```

## Airline Benchmark 最终结果

当前结果使用 tau2-bench 官方 Airline domain，共 50 个任务。

| 方法 | Success / Total | Reward | Avg. Messages | Failed Tasks |
|---|---:|---:|---:|---:|
| ReAct | 37 / 50 | 0.74 | 24.12 | 13 |
| ReAct + Memory + Reflection | 35 / 50 | 0.70 | 23.40 | 15 |
| ReAct + Airline Skill | 39 / 50 | 0.78 | 23.48 | 11 |

结论：通用 Memory + Reflection 没有提升最终 reward；针对 Airline 长流程任务设计的 workflow skill layer 有正向提升，主要作用是把长对话任务拆成更短的阶段，并给 ReAct planner 提供当前阶段的子目标提示。

详细报告：

```text
docs/AgentFlow_airline_experiment_showcase_zh.md
docs/AgentFlow_airline_experiment_showcase.md
```

## 环境准备

安装项目依赖：

```bash
pip install -e .
```

tau2-bench 需要单独安装或放到 `PYTHONPATH`。本项目的 `examples/run_tau2_batch.py` 会把本地 tau2-bench 源码路径加入 `sys.path`，如果你的路径不同，需要修改脚本里的 `TAU2_ROOT`。

## 运行方式

单任务调试：

```bash
python examples/run_tau2_single.py \
  --domain airline \
  --task-id 0 \
  --agent-name agentflow_react
```

完整 Airline ReAct baseline：

```bash
python examples/run_tau2_batch.py \
  --domain airline \
  --task-ids all \
  --agent-name agentflow_react \
  --run-name "ReAct baseline" \
  --save-root "experiments/airline_react" \
  --resume
```

完整 Airline ReAct + Memory + Reflection：

```bash
python examples/run_tau2_batch.py \
  --domain airline \
  --task-ids all \
  --agent-name agentflow_react_memory_reflection \
  --run-name "ReAct Memory Reflection" \
  --save-root "experiments/airline_react_memory_reflection" \
  --resume
```

完整 Airline ReAct + Airline Skill：

```bash
python examples/run_tau2_batch.py \
  --domain airline \
  --task-ids all \
  --agent-name agentflow_react_airline_skill \
  --run-name "ReAct Airline Skill adaptive" \
  --save-root "experiments/airline_react_airline_skill" \
  --resume
```

## 目录结构

```text
agentflow/      框架核心模块
examples/       最终 tau2 单任务和批量消融实验入口
docs/           技术报告和实验总结
experiments/    本地生成目录；GitHub 上传包中不包含完整轨迹
tests/          预留测试目录
```

## 说明

这个仓库定位是学习和研究工程项目，用来展示 Agent 框架设计、benchmark adapter、消融实验和结果分析。Airline 结果不声称 state of the art。
