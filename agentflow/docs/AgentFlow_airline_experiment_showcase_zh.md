# AgentFlow Airline Benchmark 实验展示

本文档总结 AgentFlow 在 tau2-bench Airline Domain 上的主要实验结果。项目目标不是提出新的 Agent 算法，而是搭建一个结构清晰、可运行、方便扩展和消融实验的轻量级 Agent Framework。

## 项目定位

AgentFlow 是一个轻量级 AI Agent Framework，核心设计强调模块职责清晰：

- `Planner`：决定下一步是调用工具还是输出最终回复。
- `ToolRegistry`：向 Planner 暴露可用工具的 schema。
- `Graph` / `Agent`：提供 AgentFlow 的基础执行结构。
- `tau2_adapter`：将 AgentFlow 接入 tau2-bench，同时保留 tau2 官方的 User Simulator、Environment Tools 和 Evaluator。

在 tau2-bench 评测中，AgentFlow 只替换 assistant-side agent。用户模拟器、环境工具和评测器均使用 tau2-bench 官方实现。

## Benchmark 设置

- Benchmark：`tau2-bench`
- Domain：`airline`
- 任务数量：`50`
- User Simulator：tau2 官方 `user_simulator`
- Environment / Tools：tau2 官方 Airline tools
- Evaluator：tau2 官方 evaluator
- LLM Backend：DeepSeek，通过 LiteLLM / LangChain OpenAI-compatible client 调用

## 实验版本

### 1. ReAct Baseline

基础版本使用 domain-independent ReAct Planner。Planner 接收对话历史和工具 schema，然后输出：

```text
一个 tool_call
或
一个 final_answer
```

该版本不包含领域逻辑，是 AgentFlow 的主要 baseline。

### 2. ReAct + Memory/Reflection

该版本加入通用的 prompt-level Memory 和 Reflection：

- Memory：总结近期用户信息、工具调用和工具返回结果。
- Reflection：让模型在输出前检查是否需要工具、参数是否齐全、是否可以最终回答。

该版本属于通用推理增强，不包含 Airline 专用流程逻辑。

### 3. ReAct + Airline Skill Layer

在分析前两版失败样例后，发现大量失败集中在 Airline 长对话、多阶段工具调用任务中。这类任务通常包含多个阶段：

```text
识别用户
查询 reservation / flight 状态
确认用户意图
执行写操作工具
最终回复
```

因此引入可选的 Airline Skill Layer，将长对话任务拆分为更短的 workflow phase，并向 ReAct Planner 提供当前阶段的结构化提示：

```python
{
    "intent": "book_flight | cancel_reservation | change_or_rebook | compensation | ...",
    "phase": "read_state | ask_confirmation | execute_write | finalize | converge",
    "known_info": {...},
    "current_subgoal": "...",
    "next_allowed_actions": [...],
    "must_not": [...]
}
```

需要注意的是，AgentFlow core 仍然保持 domain-independent。Airline 相关逻辑被隔离在可选 skill layer 中。

Airline Skill 也是自适应触发的：短任务默认不注入 skill hint，只有在检测到长程任务风险时才启用，例如 step 数较高、read/search 工具使用较多、或工具返回 error。

## 主实验结果

| 实验版本 | 成功数 / 总数 | Reward | 平均 Messages | 失败任务数 |
|---|---:|---:|---:|---:|
| ReAct baseline | 37 / 50 | 0.74 | 24.12 | 13 |
| ReAct + Memory/Reflection | 35 / 50 | 0.70 | 23.40 | 15 |
| ReAct + Airline Skill | 39 / 50 | 0.78 | 23.48 | 11 |

## 相对 Baseline 的变化

### Memory/Reflection

相比 ReAct baseline，Memory/Reflection 修复了一部分任务，但引入了更多 regression。

```text
修复任务：12, 27, 35, 41
退化任务：4, 15, 20, 32, 33, 42
净变化：-2 个任务
```

### Airline Skill

相比 ReAct baseline，Airline Skill 取得了正向净提升。

```text
修复任务：12, 27, 29, 35, 41
退化任务：16, 37, 42
净变化：+2 个任务
```

这说明在 tau2 Airline 这类长程工具调用任务中，垂域 workflow decomposition 比通用 Memory/Reflection 更有效。

## 逐任务结果

格式：`reward/messages`。

| Task | ReAct | Memory/Reflection | Airline Skill |
|---:|---:|---:|---:|
| 0 | 1/20 | 1/16 | 1/20 |
| 1 | 1/30 | 1/22 | 1/22 |
| 2 | 1/28 | 1/24 | 1/26 |
| 3 | 1/16 | 1/14 | 1/16 |
| 4 | 1/28 | 0/31 | 1/22 |
| 5 | 1/22 | 1/20 | 1/24 |
| 6 | 1/18 | 1/16 | 1/24 |
| 7 | 0/31 | 0/32 | 0/32 |
| 8 | 1/26 | 1/26 | 1/26 |
| 9 | 1/26 | 1/20 | 1/22 |
| 10 | 1/24 | 1/24 | 1/26 |
| 11 | 1/24 | 1/24 | 1/26 |
| 12 | 0/32 | 1/22 | 1/22 |
| 13 | 1/16 | 1/14 | 1/8 |
| 14 | 0/31 | 0/31 | 0/32 |
| 15 | 1/26 | 0/32 | 1/20 |
| 16 | 1/18 | 1/20 | 0/18 |
| 17 | 1/28 | 1/28 | 1/26 |
| 18 | 0/32 | 0/32 | 0/32 |
| 19 | 1/16 | 1/16 | 1/16 |
| 20 | 1/22 | 0/31 | 1/22 |
| 21 | 0/31 | 0/31 | 0/31 |
| 22 | 1/28 | 1/26 | 1/30 |
| 23 | 0/32 | 0/32 | 0/32 |
| 24 | 0/32 | 0/32 | 0/28 |
| 25 | 1/26 | 1/24 | 1/18 |
| 26 | 1/16 | 1/16 | 1/22 |
| 27 | 0/32 | 1/14 | 1/24 |
| 28 | 1/16 | 1/16 | 1/20 |
| 29 | 0/24 | 0/24 | 1/22 |
| 30 | 1/20 | 1/18 | 1/18 |
| 31 | 1/18 | 1/18 | 1/18 |
| 32 | 1/30 | 0/32 | 1/26 |
| 33 | 1/24 | 0/24 | 1/30 |
| 34 | 1/20 | 1/22 | 1/18 |
| 35 | 0/31 | 1/28 | 1/30 |
| 36 | 1/8 | 1/10 | 1/10 |
| 37 | 1/30 | 1/30 | 0/31 |
| 38 | 1/26 | 1/28 | 1/20 |
| 39 | 0/31 | 0/31 | 0/31 |
| 40 | 1/12 | 1/14 | 1/14 |
| 41 | 0/31 | 1/30 | 1/28 |
| 42 | 1/30 | 0/31 | 0/31 |
| 43 | 1/26 | 1/26 | 1/28 |
| 44 | 0/32 | 0/32 | 0/32 |
| 45 | 1/22 | 1/24 | 1/22 |
| 46 | 1/12 | 1/8 | 1/12 |
| 47 | 1/26 | 1/26 | 1/26 |
| 48 | 1/14 | 1/14 | 1/24 |
| 49 | 1/12 | 1/14 | 1/16 |

## 实验分析

Memory/Reflection 版本降低了平均 messages，但最终 reward 下降。这说明通用的 prompt-level 记忆和反思可能会引入额外推理噪声，导致原本正确的任务出现 regression。

Airline Skill 的提升来自更有针对性的失败修复。它不是简单增加上下文，而是将长程任务拆分为 workflow phase，让 ReAct 每轮只关注当前阶段的子目标。

核心结论可以概括为：

```text
对于 tau2 Airline 这种多阶段工具调用任务，垂域 workflow skill 比通用 Memory/Reflection 更有效。
```

## 复现实验命令

运行 ReAct baseline：

```bash
python examples/run_tau2_batch.py \
  --domain airline \
  --task-ids all \
  --agent-name agentflow_react \
  --run-name "ReAct baseline" \
  --save-root "experiments/airline_react" \
  --resume
```

运行 ReAct + Memory/Reflection：

```bash
python examples/run_tau2_batch.py \
  --domain airline \
  --task-ids all \
  --agent-name agentflow_react_memory_reflection \
  --run-name "ReAct Memory Reflection" \
  --save-root "experiments/airline_react_memory_reflection" \
  --resume
```

运行 ReAct + Airline Skill：

```bash
python examples/run_tau2_batch.py \
  --domain airline \
  --task-ids all \
  --agent-name agentflow_react_airline_skill \
  --run-name "ReAct Airline Skill adaptive" \
  --save-root "experiments/airline_react_airline_skill" \
  --resume
```

## 总结

AgentFlow 展示了一个面向 benchmark 实验的模块化 Agent Framework：

- ReAct baseline 建立了可运行的 tau2-bench 接入基础。
- Memory/Reflection 消融表明，通用推理增强并不一定带来稳定收益。
- Airline Skill Layer 通过垂域 workflow decomposition，将成功率从 37/50 提升到 39/50。

该实验说明，在工具密集型长程任务中，可靠性往往不只依赖更多 memory 或 reflection，而更依赖对业务流程阶段的建模和控制。
