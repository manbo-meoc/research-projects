# AgentFlow Airline Benchmark Showcase

This document summarizes the main Airline-domain benchmark experiments for AgentFlow. The goal is not to claim a new agent algorithm, but to demonstrate a lightweight agent framework, tau2-bench integration, and benchmark-driven ablation analysis.

## Project Goal

AgentFlow is a lightweight AI agent framework built around a clear separation of responsibilities:

- `Planner`: decides the next tool call or final response.
- `ToolRegistry`: exposes available tool schemas to the planner.
- `Graph` / `Agent`: provides the framework execution structure.
- `tau2_adapter`: connects AgentFlow to tau2-bench while preserving tau2's official user simulator, environment tools, and evaluator.

For tau2-bench evaluation, AgentFlow replaces only the assistant-side agent. The official Airline user simulator, environment, tools, and evaluator remain unchanged.

## Benchmark Setup

- Benchmark: `tau2-bench`
- Domain: `airline`
- Number of tasks: `50`
- User simulator: official tau2 `user_simulator`
- Environment and tools: official tau2 Airline tools
- Evaluator: official tau2 evaluator
- LLM backend: DeepSeek via LiteLLM / LangChain OpenAI-compatible client

## Experiment Variants

### 1. ReAct Baseline

The baseline agent uses a domain-independent ReAct planner. It receives the conversation history and registered tool schemas, then outputs either one tool call or one final answer.

This variant is intentionally simple and serves as the main framework baseline.

### 2. ReAct + Memory/Reflection

This variant adds generic prompt-level working memory and reflection:

- Memory summarizes recent user facts, tool calls, and tool observations.
- Reflection asks the model to self-check whether a tool is needed, whether arguments are known, and whether a final answer is justified.

This is a general reasoning augmentation and does not include Airline-specific workflow logic.

### 3. ReAct + Airline Skill Layer

After analyzing failures from the first two variants, many errors were found to occur in long-horizon Airline tasks. These tasks often require multiple workflow phases:

```text
identify user
inspect reservation / flight state
ask confirmation
execute write tool
finalize response
```

The Airline Skill Layer is an optional domain-specific extension that decomposes long conversations into workflow phases and provides the ReAct planner with a compact hint:

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

The core AgentFlow framework remains domain independent. Airline-specific logic is isolated in the optional skill layer.

The skill is also adaptive: it avoids affecting short, simple tasks and activates only when long-horizon risk is detected, such as high step count, repeated read-tool usage, or tool errors.

## Main Results

| Variant | Success / Total | Reward | Avg. Messages | Failed Tasks |
|---|---:|---:|---:|---|
| ReAct baseline | 37 / 50 | 0.74 | 24.12 | 13 |
| ReAct + Memory/Reflection | 35 / 50 | 0.70 | 23.40 | 15 |
| ReAct + Airline Skill | 39 / 50 | 0.78 | 23.48 | 11 |

## Pairwise Analysis

Compared with the ReAct baseline, generic Memory/Reflection improved some tasks but introduced more regressions.

```text
Memory/Reflection improved: 12, 27, 35, 41
Memory/Reflection regressed: 4, 15, 20, 32, 33, 42
Net change: -2 tasks
```

The Airline Skill Layer produced a net positive improvement.

```text
Airline Skill improved: 12, 27, 29, 35, 41
Airline Skill regressed: 16, 37, 42
Net change: +2 tasks
```

This suggests that domain-specific workflow decomposition is more effective than generic prompt-level Memory/Reflection for this Airline benchmark.

## Per-Task Results

Format: `reward/messages`.

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

## Interpretation

The Memory/Reflection variant reduced average messages but lowered final reward. This indicates that generic reasoning augmentation can introduce additional decision noise in transactional, tool-heavy tasks.

The Airline Skill Layer improved reward by targeting a more specific failure mode: long-horizon workflow discontinuity. Instead of adding more general reasoning, it decomposes the task into domain workflow phases and gives the planner a current subgoal.

The results support the following conclusion:

```text
For tau2 Airline tasks, domain-specific workflow guidance is more effective than generic prompt-level Memory/Reflection.
```

## Reproducibility

Run the ReAct baseline:

```bash
python examples/run_tau2_batch.py \
  --domain airline \
  --task-ids all \
  --agent-name agentflow_react \
  --run-name "ReAct baseline" \
  --save-root "experiments/airline_react" \
  --resume
```

Run ReAct + Memory/Reflection:

```bash
python examples/run_tau2_batch.py \
  --domain airline \
  --task-ids all \
  --agent-name agentflow_react_memory_reflection \
  --run-name "ReAct Memory Reflection" \
  --save-root "experiments/airline_react_memory_reflection" \
  --resume
```

Run ReAct + Airline Skill:

```bash
python examples/run_tau2_batch.py \
  --domain airline \
  --task-ids all \
  --agent-name agentflow_react_airline_skill \
  --run-name "ReAct Airline Skill adaptive" \
  --save-root "experiments/airline_react_airline_skill" \
  --resume
```

## Summary

AgentFlow demonstrates a modular framework for benchmark-driven agent experimentation:

- A domain-independent ReAct baseline establishes a runnable benchmark foundation.
- Generic Memory/Reflection provides an ablation study but does not improve final reward.
- A targeted Airline Skill Layer improves performance by decomposing long-horizon domain workflows.

The key engineering lesson is that reliable tool-use agents often require workflow-level control, not only more memory or reflection tokens.
