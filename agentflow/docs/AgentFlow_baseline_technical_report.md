# AgentFlow Baseline 技术报告

## 1. 项目定位

AgentFlow 是一个用于 AI Agent 学习和 Benchmark 实验的轻量级 Agent Framework。当前版本是 baseline 版本，目标不是提出新的 Agent 算法，而是搭建一个结构清晰、可运行、方便后续扩展和消融实验的基础框架。

当前框架基于 LangGraph 构建工作流，围绕 tau2-bench Benchmark 的 Airline Domain 进行后续适配。评测时仍计划使用 tau2-bench 官方的 User Simulator、Environment、Airline Tools 和 Evaluator，AgentFlow 只替换官方 Agent 部分。

当前 baseline 版本的核心目标是：

1. 建立 Agent 的最小运行闭环。
2. 明确 Planner、Tool Registry、Graph、Agent 入口之间的职责边界。
3. 保持框架 domain independent，不在框架内部实现 Airline 业务逻辑。
4. 为后续 Memory、Reflection、Review 等能力模块预留扩展空间。

## 2. Baseline 版本范围

当前版本属于 baseline agent framework，已经完成最小可运行链路：

```text
User Input
   ↓
AgentFlowAgent.run()
   ↓
LangGraph Workflow
   ↓
Planner Node
   ↓
Tool Node
   ↓
Planner Node
   ↓
Final Answer
```

当前 baseline 已包含：

- 统一运行状态 `AgentState`
- 工具注册中心 `ToolRegistry`
- 唯一 LLM 决策模块 `Planner`
- LangGraph 工作流 `graph.py`
- 对外入口 `AgentFlowAgent`
- 项目依赖和可安装配置

当前 baseline 不包含：

- Memory 模块
- Reflection 模块
- Review 或 Critic 模块
- 多 Agent 协作
- Airline Domain 工具的内置实现
- tau2-bench 官方评测脚本集成

这些模块属于后续实验阶段的扩展内容。

## 3. 当前项目结构

```text
agentflow/
├── agentflow/
│   ├── __init__.py
│   ├── agent.py
│   ├── graph.py
│   ├── planner.py
│   ├── state.py
│   └── tool_registry.py
├── configs/
├── docs/
├── examples/
├── experiments/
├── tests/
├── README.md
└── pyproject.toml
```

其中 `agentflow/` 是框架核心源码目录，`docs/` 用于保存设计文档和技术报告，`experiments/` 后续用于保存 benchmark 和消融实验结果。

## 4. 核心模块说明

### 4.1 state.py

`state.py` 定义了 LangGraph 工作流中传递的共享状态 `AgentState`。

当前状态字段包括：

- `messages`：用户、Agent、Tool 的对话历史。
- `tool_calls`：Planner 生成的工具调用请求。
- `tool_results`：Tool Node 执行工具后的返回结果。
- `final_answer`：Planner 生成的最终答案。
- `step_count`：当前运行步数，用于观察和限制循环。

该模块只负责状态结构定义，不包含业务逻辑、LLM 调用或工具执行逻辑。

### 4.2 tool_registry.py

`ToolRegistry` 是框架的工具注册中心，负责管理工具说明和真实工具函数。

它维护两类信息：

- `tool_descriptions`：提供给 Planner 的工具 schema，包括工具名、描述和参数。
- `tools`：工具名到 Python 函数的映射，用于 Tool Node 执行真实工具。

当前主要接口包括：

- `register(...)`：注册工具。
- `get_tool_schemas()`：返回所有工具说明，供 Planner 构造 prompt。
- `call_tool(name, arguments)`：根据工具名和参数执行工具。

该模块保持 domain independent。后续接入 tau2-bench Airline Domain 时，只需要把官方 Airline tools 注册进 `ToolRegistry`，框架本身不修改 Airline 工具实现。

### 4.3 planner.py

`Planner` 是当前 baseline 中唯一调用 LLM 的模块。

它的职责是：

1. 读取当前 `AgentState`。
2. 读取 `ToolRegistry` 中的工具 schema。
3. 构造 prompt。
4. 调用 LLM。
5. 输出下一步决策。

Planner 的输出只有两种：

```json
{
  "type": "tool_call",
  "name": "tool_name",
  "arguments": {}
}
```

或：

```json
{
  "type": "final_answer",
  "final_answer": "answer text"
}
```

Planner 不直接执行工具，也不管理 LangGraph 路由。这样可以保证职责清晰，方便后续替换不同 planning 策略。

本次实现中还修正了状态残留问题：

- 当 Planner 输出 `tool_call` 时，会清空旧的 `final_answer`。
- 当 Planner 输出 `final_answer` 时，会清空旧的 `tool_calls`。

这样可以避免上一轮状态影响 LangGraph 的条件路由。

### 4.4 graph.py

`graph.py` 使用 LangGraph 建立 AgentFlow 的 baseline 工作流。

当前图中只有两个业务节点：

- `planner`：调用 Planner，决定下一步是调用工具还是输出最终答案。
- `tool`：读取 `tool_calls`，调用 `ToolRegistry` 执行工具，并把结果写回 state。

工作流结构为：

```text
START
  ↓
planner
  ↓
if final_answer exists -> END
if tool_calls exists   -> tool
  ↓
planner
  ↓
END
```

Graph 只负责 workflow 控制，不负责 LLM 细节、工具定义或数据库逻辑。

### 4.5 agent.py

`agent.py` 是 AgentFlow 的对外入口。

当前提供 `AgentFlowAgent` 类，外部调用者只需要使用：

```python
from agentflow import AgentFlowAgent, ToolRegistry

agent = AgentFlowAgent(tool_registry=registry)
result = agent.run("user question")
```

`AgentFlowAgent.run()` 内部负责：

1. 把用户输入转成初始 `messages`。
2. 初始化 `AgentState`。
3. 调用 LangGraph 的 `graph.invoke()`。
4. 返回最终 state。

该入口隐藏了 LangGraph 的内部细节，便于后续接入 tau2-bench 或编写 examples。

### 4.6 __init__.py

当前 `__init__.py` 已导出：

```python
from agentflow import AgentFlowAgent, ToolRegistry
```

这样外部使用时不需要关心内部文件路径。

## 5. 安装配置修正

项目使用 `pyproject.toml` 管理依赖。当前依赖包括：

- `langgraph`
- `langchain`
- `langchain-openai`
- `pydantic`
- `python-dotenv`

在执行：

```bash
python -m pip install -e .
```

时，最初出现 setuptools 包发现错误。原因是项目根目录下存在多个顶层目录，例如：

- `agentflow`
- `configs`
- `experiments`

setuptools 无法自动判断哪个目录应该被当作 Python package。

因此在 `pyproject.toml` 中补充了明确的包发现配置：

```toml
[tool.setuptools.packages.find]
include = ["agentflow*"]
exclude = ["configs*", "docs*", "examples*", "experiments*", "tests*"]
```

修正后，项目可以通过 editable 模式成功安装。

## 6. 运行验证

当前已完成两类验证。

### 6.1 语法验证

执行了 Python 编译检查：

```bash
python -m py_compile agentflow/state.py agentflow/tool_registry.py agentflow/planner.py agentflow/graph.py agentflow/agent.py agentflow/__init__.py
```

结果通过，说明当前源码语法正确。

### 6.2 LangGraph 最小流程验证

为了避免真实调用 OpenAI API，使用 mock Planner 和一个简单 `add` 工具验证工作流。

测试逻辑：

1. DummyPlanner 第一次输出工具调用：`add(a=2, b=3)`。
2. Tool Node 调用注册工具，得到结果 `5`。
3. DummyPlanner 第二次读取 `tool_results`，输出最终答案。

验证输出：

```text
final_answer= 5
tool_results= [{'name': 'add', 'arguments': {'a': 2, 'b': 3}, 'result': 5}]
step_count= 2
```

该结果说明 baseline 工作流已经跑通：

```text
Planner -> Tool Node -> Planner -> END
```

### 6.3 包入口验证

验证了安装后的包入口可以正常导入：

```python
from agentflow import AgentFlowAgent, ToolRegistry
from agentflow.graph import build_graph
```

导入结果正常。

## 7. 当前 Baseline 的技术特点

当前 baseline 版本有几个关键特点：

1. 职责边界清晰。Planner 只做决策，ToolRegistry 只管工具，Graph 只管流程，Agent 只管入口。
2. 保持 domain independent。框架不内置 Airline 业务逻辑，后续通过注册 tau2-bench Airline tools 接入。
3. 可运行闭环已经建立。当前已经具备从用户输入到工具调用再到最终答案的完整流程。
4. 便于后续消融实验。Memory、Reflection、Review 等模块可以作为新的 node 或 Planner 前后的增强逻辑加入。

## 8. 当前限制

当前版本仍然是 baseline，因此存在以下限制：

1. Planner 的 JSON 输出依赖 prompt 约束，没有更强的结构化输出校验。
2. Tool Node 当前使用通用异常捕获，把工具错误写入 `tool_results`，还没有细分错误类型。
3. `step_count` 当前用于记录步数，循环上限主要依赖 LangGraph `recursion_limit`。
4. 尚未接入 tau2-bench Airline Domain 的真实工具。
5. 尚未实现 benchmark runner 和 evaluator 对接。
6. 尚未实现 Memory、Reflection、Review 等实验模块。

这些限制符合 baseline 阶段的范围，后续可以逐步扩展。

## 9. 后续开发方向

建议下一步按以下顺序推进：

1. 编写 `examples/` 下的最小运行示例。
2. 接入 tau2-bench Airline tools 到 `ToolRegistry`。
3. 编写 tau2-bench adapter，让 AgentFlow 可以替换官方 Agent。
4. 跑通 Airline Domain 的单条任务。
5. 跑完整 benchmark，并保存 baseline 结果。
6. 在 baseline 基础上加入 Memory、Reflection、Review 等模块。
7. 做消融实验，对比不同模块组合下的 benchmark 表现。

## 10. 总结

当前 AgentFlow 已经完成 baseline framework 的核心搭建。

它已经具备：

- 状态管理
- 工具注册
- LLM Planner
- LangGraph workflow
- 对外 Agent 入口
- 可安装项目配置
- 最小工作流验证

因此，当前版本可以作为后续 tau2-bench Airline Domain 接入和 Agent 能力消融实验的基础版本。
