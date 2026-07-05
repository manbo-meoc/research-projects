state.py
保存 Agent 运行中的状态，比如 messages、tool_calls、tool_results。

planner.py
负责让 LLM 判断下一步：回答用户，还是调用工具。

tool_registry.py
统一管理工具，外部 domain tools 会通过这里接入。

graph.py
用 LangGraph 把 Planner、Tool 等节点串成工作流。

agent.py
对外入口，外部只需要调用 AgentFlowAgent.run()。
