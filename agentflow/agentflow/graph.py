#流程控制器：控制 AI 怎么一步一步执行  START → planner → tool → planner → END
from typing import Any

from langgraph.graph import END, START, StateGraph

from agentflow.planner import Planner
from agentflow.state import AgentState
from agentflow.tool_registry import ToolRegistry


def build_graph(planner: Planner, tool_registry: ToolRegistry):
    """
    Build the AgentFlow LangGraph workflow.

    Workflow:
    START -> planner -> tool_node -> planner -> END
    """
    graph = StateGraph(AgentState)

    def planner_node(state: AgentState) -> AgentState:
        step_count = state.get("step_count", 0) + 1
        next_state = planner.plan({**state, "step_count": step_count})
        return next_state

    def tool_node(state: AgentState) -> AgentState:
        tool_calls = state.get("tool_calls", [])
        tool_results = list(state.get("tool_results", []))
        messages = list(state.get("messages", []))

        for tool_call in tool_calls:
            name = tool_call["name"]
            arguments: dict[str, Any] = tool_call.get("arguments", {})

            try:
                result = tool_registry.call_tool(name, arguments)
                tool_result = {
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                }
            except Exception as exc:  # noqa: BLE001
                tool_result = {
                    "name": name,
                    "arguments": arguments,
                    "error": str(exc),
                }

            tool_results.append(tool_result)
            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": str(tool_result),
                }
            )

        return {
            **state,
            "messages": messages,
            "tool_results": tool_results,
            "tool_calls": [],
        }

    def route_after_planner(state: AgentState) -> str:
        if state.get("final_answer"):
            return "end"
        if state.get("tool_calls"):
            return "tool"
        return "end"

    graph.add_node("planner", planner_node)
    graph.add_node("tool", tool_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "tool": "tool",
            "end": END,
        },
    )
    graph.add_edge("tool", "planner")

    return graph.compile()
