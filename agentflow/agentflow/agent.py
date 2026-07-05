#初始化 planner
#初始化 tool registry
#初始化 graph
#外部只需要调用 run()  
from typing import Any

from agentflow.graph import build_graph
from agentflow.planner import Planner
from agentflow.state import AgentState
from agentflow.tool_registry import ToolRegistry


class AgentFlowAgent:
    """
    AgentFlow framework entrypoint.

    External callers should use run() instead of calling LangGraph directly.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        model: str = "deepseek-v4-pro",
        max_steps: int = 10,
    ):
        self.tool_registry = tool_registry or ToolRegistry()
        self.planner = Planner(self.tool_registry, model=model)
        self.graph = build_graph(self.planner, self.tool_registry)
        self.max_steps = max_steps

    def run(self, user_input: str | list[dict[str, Any]]) -> AgentState:
        messages = self._build_initial_messages(user_input)
        initial_state: AgentState = {
            "messages": messages,
            "tool_calls": [],
            "tool_results": [],
            "final_answer": "",
            "step_count": 0,
        }

        return self.graph.invoke(
            initial_state,
            config={"recursion_limit": self.max_steps},
        )

    @staticmethod
    def _build_initial_messages(
        user_input: str | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(user_input, str):
            return [{"role": "user", "content": user_input}]

        return user_input
