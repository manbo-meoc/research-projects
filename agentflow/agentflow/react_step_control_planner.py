"""ReAct planner with step-aware execution control.

The base ReAct prompt is unchanged for normal turns. Extra runtime instruction is
only appended when the run is close to max steps, read/search tools are overused,
or a write tool has just succeeded.
"""

from __future__ import annotations

from typing import Any

from agentflow.react_planner import ReActPlanner
from agentflow.state import AgentState

_READ_TOOL_PREFIXES = ("get", "search", "list", "lookup", "find", "check", "status")
_WRITE_TOOL_MARKERS = (
    "book",
    "cancel",
    "update",
    "create",
    "delete",
    "modify",
    "return",
    "exchange",
    "transfer",
    "pay",
    "submit",
    "send",
)


class ReActStepControlPlanner(ReActPlanner):
    """Original ReAct with small execution-control hints under pressure."""

    def __init__(
        self,
        *args,
        max_steps: int = 30,
        pressure_step: int = 24,
        read_budget: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps
        self.pressure_step = pressure_step
        self.read_budget = read_budget

    def build_prompt(self, state: AgentState) -> str:
        prompt = super().build_prompt(state)
        hints = self._runtime_hints(state)
        if not hints:
            return prompt
        return prompt + "\n\nRuntime execution-control hints:\n" + "\n".join(
            f"- {hint}" for hint in hints
        )

    def _runtime_hints(self, state: AgentState) -> list[str]:
        messages = state.get("messages", [])
        step_count = state.get("step_count", 0)
        hints: list[str] = []

        if self._last_tool_was_successful_write(messages):
            hints.append(
                "The latest state-changing tool succeeded. Do not call more tools unless the user asked for a separate task; provide final_answer now."
            )

        read_count = self._count_read_tools(messages)
        if read_count >= self.read_budget:
            hints.append(
                "Read/search tool budget is high. Avoid further get/search/list/lookup/check calls; use existing observations to answer, take the required final action, or ask one targeted question."
            )

        if step_count >= self.pressure_step:
            remaining = max(self.max_steps - step_count, 0)
            hints.append(
                f"Only about {remaining} planner steps remain. Stop exploring; use existing observations and either execute the necessary final action or produce final_answer."
            )

        if self._last_tool_had_error(messages):
            hints.append(
                "The latest tool returned an error. Do not repeat the same call unchanged; repair the arguments once or ask one targeted question."
            )

        return hints

    @classmethod
    def _last_tool_was_successful_write(cls, messages: list[dict[str, Any]]) -> bool:
        last_tool_call_name = ""
        found_tool_result = False
        last_tool_error = False

        for message in reversed(messages):
            if message.get("role") == "tool" and not found_tool_result:
                found_tool_result = True
                last_tool_error = bool(message.get("error"))
                continue
            if found_tool_result and message.get("role") == "assistant" and message.get("tool_calls"):
                calls = message.get("tool_calls") or []
                if calls:
                    last_tool_call_name = str(calls[-1].get("name") or "")
                    break

        return found_tool_result and not last_tool_error and cls._is_write_tool(last_tool_call_name)

    @classmethod
    def _count_read_tools(cls, messages: list[dict[str, Any]]) -> int:
        count = 0
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for call in message.get("tool_calls") or []:
                    if cls._is_read_tool(str(call.get("name") or "")):
                        count += 1
        return count

    @staticmethod
    def _last_tool_had_error(messages: list[dict[str, Any]]) -> bool:
        for message in reversed(messages):
            if message.get("role") == "tool":
                return bool(message.get("error"))
        return False

    @staticmethod
    def _is_read_tool(name: str) -> bool:
        return name.lower().startswith(_READ_TOOL_PREFIXES)

    @staticmethod
    def _is_write_tool(name: str) -> bool:
        lowered = name.lower()
        if lowered.startswith(_READ_TOOL_PREFIXES):
            return False
        return any(marker in lowered for marker in _WRITE_TOOL_MARKERS)
