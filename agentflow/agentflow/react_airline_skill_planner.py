"""ReAct planner augmented with optional Airline workflow skill hints."""

from __future__ import annotations

from typing import Any

from agentflow.airline_skill import AirlineWorkflowSkill
from agentflow.react_planner import ReActPlanner
from agentflow.state import AgentState

_READ_TOOL_PREFIXES = ("get", "search", "list", "lookup", "find", "check", "status")


class ReActAirlineSkillPlanner(ReActPlanner):
    """ReAct with adaptive airline workflow decomposition hints.

    The skill stays off for normal short tasks and activates only when the
    conversation starts to look long-horizon or error-prone.
    """

    def __init__(
        self,
        *args,
        skill: AirlineWorkflowSkill | None = None,
        activation_step: int = 12,
        activation_read_count: int = 6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.skill = skill or AirlineWorkflowSkill()
        self.activation_step = activation_step
        self.activation_read_count = activation_read_count

    def build_prompt(self, state: AgentState) -> str:
        prompt = super().build_prompt(state)
        messages = state.get("messages", [])
        step_count = state.get("step_count", 0)

        if not self._should_activate_skill(messages, step_count):
            return prompt

        hint = self.skill.analyze(messages, step_count=step_count)
        return f"""{prompt}

Optional Airline workflow skill hint:
{hint.format()}

Use the skill hint only as workflow guidance for the Airline domain. The system/domain policy and actual tool observations remain authoritative.
"""

    def _should_activate_skill(self, messages: list[dict[str, Any]], step_count: int) -> bool:
        return (
            step_count >= self.activation_step
            or self._read_tool_count(messages) >= self.activation_read_count
            or self._last_tool_had_error(messages)
        )

    @staticmethod
    def _read_tool_count(messages: list[dict[str, Any]]) -> int:
        count = 0
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for call in message.get("tool_calls") or []:
                    name = str(call.get("name") or "").lower()
                    if name.startswith(_READ_TOOL_PREFIXES) or "search" in name:
                        count += 1
        return count

    @staticmethod
    def _last_tool_had_error(messages: list[dict[str, Any]]) -> bool:
        for message in reversed(messages):
            if message.get("role") == "tool":
                return bool(message.get("error"))
        return False
