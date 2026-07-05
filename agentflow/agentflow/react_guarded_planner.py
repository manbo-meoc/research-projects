"""ReAct planner with a minimal post-output guard.

This planner intentionally reuses the original ReAct prompt. It only validates
planner outputs after the LLM response, so it does not add memory, reflection, or
extra reasoning instructions.
"""

from __future__ import annotations

import json
from typing import Any

from agentflow.react_planner import ReActPlanner
from agentflow.state import AgentState

_READ_TOOL_PREFIXES = ("get", "search", "list", "lookup", "find", "check", "status")


class ReActGuardedPlanner(ReActPlanner):
    """Original ReAct planner plus minimal output validation."""

    def plan(self, state: AgentState) -> AgentState:
        prompt = self.build_prompt(state)
        response = self.llm.invoke(prompt)
        result = self._parse_json(response.content)
        guarded = self._guard_result(result, state)

        if guarded.get("type") == "tool_call":
            return {
                **state,
                "tool_calls": [
                    {
                        "name": guarded.get("name"),
                        "arguments": guarded.get("arguments", {}),
                    }
                ],
                "final_answer": "",
            }

        if guarded.get("type") == "final_answer":
            return {
                **state,
                "tool_calls": [],
                "final_answer": guarded.get("final_answer", ""),
            }

        return {**state, "tool_calls": [], "final_answer": response.content}

    def _guard_result(self, result: dict[str, Any], state: AgentState) -> dict[str, Any]:
        if result.get("type") != "tool_call":
            return result

        name = str(result.get("name") or "")
        arguments = self._normalize_arguments(result.get("arguments", {}))
        schema = self.tool_registry.tool_descriptions.get(name)

        if not schema:
            return self._ask_user(
                "I need to use one of the available tools for this request. Could you clarify the exact information or action you want me to handle?"
            )

        missing = self._missing_required_arguments(schema, arguments)
        if missing:
            return self._ask_user(
                "I need one more detail before I can continue: " + ", ".join(missing) + "."
            )

        if self._is_repeating_failed_or_successful_read_tool(name, arguments, state):
            return self._ask_user(
                "I already checked that information. Could you confirm the next detail you want me to act on?"
            )

        return {**result, "name": name, "arguments": arguments}

    @staticmethod
    def _normalize_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _missing_required_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
        parameters = schema.get("parameters") or {}
        required = parameters.get("required") or []
        return [key for key in required if arguments.get(key) in (None, "")]

    @staticmethod
    def _ask_user(message: str) -> dict[str, str]:
        return {"type": "final_answer", "final_answer": message}

    def _is_repeating_failed_or_successful_read_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        state: AgentState,
    ) -> bool:
        if not self._is_read_tool(name):
            return False

        messages = state.get("messages", [])
        last_tool_call = None
        last_tool_result_seen = False
        last_tool_had_error = False

        for message in reversed(messages):
            if message.get("role") == "tool" and not last_tool_result_seen:
                last_tool_result_seen = True
                last_tool_had_error = bool(message.get("error"))
                continue
            if message.get("role") == "assistant" and message.get("tool_calls"):
                calls = message.get("tool_calls") or []
                if calls:
                    last_tool_call = calls[-1]
                    break

        if not last_tool_result_seen or not last_tool_call:
            return False

        last_name = str(last_tool_call.get("name") or "")
        last_args = self._normalize_arguments(last_tool_call.get("arguments", {}))
        same_call = last_name == name and last_args == arguments

        return same_call and (last_tool_had_error or bool(last_args))

    @staticmethod
    def _is_read_tool(name: str) -> bool:
        lowered = name.lower()
        return lowered.startswith(_READ_TOOL_PREFIXES)
