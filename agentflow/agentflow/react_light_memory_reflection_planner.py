"""ReAct planner with structured memory and conditional reflection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentflow.react_planner import ReActPlanner
from agentflow.state import AgentState

_CONSTRAINT_MARKERS = (
    "need",
    "must",
    "prefer",
    "want",
    "would like",
    "i'd like",
    "budget",
    "date",
    "time",
    "from",
    "to",
    "passenger",
    "card",
    "refund",
    "insurance",
    "certificate",
)
_UNCERTAINTY_MARKERS = (
    "not sure",
    "uncertain",
    "unable",
    "cannot determine",
    "don't know",
    "do not know",
)


@dataclass
class LightweightMemory:
    """Structured working memory: goal, constraints, progress, last action result."""

    goal: str = "unknown"
    constraints: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)
    last_action_result: str = "none"

    def format(self) -> str:
        return json.dumps(
            {
                "goal": self.goal,
                "constraints": self.constraints,
                "progress": self.progress,
                "last_action_result": self.last_action_result,
            },
            ensure_ascii=False,
            indent=2,
        )


class LightweightMemoryBuilder:
    """Builds compact memory without copying the full conversation log."""

    def __init__(self, max_constraints: int = 8, max_progress: int = 8, max_chars: int = 260):
        self.max_constraints = max_constraints
        self.max_progress = max_progress
        self.max_chars = max_chars

    def build(self, messages: list[dict[str, Any]]) -> LightweightMemory:
        memory = LightweightMemory()
        constraints: list[str] = []
        progress: list[str] = []
        last_tool_name = "tool"

        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            lowered = content.lower()

            if role == "user" and content:
                if memory.goal == "unknown":
                    memory.goal = self._shorten(content)
                if any(marker in lowered for marker in _CONSTRAINT_MARKERS):
                    constraints.append(self._shorten(content))

            if role == "assistant" and message.get("tool_calls"):
                for call in message.get("tool_calls") or []:
                    last_tool_name = str(call.get("name") or "tool")
                    args = call.get("arguments") or {}
                    progress.append(self._shorten(f"called {last_tool_name} with {args}"))

            if role == "tool" and content:
                status = "failed" if message.get("error") else "succeeded"
                memory.last_action_result = self._shorten(f"{last_tool_name} {status}: {content}")
                progress.append(self._shorten(f"{last_tool_name} {status}"))

        memory.constraints = constraints[-self.max_constraints :]
        memory.progress = progress[-self.max_progress :]
        return memory

    def _shorten(self, text: str) -> str:
        text = " ".join(text.split())
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars].rstrip() + "..."


class ConditionalReflectionGate:
    """Triggers reflection only on failure or uncertainty."""

    def __init__(self, max_steps: int = 30):
        self.max_steps = max_steps

    def decide(self, messages: list[dict[str, Any]], step_count: int) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        last_tool = next((m for m in reversed(messages) if m.get("role") == "tool"), None)
        if last_tool and last_tool.get("error"):
            reasons.append("latest tool result was an error")

        tool_tail = self._tool_tail(messages)
        if len(tool_tail) >= 2 and tool_tail[-1] == tool_tail[-2]:
            reasons.append("same tool call repeated")

        if step_count >= self.max_steps - 5:
            reasons.append("close to max step budget")

        last_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
        last_content = str((last_assistant or {}).get("content") or "").lower()
        if any(marker in last_content for marker in _UNCERTAINTY_MARKERS):
            reasons.append("assistant expressed uncertainty")

        return bool(reasons), reasons

    @staticmethod
    def _tool_tail(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
        calls: list[tuple[str, str]] = []
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for call in message.get("tool_calls") or []:
                    name = str(call.get("name") or "")
                    args = json.dumps(call.get("arguments") or {}, sort_keys=True)
                    calls.append((name, args))
        return calls[-3:]


class ReActLightMemoryReflectionPlanner(ReActPlanner):
    """ReAct + lightweight structured memory + conditional reflection."""

    def __init__(
        self,
        *args,
        memory_builder: LightweightMemoryBuilder | None = None,
        reflection_gate: ConditionalReflectionGate | None = None,
        max_steps: int = 30,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_builder = memory_builder or LightweightMemoryBuilder()
        self.reflection_gate = reflection_gate or ConditionalReflectionGate(max_steps=max_steps)
        self.max_steps = max_steps

    def build_prompt(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        step_count = state.get("step_count", 0)
        memory = self.memory_builder.build(messages)
        should_reflect, reflection_reasons = self.reflection_gate.decide(messages, step_count)
        tool_schemas = self.tool_registry.get_tool_schemas()
        remaining_steps = max(self.max_steps - step_count, 0)

        if should_reflect:
            reflection_block = """
Conditional reflection is ON because:
{reasons}

Silently repair the next action:
- Use the latest tool result before choosing another tool.
- Avoid repeating the same failed or unnecessary tool call.
- If information is missing, ask exactly one targeted question.
- If a write/state-changing tool succeeded, finish with final_answer.
""".format(reasons="\n".join(f"- {reason}" for reason in reflection_reasons)).strip()
        else:
            reflection_block = "Conditional reflection is OFF. Continue direct ReAct planning without extra self-critique."

        return f"""
You are a domain-independent ReAct agent planner with lightweight structured memory.

Step budget:
- Current planner step: {step_count}
- Maximum planner steps: {self.max_steps}
- Approximate remaining planner steps: {remaining_steps}

Structured memory:
{memory.format()}

Reflection mode:
{reflection_block}

Rules:
- Follow the system/domain policy strictly.
- Use registered tools for facts, state changes, external lookups, and domain-specific operations.
- Use structured memory as compact task state; do not treat the full history as memory.
- Do not invent information that should come from a tool, policy, structured memory, or recent conversation.
- If required information is missing, ask exactly one targeted question.
- If a write/state-changing tool succeeds, provide a concise final_answer and stop.
- Do not claim a state-changing action is complete unless a corresponding write tool succeeded.
- Output ONLY valid JSON. Do not output markdown or extra text.

Allowed JSON outputs:

Tool action:
{{
  "type": "tool_call",
  "reason": "short reason for this action",
  "name": "tool_name",
  "arguments": {{}}
}}

Final response:
{{
  "type": "final_answer",
  "reason": "short reason the task is ready to answer",
  "final_answer": "answer to the user"
}}

Recent conversation and observations:
{json.dumps(messages[-12:], ensure_ascii=False, indent=2)}

Available tools:
{json.dumps(tool_schemas, ensure_ascii=False, indent=2)}
"""

    def plan(self, state: AgentState) -> AgentState:
        prompt = self.build_prompt(state)
        response = self.llm.invoke(prompt)
        result = self._parse_json(response.content)
        next_step = state.get("step_count", 0) + 1

        if result.get("type") == "tool_call":
            return {
                **state,
                "tool_calls": [
                    {
                        "name": result.get("name"),
                        "arguments": result.get("arguments", {}),
                    }
                ],
                "final_answer": "",
                "step_count": next_step,
            }

        if result.get("type") == "final_answer":
            return {
                **state,
                "tool_calls": [],
                "final_answer": result.get("final_answer", ""),
                "step_count": next_step,
            }

        return {**state, "tool_calls": [], "final_answer": response.content, "step_count": next_step}
