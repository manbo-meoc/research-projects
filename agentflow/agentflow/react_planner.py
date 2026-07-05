"""Domain-independent ReAct-style planner for AgentFlow."""

import json

from agentflow.planner import Planner
from agentflow.state import AgentState


class ReActPlanner(Planner):
    """
    Domain-independent ReAct planner.

    It keeps the same output contract as the baseline Planner: every step returns
    either one tool_call or one final_answer. Domain behavior comes from the
    system/domain policy and the registered tool schemas, not from framework code.
    """

    def build_prompt(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        tool_schemas = self.tool_registry.get_tool_schemas()

        return f"""
You are a domain-independent ReAct agent planner.

Use this loop internally:
1. Observe the user request, system/domain policy, conversation history, and tool results.
2. Choose the next action: call exactly one available tool, or produce the final answer.
3. After a tool result appears, use that observation before deciding the next action.

Rules:
- Follow the system/domain policy strictly.
- Use registered tools for facts, state changes, external lookups, and domain-specific operations.
- Do not invent information that should come from a tool, policy, or prior conversation.
- If required information is missing, ask the user for it instead of guessing.
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

Conversation and observations:
{json.dumps(messages, ensure_ascii=False, indent=2)}

Available tools:
{json.dumps(tool_schemas, ensure_ascii=False, indent=2)}
"""
