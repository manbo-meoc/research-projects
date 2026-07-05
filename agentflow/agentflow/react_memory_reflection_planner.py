"""ReAct planner variants with memory and reflection hooks."""

import json

from agentflow.memory import ConversationMemory
from agentflow.react_planner import ReActPlanner
from agentflow.reflection import ReflectionGuide
from agentflow.state import AgentState


class ReActMemoryPlanner(ReActPlanner):
    """Domain-independent ReAct planner with working memory context."""

    def __init__(self, *args, memory: ConversationMemory | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = memory or ConversationMemory()

    def build_prompt(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        tool_schemas = self.tool_registry.get_tool_schemas()
        memory_context = self.memory.format(messages)

        return f"""
You are a domain-independent ReAct agent planner with working memory.

Use this loop internally:
1. Observe the user request, system/domain policy, working memory, conversation history, and tool results.
2. Choose the next action: call exactly one available tool, or produce the final answer.
3. After a tool result appears, update your working understanding before deciding the next action.

Rules:
- Follow the system/domain policy strictly.
- Use working memory to keep track of user-provided facts, prior tool calls, and tool observations.
- Use registered tools for facts, state changes, external lookups, and domain-specific operations.
- Do not invent information that should come from a tool, policy, memory, or prior conversation.
- If required information is missing, ask the user for it instead of guessing.
- Output ONLY valid JSON. Do not output markdown or extra text.

Working memory:
{memory_context}

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


class ReActReflectionPlanner(ReActPlanner):
    """Domain-independent ReAct planner with prompt-level reflection."""

    def __init__(self, *args, reflection: ReflectionGuide | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reflection = reflection or ReflectionGuide()

    def build_prompt(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        tool_schemas = self.tool_registry.get_tool_schemas()
        reflection_context = self.reflection.format()

        return f"""
You are a domain-independent ReAct agent planner with reflection.

Use this loop internally:
1. Observe the user request, system/domain policy, conversation history, and tool results.
2. Reflect on whether a tool is needed or whether the task is ready for a final answer.
3. Choose the next action: call exactly one available tool, or produce the final answer.

Rules:
- Follow the system/domain policy strictly.
- Use registered tools for facts, state changes, external lookups, and domain-specific operations.
- Do not invent information that should come from a tool, policy, or prior conversation.
- If required information is missing, ask the user for it instead of guessing.
- Output ONLY valid JSON. Do not output markdown or extra text.

Reflection checklist:
{reflection_context}

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


class ReActMemoryReflectionPlanner(ReActPlanner):
    """Domain-independent ReAct planner with working memory and reflection."""

    def __init__(
        self,
        *args,
        memory: ConversationMemory | None = None,
        reflection: ReflectionGuide | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = memory or ConversationMemory()
        self.reflection = reflection or ReflectionGuide()

    def build_prompt(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        tool_schemas = self.tool_registry.get_tool_schemas()
        memory_context = self.memory.format(messages)
        reflection_context = self.reflection.format()

        return f"""
You are a domain-independent ReAct agent planner with working memory and reflection.

Use this loop internally:
1. Observe the user request, system/domain policy, working memory, conversation history, and tool results.
2. Reflect on whether a tool is needed or whether the task is ready for a final answer.
3. Choose the next action: call exactly one available tool, or produce the final answer.
4. After a tool result appears, use that observation to update your working understanding before the next action.

Rules:
- Follow the system/domain policy strictly.
- Use working memory to keep track of user-provided facts, prior tool calls, and tool observations.
- Use registered tools for facts, state changes, external lookups, and domain-specific operations.
- Do not invent information that should come from a tool, policy, memory, or prior conversation.
- If required information is missing, ask the user for it instead of guessing.
- Output ONLY valid JSON. Do not output markdown or extra text.

Working memory:
{memory_context}

Reflection checklist:
{reflection_context}

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
