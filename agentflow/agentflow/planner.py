# ChatOpenAI -> decide tool / answer, output tool_call / final_answer

import json
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agentflow.state import AgentState
from agentflow.tool_registry import ToolRegistry


class Planner:
    """
    AgentFlow baseline planner.

    The planner is the only module that calls the LLM. It returns either a
    tool_call decision or a final_answer decision as JSON-compatible state.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        model: str = "deepseek-v4-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0,
        **llm_kwargs: Any,
    ):
        self.tool_registry = tool_registry
        resolved_base_url = base_url
        resolved_api_key = api_key

        if resolved_base_url is None and model.startswith("deepseek"):
            resolved_base_url = "https://api.deepseek.com"

        if resolved_api_key is None:
            if resolved_base_url and "deepseek" in resolved_base_url:
                resolved_api_key = os.getenv("DEEPSEEK_API_KEY")
            else:
                resolved_api_key = os.getenv("OPENAI_API_KEY")

        if resolved_base_url and "deepseek" in resolved_base_url and not resolved_api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY for DeepSeek Planner.")

        self.llm = ChatOpenAI(
            model=model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            temperature=temperature,
            **llm_kwargs,
        )

    def build_prompt(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        tool_schemas = self.tool_registry.get_tool_schemas()

        return f"""
You are a tool-using agent planner.

You MUST output ONLY valid JSON.

You have two options:

1. tool_call:
{{
  "type": "tool_call",
  "name": "tool_name",
  "arguments": {{}}
}}

2. final_answer:
{{
  "type": "final_answer",
  "final_answer": "..."
}}

Conversation history:
{json.dumps(messages, ensure_ascii=False, indent=2)}

Available tools:
{json.dumps(tool_schemas, ensure_ascii=False, indent=2)}
"""

    def _parse_json(self, text: str) -> dict[str, Any]:
        """
        Parse JSON from an LLM response, falling back to a final answer.
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            return {"type": "final_answer", "final_answer": text}

        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {"type": "final_answer", "final_answer": text}

    def plan(self, state: AgentState) -> AgentState:
        prompt = self.build_prompt(state)
        response = self.llm.invoke(prompt)

        result = self._parse_json(response.content)

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
            }

        if result.get("type") == "final_answer":
            return {
                **state,
                "tool_calls": [],
                "final_answer": result.get("final_answer", ""),
            }

        return {**state, "tool_calls": [], "final_answer": response.content}
