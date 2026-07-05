"""Lightweight domain-independent memory helpers for AgentFlow."""

from __future__ import annotations

from typing import Any


class ConversationMemory:
    """
    Builds a compact working memory from the current conversation state.

    This baseline memory is intentionally simple and domain independent. It does
    not store external state; it extracts user-provided facts, assistant tool
    calls, and tool observations from the conversation passed into each turn.
    """

    def __init__(self, max_items: int = 12, max_chars_per_item: int = 500):
        self.max_items = max_items
        self.max_chars_per_item = max_chars_per_item

    def build(self, messages: list[dict[str, Any]]) -> list[str]:
        items: list[str] = []
        for message in messages:
            role = message.get("role", "unknown")
            content = str(message.get("content") or "").strip()

            if role == "user" and content:
                items.append(f"User-provided information: {content}")
            elif role == "assistant" and message.get("tool_calls"):
                items.append(f"Assistant requested tool call: {message['tool_calls']}")
            elif role == "tool" and content:
                error_marker = " error" if message.get("error") else ""
                items.append(f"Tool observation{error_marker}: {content}")

        compacted = []
        for item in items[-self.max_items :]:
            if len(item) > self.max_chars_per_item:
                item = item[: self.max_chars_per_item].rstrip() + "..."
            compacted.append(item)
        return compacted

    def format(self, messages: list[dict[str, Any]]) -> str:
        items = self.build(messages)
        if not items:
            return "No working memory yet."
        return "\n".join(f"- {item}" for item in items)
