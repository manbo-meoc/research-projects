"""Optional airline-domain workflow skill for AgentFlow.

This module is intentionally outside the framework core. It provides compact
workflow hints for tau2 Airline tasks so long conversations can be treated as
shorter business phases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_READ_PREFIXES = ("get", "search", "list", "lookup", "find", "check", "status")
_WRITE_MARKERS = (
    "book",
    "cancel",
    "update",
    "send_certificate",
    "certificate",
    "refund",
    "pay",
)
_CONFIRM_MARKERS = (
    "yes",
    "confirm",
    "confirmed",
    "go ahead",
    "please do",
    "proceed",
    "that works",
    "sounds good",
    "book it",
    "cancel it",
    "send it",
    "keep the current",
)


@dataclass
class AirlineSkillHint:
    intent: str = "unknown"
    phase: str = "understand_request"
    known_info: dict[str, Any] = field(default_factory=dict)
    current_subgoal: str = "Infer the user's next required airline workflow step."
    next_allowed_actions: list[str] = field(default_factory=lambda: ["tool_call", "final_answer"])
    stop_conditions: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)

    def format(self) -> str:
        return json.dumps(
            {
                "intent": self.intent,
                "phase": self.phase,
                "known_info": self.known_info,
                "current_subgoal": self.current_subgoal,
                "next_allowed_actions": self.next_allowed_actions,
                "stop_conditions": self.stop_conditions,
                "must_not": self.must_not,
            },
            ensure_ascii=False,
            indent=2,
        )


class AirlineWorkflowSkill:
    """Builds compact workflow hints for Airline domain conversations."""

    def __init__(self, read_budget: int = 7, pressure_step: int = 22):
        self.read_budget = read_budget
        self.pressure_step = pressure_step

    def analyze(self, messages: list[dict[str, Any]], step_count: int = 0) -> AirlineSkillHint:
        text = self._conversation_text(messages)
        tool_calls = self._tool_calls(messages)
        last_tool = self._last_tool(messages)
        known = self._known_info(messages, tool_calls)
        intent = self._infer_intent(text, tool_calls)
        user_confirmed = self._latest_user_confirmed(messages)
        read_count = sum(1 for name, _ in tool_calls if self._is_read_tool(name))
        write_success = bool(last_tool and self._is_write_tool(last_tool[0]) and last_tool[2])

        hint = AirlineSkillHint(intent=intent, known_info=known)

        if write_success:
            hint.phase = "finalize"
            hint.current_subgoal = "A state-changing airline tool just succeeded; summarize the completed action and stop."
            hint.next_allowed_actions = ["final_answer"]
            hint.stop_conditions = ["Do not call more tools after successful cancel/book/update/certificate unless the user starts a separate task."]
            hint.must_not = ["Do not continue searching reservations or flights after a successful write tool."]
            return hint

        if intent in {"cancel_reservation", "change_or_rebook"} and not known.get("reservation_id"):
            hint.phase = "read_state"
            hint.current_subgoal = "Identify the relevant reservation before taking action."
            hint.next_allowed_actions = ["get_user_details", "get_reservation_details", "ask_user"]
            hint.must_not = ["Do not cancel, update, or book until the target reservation is identified."]
            return hint

        if intent == "compensation" and not known.get("user_id"):
            hint.phase = "read_state"
            hint.current_subgoal = "Identify the user and relevant delayed/cancelled flight before compensation."
            hint.next_allowed_actions = ["get_user_details", "get_reservation_details", "get_flight_status", "ask_user"]
            hint.must_not = ["Do not send a certificate before identifying the user and reason for compensation."]
            return hint

        confirmed_write_intent = intent in {
            "cancel_reservation",
            "change_or_rebook",
            "book_flight",
            "compensation",
        } and user_confirmed

        if intent in {"cancel_reservation", "change_or_rebook", "book_flight", "compensation"} and not user_confirmed:
            hint.phase = "ask_confirmation"
            hint.current_subgoal = "Use known reservation/flight/payment facts to ask for one clear confirmation or missing detail."
            hint.next_allowed_actions = ["final_answer", "ask_user"]
            hint.stop_conditions = ["Once the user confirms, execute the required write tool rather than searching again."]
            hint.must_not = ["Do not perform a state-changing tool call without enough user confirmation required by policy."]
        else:
            hint.phase = "execute_write" if intent != "unknown" else "read_state"
            hint.current_subgoal = self._execution_subgoal(intent)
            hint.next_allowed_actions = self._allowed_actions(intent)
            hint.stop_conditions = ["After the write tool succeeds, final_answer immediately."]
            hint.must_not = [
                "Do not restart broad reservation search when the target reservation or user intent is already known."
            ]
            if confirmed_write_intent:
                hint.current_subgoal = (
                    "The user has already confirmed/proceeded. Execute the appropriate write tool now if required arguments are known; "
                    "do not ask for confirmation again."
                )
                hint.must_not.append("Do not ask the user to confirm the same booking/change/cancellation again.")

        if read_count >= self.read_budget or step_count >= self.pressure_step:
            if confirmed_write_intent:
                hint.must_not.append("Do not call another broad get/search/list tool unless a required write-tool argument is still missing.")
            else:
                hint.phase = "converge"
                hint.current_subgoal = "The conversation is long; stop broad exploration and either execute the known next action, ask one targeted question, or final_answer."
                hint.must_not.append("Do not call another broad get/search/list tool unless it is the only missing required fact.")

        if last_tool and not last_tool[2]:
            hint.current_subgoal = "Repair the latest tool error once using the error message, or ask one targeted question."
            hint.must_not.append("Do not repeat the same failed tool call with unchanged arguments.")

        return hint

    def _known_info(self, messages: list[dict[str, Any]], tool_calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        known: dict[str, Any] = {}
        for _, args in tool_calls:
            for key in ("user_id", "reservation_id", "payment_id", "flight_number", "origin", "destination", "date", "cabin"):
                if key in args and args[key]:
                    known[key] = args[key]
        for message in messages:
            content = str(message.get("content") or "")
            self._extract_ids(content, known)
        return known

    @staticmethod
    def _extract_ids(text: str, known: dict[str, Any]) -> None:
        reservation_ids = re.findall(r"\b[A-Z0-9]{6}\b", text)
        if reservation_ids:
            known.setdefault("reservation_id", reservation_ids[-1])
        user_ids = re.findall(r"\b[a-z]+_[a-z]+_\d{4}\b", text)
        if user_ids:
            known.setdefault("user_id", user_ids[-1])
        payment_ids = re.findall(r"\b(?:credit_card|gift_card|certificate)_\d+\b", text)
        if payment_ids:
            known.setdefault("payment_id", payment_ids[-1])

    @staticmethod
    def _infer_intent(text: str, tool_calls: list[tuple[str, dict[str, Any]]]) -> str:
        lowered = text.lower()
        names = " ".join(name for name, _ in tool_calls).lower()
        combined = lowered + " " + names
        if "certificate" in combined or "compensation" in combined or "delay" in combined:
            return "compensation"
        if "cancel" in combined:
            return "cancel_reservation"
        if "rebook" in combined or "change" in combined or "update_reservation" in combined:
            return "change_or_rebook"
        if "book" in combined or "search_direct_flight" in combined or "search_onestop_flight" in combined:
            return "book_flight"
        if "refund" in combined or "payment" in combined:
            return "payment_or_refund"
        return "unknown"

    @staticmethod
    def _execution_subgoal(intent: str) -> str:
        return {
            "cancel_reservation": "Execute cancel_reservation only for the confirmed target reservation, then finalize.",
            "change_or_rebook": "Execute the required update/book/cancel action for the confirmed itinerary, then finalize.",
            "book_flight": "Book the confirmed flight option with known passenger and payment details, then finalize.",
            "compensation": "If compensation is confirmed and user_id is known, send_certificate once, then finalize.",
            "payment_or_refund": "Use existing reservation/payment observations to explain or execute the required payment/refund step.",
        }.get(intent, "Read the minimum missing state, then ask one targeted question or answer.")

    @staticmethod
    def _allowed_actions(intent: str) -> list[str]:
        return {
            "cancel_reservation": ["cancel_reservation", "ask_user", "final_answer"],
            "change_or_rebook": ["update_reservation_flights", "book_reservation", "cancel_reservation", "ask_user", "final_answer"],
            "book_flight": ["book_reservation", "ask_user", "final_answer"],
            "compensation": ["send_certificate", "ask_user", "final_answer"],
            "payment_or_refund": ["ask_user", "final_answer"],
        }.get(intent, ["tool_call", "ask_user", "final_answer"])

    @staticmethod
    def _conversation_text(messages: list[dict[str, Any]]) -> str:
        return "\n".join(str(message.get("content") or "") for message in messages if message.get("role") in {"user", "assistant", "tool"})[-8000:]

    @staticmethod
    def _tool_calls(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        calls: list[tuple[str, dict[str, Any]]] = []
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for call in message.get("tool_calls") or []:
                    args = call.get("arguments") or {}
                    calls.append((str(call.get("name") or ""), args if isinstance(args, dict) else {}))
        return calls

    @staticmethod
    def _last_tool(messages: list[dict[str, Any]]) -> tuple[str, str, bool] | None:
        last_name = ""
        for message in reversed(messages):
            if message.get("role") == "tool":
                success = bool(message.get("content")) and not bool(message.get("error"))
                return (last_name, str(message.get("content") or ""), success)
            if message.get("role") == "assistant" and message.get("tool_calls"):
                calls = message.get("tool_calls") or []
                if calls:
                    last_name = str(calls[-1].get("name") or "")
        return None

    @staticmethod
    def _latest_user_confirmed(messages: list[dict[str, Any]]) -> bool:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = str(message.get("content") or "").lower()
                return any(marker in content for marker in _CONFIRM_MARKERS)
        return False

    @staticmethod
    def _is_read_tool(name: str) -> bool:
        return name.lower().startswith(_READ_PREFIXES) or "search" in name.lower()

    @staticmethod
    def _is_write_tool(name: str) -> bool:
        lowered = name.lower()
        if lowered.startswith(_READ_PREFIXES):
            return False
        return any(marker in lowered for marker in _WRITE_MARKERS)
