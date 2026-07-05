"""
tau2-bench adapter for AgentFlow baseline.

This module lets tau2-bench use AgentFlow as the assistant-side agent while
keeping tau2's official user simulator, environment tools, and evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from agentflow.planner import Planner
from agentflow.react_planner import ReActPlanner
from agentflow.react_airline_skill_planner import ReActAirlineSkillPlanner
from agentflow.react_step_control_planner import ReActStepControlPlanner
from agentflow.react_guarded_planner import ReActGuardedPlanner
from agentflow.react_light_memory_reflection_planner import ReActLightMemoryReflectionPlanner
from agentflow.react_memory_reflection_planner import (
    ReActMemoryPlanner,
    ReActMemoryReflectionPlanner,
    ReActReflectionPlanner,
)
from agentflow.state import AgentState
from agentflow.tool_registry import ToolRegistry

try:
    from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
    from tau2.data_model.message import (
        APICompatibleMessage,
        AssistantMessage,
        Message,
        MultiToolMessage,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.environment.tool import Tool
except ImportError as exc:  # pragma: no cover - tau2 is an optional integration.
    raise ImportError(
        "agentflow.tau2_adapter requires tau2-bench to be installed or available "
        "on PYTHONPATH."
    ) from exc


@dataclass
class AgentFlowTau2State:
    """Conversation state kept by tau2 between agent turns."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage] = field(default_factory=list)
    step_count: int = 0


class AgentFlowTau2Agent(HalfDuplexAgent[AgentFlowTau2State]):
    """
    Half-duplex tau2 agent backed by AgentFlow's baseline Planner.

    tau2 owns tool execution. AgentFlow only decides whether the next assistant
    message should be a tool call or a final text response.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "deepseek-v4-pro",
        llm_args: Optional[dict[str, Any]] = None,
        planner: Optional[Planner] = None,
        planner_cls: type[Planner] = Planner,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {}
        self.tool_registry = self._build_tool_registry(tools)
        self.planner = planner or planner_cls(self.tool_registry, model=llm, **self.llm_args)

    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None,
    ) -> AgentFlowTau2State:
        system_prompt = (
            "You are a customer service agent running inside tau2-bench.\n\n"
            "Follow the domain policy strictly. Use tools when needed. "
            "When a tool result answers the task, respond to the user clearly.\n\n"
            f"## Domain Policy\n{self.domain_policy}"
        )
        return AgentFlowTau2State(
            system_messages=[SystemMessage(role="system", content=system_prompt)],
            messages=list(message_history) if message_history else [],
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: AgentFlowTau2State,
    ) -> tuple[AssistantMessage, AgentFlowTau2State]:
        self._append_incoming_message(state, message)

        agent_state: AgentState = {
            "messages": self._to_agentflow_messages(
                [*state.system_messages, *state.messages]
            ),
            "tool_calls": [],
            "tool_results": [],
            "final_answer": "",
            "step_count": state.step_count,
        }
        next_state = self.planner.plan(agent_state)
        state.step_count = next_state.get("step_count", state.step_count + 1)

        response = self._to_tau2_assistant_message(next_state)
        state.messages.append(response)
        return response, state

    @staticmethod
    def _build_tool_registry(tools: list[Tool]) -> ToolRegistry:
        registry = ToolRegistry()
        for tool in tools:
            function_schema = tool.openai_schema.get("function", {})
            registry.register_schema(
                name=function_schema.get("name", tool.name),
                description=function_schema.get("description", tool.name),
                parameters=function_schema.get("parameters", {}),
            )
        return registry

    @staticmethod
    def _append_incoming_message(
        state: AgentFlowTau2State,
        message: ValidAgentInputMessage,
    ) -> None:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

    @staticmethod
    def _to_agentflow_messages(messages: list[Message]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, MultiToolMessage):
                for tool_message in message.tool_messages:
                    converted.append(
                        AgentFlowTau2Agent._to_agentflow_message(tool_message)
                    )
                continue
            converted.append(AgentFlowTau2Agent._to_agentflow_message(message))
        return converted

    @staticmethod
    def _to_agentflow_message(message: Message) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content or ""}
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.content or ""}
        if isinstance(message, AssistantMessage):
            payload: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                    for tool_call in message.tool_calls
                ]
            return payload
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "id": message.id,
                "content": message.content or "",
                "error": message.error,
                "requestor": message.requestor,
            }
        return {"role": getattr(message, "role", "unknown"), "content": str(message)}

    @staticmethod
    def _to_tau2_assistant_message(state: AgentState) -> AssistantMessage:
        tool_calls = state.get("tool_calls") or []
        if tool_calls:
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=tool_call.get("id") or f"call_{uuid4().hex[:12]}",
                        name=tool_call["name"],
                        arguments=tool_call.get("arguments", {}),
                    )
                    for tool_call in tool_calls
                ],
            )

        final_answer = state.get("final_answer") or "I could not produce an answer."
        return AssistantMessage.text(final_answer)


class AgentFlowReActTau2Agent(AgentFlowTau2Agent):
    """tau2 adapter that uses AgentFlow's ReActPlanner."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "deepseek-v4-pro",
        llm_args: Optional[dict[str, Any]] = None,
        planner: Optional[Planner] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            planner=planner,
            planner_cls=ReActPlanner,
        )


class AgentFlowReActAirlineSkillTau2Agent(AgentFlowTau2Agent):
    """tau2 adapter that uses ReAct with optional Airline workflow skill."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "deepseek-v4-pro",
        llm_args: Optional[dict[str, Any]] = None,
        planner: Optional[Planner] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            planner=planner,
            planner_cls=ReActAirlineSkillPlanner,
        )


class AgentFlowReActStepControlTau2Agent(AgentFlowTau2Agent):
    """tau2 adapter that uses ReAct with step-aware execution control."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "deepseek-v4-pro",
        llm_args: Optional[dict[str, Any]] = None,
        planner: Optional[Planner] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            planner=planner,
            planner_cls=ReActStepControlPlanner,
        )


class AgentFlowReActGuardedTau2Agent(AgentFlowTau2Agent):
    """tau2 adapter that uses the original ReAct prompt with output guard."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "deepseek-v4-pro",
        llm_args: Optional[dict[str, Any]] = None,
        planner: Optional[Planner] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            planner=planner,
            planner_cls=ReActGuardedPlanner,
        )


class AgentFlowReActMemoryTau2Agent(AgentFlowTau2Agent):
    """tau2 adapter that uses AgentFlow's ReActMemoryPlanner."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "deepseek-v4-pro",
        llm_args: Optional[dict[str, Any]] = None,
        planner: Optional[Planner] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            planner=planner,
            planner_cls=ReActMemoryPlanner,
        )


class AgentFlowReActReflectionTau2Agent(AgentFlowTau2Agent):
    """tau2 adapter that uses AgentFlow's ReActReflectionPlanner."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "deepseek-v4-pro",
        llm_args: Optional[dict[str, Any]] = None,
        planner: Optional[Planner] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            planner=planner,
            planner_cls=ReActReflectionPlanner,
        )


class AgentFlowReActMemoryReflectionTau2Agent(AgentFlowTau2Agent):
    """tau2 adapter that uses AgentFlow's ReActMemoryReflectionPlanner."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "deepseek-v4-pro",
        llm_args: Optional[dict[str, Any]] = None,
        planner: Optional[Planner] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            planner=planner,
            planner_cls=ReActMemoryReflectionPlanner,
        )


class AgentFlowReActLightMemoryReflectionTau2Agent(AgentFlowTau2Agent):
    """tau2 adapter for lightweight structured memory and conditional reflection."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "deepseek-v4-pro",
        llm_args: Optional[dict[str, Any]] = None,
        planner: Optional[Planner] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            planner=planner,
            planner_cls=ReActLightMemoryReflectionPlanner,
        )


def create_agentflow_tau2_agent(tools, domain_policy, **kwargs):
    """Factory function compatible with tau2.registry.register_agent_factory."""
    return AgentFlowTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm") or "deepseek-v4-pro",
        llm_args=kwargs.get("llm_args"),
    )


def register_agentflow_tau2_agent(name: str = "agentflow_baseline") -> None:
    """Register AgentFlow baseline in tau2's global registry."""
    from tau2.registry import registry

    if name not in registry.get_agents():
        registry.register_agent_factory(create_agentflow_tau2_agent, name)


def create_agentflow_react_tau2_agent(tools, domain_policy, **kwargs):
    """Factory function for the basic ReAct tau2 agent."""
    return AgentFlowReActTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm") or "deepseek-v4-pro",
        llm_args=kwargs.get("llm_args"),
    )


def register_agentflow_react_tau2_agent(name: str = "agentflow_react") -> None:
    """Register AgentFlow ReAct baseline in tau2's global registry."""
    from tau2.registry import registry

    if name not in registry.get_agents():
        registry.register_agent_factory(create_agentflow_react_tau2_agent, name)


def create_agentflow_react_memory_tau2_agent(tools, domain_policy, **kwargs):
    """Factory function for the ReAct + Memory tau2 agent."""
    return AgentFlowReActMemoryTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm") or "deepseek-v4-pro",
        llm_args=kwargs.get("llm_args"),
    )


def register_agentflow_react_memory_tau2_agent(
    name: str = "agentflow_react_memory",
) -> None:
    """Register AgentFlow ReAct + Memory in tau2's global registry."""
    from tau2.registry import registry

    if name not in registry.get_agents():
        registry.register_agent_factory(create_agentflow_react_memory_tau2_agent, name)


def create_agentflow_react_reflection_tau2_agent(tools, domain_policy, **kwargs):
    """Factory function for the ReAct + Reflection tau2 agent."""
    return AgentFlowReActReflectionTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm") or "deepseek-v4-pro",
        llm_args=kwargs.get("llm_args"),
    )


def register_agentflow_react_reflection_tau2_agent(
    name: str = "agentflow_react_reflection",
) -> None:
    """Register AgentFlow ReAct + Reflection in tau2's global registry."""
    from tau2.registry import registry

    if name not in registry.get_agents():
        registry.register_agent_factory(
            create_agentflow_react_reflection_tau2_agent, name
        )


def create_agentflow_react_memory_reflection_tau2_agent(
    tools, domain_policy, **kwargs
):
    """Factory function for the ReAct + Memory + Reflection tau2 agent."""
    return AgentFlowReActMemoryReflectionTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm") or "deepseek-v4-pro",
        llm_args=kwargs.get("llm_args"),
    )


def register_agentflow_react_memory_reflection_tau2_agent(
    name: str = "agentflow_react_memory_reflection",
) -> None:
    """Register AgentFlow ReAct + Memory + Reflection in tau2's global registry."""
    from tau2.registry import registry

    if name not in registry.get_agents():
        registry.register_agent_factory(
            create_agentflow_react_memory_reflection_tau2_agent, name
        )



def create_agentflow_react_light_memory_reflection_tau2_agent(
    tools, domain_policy, **kwargs
):
    """Factory function for lightweight ReAct + Memory + Conditional Reflection."""
    return AgentFlowReActLightMemoryReflectionTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm") or "deepseek-v4-pro",
        llm_args=kwargs.get("llm_args"),
    )


def register_agentflow_react_light_memory_reflection_tau2_agent(
    name: str = "agentflow_react_light_memory_reflection",
) -> None:
    """Register lightweight AgentFlow ReAct + Memory + Conditional Reflection."""
    from tau2.registry import registry

    if name not in registry.get_agents():
        registry.register_agent_factory(
            create_agentflow_react_light_memory_reflection_tau2_agent, name
        )


def create_agentflow_react_guarded_tau2_agent(tools, domain_policy, **kwargs):
    """Factory function for ReAct with a minimal output guard."""
    return AgentFlowReActGuardedTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm") or "deepseek-v4-pro",
        llm_args=kwargs.get("llm_args"),
    )


def register_agentflow_react_guarded_tau2_agent(
    name: str = "agentflow_react_guarded",
) -> None:
    """Register AgentFlow ReAct + minimal output guard."""
    from tau2.registry import registry

    if name not in registry.get_agents():
        registry.register_agent_factory(create_agentflow_react_guarded_tau2_agent, name)


def create_agentflow_react_step_control_tau2_agent(tools, domain_policy, **kwargs):
    """Factory function for ReAct with step-aware execution control."""
    return AgentFlowReActStepControlTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm") or "deepseek-v4-pro",
        llm_args=kwargs.get("llm_args"),
    )


def register_agentflow_react_step_control_tau2_agent(
    name: str = "agentflow_react_step_control",
) -> None:
    """Register AgentFlow ReAct + step-aware execution control."""
    from tau2.registry import registry

    if name not in registry.get_agents():
        registry.register_agent_factory(create_agentflow_react_step_control_tau2_agent, name)


def create_agentflow_react_airline_skill_tau2_agent(tools, domain_policy, **kwargs):
    """Factory function for ReAct with optional Airline workflow skill."""
    return AgentFlowReActAirlineSkillTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm") or "deepseek-v4-pro",
        llm_args=kwargs.get("llm_args"),
    )


def register_agentflow_react_airline_skill_tau2_agent(
    name: str = "agentflow_react_airline_skill",
) -> None:
    """Register AgentFlow ReAct + Airline workflow skill."""
    from tau2.registry import registry

    if name not in registry.get_agents():
        registry.register_agent_factory(create_agentflow_react_airline_skill_tau2_agent, name)
