#!/usr/bin/env python3
"""Run one tau2-bench task with a selectable AgentFlow agent."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    if os.environ.get(key, "").startswith("socks://"):
        os.environ.pop(key)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TAU2_ROOT = Path(os.environ.get("TAU2_ROOT", "../tau2-bench")).resolve()
TAU2_SRC = TAU2_ROOT / "src"
if TAU2_SRC.exists() and str(TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(TAU2_SRC))

from agentflow.tau2_adapter import (  # noqa: E402
    register_agentflow_react_airline_skill_tau2_agent,
    register_agentflow_react_guarded_tau2_agent,
    register_agentflow_react_light_memory_reflection_tau2_agent,
    register_agentflow_react_memory_reflection_tau2_agent,
    register_agentflow_react_memory_tau2_agent,
    register_agentflow_react_reflection_tau2_agent,
    register_agentflow_react_step_control_tau2_agent,
    register_agentflow_react_tau2_agent,
    register_agentflow_tau2_agent,
)
from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.runner import get_tasks, run_single_task  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one tau2 task with AgentFlow")
    parser.add_argument("--domain", default="airline", help="tau2 domain name")
    parser.add_argument("--task-id", default="0", help="single tau2 task id")
    parser.add_argument(
        "--agent-name",
        choices=["agentflow_baseline", "agentflow_react", "agentflow_react_airline_skill", "agentflow_react_step_control", "agentflow_react_guarded", "agentflow_react_memory", "agentflow_react_reflection", "agentflow_react_memory_reflection", "agentflow_react_light_memory_reflection"],
        default="agentflow_baseline",
    )
    parser.add_argument("--llm-agent", default="deepseek-v4-pro")
    parser.add_argument("--llm-user", default="deepseek/deepseek-chat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument(
        "--save-to",
        default=None,
        help="Path to save the SimulationRun JSON. Defaults to experiments/tau2_single/runs/.",
    )
    return parser.parse_args()


def register_agent(agent_name: str) -> None:
    if agent_name == "agentflow_baseline":
        register_agentflow_tau2_agent(agent_name)
        return
    if agent_name == "agentflow_react":
        register_agentflow_react_tau2_agent(agent_name)
        return
    if agent_name == "agentflow_react_airline_skill":
        register_agentflow_react_airline_skill_tau2_agent(agent_name)
        return
    if agent_name == "agentflow_react_step_control":
        register_agentflow_react_step_control_tau2_agent(agent_name)
        return
    if agent_name == "agentflow_react_guarded":
        register_agentflow_react_guarded_tau2_agent(agent_name)
        return
    if agent_name == "agentflow_react_memory":
        register_agentflow_react_memory_tau2_agent(agent_name)
        return
    if agent_name == "agentflow_react_reflection":
        register_agentflow_react_reflection_tau2_agent(agent_name)
        return
    if agent_name == "agentflow_react_memory_reflection":
        register_agentflow_react_memory_reflection_tau2_agent(agent_name)
        return
    if agent_name == "agentflow_react_light_memory_reflection":
        register_agentflow_react_light_memory_reflection_tau2_agent(agent_name)
        return
    raise ValueError(f"Unsupported AgentFlow agent: {agent_name}")


def default_save_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{args.domain}_{args.task_id}_{args.agent_name}.json"
    return PROJECT_ROOT / "experiments" / "tau2_single" / "runs" / filename


def save_result(result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    register_agent(args.agent_name)

    tasks = get_tasks(args.domain, task_ids=[args.task_id])
    if not tasks:
        raise ValueError(f"Task not found: domain={args.domain}, task_id={args.task_id}")

    result = run_single_task(
        TextRunConfig(
            domain=args.domain,
            agent=args.agent_name,
            llm_agent=args.llm_agent,
            llm_user=args.llm_user,
            max_steps=args.max_steps,
        ),
        tasks[0],
        seed=args.seed,
    )

    save_path = Path(args.save_to) if args.save_to else default_save_path(args)
    if not save_path.is_absolute():
        save_path = PROJECT_ROOT / save_path
    save_result(result, save_path)

    print("\n===== AgentFlow tau2 single result =====")
    print(f"domain: {args.domain}")
    print(f"task_id: {result.task_id}")
    print(f"agent: {args.agent_name}")
    print(f"reward: {result.reward_info.reward if result.reward_info else None}")
    print(f"messages: {len(result.messages)}")
    print(f"saved_to: {save_path}")


if __name__ == "__main__":
    main()
