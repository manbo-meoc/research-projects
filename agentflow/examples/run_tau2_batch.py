#!/usr/bin/env python3
"""Run multiple tau2-bench tasks with a selectable AgentFlow agent."""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Run multiple tau2 tasks with AgentFlow")
    parser.add_argument("--domain", default="airline", help="tau2 domain name")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=["0", "1", "2"],
        help="Task ids to run, or use: --task-ids all",
    )
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
        "--run-name",
        default=None,
        help="Directory name for this run.",
    )
    parser.add_argument(
        "--save-root",
        default="experiments/tau2_batch/runs",
        help="Root directory for batch results.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks whose task_<id>.json result file already exists.",
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


def main() -> None:
    args = parse_args()
    register_agent(args.agent_name)

    run_name = args.run_name or datetime.now().strftime(
        f"%Y%m%d_%H%M%S_{args.domain}_{args.agent_name}"
    )
    save_root = Path(args.save_root)
    if not save_root.is_absolute():
        save_root = PROJECT_ROOT / save_root
    save_dir = save_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_path = save_dir / "summary.json"
    summary = []
    if args.resume and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = []

    task_ids = args.task_ids
    if len(task_ids) == 1 and task_ids[0].lower() == "all":
        task_ids = [task.id for task in get_tasks(args.domain)]

    completed_task_ids = {str(item.get("task_id")) for item in summary}

    for task_id in task_ids:
        result_path = save_dir / f"task_{task_id}.json"
        if args.resume and result_path.exists():
            print("\n" + "=" * 72)
            print(f"Skipping existing {args.domain} task {task_id}: {result_path}")
            print("=" * 72)
            if str(task_id) not in completed_task_ids:
                try:
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                    reward_info = existing.get("reward_info") or {}
                    messages = existing.get("messages") or []
                    summary.append(
                        {
                            "domain": args.domain,
                            "task_id": task_id,
                            "agent": args.agent_name,
                            "reward": reward_info.get("reward"),
                            "messages": len(messages),
                            "saved_to": str(result_path),
                        }
                    )
                    completed_task_ids.add(str(task_id))
                except json.JSONDecodeError:
                    pass
            continue

        print("\n" + "=" * 72)
        print(f"Running {args.domain} task {task_id} with {args.agent_name}")
        print("=" * 72)

        tasks = get_tasks(args.domain, task_ids=[task_id])
        if not tasks:
            raise ValueError(f"Task not found: domain={args.domain}, task_id={task_id}")

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

        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        reward = result.reward_info.reward if result.reward_info else None
        item = {
            "domain": args.domain,
            "task_id": task_id,
            "agent": args.agent_name,
            "reward": reward,
            "messages": len(result.messages),
            "saved_to": str(result_path),
        }
        summary.append(item)
        completed_task_ids.add(str(task_id))
        print(f"reward: {reward}")
        print(f"messages: {len(result.messages)}")
        print(f"saved_to: {result_path}")

    summary.sort(key=lambda item: str(item.get("task_id")))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== AgentFlow tau2 batch summary =====")
    for item in summary:
        print(
            f"task={item['task_id']} agent={item['agent']} "
            f"reward={item['reward']} messages={item['messages']}"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
