from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent import Agent, OnStep
from llm_provider import LLMProvider
from typed_tool import TypedTool

PLAN_SYSTEM_PROMPT = """You are a planning agent for a coding task.
You may ONLY use read_file to inspect the repo. You cannot write files.
Reply with ONLY one JSON object per turn:

- {"tool": "read_file", "args": {"path": "..."}}
- {"done": true, "summary": "<concrete implementation plan>"}

When done, summary must list which files to create/edit and what to put in them.
"""

BUILD_SYSTEM_PROMPT = """You are a coding agent that implements a plan.
You may use read_file and write_file. Reply with ONLY one JSON object per turn:

- {"tool": "read_file", "args": {"path": "..."}}
- {"tool": "write_file", "args": {"path": "...", "content": "..."}}
- {"done": true, "summary": "<what you changed>"}
"""


@dataclass
class RunResult:
    goal: str
    plan: str
    report: str
    status: str
    events: list[dict[str, Any]] = field(default_factory=list)


def format_step(event: dict[str, Any]) -> str:
    phase = event.get("phase", "?")
    step = event.get("step", "?")
    kind = event.get("kind", "?")
    prefix = f"[{phase} {step}] {kind}"
    if kind == "tool":
        args = event.get("args") or {}
        detail = " ".join(f"{k}={v!r}" for k, v in args.items())
        return f"{prefix} {event.get('tool')} {detail}"
    if kind == "done":
        return f"{prefix} {event.get('summary')}"
    if kind == "invalid_json":
        return f"{prefix} {event.get('error')}"
    if kind == "llm":
        raw = event.get("raw") or ""
        return f"{prefix} {raw[:120]}"
    return prefix


def print_step(event: dict[str, Any]) -> None:
    print(format_step(event), flush=True)


class Orchestrator:
    """Two-node hybrid: plan (read-only) then build (read+write)."""

    def __init__(self, provider: LLMProvider, tools: dict[str, TypedTool]):
        self._provider = provider
        self._tools = tools

    def run(
        self,
        goal: str,
        *,
        plan_max_steps: int = 10,
        build_max_steps: int = 20,
        verbose: bool = True,
        on_step: OnStep | None = None,
    ) -> RunResult:
        events: list[dict[str, Any]] = []

        def make_on_step(phase: str) -> Callable[[dict[str, Any]], None]:
            def _on_step(ev: dict[str, Any]) -> None:
                row = {"phase": phase, **ev}
                events.append(row)
                if verbose:
                    print_step(row)
                if on_step:
                    on_step(row)

            return _on_step

        plan_tools = {
            name: tool
            for name, tool in self._tools.items()
            if name == "read_file"
        }
        plan_agent = Agent(
            self._provider,
            tools=plan_tools,
            system_prompt=PLAN_SYSTEM_PROMPT,
        )
        plan = plan_agent.run(
            f"GOAL: {goal}\n\nExplore if needed, then return a concrete plan.",
            max_steps=plan_max_steps,
            on_step=make_on_step("plan"),
        )

        build_agent = Agent(
            self._provider,
            tools=self._tools,
            system_prompt=BUILD_SYSTEM_PROMPT,
        )
        report = build_agent.run(
            f"GOAL: {goal}\n\nPLAN:\n{plan}\n\nImplement the plan.",
            max_steps=build_max_steps,
            on_step=make_on_step("build"),
        )

        status = "ok"
        if plan.startswith("Stopped:") or report.startswith("Stopped:"):
            status = "max_steps"
        return RunResult(
            goal=goal,
            plan=plan,
            report=report,
            status=status,
            events=events,
        )
