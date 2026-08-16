from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import Agent, OnStep
from llm_provider import LLMProvider
from tools_registry import PLAN_TOOLS
from typed_tool import TypedTool

PLAN_SYSTEM_PROMPT = """You are a planning agent for a coding task.
You may ONLY use read_file, list_dir, and glob to inspect the repo. You cannot write or edit files.
Reply with ONLY one JSON object per turn:

- {"tool": "read_file", "args": {"path": "..."}}
- {"tool": "list_dir", "args": {"path": "."}}
- {"tool": "glob", "args": {"pattern": "**/*.py"}}
- {"done": true, "summary": "<concrete implementation plan>"}

When done, summary must list which files to create/edit and what to put in them.
To finish, you MUST use {"done": true, "summary": "..."}.
Never use {"tool": "done", ...} — done is not a tool.
"""

BUILD_SYSTEM_PROMPT = """You are a coding agent that implements a plan.
You may use read_file, list_dir, glob, write_file, and edit_file.
Prefer edit_file for small changes to existing files; write_file for new files or full rewrites.
Reply with ONLY one JSON object per turn:

- {"tool": "read_file", "args": {"path": "..."}}
- {"tool": "list_dir", "args": {"path": "."}}
- {"tool": "glob", "args": {"pattern": "**/*.py"}}
- {"tool": "write_file", "args": {"path": "...", "content": "..."}}
- {"tool": "edit_file", "args": {"path": "...", "old_string": "...", "new_string": "..."}}
- {"done": true, "summary": "<what you changed>"}

To finish, you MUST use {"done": true, "summary": "..."}.
Never use {"tool": "done", ...} — done is not a tool.
"""


@dataclass
class RunResult:
    goal: str
    plan: str
    report: str
    status: str
    events: list[dict[str, Any]] = field(default_factory=list)
    trace_path: str | None = None
    total_tokens: int = 0


def format_step(event: dict[str, Any]) -> str:
    phase = event.get("phase", "?")
    step = event.get("step", "?")
    kind = event.get("kind", "?")
    ms = event.get("latency_ms")
    timing = f" {ms}ms" if ms is not None else ""
    prefix = f"[{phase} {step}] {kind}{timing}"
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
        persist_dir: Path | None = None,
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
            if name in PLAN_TOOLS
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
        plan_tokens = plan_agent.token_usage

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
        build_tokens = build_agent.token_usage

        status = "ok"
        if plan.startswith("Stopped:") or report.startswith("Stopped:"):
            status = "max_steps"

        total_prompt = plan_tokens[0] + build_tokens[0]
        total_completion = plan_tokens[1] + build_tokens[1]

        result = RunResult(
            goal=goal,
            plan=plan,
            report=report,
            status=status,
            events=events,
            total_tokens=total_prompt + total_completion
        )
        if persist_dir is not None:
            from trace_store import persist_run

            path = persist_run(result, persist_dir)
            result.trace_path = str(path)
            if verbose:
                print(f"[trace] {path}", flush=True)
        return result
