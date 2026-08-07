from dataclasses import dataclass

from agent import Agent
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
    ) -> RunResult:
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
        )

        build_agent = Agent(
            self._provider,
            tools=self._tools,
            system_prompt=BUILD_SYSTEM_PROMPT,
        )
        report = build_agent.run(
            f"GOAL: {goal}\n\nPLAN:\n{plan}\n\nImplement the plan.",
            max_steps=build_max_steps,
        )

        status = "ok"
        if plan.startswith("Stopped:") or report.startswith("Stopped:"):
            status = "max_steps"
        return RunResult(goal=goal, plan=plan, report=report, status=status)
