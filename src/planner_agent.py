import json

from helpers import safe_json_loads
from llm_provider import LLMProvider
from planner import NodeStatus, PlanDAG
from typed_tool import TypedTool

PLANNER_SYSTEM = """You are a Planner agent. Given a GOAL, produce a dependency graph
of tool calls that will satisfy it.

Output ONLY JSON in this shape (no prose, no markdown):

{"nodes": [{"id": "...", "tool": "...", "args": {...}, "deps": [...]}, ...]}

Nodes may run in parallel if their `deps` are empty or already satisfied.
The final aggregate_report node must depend on all upstream fetch/summary nodes.
Prefer id "aggregate" for that capstone node (any unique id is acceptable).
Available tools (name and schema):

<<TOOLS_JSON>>
"""


class PlannerAgent:
    def __init__(self, provider: LLMProvider, tools: dict[str, TypedTool]) -> None:
        self._provider = provider
        self._tools = tools

    def plan(self, goal: str) -> PlanDAG:
        tools_json = json.dumps([t.schema() for t in self._tools.values()], indent=2)
        system = PLANNER_SYSTEM.replace("<<TOOLS_JSON>>", tools_json)
        raw = self._provider.complete(
            system=system, user=f"GOAL: {goal}", role="planner"
        )
        return PlanDAG.from_json(safe_json_loads(raw))

    def replan(self, goal: str, failed_dag: PlanDAG) -> PlanDAG:
        """Re-plan with context about what failed"""
        tools_json = json.dumps([t.schema() for t in self._tools.values()], indent=2)
        system = PLANNER_SYSTEM.replace("<<TOOLS_JSON>>", tools_json)
        failures = []
        for node in failed_dag.nodes.values():
            if node.status == NodeStatus.FAILED:
                failures.append(
                    f"Node {node.id} (tool={node.tool}, args={node.args}) failed: {node.error}"
                )
        failure_context = "\n".join(failures)
        user_prompt = (
            f"GOAL: {goal}\n\n"
            f"Previous plan failed with these errors:\n{failure_context}\n\n"
            f"Please provide an amended plan that avoids these failures."
        )

        raw = self._provider.complete(system=system, user=user_prompt, role="replanner")
        return PlanDAG.from_json(safe_json_loads(raw))
