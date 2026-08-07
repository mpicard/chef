import json

from llm_provider import LLMProvider, safe_json_loads
from planner import NodeStatus, PlanDAG
from typed_tool import TypedTool

PLANNER_SYSTEM = """You are a Planner agent. Given a GOAL, produce a dependency graph
of tool calls that will satisfy it.

Output ONLY a JSON object (never a bare array) in this exact shape:

{"nodes": [{"id": "...", "tool": "...", "args": {...}, "deps": [...]}, ...]}

Rules:
- Root value MUST be an object with key "nodes".
- Each node MUST be an object with keys id, tool, args, deps.
- Do NOT return a list of tool names.
- Nodes may run in parallel if their `deps` are empty or already satisfied.
- The final aggregate_report node must depend on all upstream fetch/summary nodes.
- Prefer id "aggregate" for that capstone node (any unique id is acceptable).

Available tools (name and schema):

<<TOOLS_JSON>>
"""


class PlannerAgent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: dict[str, TypedTool],
        max_attempts: int = 3,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._max_attempts = max_attempts

    def plan(self, goal: str) -> PlanDAG:
        tools_json = json.dumps([t.schema() for t in self._tools.values()], indent=2)
        system = PLANNER_SYSTEM.replace("<<TOOLS_JSON>>", tools_json)
        last_err: Exception | None = None
        for attempt in range(self._max_attempts):
            user = f"GOAL: {goal}"
            if last_err is not None:
                user += (
                    f"\n\nPrevious output was invalid ({last_err}). "
                    'Return ONLY a JSON object: {"nodes": [...]}.'
                )
            raw = self._provider.complete(
                system=system, user=user, role="planner"
            )
            try:
                return PlanDAG.from_json(safe_json_loads(raw))
            except (ValueError, TypeError, KeyError) as e:
                last_err = e
        raise ValueError(f"Planner failed after {self._max_attempts} attempts: {last_err}")

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

        last_err: Exception | None = None
        for attempt in range(self._max_attempts):
            user = user_prompt
            if last_err is not None:
                user += (
                    f"\n\nPrevious output was invalid ({last_err}). "
                    'Return ONLY a JSON object: {"nodes": [...]}.'
                )
            raw = self._provider.complete(
                system=system, user=user, role="replanner"
            )
            try:
                return PlanDAG.from_json(safe_json_loads(raw))
            except (ValueError, TypeError, KeyError) as e:
                last_err = e
        raise ValueError(
            f"Replanner failed after {self._max_attempts} attempts: {last_err}"
        )
