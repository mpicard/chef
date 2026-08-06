from collections.abc import Callable

from execute_dag import execute_dag
from planner import PlanDAG, PlanNode
from typed_tool import TypedTool


class WorkerAgent:
    """Runs the DAG produced by the PlannerAgent"""

    # TODO: split worker into fetcher (cheap lookups) and writer (llm composition)

    def __init__(self, tools: dict[str, TypedTool]) -> None:
        self._tools = tools

    async def execute(
        self, dag: PlanDAG, on_step: Callable[[PlanNode, float], None] | None = None
    ) -> None:
        await execute_dag(dag, self._tools, on_step=on_step)
