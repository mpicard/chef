import asyncio
import random
import time
from collections.abc import Callable

from classify_error import RecoveryAction, classify_and_recover
from planner import NodeStatus, PlanDAG, PlanNode
from typed_tool import TypedTool

MAX_CONCURRENT = 5


async def execute_dag(
    dag: PlanDAG,
    tools: dict[str, TypedTool],
    on_step: Callable[[PlanNode, float], None] | None = None,
):
    """Walk the DAG in waves"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def run_node(node) -> None:
        node.status = NodeStatus.RUNNING
        node.attempts += 1

        async with semaphore:
            t0 = time.monotonic()
            try:
                tool = tools[node.tool]
                node.result = await asyncio.to_thread(tool.run, node.args)
                node.status = NodeStatus.DONE
            except Exception as exc:  # noqa: BLE001
                node.error = f"{type(exc).__name__}: {exc}"
                attempt = getattr(node, "attempts", 0)
                _cls, action = classify_and_recover(node.error, attempt)

                match action:
                    case RecoveryAction.RETRY:
                        await asyncio.sleep(backoff_seconds(attempt))  # todo attempt
                        node.status = NodeStatus.PENDING
                    case RecoveryAction.REPROMPT:
                        # append node.error to the LLM context; ask for corrected args
                        # then re-run with the new args
                        node.status = NodeStatus.PENDING
                    case RecoveryAction.REPLAN:
                        # remove the unknown entity from the goal/plan; do not retry
                        # rebuild DAG without that node
                        node.status = NodeStatus.FAILED
                    case RecoveryAction.HALT:
                        node.status = NodeStatus.FAILED
                        return
                node.error = f"{type(exc).__name__}: {exc}"
                node.status = NodeStatus.FAILED
            finally:
                if on_step:
                    on_step(node, time.monotonic() - t0)

    while not dag.is_done():
        ready = dag.ready_nodes()

        if not ready:
            break

        await asyncio.gather(*[run_node(n) for n in ready])


def backoff_seconds(attempt: int, base: float = 0.5, cap: float = 30.0) -> float:
    ceiling = min(cap, base * (2**attempt))
    return random.uniform(0, ceiling)
