import json
from dataclasses import dataclass
import time
from typing import Any

from budget_multi import BudgetMulti
from classify_error import ErrorClass, classify_error
from critic_agent import CriticAgent
from llm_provider import LLMProvider
from memory_store import MemoryStore
from planner import NodeStatus, PlanDAG, PlanNode
from planner_agent import PlannerAgent
from tracer import Tracer
from typed_tool import TypedTool
from verdict import deterministic_check_report
from worker_agent import WorkerAgent

COST_PER_1K_TOKENS = 0.003


@dataclass
class RunResult:
    goal: str
    report: str | None
    verdict: Any
    dag: PlanDAG
    tracer: Tracer
    budget: BudgetMulti
    status: str
    replan_count: int = 0


class Orchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        tools: dict[str, TypedTool],
        memory: MemoryStore,
        max_replans: int = 2,
    ):
        self._provider = provider
        self._tools = tools
        self._memory = memory
        self._max_replans = max_replans
        self.planner = PlannerAgent(provider, tools)
        self.worker = WorkerAgent(tools)
        self.critic = CriticAgent(provider)

    async def run(
        self,
        goal: str,
        required_cities: list[str],
        budget: BudgetMulti | None = None,
    ) -> RunResult:
        budget = budget or BudgetMulti()
        tracer = Tracer()
        replan_count = 0

        dag = None
        for attempt in range(self._max_replans + 1):
            t0 = time.monotonic()

            if attempt == 0:
                dag = self.planner.plan(goal)
                plan_type = "initial_plan"
            else:
                dag = self.planner.replan(goal, dag)
                plan_type = "replan"
                replan_count += 1

            plan_latency = int((time.monotonic() - t0) * 1000)
            est_tokens = (
                len(goal)
                + sum(len(n.tool) + len(json.dumps(n.args)) for n in dag.nodes.values())
            ) // 4
            est_cost = est_tokens * COST_PER_1K_TOKENS / 1000
            budget.charge(tokens=est_tokens, cost=est_cost)
            tracer.record(
                step_id=tracer.new_step_id("plan"),
                parent_id=None,
                timestamp=time.monotonic(),
                role="planner",
                action={"type": plan_type, "goal": goal, "attempt": attempt},
                result={"n_nodes": len(dag.nodes)},
                latency_ms=plan_latency,
                tokens=est_tokens,
                cost_eur=est_cost,
                budget_pressure=budget.pressure(),
            )

            def on_step(node: PlanNode, wall_s: float) -> None:
                tool = self._tools[node.tool]
                toks = int(max(20, tool.cost_hint * 200))
                budget.charge(
                    tokens=toks, cost=toks * COST_PER_1K_TOKENS / 1000, tool_calls=1
                )
                tracer.record(
                    step_id=tracer.new_step_id("w"),
                    parent_id=f"plan-{len([e for e in tracer.events if e.role == 'planner']) - 1:04d}",
                    timestamp=time.time(),
                    role="worker",
                    action={"type": "tool_call", "tool": node.tool, "args": node.args},
                    result=node.result
                    if node.status == NodeStatus.DONE
                    else node.error,
                    latency_ms=int(wall_s * 1000),
                    tokens=toks,
                    cost_eur=toks * COST_PER_1K_TOKENS / 1000,
                    budget_pressure=budget.pressure(),
                )

            try:
                await self.worker.execute(dag, on_step=on_step)
            except Exception as exc:  # noqa: BLE001
                if attempt < self._max_replans and budget.has_room():
                    continue
                return RunResult(
                    goal=goal,
                    report=None,
                    verdict=None,
                    dag=dag,
                    tracer=tracer,
                    budget=budget,
                    status=f"failed_exceute:{exc}",
                    replan_count=replan_count,
                )

            # check if we need to re-plan
            if dag.any_failed():
                if attempt < self._max_replans and budget.has_room():
                    should_replan = False
                    for node in dag.nodes.values():
                        if node.status == NodeStatus.FAILED:
                            err_class = classify_error(node.error or "")
                            if err_class == ErrorClass.UNKNOWN_ENTITY:
                                should_replan = True
                                break
                    if should_replan:
                        continue  # replan
                # cant replan
                return RunResult(
                    goal=goal,
                    report=None,
                    verdict=None,
                    dag=dag,
                    tracer=tracer,
                    budget=budget,
                    status="failed_execute",
                    replan_count=replan_count,
                )
            # sucess
            break

        agg_node = dag.aggregate_node()
        if agg_node is None or agg_node.status != NodeStatus.DONE:
            return RunResult(
                goal=goal,
                report=None,
                verdict=None,
                dag=dag,
                tracer=tracer,
                budget=budget,
                status="failed_execute",
                replan_count=replan_count,
            )

        report = agg_node.result

        if budget.pressure() > 0.9:
            verdict = deterministic_check_report(report, required_cities)
        else:
            verdict = self.critic.judge(report, goal, required_cities)

        tracer.record(
            step_id=tracer.new_step_id("c"),
            parent_id=f"plan-{len([e for e in tracer.events if e.role == 'planner']) - 1:04d}",
            timestamp=time.time(),
            role="critic",
            action={"type": "verify"},
            result={"passed": verdict.passed, "tier": verdict.tier},
            latency_ms=0,
            tokens=100,
            cost_eur=100 * COST_PER_1K_TOKENS / 1000,
            budget_pressure=budget.pressure(),
            verdict={
                "passed": verdict.passed,
                "tier": verdict.tier,
                "reason": verdict.reason,
            },
        )

        # save outcome in episodic memory for future runs
        self._memory.add(
            f"Task: {goal[:120]} -> outcome: "
            f"{'pass' if verdict.passed else 'fail'} ({verdict.reason})",
            kind="episodic",
        )

        status = "ok" if verdict.passed else "failed_verify"
        if replan_count > 0:
            status = f"{status}_replanned_{replan_count}x"
        if not budget.has_room():
            status = "budget"
        return RunResult(
            goal=goal,
            report=report,
            verdict=verdict,
            dag=dag,
            tracer=tracer,
            budget=budget,
            status=status,
            replan_count=replan_count,
        )
