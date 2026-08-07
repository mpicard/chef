import argparse
from dataclasses import dataclass

from llm import llm
from orchestrator import Orchestrator
from tools_registry import TOOLS


@dataclass
class CliArgs:
    goal: str
    plan_steps: int
    build_steps: int
    quiet: bool


def parse_args(argv: list[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        prog="chef",
        description="Plan/build coding agent over the local repo.",
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default="Create sandbox/hello.py that prints Hello, chef",
        help="Task for the agent (default: hello-world demo)",
    )
    parser.add_argument(
        "--plan-steps",
        type=int,
        default=10,
        metavar="N",
        help="Max plan-phase ReAct steps (default: 10)",
    )
    parser.add_argument(
        "--build-steps",
        type=int,
        default=20,
        metavar="N",
        help="Max build-phase ReAct steps (default: 20)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Hide per-step trace lines",
    )
    ns = parser.parse_args(argv)
    return CliArgs(
        goal=ns.goal,
        plan_steps=ns.plan_steps,
        build_steps=ns.build_steps,
        quiet=ns.quiet,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    orch = Orchestrator(provider=llm, tools=TOOLS)
    result = orch.run(
        args.goal,
        plan_max_steps=args.plan_steps,
        build_max_steps=args.build_steps,
        verbose=not args.quiet,
    )
    print(f"Status:  {result.status}")
    print()
    print("--- PLAN ---")
    print(result.plan)
    print()
    print("--- SUMMARY ---")
    print(result.report)
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
