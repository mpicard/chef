import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from llm_provider import OpenAICompatibleProvider
from orchestrator import Orchestrator
from tools_registry import REPO_ROOT, TOOLS

_ = load_dotenv()

llm = OpenAICompatibleProvider(
    model="Qwen3.6-35B-A3B-4bit",
    base_url="http://192.168.178.45:11434/v1",
    api_key=os.getenv("OPENAI_API_KEY", ""),
)


@dataclass
class CliArgs:
    goal: str
    plan_steps: int
    build_steps: int
    quiet: bool
    persist: bool
    runs_dir: Path


def parse_args(argv: list[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        prog="chef",
        description="Plan/build coding agent over the local repo.",
    )
    _ = parser.add_argument(
        "goal",
        nargs="?",
        default="Create sandbox/hello.py that prints Hello, chef",
        help="Task for the agent (default: hello-world demo)",
    )
    _ = parser.add_argument(
        "--plan-steps",
        type=int,
        default=10,
        metavar="N",
        help="Max plan-phase ReAct steps (default: 10)",
    )
    _ = parser.add_argument(
        "--build-steps",
        type=int,
        default=20,
        metavar="N",
        help="Max build-phase ReAct steps (default: 20)",
    )
    _ = parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Hide per-step trace lines",
    )
    _ = parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not write a JSONL trace under runs/",
    )
    _ = parser.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "runs",
        help="Directory for JSONL traces (default: <repo>/runs)",
    )
    ns = parser.parse_args(argv)
    return CliArgs(
        goal=ns.goal,
        plan_steps=ns.plan_steps,
        build_steps=ns.build_steps,
        quiet=ns.quiet,
        persist=not ns.no_persist,
        runs_dir=ns.runs_dir,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    orch = Orchestrator(provider=llm, tools=TOOLS)
    result = orch.run(
        args.goal,
        plan_max_steps=args.plan_steps,
        build_max_steps=args.build_steps,
        verbose=not args.quiet,
        persist_dir=args.runs_dir if args.persist else None,
    )
    print(f"Status:  {result.status}")
    if result.trace_path:
        print(f"Trace:   {result.trace_path}")
    print()
    print("--- PLAN ---")
    print(result.plan)
    print()
    print("--- SUMMARY ---")
    print(result.report)
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
