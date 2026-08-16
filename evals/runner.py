#!/usr/bin/env python3
"""Evaluation runner for Chef's agentic harness.

Runs eval cases and reports results as a summary table.

Usage:
    python evals/runner.py                    # run all happy path cases
    python evals/runner.py --case 01 02       # run specific cases
    python evals/runner.py -q                 # quiet mode (only summary)
    python evals/runner.py --verbose          # show plan and report
"""

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add src/ to path for importing Chef modules
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Load environment variables
from dotenv import load_dotenv

env_path = _PROJECT_ROOT / ".env"
if env_path.exists():
    _ = load_dotenv(env_path)

# Import from Chef (src/)
# Import from evals (same directory)
# Relative imports when run as module; absolute when run directly as __main__.
try:
    from .llm_provider import OpenAICompatibleProvider
    from .loader import list_case_ids, load_goal, resolve_case_id
    from .orchestrator import Orchestrator, RunResult
    from .tools_registry import make_tools
except ImportError:
    from loader import list_case_ids, load_goal, resolve_case_id

    from llm_provider import OpenAICompatibleProvider
    from orchestrator import Orchestrator, RunResult
    from tools_registry import make_tools


@dataclass
class EvalResult:
    """Result of running a single eval case."""

    case_id: str
    passed: bool
    status: str  # "ok" | "max_steps" | "error"
    plan_steps: int
    build_steps: int
    total_steps: int
    wall_time_ms: int
    total_tokens: int
    summary: str  # plan summary
    report: str  # build report
    error: str | None = None


def _grade_case(case_id: str, result: RunResult, output_dir: Path) -> tuple[bool, str]:
    """Import and run the case's grade.py."""
    case_dir = _PROJECT_ROOT / "evals" / "cases" / case_id
    grade_path = case_dir / "grade.py"

    if not grade_path.exists():
        return False, "no grade.py found"

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"grade_{case_id}", str(grade_path)
    )
    if spec is None or spec.loader is None:
        return False, f"could not load grade module for {case_id}"
    grade_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grade_module)

    try:
        return grade_module.grade(result, output_dir)
    except (TypeError, ValueError, AssertionError) as e:
        return False, f"grade error: {e}"


def run_case(orch: Orchestrator, case_id: str, output_dir: Path) -> EvalResult:
    """Run a single eval case and return its result."""
    goal = load_goal(case_id)
    t0 = time.monotonic()

    result = orch.run(goal, verbose=False)

    passed, _grade_msg = _grade_case(case_id, result, output_dir)

    plan_events = [e for e in result.events if e.get("phase") == "plan"]
    build_events = [e for e in result.events if e.get("phase") == "build"]

    return EvalResult(
        case_id=case_id,
        passed=passed,
        status=result.status,
        plan_steps=len(plan_events),
        build_steps=len(build_events),
        total_steps=len(result.events),
        wall_time_ms=int((time.monotonic() - t0) * 1000),
        total_tokens=result.total_tokens,
        summary=result.plan,
        report=result.report,
    )


def run_suite(
    case_ids: list[str] | None,
    output_dir: Path,
    model: str,
    base_url: str,
) -> list[EvalResult]:
    """Run all cases (or a subset) and return results."""
    if case_ids is None:
        case_ids = list_case_ids()

    results = []
    for raw_id in sorted(case_ids):
        case_id = resolve_case_id(raw_id)
        case_output = output_dir / case_id
        shutil.rmtree(case_output)
        case_output.mkdir(parents=True, exist_ok=True)

        tools = make_tools(root=case_output)
        llm = OpenAICompatibleProvider(
            model=model,
            base_url=os.getenv("OPENAI_BASE_URL", base_url),
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        orch = Orchestrator(provider=llm, tools=tools)

        try:
            print(f"[{case_id}] Running...", flush=True)
            shutil.rmtree(case_output, ignore_errors=True)
            case_output.mkdir(parents=True, exist_ok=True)
            r = run_case(orch, case_id, case_output)
            print(f"[{case_id}] Done [{r.status}] ({r.total_steps} steps, {r.wall_time_ms}ms)", flush=True)
        except (OSError, PermissionError, shutil.Error) as e:
            r = EvalResult(
                case_id=case_id,
                passed=False,
                status="error",
                plan_steps=0,
                build_steps=0,
                total_steps=0,
                wall_time_ms=0,
                total_tokens=0,
                summary="",
                report="",
                error=str(e),
            )

        results.append(r)

    return results


def print_results(results: list[EvalResult]):
    """Print results as an ASCII table."""
    print()
    header = (
        f"{'Case':<30} {'Status':<12} {'Plan':>5} {'Build':>6} "
        f"{'Total':>6} {'Time':>8} {'Pass':>5}"
    )
    separator = "-" * 79
    print(header)
    print(separator)

    for r in results:
        line = (
            f"{r.case_id:<30} {r.status:<12} {r.plan_steps:>5} {r.build_steps:>6} "
            f"{r.total_steps:>6} {f'{r.wall_time_ms}ms':>8} {f'{r.total_tokens}' :>8} "
            f"{('✓' if r.passed else '✗'):>5}"
        )
        print(line)
        if r.error:
            print(f"  Error: {r.error}")

    print(separator)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"Pass rate: {passed}/{total} ({passed / total * 100:.0f}%)")

    statuses: dict[str,int] = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    if statuses:
        print(
            f"Status breakdown: {', '.join(f'{k}={v}' for k, v in sorted(statuses.items()))}"
        )


def main():
    parser = argparse.ArgumentParser(
        prog="chef-eval",
        description="Run eval cases and report results.",
    )
    _ = parser.add_argument(
        "--case",
        nargs="+",
        help="Specific case IDs to run (default: all)",
    )
    _ = parser.add_argument(
        "-q", "--quiet", action="store_true", help="Hide per-case output"
    )
    _ = parser.add_argument("--verbose", action="store_true", help="Show plan and report")
    _ = parser.add_argument("--model", default="Qwen3.6-35B-A3B-4bit", help="LLM model name")
    _ = parser.add_argument(
        "--base-url", default="http://127.0.0.1:11434/v1", help="LLM base URL"
    )

    args = parser.parse_args()

    output_dir = _PROJECT_ROOT / "runs" / "eval-output"
    results = run_suite(
        case_ids=list(args.case) if args.case else None,
        output_dir=output_dir,
        model=str(args.model),
        base_url=str(args.base_url),
    )

    # Per-case summary (unless quiet)
    if not args.quiet:
        print()
        for r in results:
            indicator = "✓" if r.passed else "✗"
            print(f"  {indicator} {r.case_id} [{r.status}]")
            if args.verbose:
                print(f"      Plan:  {r.summary[:200]}")
                print(f"      Report: {r.report[:200]}")

    print_results(results)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
