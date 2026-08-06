from dataclasses import dataclass

from helpers import safe_json_loads
from llm_provider import LLMProvider


@dataclass
class Verdict:
    passed: bool
    tier: str  # deterministic | diff | llm_judge
    reason: str = ""


def deterministic_check_report(report: str, required_cities: list[str]) -> Verdict:
    """The report must be non-empty and mention every required city."""
    if not report or not report.strip():
        return Verdict(False, "deterministic", "Empty report.")

    missing = [c for c in required_cities if c.lower() not in report.lower()]
    if missing:
        return Verdict(False, "deterministic", f"Missing cities: {missing}")

    return Verdict(True, "deterministic", "All required cities present.")


def llm_judge_report(report: str, goal: str, provider: LLMProvider) -> Verdict:
    """Ask ::M to grade the report against the goal. Returns JSON."""

    raw = provider.complete(
        system=(
            "You are a strict grader. Given a GOAL and a REPORT, decide whether "
            "the report satisfies the goal. Respond ONLY with JSON: "
            '{"pass":  true|false, "reason":"..."}.'
        ),
        user=f"GOAL:\n{goal}\n\nREPORT:\n{report}",
        role="critic",
    )
    try:
        data = safe_json_loads(raw)
        return Verdict(bool(data["pass"]), "llm_judge", data.get("reason", ""))
    except Exception as e:  # noqa: BLE001
        return Verdict(False, "llm_judge", f"Judge output malformed: {e}")


def verify_report(
    report: str, goal: str, required_cities: list[str], provider: LLMProvider
) -> Verdict:
    det = deterministic_check_report(report, required_cities)
    if not det.passed:
        return det
    return llm_judge_report(report, goal, provider)
