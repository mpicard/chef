from llm_provider import LLMProvider
from verdict import Verdict, verify_report


class CriticAgent:
    """Thin wrapper; keeps critic logic swappable without touching the orchestrator."""

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    def judge(self, report: str, goal: str, required_cities: list[str]) -> Verdict:
        return verify_report(report, goal, required_cities, self._provider)
