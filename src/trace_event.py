from dataclasses import dataclass
from typing import Any


@dataclass
class TraceEvent:
    step_id: str
    parent_id: str | None
    timestamp: float
    role: str
    action: dict
    result: Any
    latency_ms: int
    tokens: int
    cost_eur: float
    budget_pressure: float
    verdict: dict | None = None

    def to_json(self) -> dict:
        return {
            "step_id": self.step_id,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp,
            "role": self.role,
            "action": self.action,
            "result": _truncate(self.result, 200),
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "cost_eur": round(self.cost_eur, 6),
            "budget_pressure": round(self.budget_pressure, 3),
            "verdict": self.verdict,
        }


def _truncate(x: Any, n: int) -> Any:
    s = str(x)
    return s if len(s) <= n else s[:n] + f"...(+{len(s) - n} chars)"
