import time
from dataclasses import dataclass, field


@dataclass
class BudgetMulti:
    max_tokens: int = 10_000
    max_tool_calls: int = 12
    max_wall_seconds: float = 240.0
    max_cost_eur: float = 0.50

    tokens_used: int = 0
    tool_calls_used: int = 0
    cost_eur: float = 0.00
    _started_at: float = field(default_factory=time.monotonic, repr=False)

    def pressure(self) -> float:
        # drives graceful shutdown, switch to cheaper models, ask for confirmation, etc
        return max(
            self.tokens_used / self.max_tokens,
            self.tool_calls_used / self.max_tool_calls,
            self.elapsed() / self.max_wall_seconds,
            self.cost_eur / self.max_cost_eur,
        )

    def elapsed(self) -> float:
        return time.monotonic() - self._started_at

    def has_room(self) -> bool:
        return (
            self.tokens_used < self.max_tokens
            and self.tool_calls_used < self.max_tool_calls
            and self.elapsed() < self.max_wall_seconds
            and self.cost_eur < self.max_cost_eur
        )

    def charge(
        self, *, tokens: int = 0, cost: float = 0.0, tool_calls: int = 0
    ) -> None:
        self.tokens_used += tokens
        self.cost_eur += cost
        self.tool_calls_used += tool_calls
