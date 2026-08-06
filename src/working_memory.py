from dataclasses import dataclass, field


@dataclass
class WorkingMemory:
    goal: str
    plan_summary: str = ""
    recent_results: list[str] = field(default_factory=list)
    max_recent: int = 3

    def add_result(self, result: str) -> None:
        self.recent_results.append(result)
        if len(self.recent_results) > self.max_recent:
            self.recent_results = self.recent_results[-self.max_recent :]

    def to_prompt(self) -> str:
        lines = [f"Goal: {self.goal}"]
        if self.plan_summary:
            lines.append(f"Plan: {self.plan_summary}")
        if self.recent_results:
            lines.append("Recent:")
            for r in self.recent_results[-5:]:
                lines.append(f"- {r}")
        return "\n".join(lines)
