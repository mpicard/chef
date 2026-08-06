from dataclasses import dataclass

from trace_event import TraceEvent


@dataclass
class Tracer:
    """append-only log of structured events"""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, **kwargs) -> TraceEvent:
        event = TraceEvent(**kwargs)
        self.events.append(event)
        return event

    def dump(self) -> list[dict]:
        return [e.to_json() for e in self.events]

    def new_step_id(self, prefix: str = "s") -> str:
        return f"{prefix}-{len(self.events):04d}"
