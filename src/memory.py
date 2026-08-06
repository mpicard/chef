from dataclasses import dataclass, field
from typing import Literal

MemoryKind = Literal["episodic", "semantic"]


@dataclass
class Memory:
    content: str
    kind: MemoryKind
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)
