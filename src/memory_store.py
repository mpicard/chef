from dataclasses import dataclass, field

from embedder import Embedder
from memory import Memory, MemoryKind
from similarity import cosine, jaccard


@dataclass
class MemoryStore:
    memories: list[Memory] = field(default_factory=list)
    embedder: Embedder = field(default_factory=Embedder)
    _cache: dict[str, list[float] | None] = field(default_factory=dict, repr=False)

    def add(self, content: str, kind: MemoryKind) -> Memory:
        emb = self.embedder.encode(content) if self.embedder.available else None
        m = Memory(content=content, kind=kind, embedding=emb)
        self.memories.append(m)
        return m

    def score(
        self, query: str, memory: Memory, query_embedding: list[float] | None = None
    ) -> float:
        # Prefer embeddings
        if query_embedding is not None and memory.embedding is not None:
            return cosine(query_embedding, memory.embedding)
        return jaccard(query, memory.content)

    def retrieve(
        self, goal: str, k: int = 3, kind: MemoryKind | None = None
    ) -> list[Memory]:
        q_embedding = self._embed_query(goal)
        candidates = [m for m in self.memories if kind is None or m.kind == kind]
        ranked = sorted(
            candidates, key=lambda m: self.score(goal, m, q_embedding), reverse=True
        )
        return ranked[:k]

    def _embed_query(self, goal: str) -> list[float] | None:
        if not self.embedder.available:
            return None
        if goal not in self._cache:
            self._cache[goal] = self.embedder.encode(goal)
        return self._cache[goal]

    def clear_query_cache(self) -> None:
        self._cache.clear()
