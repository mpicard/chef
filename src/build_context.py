from memory_store import MemoryStore
from working_memory import WorkingMemory


def build_context(
    working: WorkingMemory, store: MemoryStore, budget_chars: int = 4000
) -> str:
    """Assemble working memory + retrieve memories, respecting a char budget"""

    pieces = [working.to_prompt()]
    used = len(pieces[0])

    for kind in ("episodic", "semantic"):
        for m in store.retrieve(working.goal, k=3, kind=kind):
            snippet = f"[{kind}] {m.content}"
            if used + len(snippet) + 1 > budget_chars:
                return "\n".join(pieces) + "\n(...truncated at budget...)"

            pieces.append(snippet)
            used += len(snippet) + 1

    return "\n".join(pieces)
