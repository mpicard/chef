from mock_embedder import MockEmbedder

from memory import Memory
from memory_store import MemoryStore


def test_retrieve_encodes_goal_once_across_kinds():
    mock = MockEmbedder()
    store = MemoryStore(embedder=mock)
    store.memories = [
        Memory(
            "Paris population was 12M",
            "episodic",
            embedding=mock.encode("Paris population was 12M"),
        ),
        Memory(
            "Paris is capital of France",
            "semantic",
            embedding=mock.encode("Paris is capital of France"),
        ),
        Memory(
            "Berlin timezone Europe/Berlin",
            "semantic",
            embedding=mock.encode("Berlin timezone Europe/Berlin"),
        ),
    ]
    mock.calls.clear()  # ignore seeding encodes

    store.retrieve("summarise Paris", k=3, kind="episodic")
    store.retrieve("summarise Paris", k=3, kind="semantic")

    assert mock.calls == ["summarise Paris"]
    assert "summarise Paris" in store._cache


def test_retrieve_does_not_reencode_per_candidate():
    mock = MockEmbedder()
    store = MemoryStore(embedder=mock)
    store.memories = [
        Memory(f"fact {i}", "semantic", embedding=[float(i)]) for i in range(10)
    ]
    mock.calls.clear()

    store.retrieve("goal", k=3, kind="semantic")

    assert mock.calls == ["goal"]


def test_different_goals_cache_miss():
    mock = MockEmbedder()
    store = MemoryStore(embedder=mock)
    store.memories = [Memory("x", "semantic", embedding=[1.0])]
    mock.calls.clear()

    store.retrieve("goal A", k=1)
    store.retrieve("goal B", k=1)

    assert mock.calls == ["goal A", "goal B"]


def test_clear_query_cache_forces_reencode():
    mock = MockEmbedder()
    store = MemoryStore(embedder=mock)
    store.memories = [Memory("x", "semantic", embedding=[1.0])]
    mock.calls.clear()

    store.retrieve("same", k=1)
    store.clear_query_cache()
    store.retrieve("same", k=1)

    assert mock.calls == ["same", "same"]
