from build_context import build_context
from embedder import Embedder
from memory_store import MemoryStore
from working_memory import WorkingMemory


def test_integration_build_context():
    store = MemoryStore(embedder=Embedder(enabled=False))
    store.add("Queried Paris population; tool return 1M", kind="episodic")
    store.add("Paris is the capital of France", kind="semantic")
    store.add("Berlin timezone is Europe/Berlin", kind="semantic")
    working = WorkingMemory(
        goal="summarize Paris",
        plan_summary="get_population -> summarize_city",
    )
    working.add_result("population=12000000")
    assert (
        build_context(working, store, budget_chars=500)
        == """Goal: summarize Paris
Plan: get_population -> summarize_city
Recent:
- population=12000000
[episodic] Queried Paris population; tool return 1M
[semantic] Paris is the capital of France
[semantic] Berlin timezone is Europe/Berlin"""
    )
