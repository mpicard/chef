import json
from pathlib import Path

from orchestrator import RunResult
from trace_store import persist_run


def test_persist_run_writes_meta_and_events(tmp_path: Path):
    result = RunResult(
        goal="g",
        plan="p",
        report="r",
        status="ok",
        events=[
            {"phase": "plan", "kind": "llm", "step": 0, "latency_ms": 1},
            {"phase": "build", "kind": "done", "step": 0, "summary": "r"},
        ],
    )
    path = persist_run(result, tmp_path / "runs", stamp="testrun")
    assert path == tmp_path / "runs" / "testrun.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    meta = json.loads(lines[0])
    assert meta["type"] == "meta"
    assert meta["goal"] == "g"
    assert meta["n_events"] == 2
    ev0 = json.loads(lines[1])
    assert ev0["type"] == "event"
    assert ev0["phase"] == "plan"
    assert ev0["kind"] == "llm"


def test_orchestrator_persist_sets_trace_path(tmp_path: Path):
    import json as _json

    from orchestrator import Orchestrator
    from test_agent import ScriptedProvider
    from tools_registry import make_tools

    tools = make_tools(root=tmp_path)
    provider = ScriptedProvider(
        [
            _json.dumps({"done": True, "summary": "plan"}),
            _json.dumps({"done": True, "summary": "build"}),
        ]
    )
    runs = tmp_path / "runs"
    orch = Orchestrator(provider, tools)
    result = orch.run("goal", verbose=False, persist_dir=runs)
    assert result.trace_path is not None
    assert Path(result.trace_path).is_file()
    assert Path(result.trace_path).parent == runs
