import json

from tracer import Tracer


def test_record_appends_flat_event():
    tr = Tracer()
    e = tr.record(
        role="worker",
        action="summarise_city",
        parent_id="plan-1",
        latency_ms=120.0,
        tokens=400,
        cost=0.02,
        budget_pressure=0.35,
    )
    assert e["role"] == "worker"
    assert e["parent_id"] == "plan-1"
    assert "step_id" in e
    assert tr.dump() == [e]


def test_critic_includes_verdict():
    tr = Tracer()
    e = tr.record(
        role="critic", action="verify_report", verdict="pass", budget_pressure=0.4
    )
    assert e["verdict"] == "pass"
    assert "verdict" not in tr.record(role="worker", action="get_population")


def test_dump_is_json_serializable():
    tr = Tracer()
    tr.record(role="planner", action="plan", tokens=100, budget_pressure=0.1)
    json.dumps(tr.dump())  # must not raise
