from budget_multi import BudgetMulti
from classify_error import ErrorClass, classify_error


def test_pressure_is_max_across_dimensions():
    b = BudgetMulti(
        max_tokens=100, max_tool_calls=10, max_wall_seconds=1e9, max_cost_eur=1.0
    )
    b.record_tokens(50)  # 0.5
    b.record_tool_call(0.0)  # 0.1
    assert 0.49 < b.pressure() < 0.51


def test_classify_error_classes():
    assert classify_error("Rate limit 429") is ErrorClass.TRANSIENT
    assert classify_error("ValidationError: city") is ErrorClass.VALIDATION
    assert classify_error("KeyError: 'Atlantis'") is ErrorClass.UNKNOWN_ENTITY
    assert classify_error("Policy forbidden") is ErrorClass.POLICY
