from budget_multi import BudgetMulti


def verify_report(
    report, goal, required_cities, provider, budget: BudgetMulti
) -> Verdict:
    det = deterministic_check_report(report, required_cities)

    if not det.passed:
        return det  # chear tier caught it - don't waist tokens

    if not budget.allow_llm_judge():
        return det

    # deterministic passed -> escalate to LLM judge for subjective quality
    return llm_judge_report(report, goal, provider)
