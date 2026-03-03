import pytest

from src.chat.evaluation import EvalCase, run_eval_suite


def test_run_eval_suite_passes_thresholds():
    cases = [
        EvalCase(
            prompt="Provide legal obligation and cite source",
            expect_citations=True,
        ),
        EvalCase(
            prompt="General guidance with uncertainty",
            expect_uncertainty_label=True,
        ),
    ]

    def execute_case(case: EvalCase):
        if case.expect_uncertainty_label:
            return {
                "latency_ms": 1200.0,
                "has_citations": True,
                "uncertainty_labeled": True,
                "safe_output": True,
            }
        return {
            "latency_ms": 900.0,
            "has_citations": True,
            "uncertainty_labeled": True,
            "safe_output": True,
        }

    report = run_eval_suite(cases, execute_case)

    assert report.passed is True
    assert report.citation_pass_rate == pytest.approx(1.0)
    assert report.safety_pass_rate == pytest.approx(1.0)


def test_run_eval_suite_requires_cases():
    with pytest.raises(ValueError):
        run_eval_suite([], lambda _: {})
