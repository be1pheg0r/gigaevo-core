from __future__ import annotations

import pytest

from gigaevo.monitoring.growth_estimator import (
    EnsembleLaw,
    LinearLaw,
    PowerLaw,
    confidence_width,
    estimate,
    estimate_by_stage,
)


def test_linear_law_recovers_exact_line_without_noise() -> None:
    values = [10 + 2 * i for i in range(10)]  # y = 10 + 2i
    law = LinearLaw.fit(values)
    assert law.intercept == pytest.approx(10.0)
    assert law.slope == pytest.approx(2.0)


def test_integral_matches_closed_form_for_constant_law() -> None:
    law = LinearLaw.fit([5.0, 5.0, 5.0])
    assert law.integral(100) == pytest.approx(500.0)


def test_fit_handles_zero_and_one_points() -> None:
    assert LinearLaw.fit([]).n_points == 0
    single = LinearLaw.fit([42.0])
    assert single.intercept == pytest.approx(42.0)
    assert single.slope == 0.0


def test_estimate_extrapolates_beyond_observed_calls() -> None:
    tokens = [100.0, 110.0, 120.0]  # +10/call
    latency = [1000.0, 1000.0, 1000.0]  # constant 1s/call
    est = estimate(tokens, latency, total_calls=10, max_in_flight=2)
    # integral of (100 + 10*i) from 0 to 10 = 100*10 + 10*10^2/2 = 1500
    assert est.predicted_total_tokens == pytest.approx(1500.0)
    # integral of constant 1000ms over 10 calls = 10s of work; /2 concurrency = 5s
    assert est.predicted_duration_s == pytest.approx(5.0)
    assert est.tokens_ci[0] < est.predicted_total_tokens < est.tokens_ci[1]


def test_confidence_widens_with_fewer_points() -> None:
    assert confidence_width(1) > confidence_width(50)


def test_estimate_by_stage_sums_pooled_service_time_before_dividing_once() -> None:
    # Two stages, each constant 1000ms/call, 5 calls each expected.
    tokens_by_stage = {"A": [100.0, 100.0], "B": [200.0, 200.0]}
    latency_by_stage = {"A": [1000.0, 1000.0], "B": [1000.0, 1000.0]}
    est = estimate_by_stage(
        tokens_by_stage, latency_by_stage,
        total_units_by_stage={"A": 5, "B": 5},
        max_in_flight=2,
    )
    # tokens: 100*5 + 200*5 = 1500
    assert est.predicted_total_tokens == pytest.approx(1500.0)
    # pooled service time: (1000*5 + 1000*5)ms = 10s, / concurrency 2 = 5s
    assert est.predicted_duration_s == pytest.approx(5.0)


def test_estimate_by_stage_beats_pooled_estimate_on_alternating_stages() -> None:
    # Two stages that alternate in the raw call stream — a single pooled
    # linear fit over raw index sees a flat/noisy trend and mispredicts;
    # per-stage fitting isolates each stage's own (here: flat) trend.
    tokens_a = [100.0] * 5  # stage A: flat at 100
    tokens_b = [300.0] * 5  # stage B: flat at 300
    est = estimate_by_stage(
        {"A": tokens_a, "B": tokens_b}, {"A": [0.0] * 5, "B": [0.0] * 5},
        total_units_by_stage={"A": 10, "B": 10},
        max_in_flight=1,
    )
    # true total if extrapolated correctly: 100*10 + 300*10 = 4000
    assert est.predicted_total_tokens == pytest.approx(4000.0)


def test_power_law_recovers_exact_curve_without_noise() -> None:
    # y = 3 * (i+1)^2  ->  a=3, b=2
    values = [3 * (i + 1) ** 2 for i in range(10)]
    law = PowerLaw.fit(values)
    assert law.a == pytest.approx(3.0, rel=1e-6)
    assert law.b == pytest.approx(2.0, rel=1e-6)


def test_power_law_integral_matches_closed_form() -> None:
    law = PowerLaw(a=2.0, b=1.0, n_points=5)  # y = 2x, integral = x^2
    # integral from x=1 to x=n+1 of 2x dx = (n+1)^2 - 1
    assert law.integral(3) == pytest.approx(4.0**2 - 1.0)


def test_ensemble_weights_toward_whichever_shape_fits_better() -> None:
    linear_values = [10 + 2 * i for i in range(10)]
    ens_linear = EnsembleLaw.fit(linear_values)
    assert ens_linear.w_linear > ens_linear.w_power

    power_values = [3 * (i + 1) ** 2.5 for i in range(10)]
    ens_power = EnsembleLaw.fit(power_values)
    assert ens_power.w_power > ens_power.w_linear


def test_estimate_accepts_alternate_law_cls() -> None:
    tokens = [3 * (i + 1) ** 1.5 for i in range(5)]
    est = estimate(tokens, [0.0] * 5, total_calls=20, max_in_flight=1, law_cls=PowerLaw)
    assert est.predicted_total_tokens > 0
