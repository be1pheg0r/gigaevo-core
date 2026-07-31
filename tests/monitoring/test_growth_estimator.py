from __future__ import annotations

import pytest

from gigaevo.monitoring.growth_estimator import (
    EnsembleLaw,
    LinearLaw,
    PowerLaw,
    RobustPowerLaw,
    achieved_concurrency,
    confidence_width,
    estimate,
    estimate_by_stage,
    estimate_duration_by_stage,
    fit_ttft_tpot,
    tail_integral,
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

    power_values = [3 * (i + 1) ** 1.7 for i in range(10)]
    ens_power = EnsembleLaw.fit(power_values)
    assert ens_power.w_power > ens_power.w_linear


def test_estimate_accepts_alternate_law_cls() -> None:
    tokens = [3 * (i + 1) ** 1.5 for i in range(5)]
    est = estimate(tokens, [0.0] * 5, total_calls=20, max_in_flight=1, law_cls=PowerLaw)
    assert est.predicted_total_tokens > 0


def test_robust_power_law_ignores_a_single_outlier_point() -> None:
    # One anomalously tiny first value (e.g. a near-empty cold-start call)
    # should not dominate the fit the way it would under OLS.
    values = [1.0, 500.0, 520.0, 540.0, 560.0]
    robust = RobustPowerLaw.fit(values)
    ols = PowerLaw.fit(values)
    # Both clamp b<=2, but the robust fit's extrapolation stays far more
    # conservative than the OLS fit dragged around by the outlier.
    assert robust.integral(50) < ols.integral(50)


def test_bounded_integral_caps_runaway_extrapolation() -> None:
    # A single tiny first point can send OLS log-log slope near the clamp
    # ceiling; integrating that far past the observed range must not blow
    # up to orders of magnitude beyond anything plausible.
    values = [1.0, 500.0, 520.0, 540.0, 560.0]
    est = estimate(values, [0.0] * 5, total_calls=200, max_in_flight=1, law_cls=PowerLaw)
    # Flat (no-growth) extrapolation would be ~mean(values)*200 ≈ 84,400.
    assert est.predicted_total_tokens < 84_400 * 10


def test_fit_ttft_tpot_recovers_exact_linear_relationship() -> None:
    tokens_out = [100.0, 200.0, 300.0, 400.0]
    latency_ms = [1000.0 + 20.0 * t for t in tokens_out]  # ttft=1000, tpot=20
    ttft, tpot = fit_ttft_tpot(tokens_out, latency_ms)
    assert ttft == pytest.approx(1000.0)
    assert tpot == pytest.approx(20.0)


def test_estimate_duration_by_stage_uses_tokens_out_not_call_index() -> None:
    # Latency is flat regardless of call index (no growth trend), but
    # scales with tokens_out — the TTFT/TPOT model should recover that
    # even though a growth-law-over-index fit would see nothing.
    tokens_out_by_stage = {"A": [100.0, 200.0, 200.0, 100.0]}  # no trend over index
    latency_by_stage = {"A": [1100.0, 2100.0, 2100.0, 1100.0]}  # ttft=100, tpot=10
    # 8 total units, 4 observed -> only the 4 remaining are predicted.
    duration_s, ci = estimate_duration_by_stage(
        tokens_out_by_stage, latency_by_stage,
        total_units_by_stage={"A": 8}, max_in_flight=1,
    )
    # mean tokens_out=150 -> per-call latency ~= 100+10*150=1600ms; *4 remaining = 6.4s
    assert duration_s == pytest.approx(6.4, rel=0.05)
    assert ci[0] <= duration_s <= ci[1]


def test_duration_anchors_on_elapsed_and_divides_by_achieved_concurrency() -> None:
    tokens_out_by_stage = {"A": [100.0, 100.0]}
    latency_by_stage = {"A": [1100.0, 1100.0]}  # ttft=0, tpot=11 -> 1100ms/unit
    duration_s, _ = estimate_duration_by_stage(
        tokens_out_by_stage, latency_by_stage,
        total_units_by_stage={"A": 6}, max_in_flight=8,
        elapsed_s=500.0, concurrency=2.0,
    )
    # 4 remaining units * 1100ms = 4.4s of service, at concurrency 2 -> 2.2s
    assert duration_s == pytest.approx(502.2, rel=0.02)


def test_duration_never_goes_negative_on_a_noisy_ttft_fit() -> None:
    # Latency FALLS as tokens_out rises — OLS hands back a negative slope,
    # which used to produce a negative predicted duration.
    duration_s, _ = estimate_duration_by_stage(
        {"A": [100.0, 200.0, 300.0]}, {"A": [3000.0, 2000.0, 1000.0]},
        total_units_by_stage={"A": 20}, max_in_flight=4,
    )
    assert duration_s >= 0.0


def test_achieved_concurrency_measures_parallelism_not_the_dispatch_cap() -> None:
    # 20 calls of 10s each all completing within a 25s window => ~8 in parallel,
    # even though max_in_flight says 2.
    calls = [(5.0 + i * 1.0, 10_000.0) for i in range(20)]
    conc = achieved_concurrency(calls, now_s=25.0, max_in_flight=2, shrink_k=0)
    assert conc > 4.0
    # ...but with only 20 calls of evidence the default shrinkage still keeps
    # it close to the max_in_flight prior.
    shrunk = achieved_concurrency(calls, now_s=25.0, max_in_flight=2)
    assert conc > shrunk > 2.0


def test_achieved_concurrency_falls_back_to_dispatch_cap_before_any_calls() -> None:
    assert achieved_concurrency([], now_s=100.0, max_in_flight=8) == 8.0


def test_tail_integral_never_undercuts_what_was_already_observed() -> None:
    # Right-skewed buckets: a log-space fit recovers the geometric mean and
    # would extrapolate below the observed arithmetic mean without smearing.
    values = [100.0, 120.0, 110.0, 5000.0, 130.0, 115.0]
    law = RobustPowerLaw.fit(values)
    tail = tail_integral(law, values, 12)
    assert tail >= 0.0
    # 6 more buckets should be worth roughly what the observed 6 were worth,
    # not the (much smaller) median-based extrapolation.
    assert tail > sum(values) * 0.3
