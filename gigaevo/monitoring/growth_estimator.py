"""Growth-law estimator for live cost prediction.

Fits a per-call growth law (linear) from calls observed so far, integrates
it out to the total call count for cumulative tokens, and applies Little's
Law (wall time = total service time / concurrency) for duration. Refit
after every call, so the estimate is usable from the first few calls of a
run rather than only converging by the end.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import stats as sp_stats


@dataclass
class LinearLaw:
    """y(i) = intercept + slope * i, least-squares fit over observed points
    at indices 0..n-1."""

    intercept: float
    slope: float
    n_points: int

    @classmethod
    def fit(cls, values: list[float]) -> LinearLaw:
        n = len(values)
        if n == 0:
            return cls(intercept=0.0, slope=0.0, n_points=0)
        if n == 1:
            return cls(intercept=values[0], slope=0.0, n_points=1)
        idx = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(idx, np.asarray(values, dtype=float), 1)
        return cls(intercept=float(intercept), slope=float(slope), n_points=n)

    def value_at(self, i: float) -> float:
        return self.intercept + self.slope * i

    def integral(self, n: float) -> float:
        """Cumulative sum of the fitted line over calls 0..n."""
        return max(0.0, self.intercept * n + self.slope * n * n / 2.0)


@dataclass
class PowerLaw:
    """y(x) = a * x^b, x = index + 1 (avoids log(0) at index 0).

    Fit via ordinary least squares in log-log space — the standard way to
    fit a power law without nonlinear optimization.
    """

    a: float
    b: float
    n_points: int

    @classmethod
    def fit(cls, values: list[float]) -> PowerLaw:
        n = len(values)
        if n == 0:
            return cls(a=0.0, b=0.0, n_points=0)
        pos = np.maximum(np.asarray(values, dtype=float), 1e-6)
        if n == 1:
            return cls(a=float(pos[0]), b=0.0, n_points=1)
        xs = np.log(np.arange(1, n + 1, dtype=float))
        ys = np.log(pos)
        b, _ = np.polyfit(xs, ys, 1)
        # Clamp the exponent: genetic-programming bloat literature finds
        # program-size growth is sub-quadratic even in the worst case, and
        # a single noisy early point (e.g. a near-empty first call) can
        # otherwise send b > 2 and blow up ``integral()`` by orders of
        # magnitude when extrapolated far past the observed points.
        b = float(np.clip(b, -1.0, 2.0))
        log_a = ys.mean() - b * xs.mean()
        return cls(a=float(np.exp(log_a)), b=b, n_points=n)

    def value_at(self, i: float) -> float:
        return self.a * (i + 1) ** self.b

    def integral(self, n: float) -> float:
        """Cumulative sum over calls 0..n (integral of a*x^b, x from 1 to n+1)."""
        if n <= 0:
            return 0.0
        if abs(self.b + 1) < 1e-9:
            return max(0.0, self.a * math.log(n + 1))
        return max(0.0, self.a / (self.b + 1) * ((n + 1) ** (self.b + 1) - 1))


@dataclass
class RobustPowerLaw:
    """y(x) = a * x^b, x = index + 1, fit via Theil-Sen (median of all
    pairwise slopes) in log-log space instead of ordinary least squares.

    OLS in log-log space is skewed by a single noisy point (e.g. a
    near-empty first call, common in this system's cold-start calls);
    Theil-Sen has a ~29% breakdown point and stays reliable through that.
    Validated on real logs as the best of several token-count estimators
    (tools/pipeline_15tasks/estimation_research).
    """

    a: float
    b: float
    n_points: int

    @classmethod
    def fit(cls, values: list[float]) -> RobustPowerLaw:
        n = len(values)
        if n == 0:
            return cls(a=0.0, b=0.0, n_points=0)
        pos = np.maximum(np.asarray(values, dtype=float), 1e-6)
        if n == 1:
            return cls(a=float(pos[0]), b=0.0, n_points=1)
        xs = np.log(np.arange(1, n + 1, dtype=float))
        ys = np.log(pos)
        b, _, _, _ = sp_stats.theilslopes(ys, xs)
        b = float(np.clip(b, -1.0, 2.0))  # see PowerLaw.fit
        log_a = float(np.median(ys - b * xs))
        return cls(a=math.exp(log_a), b=b, n_points=n)

    def value_at(self, i: float) -> float:
        return self.a * (i + 1) ** self.b

    def integral(self, n: float) -> float:
        if n <= 0:
            return 0.0
        if abs(self.b + 1) < 1e-9:
            return max(0.0, self.a * math.log(n + 1))
        return max(0.0, self.a / (self.b + 1) * ((n + 1) ** (self.b + 1) - 1))


def fit_ttft_tpot(tokens_out: list[float], latency_ms: list[float]) -> tuple[float, float]:
    """Fit latency_ms = ttft + tpot * tokens_out (OLS).

    The physical model behind LLM decode latency: a roughly-constant
    time-to-first-token plus a per-output-token decode cost. Validated as
    a far better duration predictor than fitting latency as a growth law
    over call index — latency correlates weakly with call order (r≈0.23
    on real logs) but moderately with tokens_out (r≈0.57), and any
    index-based curve fit is dominated by backpressure noise instead of a
    real trend (tools/pipeline_15tasks/estimation_research).
    """
    n = len(tokens_out)
    if n == 0:
        return 0.0, 0.0
    if n < 2:
        return latency_ms[0], 0.0
    xs = np.asarray(tokens_out, dtype=float)
    ys = np.asarray(latency_ms, dtype=float)
    if np.ptp(xs) == 0:
        return float(ys.mean()), 0.0
    tpot, ttft = np.polyfit(xs, ys, 1)
    return float(ttft), float(tpot)


def _residual_ss(law, values: list[float]) -> float:
    idx = np.arange(len(values), dtype=float)
    fitted = np.array([law.value_at(i) for i in idx])
    return float(np.sum((fitted - np.asarray(values, dtype=float)) ** 2))


@dataclass
class EnsembleLaw:
    """LinearLaw + PowerLaw combined, weighted by inverse in-sample residual
    on the points observed so far — whichever shape fits the current run's
    own data better gets more weight. No historical training data needed;
    the weighting is recomputed from the same points passed to ``fit``.
    """

    linear: LinearLaw
    power: PowerLaw
    w_linear: float
    w_power: float
    n_points: int

    @classmethod
    def fit(cls, values: list[float]) -> EnsembleLaw:
        n = len(values)
        linear = LinearLaw.fit(values)
        power = PowerLaw.fit(values)
        if n < 3:
            return cls(linear, power, 0.5, 0.5, n)
        sse_linear = _residual_ss(linear, values) + 1e-9
        sse_power = _residual_ss(power, values) + 1e-9
        w_linear = (1 / sse_linear) / (1 / sse_linear + 1 / sse_power)
        return cls(linear, power, w_linear, 1.0 - w_linear, n)

    def value_at(self, i: float) -> float:
        return self.w_linear * self.linear.value_at(i) + self.w_power * self.power.value_at(i)

    def integral(self, n: float) -> float:
        return self.w_linear * self.linear.integral(n) + self.w_power * self.power.integral(n)


def _bounded_integral(law, values: list[float], n: float) -> float:
    """``law.integral(n)``, backstopped against runaway extrapolation.

    A single noisy point (e.g. a near-empty first call) can still distort
    the fitted intercept/scale enough to blow up the integral by orders of
    magnitude even with a clamped exponent (PowerLaw.fit clamps the slope,
    not the scale). Cap at a generous multiple of the flat (no-growth)
    extrapolation — high enough that real growth trends pass through
    untouched, low enough to catch orders-of-magnitude blowups.
    """
    raw = law.integral(n)
    if not values:
        return raw
    flat = (sum(values) / len(values)) * n
    cap = max(flat * 8.0, max(values) * 2.0)
    return min(raw, cap) if cap > 0 else raw


def tail_integral(law, values: list[float], n: float,
                  fit_values: list[float] | None = None) -> float:
    """Extrapolate ONLY the unobserved tail (buckets ``len(values)``..``n``).

    Two biases this removes, both measured on the 9 collected 100-mutant runs
    (tools/cost_ablation, 2026-07-31):

    1. *Re-predicting the past.* ``_bounded_integral`` fits a law to the
       observed buckets and then integrates from zero, so the already-observed
       part is replaced by the fit's own (lossy) reconstruction of it. On real
       logs the final token estimate came out BELOW the sum the estimator had
       already watched go by. Taking ``sum(values)`` as ground truth and
       predicting only what is left makes the estimate monotonically
       self-correcting and guarantees ``pred >= observed``.
    2. *Retransformation bias.* PowerLaw/RobustPowerLaw fit in log space, so
       exponentiating recovers the GEOMETRIC mean (Theil-Sen: the median);
       what we extrapolate is a SUM, which needs the ARITHMETIC mean. For
       right-skewed token/latency buckets the geometric mean sits well below
       it — a systematic ~-20% undercount. Duan's smearing estimator fixes
       this: rescale the law by the factor that makes it reproduce the
       observed sum in-sample, then extrapolate with that scale.

    Measured effect (median |error| of the final token estimate across the 9
    runs): 22% -> 13%, and residual error then equals exactly the share of
    tokens the LLM_CALL event stream misses, nothing else.

    ``fit_values``: same-length series with outlier buckets replaced by the
    median of the rest (see CostMonitorHook._winsorised). The tail is shaped
    by ``fit_values`` — a transient spike must not be extrapolated over the
    whole remaining run — while ``values`` still anchors the already-observed
    part, because the outlier's cost was genuinely incurred.
    """
    k = len(values)
    if k == 0 or n <= k:
        return 0.0
    fv = fit_values if fit_values and len(fit_values) == k else values
    scale = 1.0
    if k >= 2:
        in_sample = law.integral(k)
        if in_sample > 0:
            scale = min(max(sum(fv) / in_sample, 0.2), 5.0)  # Duan smearing
    tail = (law.integral(n) - law.integral(k)) * scale
    flat = (sum(fv) / k) * (n - k)  # no-growth backstop, see _bounded_integral
    # Cap the tail at a multiple of the flat projection, tightened while
    # evidence is thin: a power law fitted on ~10 buckets and extrapolated to
    # 100 can multiply by (100/10)^2 on nothing but noise. Relaxes back to the
    # old generous 8x ceiling once enough buckets support the trend.
    cap_mult = 1.5 + 6.5 * k / (k + 20)
    return min(max(tail, 0.0), flat * cap_mult)


def _resample_tail_totals(
    law_cls: type, values: list[float], n: float, noise_fn, n_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shared driver for the resampling-based CI methods below: refit
    ``law_cls`` on a noised copy of ``values`` and extrapolate with
    :func:`tail_integral`, ``n_iter`` times. ``values`` still anchors the
    real observed sum (see tail_integral) — only the tail *shape* varies
    across iterations, via ``fit_values``.
    """
    k = len(values)
    law = law_cls.fit(values)
    fitted = np.array([law.value_at(i) for i in range(k)])
    resid = np.asarray(values, dtype=float) - fitted
    base = sum(values)
    totals = np.empty(n_iter)
    for i in range(n_iter):
        sim_values = list(np.maximum(fitted + noise_fn(resid, rng), 1e-6))
        sim_law = law_cls.fit(sim_values)
        totals[i] = base + tail_integral(sim_law, values, n, fit_values=sim_values)
    return totals


def tail_ci_bootstrap(
    law_cls: type, values: list[float], n: float, *,
    n_iter: int = 200, alpha: float = 0.1, rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Residual-bootstrap CI for the anchored total (observed sum + tail).

    Resamples the in-sample fit residuals with replacement, refits the
    growth law on the perturbed series, and re-extrapolates — repeated
    ``n_iter`` times to build an empirical distribution over the tail total,
    then takes the ``alpha``/``1-alpha`` percentiles. No assumption on the
    residual distribution's shape, unlike :func:`tail_ci_montecarlo`; needs
    >=2 points to have any residuals to resample.
    """
    k = len(values)
    total_point = sum(values)
    if k < 2 or n <= k:
        return (total_point, total_point)
    rng = rng or np.random.default_rng(0)
    totals = _resample_tail_totals(
        law_cls, values, n,
        lambda resid, r: r.choice(resid, size=len(resid), replace=True),
        n_iter, rng)
    lo, hi = np.percentile(totals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(max(lo, total_point)), float(max(hi, total_point))


def tail_ci_montecarlo(
    law_cls: type, values: list[float], n: float, *,
    n_iter: int = 200, alpha: float = 0.1, rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Parametric Monte Carlo CI: assumes in-sample residuals are i.i.d.
    Normal(0, sigma) (sigma estimated from those residuals), draws fresh
    Gaussian noise each iteration instead of resampling the observed
    residuals, refits and re-extrapolates. Cheaper than the bootstrap and
    smoother with few points, but under-covers if the true residuals are
    skewed (token/latency buckets typically are).
    """
    k = len(values)
    total_point = sum(values)
    if k < 2 or n <= k:
        return (total_point, total_point)
    rng = rng or np.random.default_rng(0)
    law0 = law_cls.fit(values)
    fitted0 = np.array([law0.value_at(i) for i in range(k)])
    sigma = float(np.std(np.asarray(values, dtype=float) - fitted0)) or 1e-6
    totals = _resample_tail_totals(
        law_cls, values, n,
        lambda resid, r: r.normal(0.0, sigma, size=len(resid)),
        n_iter, rng)
    lo, hi = np.percentile(totals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(max(lo, total_point)), float(max(hi, total_point))


def tail_ci_bayesian(
    law_cls: type, values: list[float], n: float, *, alpha: float = 0.1,
) -> tuple[float, float]:
    """Closed-form Bayesian CI for the anchored total.

    Normal-Inverse-Gamma conjugate posterior over the fit residual variance
    (weak prior a0=1, b0=max(var, 1)), giving a Student-t posterior
    predictive for the sum of the remaining tail points. Deliberately
    conjugate/closed-form rather than MCMC — a single principled uncertainty
    model without simulation cost — at the price of assuming Normal
    residuals and treating the ``n - k`` future points as independent draws
    (their variances add; no growth-law correlation between them is
    modelled).
    """
    k = len(values)
    total_point = sum(values)
    if k < 2 or n <= k:
        return (total_point, total_point)
    law = law_cls.fit(values)
    fitted = np.array([law.value_at(i) for i in range(k)])
    resid = np.asarray(values, dtype=float) - fitted
    a0, b0 = 1.0, max(float(np.var(resid)), 1.0)
    a_n = a0 + k / 2.0
    b_n = b0 + 0.5 * float(np.sum(resid ** 2))
    remaining = n - k
    tail_point = tail_integral(law, values, n)
    scale = math.sqrt(b_n / a_n * (1.0 + 1.0 / k) * remaining)
    t_crit = float(sp_stats.t.ppf(1 - alpha / 2, df=2 * a_n))
    half = t_crit * scale
    lo, hi = total_point + tail_point - half, total_point + tail_point + half
    return float(max(lo, total_point)), float(hi)


CI_METHODS = {
    "bootstrap": tail_ci_bootstrap,
    "montecarlo": tail_ci_montecarlo,
    "bayesian": tail_ci_bayesian,
}


def relative_tail_width(ci_method: str, law_cls: type, values: list[float], n: float) -> float:
    """Turn one of :data:`CI_METHODS`' absolute (lo, hi) bands into a
    relative half-width comparable to :func:`confidence_width`'s role, so
    any of the three probabilistic methods can be swapped in wherever the
    1/sqrt(n) heuristic is used (:func:`estimate_by_stage`,
    :func:`estimate_duration_by_stage`) without touching the surrounding
    anchoring/tail_mult/concurrency machinery. Falls back to the heuristic
    when there isn't enough data for the method to produce a band.
    """
    k = len(values)
    if k < 2 or n <= k:
        return confidence_width(k)
    lo, hi = CI_METHODS[ci_method](law_cls, values, n)
    tail_point = tail_integral(law_cls.fit(values), values, n)
    if tail_point <= 0:
        return confidence_width(k)
    return float(np.clip((hi - lo) / (2 * tail_point), 0.05, 2.0))


def confidence_width(n_points: int) -> float:
    """Relative half-width of the prediction interval. Shrinks as evidence
    accumulates; wide by default so a 1-2 point fit doesn't claim precision
    it doesn't have.

    ponytail: 1/sqrt(n) heuristic, not a real prediction interval — swap
    if validation shows it's too loose/tight.
    """
    return max(0.15, min(1.0, 1.0 / math.sqrt(max(n_points, 1))))


@dataclass
class GrowthEstimate:
    predicted_total_tokens: float
    predicted_duration_s: float
    tokens_ci: tuple[float, float]
    duration_ci: tuple[float, float]
    n_points: int


def estimate(
    tokens_per_call: list[float],
    latency_ms_per_call: list[float],
    *,
    total_calls: int,
    max_in_flight: int,
    law_cls: type = LinearLaw,
) -> GrowthEstimate:
    """Fit growth laws from calls seen so far, integrate out to
    ``total_calls`` for cumulative tokens, and apply Little's Law
    (wall_time = total_service_time / concurrency) for duration.
    """
    token_law = law_cls.fit(tokens_per_call)
    latency_law = law_cls.fit(latency_ms_per_call)

    total_tokens = _bounded_integral(token_law, tokens_per_call, total_calls)
    total_latency_s = _bounded_integral(latency_law, latency_ms_per_call, total_calls) / 1000.0
    duration_s = total_latency_s / max(max_in_flight, 1)

    w_tok = confidence_width(token_law.n_points)
    w_dur = confidence_width(latency_law.n_points)
    return GrowthEstimate(
        predicted_total_tokens=total_tokens,
        predicted_duration_s=duration_s,
        tokens_ci=(total_tokens * (1 - w_tok), total_tokens * (1 + w_tok)),
        duration_ci=(duration_s * (1 - w_dur), duration_s * (1 + w_dur)),
        n_points=token_law.n_points,
    )


def estimate_by_stage(
    tokens_by_stage: dict[str, list[float]],
    latency_by_stage: dict[str, list[float]],
    *,
    total_units_by_stage: dict[str, int],
    max_in_flight: int,
    law_cls: type = LinearLaw,
    fit_tokens_by_stage: dict[str, list[float]] | None = None,
    fit_latency_by_stage: dict[str, list[float]] | None = None,
    ci_method: str | None = None,
) -> GrowthEstimate:
    """Same as :func:`estimate`, but fits one growth law per call stage
    (e.g. ``MutationSuggestionAgent`` vs ``MutationAgent``) instead of one
    law over the pooled, heterogeneous call stream.

    Different stages have structurally different token/latency profiles —
    pooling them makes the index axis alternate between two distributions
    instead of following a single growth trend, which breaks the linear
    fit. Fitting per stage and summing keeps each law honest.

    Little's Law is applied ONCE, on the pooled total service time across
    all stages (not per stage then divided again) — the stages share the
    same ``max_in_flight`` concurrency pool.
    """
    total_tokens = 0.0
    tok_ci_lo = tok_ci_hi = 0.0
    total_latency_s = 0.0
    lat_ci_lo_s = lat_ci_hi_s = 0.0
    n_points = 0

    for stage, tokens in tokens_by_stage.items():
        latency = latency_by_stage.get(stage, [])
        n = total_units_by_stage.get(stage, len(tokens))
        # Outlier-suppressed copies shape the extrapolation; the raw series
        # still anchors the observed part. See :func:`tail_integral`.
        fit_tok = (fit_tokens_by_stage or {}).get(stage, tokens)
        fit_lat = (fit_latency_by_stage or {}).get(stage, latency)

        tok_law = law_cls.fit(fit_tok)
        lat_law = law_cls.fit(fit_lat)
        tok_total = sum(tokens) + tail_integral(tok_law, tokens, n, fit_tok)
        lat_total_s = (sum(latency) + tail_integral(lat_law, latency, n, fit_lat)) / 1000.0
        if ci_method:
            w_tok = relative_tail_width(ci_method, law_cls, fit_tok, n)
            w_lat = relative_tail_width(ci_method, law_cls, fit_lat, n)
        else:
            w_tok = confidence_width(tok_law.n_points)
            w_lat = confidence_width(lat_law.n_points)

        total_tokens += tok_total
        tok_ci_lo += tok_total * (1 - w_tok)
        tok_ci_hi += tok_total * (1 + w_tok)
        total_latency_s += lat_total_s
        lat_ci_lo_s += lat_total_s * (1 - w_lat)
        lat_ci_hi_s += lat_total_s * (1 + w_lat)
        n_points += tok_law.n_points

    duration_s = total_latency_s / max(max_in_flight, 1)
    return GrowthEstimate(
        predicted_total_tokens=total_tokens,
        predicted_duration_s=duration_s,
        tokens_ci=(tok_ci_lo, tok_ci_hi),
        duration_ci=(lat_ci_lo_s / max(max_in_flight, 1), lat_ci_hi_s / max(max_in_flight, 1)),
        n_points=n_points,
    )


def achieved_concurrency(
    calls: list[tuple[float, float]],
    *,
    now_s: float,
    max_in_flight: int,
    window_frac: float = 0.5,
    min_window_s: float = 300.0,
    shrink_k: int = 80,
) -> float:
    """How many LLM calls the system is ACTUALLY running at once, measured as
    service-time-per-wall-second over a trailing window.

    ``max_in_flight`` is the DAG's mutant-dispatch cap, not the LLM
    concurrency: one mutant DAG issues several LLM calls that overlap, and
    conversely a low-accept-rate or slow-validator task never fills the pool.
    Measured on the 9 collected runs the true value ranged from 4.2 to 14.5
    against a constant ``max_in_flight=8`` — and Little's Law divided by 8
    reproduced the whole observed duration-error spread (-47%..+80%) on its
    own. Nothing else in the duration model was materially wrong.

    Measured over a trailing window rather than from t0, because the run ramps
    up and a cumulative average never lives that cold start down. Shrunk
    toward ``max_in_flight`` while call evidence is thin, so the first few
    calls (when the ramp reads as near-zero concurrency) can't blow the
    estimate up. ``shrink_k`` / ``min_window_s`` were swept on those 9 runs;
    80 / 300s minimised error at 25% and 50% progress without hurting the
    final estimate.

    ``calls``: ``(completion_time_s, latency_ms)`` for every call so far.
    """
    if not calls:
        return float(max_in_flight)
    t_first = min(ts - lat / 1000.0 for ts, lat in calls)
    if now_s - t_first <= 1.0:
        return float(max_in_flight)
    w_start = max(t_first, now_s - max(min_window_s, (now_s - t_first) * window_frac))
    service_s = sum(lat for ts, lat in calls if ts > w_start) / 1000.0
    span_s = now_s - w_start
    if span_s <= 1.0 or service_s <= 0:
        return float(max_in_flight)
    n = len(calls)
    w = n / (n + shrink_k)
    blended = w * (service_s / span_s) + (1 - w) * max_in_flight
    return min(max(blended, 0.5), 64.0)


def estimate_duration_by_stage(
    tokens_out_by_stage: dict[str, list[float]],
    latency_by_stage: dict[str, list[float]],
    *,
    total_units_by_stage: dict[str, int],
    max_in_flight: int,
    token_law_cls: type = RobustPowerLaw,
    nonllm_duration_by_stage: dict[str, list[float]] | None = None,
    nonllm_law_cls: type = RobustPowerLaw,
    elapsed_s: float = 0.0,
    concurrency: float | None = None,
    tail_mult: float = 1.0,
    fit_tokens_out_by_stage: dict[str, list[float]] | None = None,
    fit_latency_by_stage: dict[str, list[float]] | None = None,
    fit_nonllm_by_stage: dict[str, list[float]] | None = None,
    ci_method: str | None = None,
) -> tuple[float, tuple[float, float]]:
    """Predict total wall-clock duration via the TTFT+TPOT physical model
    (:func:`fit_ttft_tpot`) per stage instead of fitting latency itself as
    a growth law over call index — latency doesn't follow a growth trend
    in this system (dominated by backpressure noise, r≈0.23 with call
    index vs r≈0.57 with tokens_out); see :func:`fit_ttft_tpot`.

    ``nonllm_duration_by_stage``: bucketed wall-clock duration_ms for
    non-LLM pipeline stages (e.g. CallValidatorFunction, CallProgramFunction)
    that the LLM-latency model above never sees. Historical validation
    (2026-07-27 cost-model report, 21 real runs) found these dominate 42-94%
    of wall time on some tasks, and LLM-latency-only duration estimates
    underestimate systematically as a result (-33% median bias measured
    there). Fit directly as a growth law over duration_ms (no tokens_out
    concept applies to a validator/executor stage) and folded into the same
    Little's Law division as the LLM contribution below, not a second one.

    ``elapsed_s`` / ``concurrency``: the estimate is ANCHORED — wall time
    already spent is known exactly, so only the REMAINING service time is
    predicted, and it is divided by the concurrency the system is actually
    achieving (see :func:`achieved_concurrency`) rather than by the
    ``max_in_flight`` dispatch cap. ``concurrency=None`` falls back to
    ``max_in_flight`` (old behaviour).

    ``tail_mult``: CostMonitorAgent's calibration multiplier. Applied to the
    remaining-work term only — scaling the elapsed term too would let the
    agent "correct" wall time that has already been measured.

    ``ci_method``: swaps the 1/sqrt(n) heuristic width for one of
    :data:`CI_METHODS` (via :func:`relative_tail_width`), weighted by each
    stage's share of the remaining service time. ``None`` (default) keeps
    the exact original heuristic behaviour.

    Returns ``(duration_s, (ci_low_s, ci_high_s))``.
    """
    remaining_ms = 0.0
    n_points = 0
    weighted_w_num = 0.0  # sum(w_stage * remaining_ms_stage), only used if ci_method

    for stage, tokens_out in tokens_out_by_stage.items():
        latency = latency_by_stage.get(stage, [])
        n = total_units_by_stage.get(stage, len(tokens_out))
        k = len(tokens_out)
        n_points += k
        if n <= k:
            continue

        fit_out = (fit_tokens_out_by_stage or {}).get(stage, tokens_out)
        fit_lat = (fit_latency_by_stage or {}).get(stage, latency)
        ttft, tpot = fit_ttft_tpot(fit_out, fit_lat)
        # OLS is unconstrained and a noisy fit can hand back a negative
        # intercept or slope, which turned into negative predicted durations
        # on real logs. Both are physically non-negative.
        ttft, tpot = max(ttft, 0.0), max(tpot, 0.0)
        tail_out = tail_integral(token_law_cls.fit(fit_out), tokens_out, n, fit_out)
        stage_remaining_ms = ttft * (n - k) + tpot * tail_out
        remaining_ms += stage_remaining_ms
        if ci_method:
            weighted_w_num += relative_tail_width(ci_method, token_law_cls, fit_out, n) * stage_remaining_ms

    for stage, durations in (nonllm_duration_by_stage or {}).items():
        n = total_units_by_stage.get(stage, len(durations))
        fit_dur = (fit_nonllm_by_stage or {}).get(stage, durations)
        stage_remaining_ms = tail_integral(nonllm_law_cls.fit(fit_dur), durations, n, fit_dur)
        remaining_ms += stage_remaining_ms
        n_points += len(durations)
        if ci_method:
            weighted_w_num += relative_tail_width(ci_method, nonllm_law_cls, fit_dur, n) * stage_remaining_ms

    conc = concurrency if concurrency and concurrency > 0 else max(max_in_flight, 1)
    duration_s = max(0.0, elapsed_s + remaining_ms * tail_mult / 1000.0 / conc)
    w = (weighted_w_num / remaining_ms) if (ci_method and remaining_ms > 0) else confidence_width(n_points)
    # Only the predicted part carries uncertainty; elapsed time is measured.
    half = (duration_s - elapsed_s) * w
    return duration_s, (duration_s - half, duration_s + half)
