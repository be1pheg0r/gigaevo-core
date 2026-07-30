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
import statistics as st


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
        mean_i = (n - 1) / 2.0
        mean_y = sum(values) / n
        cov = sum((i - mean_i) * (y - mean_y) for i, y in enumerate(values))
        var = sum((i - mean_i) ** 2 for i in range(n))
        slope = cov / var if var > 0 else 0.0
        intercept = mean_y - slope * mean_i
        return cls(intercept=intercept, slope=slope, n_points=n)

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
        pos = [max(v, 1e-6) for v in values]
        if n == 1:
            return cls(a=pos[0], b=0.0, n_points=1)
        xs = [math.log(i + 1) for i in range(n)]
        ys = [math.log(v) for v in pos]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var = sum((x - mean_x) ** 2 for x in xs)
        b = cov / var if var > 0 else 0.0
        # Clamp the exponent: genetic-programming bloat literature finds
        # program-size growth is sub-quadratic even in the worst case, and
        # a single noisy early point (e.g. a near-empty first call) can
        # otherwise send b > 2 and blow up ``integral()`` by orders of
        # magnitude when extrapolated far past the observed points.
        b = max(-1.0, min(b, 2.0))
        log_a = mean_y - b * mean_x
        return cls(a=math.exp(log_a), b=b, n_points=n)

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
        pos = [max(v, 1e-6) for v in values]
        if n == 1:
            return cls(a=pos[0], b=0.0, n_points=1)
        xs = [math.log(i + 1) for i in range(n)]
        ys = [math.log(v) for v in pos]
        slopes = [
            (ys[j] - ys[i]) / (xs[j] - xs[i])
            for i in range(n) for j in range(i + 1, n)
            if xs[j] != xs[i]
        ]
        b = st.median(slopes) if slopes else 0.0
        b = max(-1.0, min(b, 2.0))  # see PowerLaw.fit
        log_a = st.median(y - b * x for x, y in zip(xs, ys))
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
    mean_x = sum(tokens_out) / n
    mean_y = sum(latency_ms) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(tokens_out, latency_ms))
    var = sum((x - mean_x) ** 2 for x in tokens_out)
    tpot = cov / var if var > 0 else 0.0
    ttft = mean_y - tpot * mean_x
    return ttft, tpot


def _residual_ss(law, values: list[float]) -> float:
    return sum((law.value_at(i) - y) ** 2 for i, y in enumerate(values))


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

        tok_law = law_cls.fit(tokens)
        lat_law = law_cls.fit(latency)
        tok_total = _bounded_integral(tok_law, tokens, n)
        lat_total_s = _bounded_integral(lat_law, latency, n) / 1000.0
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


def estimate_duration_by_stage(
    tokens_out_by_stage: dict[str, list[float]],
    latency_by_stage: dict[str, list[float]],
    *,
    total_units_by_stage: dict[str, int],
    max_in_flight: int,
    token_law_cls: type = RobustPowerLaw,
) -> tuple[float, tuple[float, float]]:
    """Predict total wall-clock duration via the TTFT+TPOT physical model
    (:func:`fit_ttft_tpot`) per stage instead of fitting latency itself as
    a growth law over call index — latency doesn't follow a growth trend
    in this system (dominated by backpressure noise, r≈0.23 with call
    index vs r≈0.57 with tokens_out); see :func:`fit_ttft_tpot`.

    Returns ``(duration_s, (ci_low_s, ci_high_s))``.
    """
    total_latency_ms = 0.0
    n_points = 0

    for stage, tokens_out in tokens_out_by_stage.items():
        latency = latency_by_stage.get(stage, [])
        n = total_units_by_stage.get(stage, len(tokens_out))

        ttft, tpot = fit_ttft_tpot(tokens_out, latency)
        tok_law = token_law_cls.fit(tokens_out)
        tok_out_total = _bounded_integral(tok_law, tokens_out, n)
        mean_tok_out = tok_out_total / max(n, 1)

        total_latency_ms += (ttft + tpot * mean_tok_out) * n
        n_points += len(tokens_out)

    duration_s = max(0.0, total_latency_ms / 1000.0 / max(max_in_flight, 1))
    w = confidence_width(n_points)
    return duration_s, (duration_s * (1 - w), duration_s * (1 + w))
