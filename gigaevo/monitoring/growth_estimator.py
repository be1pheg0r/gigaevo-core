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

    total_tokens = token_law.integral(total_calls)
    total_latency_s = latency_law.integral(total_calls) / 1000.0
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
        tok_total = tok_law.integral(n)
        lat_total_s = lat_law.integral(n) / 1000.0
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
