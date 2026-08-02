"""Cost Monitor Hook — call CostMonitorAgent every N mutants.

Designed as a post-step hook for the evolution engine.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time

from loguru import logger

from gigaevo.llm.agents.cost_monitor import (
    CostMonitorAgent,
    LlmCallRecord,
    _ToolSet,
)
from gigaevo.monitoring.cost_predictor import CostPrediction
from gigaevo.monitoring.emit import emit, subscribe
from gigaevo.monitoring.events import (
    BackpressureSample,
    CostAgentAdjustment,
    LLMCall,
    MutationAttempted,
    StageExec,
)
from gigaevo.monitoring.growth_estimator import (
    RobustPowerLaw,
    achieved_concurrency,
    estimate_by_stage,
    estimate_duration_by_stage,
)


def _clamp_step(current: float, proposed: float, max_rel_step: float = 0.3) -> float:
    """Limit how far one calibration call can move a multiplier relative to
    its current value — caps the blast radius of a single (possibly wrong)
    LLM judgment call, instead of letting it jump anywhere in the full
    accepted range (0.5–2.0 golden / 0.3–3.0 growth) in one step."""
    lo, hi = current * (1 - max_rel_step), current * (1 + max_rel_step)
    return max(lo, min(hi, proposed))


# Minimum relative change accepted from an agent lever.
LEVER_DEADBAND = 0.05

# Recent reversals are damped to prevent controller oscillation.
REVERSAL_WINDOW = 2

# Minimum evidence required for levers that affect all remaining work.
MIN_SUSTAINED_CALLS = 5


# Recurring non-LLM stages included in duration projection.
NON_LLM_DURATION_STAGES = frozenset(
    {
        "CallProgramFunction",
        "CallValidatorFunction",
        "IntraMemoryStage",
    }
)

# Observer calls are excluded from the telemetry they inspect.
OBSERVER_STAGES = frozenset({"CostMonitorAgent", "NoOpCostMonitorAgent"})

# Empirical multiplier for the unobserved tail at each progress checkpoint.
# Recompute with ``tools/cost_ablation/calibrate_tail.py``.
TAIL_CALIBRATION = (
    (0.05, 1.83),
    (0.10, 1.90),
    (0.15, 1.60),
    (0.20, 1.43),
    (0.25, 1.22),
    (0.35, 1.07),
    (0.50, 1.03),
    (1.00, 1.00),
)


class CostMonitorHook:
    """Runs CostMonitorAgent every N mutants, feeds results into CostPrediction."""

    def __init__(
        self,
        agent: CostMonitorAgent,
        prediction: CostPrediction,
        interval: int = 5,
        clock=time.monotonic,
        max_agent_calls: int = 12,
        cooldown_attempts: int = 5,
        warmup_attempts: int = 5,
        min_leverage: float = 0.15,
        ci_method: str | None = None,
        aci_alpha: float = 0.10,
        aci_gamma: float = 0.05,
        disagreement_gap: float = 0.15,
        max_relative_step: float = 0.30,
        reversal_step_factor: float = 0.50,
        lever_deadband: float = LEVER_DEADBAND,
        reversal_window: int = REVERSAL_WINDOW,
        min_sustained_calls: int = MIN_SUSTAINED_CALLS,
        non_llm_duration_stages: list[str] | tuple[str, ...] = tuple(
            NON_LLM_DURATION_STAGES
        ),
        observer_stages: list[str] | tuple[str, ...] = tuple(OBSERVER_STAGES),
        tail_calibration: list[list[float]] | tuple[tuple[float, float], ...] = (
            TAIL_CALIBRATION
        ),
        concurrency_window_frac: float = 0.50,
        concurrency_min_window_s: float = 300.0,
        concurrency_shrink_k: int = 80,
        min_effective_concurrency: float = 0.50,
        max_effective_concurrency: float = 64.0,
        program_size_min_samples: int = 8,
        program_size_window_divisor: int = 4,
        projection_recent_min_samples: int = 4,
        aci_scale_min: float = 0.50,
        aci_scale_max: float = 5.0,
        golden_ratio_min: float = 0.50,
        golden_ratio_max: float = 2.0,
        growth_rate_min: float = 0.30,
        growth_rate_max: float = 3.0,
        concurrency_mult_min: float = 0.30,
        concurrency_mult_max: float = 3.0,
        recent_calls_limit: int = 20,
        fallback_backpressure_util: float = 0.80,
    ):
        self._agent = agent
        self._pred = prediction
        self._interval = interval
        self._counter = 0
        self._clock = clock
        # ``None`` selects the default 1/sqrt(n) interval heuristic.
        self._ci_method = ci_method
        self._t0 = clock()
        # Completion time and latency for measured-concurrency estimation.
        self._call_times: list[tuple[float, float]] = []
        self._call_history: list[LlmCallRecord] = []
        # Both attempt and accepted-mutant paths flush; this index deduplicates them.
        self._history_flush_idx = 0
        self._tokens_by_stage: dict[str, list[float]] = {}
        self._tokens_out_by_stage: dict[str, list[float]] = {}
        self._latency_by_stage: dict[str, list[float]] = {}
        # ``max_mutants`` and this counter both measure mutation attempts.
        self._attempts = 0
        self._concurrency = float(prediction.max_in_flight)
        # Map agent-visible call indices to growth-law bucket positions.
        self._call_bucket: dict[int, tuple[str, int]] = {}
        self._flagged_buckets: dict[str, set[int]] = {}
        # Agent scheduling is driven by interval breaches and progress disagreement.
        self._agent_due = False
        self._agent_calls = 0
        # Claimed before coroutine startup to prevent duplicate dispatch.
        self._agent_running = False
        self._last_agent_attempt = -(10**9)
        self._prev_ci: tuple[float, float] | None = None
        self._trigger_reason = ""
        # Preserve the previous adjustment for closed-loop feedback.
        self._last_adjustment: dict | None = None
        self._max_agent_calls = max_agent_calls
        self._cooldown_attempts = cooldown_attempts
        # Minimum gap between token and clock progress before waking the agent.
        self._disagreement_gap = disagreement_gap
        self._warmup_attempts = warmup_attempts
        self._min_leverage = min_leverage
        self._max_relative_step = max_relative_step
        self._reversal_step_factor = reversal_step_factor
        self._lever_deadband = lever_deadband
        self._reversal_window = reversal_window
        self._min_sustained_calls = min_sustained_calls
        self._non_llm_duration_stages = frozenset(non_llm_duration_stages)
        self._observer_stages = frozenset(observer_stages)
        self._tail_calibration_points = tuple(
            (float(progress), float(multiplier))
            for progress, multiplier in tail_calibration
        )
        if not self._tail_calibration_points:
            raise ValueError("tail_calibration must contain at least one point")
        self._concurrency_window_frac = concurrency_window_frac
        self._concurrency_min_window_s = concurrency_min_window_s
        self._concurrency_shrink_k = concurrency_shrink_k
        self._min_effective_concurrency = min_effective_concurrency
        self._max_effective_concurrency = max_effective_concurrency
        self._program_size_min_samples = program_size_min_samples
        self._program_size_window_divisor = program_size_window_divisor
        self._projection_recent_min_samples = projection_recent_min_samples
        self._aci_scale_min = aci_scale_min
        self._aci_scale_max = aci_scale_max
        self._golden_ratio_min = golden_ratio_min
        self._golden_ratio_max = golden_ratio_max
        self._growth_rate_min = growth_rate_min
        self._growth_rate_max = growth_rate_max
        self._concurrency_mult_min = concurrency_mult_min
        self._concurrency_mult_max = concurrency_mult_max
        self._recent_calls_limit = recent_calls_limit
        self._fallback_backpressure_util = fallback_backpressure_util
        # Online interval calibration (see _update_aci). alpha is the target
        # miscoverage rate, i.e. also the target agent wakeup rate; gamma is
        # the step. gamma=0 disables it and restores the model-only width.
        self._aci_alpha = aci_alpha
        self._aci_gamma = aci_gamma
        self._aci_scale = 1.0
        # Intervals published but not yet contradicted by the elapsed clock.
        self._pending_intervals: list[tuple[float, float]] = []
        self._flushes = 0
        self._miscoverages = 0
        self._agent_task: asyncio.Task | None = None
        # Kept only to report what watching cost — never fed to the estimator.
        self._observer_calls = 0
        self._observer_latency_ms = 0.0
        # Call indices already winsorised, so the agent can be told not to
        # spend another wakeup rediscovering a spike it has handled.
        self._flagged_calls: set[int] = set()
        # lever name -> (direction of last accepted move, wakeup it happened on)
        self._lever_dir: dict[str, tuple[int, int]] = {}
        # stage -> attempt index of every bucket it produced; see _project_units.
        self._bucket_attempts: dict[str, list[int]] = {}
        # (horizon, alpha) -> (tokens, ci); see project_tokens. Dropped on flush.
        self._projection_memo: dict[
            tuple[int, float], tuple[float, tuple[float, float]]
        ] = {}
        subscribe(LLMCall.event, self._on_llm_call)
        subscribe(MutationAttempted.event, self._on_mutation_attempted)
        subscribe(BackpressureSample.event, self._on_backpressure_sample)
        subscribe(StageExec.event, self._on_stage_exec)
        self._last_bp: BackpressureSample | None = None
        # Same drain-on-flush pattern as _call_history/_history_flush_idx,
        # but for non-LLM stage wall-clock time (see NON_LLM_DURATION_STAGES).
        self._stage_exec_history: list[tuple[str, float]] = []
        self._nonllm_flush_idx = 0
        self._nonllm_duration_by_stage: dict[str, list[float]] = {}

    def _on_llm_call(self, event: LLMCall) -> None:
        """Live subscriber — feeds every real LLM call into the history.

        Skips the monitor's own calls (see OBSERVER_STAGES): they are the cost
        of watching, not the cost of the run, and letting them in made the
        observer the biggest outlier in its own telemetry.
        """
        if event.stage in self._observer_stages:
            self._observer_calls += 1
            self._observer_latency_ms += event.latency_ms
            return
        self._call_times.append((self._clock() - self._t0, event.latency_ms))
        self.add_llm_call(
            event.tokens_in, event.tokens_out, event.latency_ms, event.stage
        )

    def _on_mutation_attempted(self, event: MutationAttempted) -> None:
        self._attempts += 1
        self._flush_mutant_bucket()
        self._maybe_dispatch_agent()

    def _maybe_dispatch_agent(self) -> None:
        """Dispatch at most one background agent call from the attempt path."""
        if not self._can_dispatch_agent():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (tests / offline replay) — __call__ will pick it up
        self._claim_agent()
        self._agent_task = loop.create_task(
            self._run_agent(), name="cost-monitor-agent"
        )

    def _can_dispatch_agent(self) -> bool:
        return (
            self._agent_due
            and self._agent is not None
            and not self._agent_running
            and (self._agent_task is None or self._agent_task.done())
        )

    def _claim_agent(self) -> None:
        """Claim a pending wakeup before either dispatch path starts it."""
        self._agent_due = False
        self._agent_running = True
        self._agent_calls += 1
        self._last_agent_attempt = self._attempts

    def _on_backpressure_sample(self, event: BackpressureSample) -> None:
        self._last_bp = event

    def _on_stage_exec(self, event: StageExec) -> None:
        # A cache "hit" returns near-instantly and isn't representative of
        # future cost for that stage — only count real executions.
        if event.stage in self._non_llm_duration_stages and event.decision != "hit":
            self._stage_exec_history.append((event.stage, event.duration_ms))

    def _winsorised(self, by_stage: dict[str, list[float]]) -> dict[str, list[float]]:
        """Replace flagged buckets with the stage median without shifting indices."""
        if not self._flagged_buckets:
            return by_stage
        out: dict[str, list[float]] = {}
        for stage, vals in by_stage.items():
            flagged = self._flagged_buckets.get(stage)
            if not flagged:
                out[stage] = vals
                continue
            keep = [v for i, v in enumerate(vals) if i not in flagged]
            med = statistics.median(keep) if keep else 0.0
            out[stage] = [med if i in flagged else v for i, v in enumerate(vals)]
        return out

    def _rotate_interval(
        self, duration_ci: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Publish the current interval and return the usable previous one."""
        prev, self._prev_ci = self._prev_ci, duration_ci
        return prev if (prev and prev[1] > prev[0] > 0) else None

    def _falsified_by_elapsed(self, elapsed_s: float) -> int:
        """Count pending intervals whose upper bound is below elapsed time."""
        if not self._pending_intervals:
            return 0
        falsified = sum(1 for _, hi in self._pending_intervals if hi < elapsed_s)
        if falsified:
            self._pending_intervals = [
                iv for iv in self._pending_intervals if iv[1] >= elapsed_s
            ]
        return falsified

    def _update_aci(self, miscovered: bool) -> None:
        """Update the clipped ACI multiplier from interval miscoverage."""
        if self._aci_gamma <= 0:
            return
        err = 1.0 if miscovered else 0.0
        self._aci_scale = float(
            min(
                self._aci_scale_max,
                max(
                    self._aci_scale_min,
                    self._aci_scale
                    * math.exp(self._aci_gamma * (err - self._aci_alpha)),
                ),
            )
        )

    def _widen_for_falsified(self, n: int) -> None:
        """One ACI step per interval the elapsed clock has already outlived.

        Weighted harder than a breach: a breach says the estimator moved more
        than it expected to, which is ordinary; an interval the run has
        already outrun is simply false, and the only fix is a wider band.
        """
        if self._aci_gamma <= 0 or n <= 0:
            return
        self._aci_scale = float(
            min(
                self._aci_scale_max,
                max(
                    self._aci_scale_min,
                    self._aci_scale
                    * math.exp(self._aci_gamma * n * (1.0 - self._aci_alpha)),
                ),
            )
        )

    def _check_agent_trigger(
        self, duration_s: float, prev: tuple[float, float] | None, miscovered: bool
    ) -> None:
        """Decide whether this flush should wake CostMonitorAgent.

        Wakes it when the estimator surprises itself — the new estimate falls
        outside the confidence interval the previous one published — plus one
        guaranteed early call so a run that never surprises anyone still gets
        looked at. Gated by a cooldown, a per-run budget, and remaining
        leverage: past ~85% of predicted wall time the agent's multiplier
        scales an almost-empty tail and can only add noise.
        """
        if self._agent_due or self._agent_calls >= self._max_agent_calls:
            return
        if self._attempts - self._last_agent_attempt < self._cooldown_attempts:
            return
        elapsed = self._clock() - self._t0
        if duration_s > 0 and (duration_s - elapsed) / duration_s < self._min_leverage:
            return
        if self._attempts >= self._warmup_attempts and self._agent_calls == 0:
            self._agent_due = True
            self._trigger_reason = f"first look after {self._attempts} attempts"
            return
        # Prefer disagreement between independent token and clock projections.
        gap = self._progress_disagreement(duration_s, elapsed)
        if gap is not None and abs(gap) >= self._disagreement_gap:
            self._agent_due = True
            lag = "clock" if gap > 0 else "tokens"
            self._trigger_reason = (
                f"token progress and clock progress disagree by {abs(gap):.0%} at attempt "
                f"{self._attempts} ({lag} is ahead) — one of them is measuring "
                f"something the other cannot see"
            )
            return
        if miscovered and prev:
            direction = "above" if duration_s > prev[1] else "below"
            self._agent_due = True
            self._trigger_reason = (
                f"estimate jumped {direction} its own interval at attempt "
                f"{self._attempts}: {duration_s:.0f}s vs [{prev[0]:.0f}, {prev[1]:.0f}]s"
            )

    def _progress_disagreement(
        self, duration_s: float, elapsed_s: float
    ) -> float | None:
        """Return clock progress minus token progress.

        Tokens say ``observed / predicted_total``; the clock says
        ``elapsed / predicted_duration``. Positive values indicate clock lag;
        negative values indicate token growth. Agent overrides are removed
        before comparison to keep the trigger independent of its response.

        None while either projection is too thin to compare.
        """
        observed = sum(sum(v) for v in self._tokens_by_stage.values())
        total = self._pred.predicted_total_tokens
        if not (total > 0 and duration_s > 0 and observed > 0):
            return None

        # Undo the levers: remaining work scales with tail_mult and inversely
        # with the concurrency multiplier, so multiplying the remaining term
        # by conc/tail recovers what the model would have said untouched.
        conc = self._pred.llm_concurrency_override
        conc = (
            conc
            if self._concurrency_mult_min <= conc <= self._concurrency_mult_max
            else 1.0
        )
        tail = 1.0
        if (
            self._golden_ratio_min
            <= self._pred.llm_golden_override
            <= self._golden_ratio_max
        ):
            tail *= self._pred.llm_golden_override
        if (
            self._growth_rate_min
            <= self._pred.llm_growth_override
            <= self._growth_rate_max
        ):
            tail *= self._pred.llm_growth_override
        raw_duration = elapsed_s + max(duration_s - elapsed_s, 0.0) * conc / max(
            tail, 1e-6
        )
        if raw_duration <= 0:
            return None

        tok_done = observed / total
        dur_done = elapsed_s / raw_duration
        if not (0.0 < tok_done <= 1.0 and 0.0 < dur_done <= 1.0):
            return None
        return dur_done - tok_done

    def _program_size_proxy(self) -> tuple[int, int]:
        """Estimate baseline and current program size from mutation input tokens."""
        ins = [
            r.tokens_in
            for r in self._call_history
            if r.stage == "MutationAgent" and r.tokens_in
        ]
        if len(ins) < self._program_size_min_samples:
            return 0, 0
        k = max(len(ins) // self._program_size_window_divisor, 2)
        return int(statistics.median(ins[:k])), int(statistics.median(ins[-k:]))

    def _tail_calibration(self) -> float:
        """Interpolate ``TAIL_CALIBRATION`` at the current attempt progress."""
        cap = max(self._pred.max_mutants, 1)
        p = min(max(self._attempts / cap, 0.0), 1.0)
        pts = self._tail_calibration_points
        if p <= pts[0][0]:
            return pts[0][1]
        for (p0, m0), (p1, m1) in zip(pts, pts[1:]):
            if p <= p1:
                return m0 + (m1 - m0) * (p - p0) / (p1 - p0)
        return pts[-1][1]

    def _project_units(self, stage: str, seen: int, horizon: int) -> int:
        """Project stage firings to ``horizon`` from the recent attempt rate."""
        hist = self._bucket_attempts.get(stage) or []
        rate = seen / max(self._attempts, 1)
        if (
            len(hist) >= self._projection_recent_min_samples
            and self._attempts >= self._projection_recent_min_samples
        ):
            lo = self._attempts // 2
            span = max(self._attempts - lo, 1)
            rate = max(rate, sum(1 for a in hist if a > lo) / span)
        return max(1, round(seen + rate * max(horizon - self._attempts, 0)))

    def _flush_mutant_bucket(self) -> None:
        """Sum calls since the last flush per stage, refit the growth law,
        and write the new estimate into the shared CostPrediction.

        Runs on every accepted mutant AND every mutation attempt, not
        gated by ``interval`` — only the LLM-agent calibration call in
        ``__call__`` is throttled to every N accepted mutants.
        """
        new_calls = self._call_history[self._history_flush_idx :]
        self._history_flush_idx = len(self._call_history)
        if not new_calls:
            return
        self._projection_memo.clear()  # the buckets these were fitted on moved

        stage_tokens: dict[str, float] = {}
        stage_tokens_out: dict[str, float] = {}
        stage_latency: dict[str, float] = {}
        for rec in new_calls:
            stage_tokens[rec.stage] = (
                stage_tokens.get(rec.stage, 0.0) + rec.tokens_in + rec.tokens_out
            )
            stage_tokens_out[rec.stage] = (
                stage_tokens_out.get(rec.stage, 0.0) + rec.tokens_out
            )
            stage_latency[rec.stage] = (
                stage_latency.get(rec.stage, 0.0) + rec.latency_ms
            )
        for stage, tokens in stage_tokens.items():
            self._bucket_attempts.setdefault(stage, []).append(self._attempts)
            self._tokens_by_stage.setdefault(stage, []).append(tokens)
            self._tokens_out_by_stage.setdefault(stage, []).append(
                stage_tokens_out[stage]
            )
            self._latency_by_stage.setdefault(stage, []).append(stage_latency[stage])
        for rec in new_calls:
            bucket_pos = len(self._tokens_by_stage.get(rec.stage, [])) - 1
            if bucket_pos >= 0:
                self._call_bucket[rec.index] = (rec.stage, bucket_pos)

        new_stage_execs = self._stage_exec_history[self._nonllm_flush_idx :]
        self._nonllm_flush_idx = len(self._stage_exec_history)
        if new_stage_execs:
            nonllm_stage_totals: dict[str, float] = {}
            for stage, duration_ms in new_stage_execs:
                nonllm_stage_totals[stage] = (
                    nonllm_stage_totals.get(stage, 0.0) + duration_ms
                )
            for stage, total_ms in nonllm_stage_totals.items():
                self._bucket_attempts.setdefault(stage, []).append(self._attempts)
                self._nonllm_duration_by_stage.setdefault(stage, []).append(total_ms)

        total_units_by_stage = {
            stage: self._project_units(stage, len(series), self._pred.max_mutants)
            for stage, series in self._tokens_by_stage.items()
        }
        total_units_by_stage.update(
            {
                stage: self._project_units(stage, len(series), self._pred.max_mutants)
                for stage, series in self._nonllm_duration_by_stage.items()
            }
        )
        fit_tokens = self._winsorised(self._tokens_by_stage)
        fit_tokens_out = self._winsorised(self._tokens_out_by_stage)
        fit_latency = self._winsorised(self._latency_by_stage)
        fit_nonllm = self._winsorised(self._nonllm_duration_by_stage)
        est = estimate_by_stage(
            self._tokens_by_stage,
            self._latency_by_stage,
            total_units_by_stage=total_units_by_stage,
            max_in_flight=self._pred.max_in_flight,
            law_cls=RobustPowerLaw,
            fit_tokens_by_stage=fit_tokens,
            fit_latency_by_stage=fit_latency,
            ci_method=self._ci_method,
        )
        # Agent overrides scale only the unobserved tail.
        tail_mult = 1.0
        if (
            self._golden_ratio_min
            <= self._pred.llm_golden_override
            <= self._golden_ratio_max
        ):
            tail_mult *= self._pred.llm_golden_override
        if (
            self._growth_rate_min
            <= self._pred.llm_growth_override
            <= self._growth_rate_max
        ):
            tail_mult *= self._pred.llm_growth_override
        elapsed_s = self._clock() - self._t0
        self._concurrency = achieved_concurrency(
            self._call_times,
            now_s=elapsed_s,
            max_in_flight=self._pred.max_in_flight,
            window_frac=self._concurrency_window_frac,
            min_window_s=self._concurrency_min_window_s,
            shrink_k=self._concurrency_shrink_k,
            min_concurrency=self._min_effective_concurrency,
            max_concurrency=self._max_effective_concurrency,
        )
        if (
            self._concurrency_mult_min
            <= self._pred.llm_concurrency_override
            <= self._concurrency_mult_max
        ):
            self._concurrency = max(
                self._min_effective_concurrency,
                self._concurrency * self._pred.llm_concurrency_override,
            )
        # Duration uses the TTFT+TPOT physical model (latency ~ tokens_out),
        # not a growth law over call index — latency doesn't follow a growth
        # trend in this system (see estimate_duration_by_stage docstring).
        duration_s, duration_ci = estimate_duration_by_stage(
            self._tokens_out_by_stage,
            self._latency_by_stage,
            total_units_by_stage=total_units_by_stage,
            max_in_flight=self._pred.max_in_flight,
            nonllm_duration_by_stage=self._nonllm_duration_by_stage,
            elapsed_s=elapsed_s,
            concurrency=self._concurrency,
            # The model's own calibration multiplies with the agent's lever:
            # the table is what the estimator is known to be short by, the
            # lever is the agent's correction on top of an unbiased baseline.
            tail_mult=tail_mult * self._tail_calibration(),
            fit_tokens_out_by_stage=fit_tokens_out,
            fit_latency_by_stage=fit_latency,
            fit_nonllm_by_stage=fit_nonllm,
            ci_method=self._ci_method,
            ci_alpha=self._aci_alpha,
            width_scale=self._aci_scale,
        )
        # Predict with the current width, observe, then update it — the online
        # order ACI requires; updating first would grade the interval against
        # the very estimate that widened it.
        prev_ci = self._rotate_interval(duration_ci)
        miscovered = bool(prev_ci and not (prev_ci[0] <= duration_s <= prev_ci[1]))
        self._check_agent_trigger(duration_s, prev_ci, miscovered)
        self._update_aci(miscovered)
        # Second, independent signal: intervals the clock has already outlived.
        self._widen_for_falsified(self._falsified_by_elapsed(elapsed_s))
        self._pending_intervals.append(duration_ci)
        if len(self._pending_intervals) > 500:
            del self._pending_intervals[:250]
        self._flushes += 1
        self._miscoverages += int(miscovered)
        self._pred.predicted_total_tokens = int(est.predicted_total_tokens)
        self._pred.predicted_duration_s = duration_s
        self._pred.token_ci_low, self._pred.token_ci_high = (
            int(est.tokens_ci[0]),
            int(est.tokens_ci[1]),
        )
        self._pred.ci_low_s, self._pred.ci_high_s = duration_ci
        logger.info(
            "[CostMonitorHook] mutant={} {}", self._counter, self._pred._log_estimate()
        )
        # Compact single-line JSON alongside the human-readable block above —
        # consumers (e.g. tools/task_builder_web) parse this instead of the
        # multi-line text, which is fragile to regex across interleaved logs.
        logger.info(
            "[CostMonitorHookJSON] {}",
            json.dumps(
                {
                    "mutant": self._counter,
                    "predicted_tokens": self._pred.predicted_total_tokens,
                    "token_ci_low": self._pred.token_ci_low,
                    "token_ci_high": self._pred.token_ci_high,
                    "predicted_duration_s": self._pred.predicted_duration_s,
                    "ci_low_s": self._pred.ci_low_s,
                    "ci_high_s": self._pred.ci_high_s,
                    "elapsed_s": elapsed_s,
                    "attempts": self._attempts,
                    "concurrency": self._concurrency,
                    "agent_due": self._agent_due,
                    "aci_scale": self._aci_scale,
                    "miscoverage_rate": self._miscoverages / max(self._flushes, 1),
                }
            ),
        )

    # ---------------------------------------------------------- budget mode
    #
    # The live estimate answers "what will THIS run cost", with the horizon
    # fixed at `max_mutants`. Budget mode asks the two questions you actually
    # have before committing GPU: what would N attempts cost, and if that is
    # more than I have, how many attempts do I get. Same per-stage growth
    # laws, re-integrated to a different horizon — no second cost model.

    def project_tokens(
        self, attempts_target: int, alpha: float = 0.1
    ) -> tuple[float, tuple[float, float]]:
        """Total tokens this run would spend if it ran to ``attempts_target``.

        Each stage fires at its own observed rate per attempt (some stages
        skip-cascade), so the horizon scales that rate rather than the raw
        bucket count — the same projection ``_flush_mutant_bucket`` makes
        against ``max_mutants``.

        ``alpha`` sets the returned band's tail mass: 0.1 is the 90% interval,
        0.5 the quartiles. Budget mode quotes the quartiles — a 90% band on a
        10-attempt probe spans a factor of two and answers nothing.
        """
        if not self._tokens_by_stage:
            return 0.0, (0.0, 0.0)
        # `affordable_attempts` bisects over this, and the montecarlo CI is not
        # cheap; the hook's buckets do not change while a budget question is
        # being answered, so the horizon is a sound cache key. Invalidated by
        # the flush that appends a new bucket.
        key = (attempts_target, alpha)
        hit = self._projection_memo.get(key)
        if hit is not None:
            return hit
        units = {
            stage: self._project_units(stage, len(series), attempts_target)
            for stage, series in self._tokens_by_stage.items()
        }
        est = estimate_by_stage(
            self._tokens_by_stage,
            self._latency_by_stage,
            total_units_by_stage=units,
            max_in_flight=self._pred.max_in_flight,
            law_cls=RobustPowerLaw,
            fit_tokens_by_stage=self._winsorised(self._tokens_by_stage),
            fit_latency_by_stage=self._winsorised(self._latency_by_stage),
            ci_method=self._ci_method,
            ci_alpha=alpha,
        )
        out = (est.predicted_total_tokens, est.tokens_ci)
        self._projection_memo[key] = out
        return out

    def affordable_attempts(self, budget_tokens: float, *, hi: int = 100_000) -> int:
        """Largest attempt count whose projected spend stays inside the budget.

        Returns 0 when even what has already been observed exceeds the budget —
        those tokens are spent, and the projection can never go below
        ``sum(observed)``. Bisection is valid because the projection is
        monotone in the horizon: more attempts only ever add tail.
        """
        # The interval method sets the WIDTH only; the point estimate is
        # identical either way, and montecarlo costs ~0.6 s a call against
        # ~0.003 s for the heuristic — far too much for a 17-step bisection.
        method, self._ci_method = self._ci_method, None
        memo, self._projection_memo = self._projection_memo, {}
        try:
            spend = lambda n: self.project_tokens(n)[0]  # noqa: E731
            lo = max(self._attempts, 1)
            if spend(lo) > budget_tokens:
                return 0
            if spend(hi) <= budget_tokens:
                return hi
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if spend(mid) <= budget_tokens:
                    lo = mid
                else:
                    hi = mid
            return lo
        finally:
            self._ci_method, self._projection_memo = method, memo

    async def __call__(self) -> None:
        """Engine post_step_hook — fires when a mutant is ACCEPTED.

        Kept as a second dispatch point so an accepted mutant can still pick
        up a pending trigger promptly; the primary path is
        ``_maybe_dispatch_agent`` off the attempt clock.
        """
        self._counter += 1
        self._flush_mutant_bucket()
        if self._can_dispatch_agent():
            self._claim_agent()
            await self._run_agent()

    def _blunt_lever_allowed(self, sustained_over_calls) -> bool:
        """Return whether a blunt lever has enough persistence evidence."""
        try:
            n = int(sustained_over_calls)
        except (TypeError, ValueError):
            n = 0
        if n >= self._min_sustained_calls:
            return True
        logger.info(
            "[CostMonitorHook] blunt lever refused: sustained_over_calls={} < {}",
            n,
            self._min_sustained_calls,
        )
        return False

    def _gate_lever(self, name: str, current: float, proposed: float) -> float | None:
        """Apply deadband and reversal damping to a proposed lever value."""
        if current > 0 and abs(proposed - current) / current < self._lever_deadband:
            logger.info(
                "[CostMonitorHook] {} move ignored: {:.3f}->{:.3f} inside deadband",
                name,
                current,
                proposed,
            )
            return None
        direction = 1 if proposed > current else -1
        prev_dir, prev_at = self._lever_dir.get(name, (0, -(10**9)))
        reversing = (
            prev_dir != 0
            and direction != prev_dir
            and (self._agent_calls - prev_at) <= self._reversal_window
        )
        step = self._max_relative_step
        if reversing:
            step *= self._reversal_step_factor
        value = _clamp_step(current, proposed, step)
        self._lever_dir[name] = (direction, self._agent_calls)
        if reversing:
            logger.info(
                "[CostMonitorHook] {} reverses its last move — half step to {:.3f}",
                name,
                value,
            )
        return value

    async def _run_agent(self) -> None:
        """Run one wakeup. The caller must have claimed it via ``_claim_agent``."""
        trigger = self._trigger_reason

        current_golden = (
            self._pred.llm_golden_override
            if self._pred.llm_golden_override > 0
            else 1.0
        )
        current_growth = (
            self._pred.llm_growth_override
            if self._pred.llm_growth_override > 0
            else 1.0
        )
        current_cold = (
            self._pred.llm_cold_override if self._pred.llm_cold_override > 0 else 0.57
        )
        current_conc_mult = (
            self._pred.llm_concurrency_override
            if self._pred.llm_concurrency_override > 0
            else 1.0
        )
        if self._last_bp is not None:
            bp_util = self._last_bp.in_flight / self._last_bp.max_in_flight
            bp_in_flight, bp_max_in_flight = (
                self._last_bp.in_flight,
                self._last_bp.max_in_flight,
            )
        else:
            bp_util = self._fallback_backpressure_util
            bp_in_flight, bp_max_in_flight = 0, self._pred.max_in_flight
        base_len, cur_len = self._program_size_proxy()
        tools = _ToolSet(
            recent_calls=self._call_history[-self._recent_calls_limit :],
            backpressure_util=bp_util,
            in_flight=bp_in_flight,
            max_in_flight=bp_max_in_flight,
            current_cold=current_cold,
            current_golden=current_golden,
            current_growth=current_growth,
            achieved_concurrency=self._concurrency,
            current_concurrency_mult=current_conc_mult,
            elapsed_s=self._clock() - self._t0,
            attempts_done=self._attempts,
            max_mutants=self._pred.max_mutants,
            predicted_duration_s=self._pred.predicted_duration_s,
            trigger_reason=trigger,
            last_adjustment=self._last_adjustment,
            already_flagged=sorted(self._flagged_calls),
            miscoverage_rate=(self._miscoverages / self._flushes)
            if self._flushes
            else None,
            miscoverage_target=self._aci_alpha,
            width_scale=self._aci_scale,
            baseline_program_len=base_len,
            current_program_len=cur_len,
        )
        self._agent.tools = tools

        try:
            messages = self._agent.build_prompt({"messages": []})
            state = {"messages": messages}
            result = await self._agent.acall_llm(state)
            result = self._agent.parse_response(result)

            adjustments = result.get("cost_adjustments", {})
            self._log_agent_trace(
                tools,
                trigger,
                adjustments,
                before={
                    "golden_ratio": current_golden,
                    "growth_rate_mult": current_growth,
                    "concurrency_mult": current_conc_mult,
                },
            )
            if adjustments:
                golden = adjustments.get("golden_ratio", -1)
                growth = adjustments.get("growth_rate_mult", -1)
                conc_mult = adjustments.get("concurrency_mult", -1)
                reasoning = adjustments.get("reasoning", "")
                outliers = adjustments.get("flag_outlier_indices", []) or []
                self._flag_outliers(outliers)

                sustained = adjustments.get("sustained_over_calls", 0)
                # golden/growth are the blunt levers: they scale everything
                # still ahead, so they need evidence that the deviation
                # persisted, not one surprising call.
                blunt_ok = self._blunt_lever_allowed(sustained)
                if golden > 0 and blunt_ok:
                    v = self._gate_lever("golden_ratio", current_golden, golden)
                    if v is not None:
                        self._pred.llm_golden_override = v
                if growth > 0 and blunt_ok:
                    v = self._gate_lever("growth_rate_mult", current_growth, growth)
                    if v is not None:
                        self._pred.llm_growth_override = v
                if conc_mult > 0:
                    v = self._gate_lever(
                        "concurrency_mult", current_conc_mult, conc_mult
                    )
                    if v is not None:
                        self._pred.llm_concurrency_override = v
                if reasoning:
                    self._pred.llm_reasoning = reasoning

                self._last_adjustment = {
                    "attempt": self._attempts,
                    "golden": golden,
                    "growth": growth,
                    "concurrency": conc_mult,
                    "outliers": len(outliers),
                    "reasoning": reasoning,
                    "predicted_duration_s": self._pred.predicted_duration_s,
                    "elapsed_s": self._clock() - self._t0,
                }
                logger.info(
                    "[CostMonitorHook] adjustments: golden={} growth={} conc={} "
                    "sustained={} outliers={} trigger={!r} reason={}",
                    golden,
                    growth,
                    conc_mult,
                    sustained,
                    len(outliers),
                    trigger,
                    reasoning,
                )
                emit(
                    CostAgentAdjustment(
                        mutant_index=self._counter,
                        golden_ratio=golden,
                        growth_rate_mult=growth,
                        concurrency_mult=conc_mult,
                        flag_outlier_indices=adjustments.get(
                            "flag_outlier_indices", []
                        ),
                        skip_calibration=adjustments.get("skip_calibration", False),
                        reasoning=reasoning,
                    )
                )
        except Exception as e:
            logger.warning("[CostMonitorHook] agent call failed: {}", e)
        finally:
            # Release the claim only once this wakeup is fully done, so the
            # other dispatch path cannot start a second concurrent call.
            self._agent_running = False

    def _log_agent_trace(
        self, tools, trigger: str, adjustments: dict, before: dict | None = None
    ) -> None:
        """Log the evidence and adjustments for one agent wakeup."""

        def _lever(key: str) -> float:
            try:
                return float(adjustments.get(key, -1))
            except (TypeError, ValueError):
                return -1.0

        outliers = adjustments.get("flag_outlier_indices", []) or []
        skip = bool(adjustments.get("skip_calibration", False))
        levers = {
            "golden_ratio": _lever("golden_ratio"),
            "growth_rate_mult": _lever("growth_rate_mult"),
            "concurrency_mult": _lever("concurrency_mult"),
        }

        cause = str(adjustments.get("cause", "") or "")
        actions = []
        if any(v > 0 for v in levers.values()):
            actions.append("adjust_model")
        if outliers:
            actions.append("flag_as_outlier")
        if skip:
            actions.append("skip_next_calibration")

        def _safe(fn, *args) -> str:
            try:
                return str(fn(*args))[:4000]
            except Exception as exc:  # noqa: BLE001 - a broken tool must not kill the run
                return f"<unavailable: {exc}>"

        try:
            trace = {
                "attempt": self._attempts,
                "mutant": self._counter,
                "trigger": trigger,
                "cause": cause,
                "actions": actions,
                "levers": levers,
                "levers_before": before or {},
                "flag_outlier_indices": list(outliers),
                "skip_calibration": skip,
                "reasoning": adjustments.get("reasoning", ""),
                "evidence": {
                    "get_trigger": _safe(tools.get_trigger),
                    "get_calibration": _safe(tools.get_calibration),
                    "get_progress": _safe(tools.get_progress),
                    "get_backpressure": _safe(tools.get_backpressure),
                    "get_model_params": _safe(tools.get_model_params),
                    "get_last_adjustment_outcome": _safe(
                        tools.get_last_adjustment_outcome
                    ),
                    "get_recent_calls": _safe(tools.get_recent_calls, 15),
                },
            }
            logger.info(
                "[CostMonitorAgentTrace] {}", json.dumps(trace, ensure_ascii=False)
            )
        except Exception as exc:  # noqa: BLE001 - tracing is never worth a crash
            logger.warning("[CostMonitorHook] could not write agent trace: {}", exc)

    def _flag_outliers(self, call_indices: list[int]) -> None:
        """Map flagged call indices to buckets excluded from tail fitting."""
        for idx in call_indices:
            self._flagged_calls.add(idx)
            entry = self._call_bucket.get(idx)
            if entry is None:
                continue
            stage, bucket_pos = entry
            self._flagged_buckets.setdefault(stage, set()).add(bucket_pos)
        if call_indices:
            self._pred.llm_outlier_indices = sorted(
                set(self._pred.llm_outlier_indices) | set(call_indices)
            )

    def add_llm_call(
        self, tokens_in: int, tokens_out: int, latency_ms: float, stage: str
    ) -> None:
        """Record an LLM call for the agent to analyse."""
        self._call_history.append(
            LlmCallRecord(
                index=len(self._call_history),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                stage=stage,
            )
        )
