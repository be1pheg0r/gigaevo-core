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
from gigaevo.monitoring.events import BackpressureSample, CostAgentAdjustment, LLMCall, MutationAttempted, StageExec
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


# Below this relative change a lever move is noise, not a decision: it costs
# a wake-up and moves the forecast by less than the flush-to-flush jitter.
# Standard deadband — a controller without one chatters around its setpoint.
LEVER_DEADBAND = 0.05

# A move that reverses the previous move on the same lever within this many
# wake-ups is taken at half step. Reversal is the signature of a controller
# hunting rather than converging, and halving is the cheapest damping there is.
REVERSAL_WINDOW = 2

# Blunt levers (golden_ratio, growth_rate_mult) apply to everything still
# ahead, so the system prompt only allows them for a deviation that PERSISTED.
# That was advice the model could ignore; now it has to name the evidence, and
# a claim below this many calls is refused.
MIN_SUSTAINED_CALLS = 5


# Non-LLM pipeline stages whose wall-clock time the LLM-latency duration
# model is otherwise blind to. Named explicitly (rather than "everything
# that isn't an LLM stage") so the growth-law fit only sees stages known to
# be real, recurring per-mutant cost — see estimate_duration_by_stage's
# nonllm_duration_by_stage docstring for the historical evidence.
NON_LLM_DURATION_STAGES = frozenset({
    "CallProgramFunction", "CallValidatorFunction", "IntraMemoryStage",
})

# The monitor's own inference is emitted as an LLM_CALL like every other agent
# (LangGraphAgent.acall_llm tags the event with its own class name), so without
# this it lands in the very telemetry the cost model is fitted on. Measured on
# alphaevolve runs it was 4.7-17.3% of all observed LLM latency, with single
# calls of 280-366s against a run median of 25-36s — i.e. the biggest outliers
# in the run were the observer, and the agent was seen flagging them as
# anomalies. It also made the ablation unfair: the agentless arm has no such
# calls at all, so "with agent" was measuring a different workload.
# Listed by name rather than read off the live agent so an offline replay of an
# older log strips them too.
OBSERVER_STAGES = frozenset({"CostMonitorAgent", "NoOpCostMonitorAgent"})


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
    ):
        self._agent = agent
        self._pred = prediction
        self._interval = interval
        self._counter = 0
        self._clock = clock
        # Which CI-width method feeds tokens_ci/duration_ci — None keeps the
        # original 1/sqrt(n) heuristic; see growth_estimator.CI_METHODS for
        # the probabilistic alternatives (bootstrap/montecarlo/bayesian).
        self._ci_method = ci_method
        self._t0 = clock()
        # (completion_time_s, latency_ms) per LLM call — the raw material for
        # measuring the concurrency the system actually achieves.
        self._call_times: list[tuple[float, float]] = []
        self._call_history: list[LlmCallRecord] = []
        # Per-stage growth-law state. The hook flushes once per accepted
        # mutant (post_step_hook, see ingestor.py) AND once per mutation
        # attempt (MUTATION_ATTEMPTED, see mutant_task.py) — the latter
        # guarantees at least one estimate appears even for low/zero-accept
        # domains, where post_step_hook (gated on an ACCEPT landing) may
        # never fire within a short/small run. ``_history_flush_idx``
        # dedupes: whichever trigger fires first drains ``_call_history``,
        # the other is then a no-op for that batch of calls.
        self._history_flush_idx = 0
        self._tokens_by_stage: dict[str, list[float]] = {}
        self._tokens_out_by_stage: dict[str, list[float]] = {}
        self._latency_by_stage: dict[str, list[float]] = {}
        # Mutation attempts observed so far (MUTATION_ATTEMPTED fires per DAG
        # dispatch, before accept/reject). ``max_mutants`` caps ATTEMPTS, not
        # accepted mutants (MaxMutantsStopper watches
        # engine.metrics.mutations_created) — since flushes now track
        # attempts 1:1, this is the correct denominator for projecting each
        # stage's total firing count, no accept-rate correction needed.
        self._attempts = 0
        self._concurrency = float(prediction.max_in_flight)
        # Which bucket each LLM call landed in, so the agent's
        # flag_outlier_indices (call indices, as shown by get_recent_calls)
        # can be turned into bucket positions to winsorise.
        self._call_bucket: dict[int, tuple[str, int]] = {}
        self._flagged_buckets: dict[str, set[int]] = {}
        # Agent scheduling: fire when the estimator is SURPRISED (the new
        # estimate falls outside the interval the previous one claimed),
        # not on a fixed tick. Measured on the 9 collected runs: the old
        # every-5-accepted-mutants tick gave 0-6 calls per 100-attempt run
        # with the first one landing anywhere from attempt 10 to 92 (and
        # never at all on a 0%-accept task), while a raw "prediction moved
        # >30%" trigger fired only during cold start and never once in the
        # final third. Breaching the previous confidence interval is
        # self-normalising — the interval is wide early and narrow late —
        # and fires ~10 times per run, spread across the whole run.
        self._agent_due = False
        self._agent_calls = 0
        # Claimed at DISPATCH, not inside the coroutine: a task created by
        # ``_maybe_dispatch_agent`` has not started running when ``__call__``
        # next fires, so clearing ``_agent_due`` in ``_run_agent`` left a
        # window where the accept path started a second, concurrent agent
        # call on the same evidence. Measured on the full alphaevolve
        # ablation: 19 of 60 wakeups were the same trigger handled twice.
        self._agent_running = False
        self._last_agent_attempt = -10**9
        self._prev_ci: tuple[float, float] | None = None
        self._trigger_reason = ""
        # Closed-loop feedback: what the previous adjustment was and what the
        # estimate looked like when it was made, so the next call can be told
        # whether it helped.
        self._last_adjustment: dict | None = None
        self._max_agent_calls = max_agent_calls
        self._cooldown_attempts = cooldown_attempts
        self._warmup_attempts = warmup_attempts
        self._min_leverage = min_leverage
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
        if event.stage in OBSERVER_STAGES:
            self._observer_calls += 1
            self._observer_latency_ms += event.latency_ms
            return
        self._call_times.append((self._clock() - self._t0, event.latency_ms))
        self.add_llm_call(event.tokens_in, event.tokens_out, event.latency_ms, event.stage)

    def _on_mutation_attempted(self, event: MutationAttempted) -> None:
        self._attempts += 1
        self._flush_mutant_bucket()
        self._maybe_dispatch_agent()

    def _maybe_dispatch_agent(self) -> None:
        """Run the agent off the ATTEMPT clock, not the accepted-mutant clock.

        ``__call__`` is the engine's post_step_hook and only fires when a
        mutant is accepted, so on a low- or zero-accept task the agent used to
        be called late or never (measured: 0 calls on both toy_kadane runs).
        The trigger lives in the flush, which runs on every attempt; dispatch
        the call as a background task so the synchronous event path is not
        blocked, and never more than one at a time.
        """
        if not self._can_dispatch_agent():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (tests / offline replay) — __call__ will pick it up
        self._claim_agent()
        self._agent_task = loop.create_task(self._run_agent(), name="cost-monitor-agent")

    def _can_dispatch_agent(self) -> bool:
        return (
            self._agent_due
            and self._agent is not None
            and not self._agent_running
            and (self._agent_task is None or self._agent_task.done())
        )

    def _claim_agent(self) -> None:
        """Take ownership of the pending wakeup.

        Both dispatch paths call this BEFORE the coroutine starts, so exactly
        one of them can win a given trigger — see ``_agent_running``.
        """
        self._agent_due = False
        self._agent_running = True
        self._agent_calls += 1
        self._last_agent_attempt = self._attempts

    def _on_backpressure_sample(self, event: BackpressureSample) -> None:
        self._last_bp = event

    def _on_stage_exec(self, event: StageExec) -> None:
        # A cache "hit" returns near-instantly and isn't representative of
        # future cost for that stage — only count real executions.
        if event.stage in NON_LLM_DURATION_STAGES and event.decision != "hit":
            self._stage_exec_history.append((event.stage, event.duration_ms))

    def _winsorised(self, by_stage: dict[str, list[float]]) -> dict[str, list[float]]:
        """Copy of ``by_stage`` with agent-flagged buckets replaced by the
        median of that stage's remaining buckets.

        Replaced, not dropped: the bucket index IS the growth law's x axis,
        so removing an entry would shift every later point. The spike's real
        cost still anchors the observed part (see growth_estimator.tail_integral) —
        only its influence on the extrapolated tail is neutralised.
        """
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

    def _rotate_interval(self, duration_ci: tuple[float, float]) -> tuple[float, float] | None:
        """Publish this flush's interval, return the usable previous one.

        The comparison "did the new estimate land inside what the previous one
        promised" is the single signal behind both the agent trigger and the
        width calibration, so it is derived once here. It used to be computed
        inside ``_check_agent_trigger`` off ``self._pred.ci_*``, which at that
        point still held the PREVIOUS flush's interval — the comparison was
        against the interval published two flushes back, one step staler than
        intended.
        """
        prev, self._prev_ci = self._prev_ci, duration_ci
        return prev if (prev and prev[1] > prev[0] > 0) else None

    def _falsified_by_elapsed(self, elapsed_s: float) -> int:
        """Past intervals the run has already outlived.

        Self-consistency and accuracy are different things, and on the
        collected runs they disagreed loudly: breaches sat at 4-15% (looking
        calibrated) while only 64-68% of intervals contained the eventual
        truth. An estimator that drifts smoothly stays consistent with itself
        while being steadily wrong, so grading on breaches alone would tune
        the wrong quantity.

        The final duration is unknown mid-run, but ``elapsed`` is a hard lower
        bound on it: once wall time passes an interval's upper edge, that
        interval is definitively wrong — no future information can rescue it.
        That is a real, one-sided coverage signal available online, and it
        points the right way, since the measured failure is systematic
        UNDER-estimation early in the run.
        """
        if not self._pending_intervals:
            return 0
        falsified = sum(1 for _, hi in self._pending_intervals if hi < elapsed_s)
        if falsified:
            self._pending_intervals = [iv for iv in self._pending_intervals if iv[1] >= elapsed_s]
        return falsified

    def _update_aci(self, miscovered: bool) -> None:
        """Adaptive Conformal Inference on the interval half-width.

        Gibbs & Candès (NeurIPS 2021): treat the miscoverage level as a single
        parameter re-estimated online from hit/miss feedback. It buys a
        long-run coverage guarantee with no exchangeability assumption, which
        matters here because a run is a distribution shift by construction
        (cold start -> steady state).

        Measured on 20 runs of the alphaevolve ablation, the model-based
        interval covered the eventual truth 64-68% of the time while aiming
        at 90%: it claimed precision it did not have. Multiplicative update
        so the scale stays positive; clipped so one weird flush cannot make
        the band useless in either direction.

        Second effect, and the reason this sits in the control loop rather
        than in the estimator: miscoverage IS the agent's wakeup condition,
        so driving its frequency to ``aci_alpha`` turns the wakeup rate into
        a quantity we set instead of one we discover.
        """
        if self._aci_gamma <= 0:
            return
        err = 1.0 if miscovered else 0.0
        self._aci_scale = float(min(5.0, max(0.5, self._aci_scale * math.exp(
            self._aci_gamma * (err - self._aci_alpha)))))

    def _widen_for_falsified(self, n: int) -> None:
        """One ACI step per interval the elapsed clock has already outlived.

        Weighted harder than a breach: a breach says the estimator moved more
        than it expected to, which is ordinary; an interval the run has
        already outrun is simply false, and the only fix is a wider band.
        """
        if self._aci_gamma <= 0 or n <= 0:
            return
        self._aci_scale = float(min(5.0, max(0.5, self._aci_scale * math.exp(
            self._aci_gamma * n * (1.0 - self._aci_alpha)))))

    def _check_agent_trigger(self, duration_s: float,
                             prev: tuple[float, float] | None, miscovered: bool) -> None:
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
        if miscovered and prev:
            direction = "above" if duration_s > prev[1] else "below"
            self._agent_due = True
            self._trigger_reason = (
                f"estimate jumped {direction} its own interval at attempt "
                f"{self._attempts}: {duration_s:.0f}s vs [{prev[0]:.0f}, {prev[1]:.0f}]s"
            )

    def _flush_mutant_bucket(self) -> None:
        """Sum calls since the last flush per stage, refit the growth law,
        and write the new estimate into the shared CostPrediction.

        Runs on every accepted mutant AND every mutation attempt, not
        gated by ``interval`` — only the LLM-agent calibration call in
        ``__call__`` is throttled to every N accepted mutants.
        """
        new_calls = self._call_history[self._history_flush_idx:]
        self._history_flush_idx = len(self._call_history)
        if not new_calls:
            return

        stage_tokens: dict[str, float] = {}
        stage_tokens_out: dict[str, float] = {}
        stage_latency: dict[str, float] = {}
        for rec in new_calls:
            stage_tokens[rec.stage] = stage_tokens.get(rec.stage, 0.0) + rec.tokens_in + rec.tokens_out
            stage_tokens_out[rec.stage] = stage_tokens_out.get(rec.stage, 0.0) + rec.tokens_out
            stage_latency[rec.stage] = stage_latency.get(rec.stage, 0.0) + rec.latency_ms
        for stage, tokens in stage_tokens.items():
            self._tokens_by_stage.setdefault(stage, []).append(tokens)
            self._tokens_out_by_stage.setdefault(stage, []).append(stage_tokens_out[stage])
            self._latency_by_stage.setdefault(stage, []).append(stage_latency[stage])
        for rec in new_calls:
            bucket_pos = len(self._tokens_by_stage.get(rec.stage, [])) - 1
            if bucket_pos >= 0:
                self._call_bucket[rec.index] = (rec.stage, bucket_pos)

        new_stage_execs = self._stage_exec_history[self._nonllm_flush_idx:]
        self._nonllm_flush_idx = len(self._stage_exec_history)
        if new_stage_execs:
            nonllm_stage_totals: dict[str, float] = {}
            for stage, duration_ms in new_stage_execs:
                nonllm_stage_totals[stage] = nonllm_stage_totals.get(stage, 0.0) + duration_ms
            for stage, total_ms in nonllm_stage_totals.items():
                self._nonllm_duration_by_stage.setdefault(stage, []).append(total_ms)

        # Extrapolate each stage's total firing count from its observed
        # rate-per-attempt so far (some stages skip-cascade and don't fire
        # on every attempt — see lineage_memory_pipeline.py archive gating).
        # ``self._attempts`` is the flush-cadence denominator (buckets track
        # attempts via ``_on_mutation_attempted`` above), so this projects
        # directly against ``max_mutants`` (attempts cap) with no
        # accept-rate correction needed. Falls back to a denominator of 1
        # when no attempts have been observed yet (e.g. tests that emit
        # LLM_CALL without MUTATION_ATTEMPTED).
        total_units_by_stage = {
            stage: max(1, round(self._pred.max_mutants * len(series) / max(self._attempts, 1)))
            for stage, series in self._tokens_by_stage.items()
        }
        total_units_by_stage.update({
            stage: max(1, round(self._pred.max_mutants * len(series) / max(self._attempts, 1)))
            for stage, series in self._nonllm_duration_by_stage.items()
        })
        fit_tokens = self._winsorised(self._tokens_by_stage)
        fit_tokens_out = self._winsorised(self._tokens_out_by_stage)
        fit_latency = self._winsorised(self._latency_by_stage)
        fit_nonllm = self._winsorised(self._nonllm_duration_by_stage)
        est = estimate_by_stage(
            self._tokens_by_stage, self._latency_by_stage,
            total_units_by_stage=total_units_by_stage,
            max_in_flight=self._pred.max_in_flight,
            law_cls=RobustPowerLaw,
            fit_tokens_by_stage=fit_tokens,
            fit_latency_by_stage=fit_latency,
            ci_method=self._ci_method,
        )
        # Apply the CostMonitorAgent's live overrides (Layer 4): golden_ratio
        # is a safety margin, growth_rate_mult reacts to a sustained
        # size/latency trend the agent detected. Sentinel -1 (out of range)
        # means no change. These now scale the REMAINING work only — the
        # elapsed part of the estimate is measured wall time, not a guess the
        # agent has any business correcting.
        tail_mult = 1.0
        if 0.5 <= self._pred.llm_golden_override <= 2.0:
            tail_mult *= self._pred.llm_golden_override
        if 0.3 <= self._pred.llm_growth_override <= 3.0:
            tail_mult *= self._pred.llm_growth_override
        elapsed_s = self._clock() - self._t0
        self._concurrency = achieved_concurrency(
            self._call_times, now_s=elapsed_s, max_in_flight=self._pred.max_in_flight,
        )
        if 0.3 <= self._pred.llm_concurrency_override <= 3.0:
            self._concurrency = max(0.5, self._concurrency * self._pred.llm_concurrency_override)
        # Duration uses the TTFT+TPOT physical model (latency ~ tokens_out),
        # not a growth law over call index — latency doesn't follow a growth
        # trend in this system (see estimate_duration_by_stage docstring).
        duration_s, duration_ci = estimate_duration_by_stage(
            self._tokens_out_by_stage, self._latency_by_stage,
            total_units_by_stage=total_units_by_stage,
            max_in_flight=self._pred.max_in_flight,
            nonllm_duration_by_stage=self._nonllm_duration_by_stage,
            elapsed_s=elapsed_s,
            concurrency=self._concurrency,
            tail_mult=tail_mult,
            fit_tokens_out_by_stage=fit_tokens_out,
            fit_latency_by_stage=fit_latency,
            fit_nonllm_by_stage=fit_nonllm,
            ci_method=self._ci_method,
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
        if len(self._pending_intervals) > 500:      # ponytail: cap, not a ring buffer
            del self._pending_intervals[:250]
        self._flushes += 1
        self._miscoverages += int(miscovered)
        self._pred.predicted_total_tokens = int(est.predicted_total_tokens)
        self._pred.predicted_duration_s = duration_s
        self._pred.token_ci_low, self._pred.token_ci_high = (
            int(est.tokens_ci[0]), int(est.tokens_ci[1])
        )
        self._pred.ci_low_s, self._pred.ci_high_s = duration_ci
        logger.info("[CostMonitorHook] mutant={} {}", self._counter, self._pred._log_estimate())
        # Compact single-line JSON alongside the human-readable block above —
        # consumers (e.g. tools/task_builder_web) parse this instead of the
        # multi-line text, which is fragile to regex across interleaved logs.
        logger.info(
            "[CostMonitorHookJSON] {}",
            json.dumps({
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
            }),
        )

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
        """Did the agent back a blunt lever with a persistence claim?

        The system prompt has always said golden/growth are only for a
        deviation that PERSISTED across several calls — advice the model was
        free to ignore, and did. Requiring the number turns it into something
        that can be refused.
        """
        try:
            n = int(sustained_over_calls)
        except (TypeError, ValueError):
            n = 0
        if n >= MIN_SUSTAINED_CALLS:
            return True
        logger.info("[CostMonitorHook] blunt lever refused: sustained_over_calls={} < {}",
                    n, MIN_SUSTAINED_CALLS)
        return False

    def _gate_lever(self, name: str, current: float, proposed: float) -> float | None:
        """Deadband + reversal damping. ``None`` means the move was refused.

        Two standard controller guards the loop was missing: a change too
        small to be a decision is dropped, and a change that reverses the
        previous one on the same lever within a couple of wake-ups is taken
        at half step instead of letting the controller hunt.
        """
        if current > 0 and abs(proposed - current) / current < LEVER_DEADBAND:
            logger.info("[CostMonitorHook] {} move ignored: {:.3f}->{:.3f} inside deadband",
                        name, current, proposed)
            return None
        direction = 1 if proposed > current else -1
        prev_dir, prev_at = self._lever_dir.get(name, (0, -10**9))
        reversing = prev_dir != 0 and direction != prev_dir and \
            (self._agent_calls - prev_at) <= REVERSAL_WINDOW
        step = 0.3 / 2 if reversing else 0.3
        value = _clamp_step(current, proposed, step)
        self._lever_dir[name] = (direction, self._agent_calls)
        if reversing:
            logger.info("[CostMonitorHook] {} reverses its last move — half step to {:.3f}",
                        name, value)
        return value

    async def _run_agent(self) -> None:
        """Run one wakeup. The caller must have claimed it via ``_claim_agent``."""
        trigger = self._trigger_reason

        # Build tool context from real live state, not fixed placeholders —
        # previously the agent always saw cold=0.57/golden=1.1/growth=1.0 and
        # backpressure_util=0.8 regardless of what was actually happening, so
        # its transient-vs-regime-change judgment was made half-blind.
        current_golden = self._pred.llm_golden_override if self._pred.llm_golden_override > 0 else 1.0
        current_growth = self._pred.llm_growth_override if self._pred.llm_growth_override > 0 else 1.0
        current_cold = self._pred.llm_cold_override if self._pred.llm_cold_override > 0 else 0.57
        current_conc_mult = (
            self._pred.llm_concurrency_override if self._pred.llm_concurrency_override > 0 else 1.0
        )
        if self._last_bp is not None:
            bp_util = self._last_bp.in_flight / self._last_bp.max_in_flight
            bp_in_flight, bp_max_in_flight = self._last_bp.in_flight, self._last_bp.max_in_flight
        else:
            bp_util, bp_in_flight, bp_max_in_flight = 0.8, 0, 8
        tools = _ToolSet(
            recent_calls=self._call_history[-20:],
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
            miscoverage_rate=(self._miscoverages / self._flushes) if self._flushes else None,
            miscoverage_target=self._aci_alpha,
            width_scale=self._aci_scale,
        )
        self._agent.tools = tools

        try:
            # Build messages from the prompt
            messages = self._agent.build_prompt({"messages": []})
            state = {"messages": messages}
            # Run agent — acall_llm from base class
            result = await self._agent.acall_llm(state)
            result = self._agent.parse_response(result)

            adjustments = result.get("cost_adjustments", {})
            self._log_agent_trace(tools, trigger, adjustments, before={
                "golden_ratio": current_golden,
                "growth_rate_mult": current_growth,
                "concurrency_mult": current_conc_mult,
            })
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
                    v = self._gate_lever("concurrency_mult", current_conc_mult, conc_mult)
                    if v is not None:
                        self._pred.llm_concurrency_override = v
                if reasoning:
                    self._pred.llm_reasoning = reasoning

                self._last_adjustment = {
                    "attempt": self._attempts,
                    "golden": golden, "growth": growth, "concurrency": conc_mult,
                    "outliers": len(outliers),
                    "reasoning": reasoning,
                    "predicted_duration_s": self._pred.predicted_duration_s,
                    "elapsed_s": self._clock() - self._t0,
                }
                logger.info(
                    "[CostMonitorHook] adjustments: golden={} growth={} conc={} "
                    "sustained={} outliers={} trigger={!r} reason={}",
                    golden, growth, conc_mult, sustained, len(outliers), trigger, reasoning)
                emit(CostAgentAdjustment(
                    mutant_index=self._counter,
                    golden_ratio=golden,
                    growth_rate_mult=growth,
                    concurrency_mult=conc_mult,
                    flag_outlier_indices=adjustments.get("flag_outlier_indices", []),
                    skip_calibration=adjustments.get("skip_calibration", False),
                    reasoning=reasoning,
                ))
        except Exception as e:
            logger.warning("[CostMonitorHook] agent call failed: {}", e)
        finally:
            # Release the claim only once this wakeup is fully done, so the
            # other dispatch path cannot start a second concurrent call.
            self._agent_running = False

    def _log_agent_trace(self, tools, trigger: str, adjustments: dict,
                          before: dict | None = None) -> None:
        """One JSON line per wakeup: the evidence in, the tool calls out.

        The observability tools are embedded into the prompt rather than
        called back by the model (see CostMonitorAgent.build_prompt), so
        nothing downstream could otherwise tell *why* a lever moved — only
        that it did. This records the same tool outputs the agent was shown,
        so a wakeup can be re-read after the fact. Emitted even when the
        agent changed nothing: a monitor that keeps declining is a finding.
        """
        def _lever(key: str) -> float:
            try:
                return float(adjustments.get(key, -1))
            except (TypeError, ValueError):
                return -1.0

        outliers = adjustments.get("flag_outlier_indices", []) or []
        skip = bool(adjustments.get("skip_calibration", False))
        levers = {"golden_ratio": _lever("golden_ratio"),
                  "growth_rate_mult": _lever("growth_rate_mult"),
                  "concurrency_mult": _lever("concurrency_mult")}

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
                "actions": actions,
                "levers": levers,
                # what each lever was before this wakeup, so a reader can say
                # "changed A from B to C" instead of just "set A to C"
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
                    "get_last_adjustment_outcome": _safe(tools.get_last_adjustment_outcome),
                    "get_recent_calls": _safe(tools.get_recent_calls, 15),
                },
            }
            logger.info("[CostMonitorAgentTrace] {}", json.dumps(trace, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - tracing is never worth a crash
            logger.warning("[CostMonitorHook] could not write agent trace: {}", exc)

    def _flag_outliers(self, call_indices: list[int]) -> None:
        """Turn the agent's flagged CALL indices into flagged bucket positions.

        ``flag_outlier_indices`` used to be written to CostPrediction and the
        COST_AGENT_ADJUSTMENT event and read by nobody — the same dead-code
        shape the golden/growth overrides had. Now a flagged call winsorises
        its bucket out of the growth-law fit (see :meth:`_winsorised`), which
        is the surgical action a surprise-triggered agent actually needs: it
        answers "transient spike" without touching the multipliers.
        """
        for idx in call_indices:
            self._flagged_calls.add(idx)
            entry = self._call_bucket.get(idx)
            if entry is None:
                continue
            stage, bucket_pos = entry
            self._flagged_buckets.setdefault(stage, set()).add(bucket_pos)
        if call_indices:
            self._pred.llm_outlier_indices = sorted(
                set(self._pred.llm_outlier_indices) | set(call_indices))

    def add_llm_call(self, tokens_in: int, tokens_out: int, latency_ms: float, stage: str) -> None:
        """Record an LLM call for the agent to analyse."""
        self._call_history.append(LlmCallRecord(
            index=len(self._call_history),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            stage=stage,
        ))