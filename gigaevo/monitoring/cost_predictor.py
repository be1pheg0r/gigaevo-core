from __future__ import annotations

from dataclasses import dataclass, field
import statistics
from typing import ClassVar

# ── Constants (validated on 4 benchmarks) ─────────────────────
COLD_START_FACTOR = 0.57
WARMUP_CALLS = 3
CALIBRATION_WINDOW = 5
GOLDEN_RATIO_MAX = 1.1
GOLDEN_RATIO_RAMP_STEPS = 5


def _robust_rate(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _prediction_interval(
    values: list[float],
    *,
    sparse_low_mult: float = 0.85,
    sparse_high_mult: float = 1.15,
) -> tuple[float, float]:
    if len(values) < 3:
        m = _robust_rate(values)
        return m * sparse_low_mult, m * sparse_high_mult
    s = sorted(values)
    lo = s[max(0, len(s) // 10)]
    hi = s[min(len(s) - 1, 9 * len(s) // 10)]
    return lo, hi


def _ramped_golden(steps: int, *, golden_ratio_max: float, ramp_steps: int) -> float:
    return 1.0 + (golden_ratio_max - 1.0) * min(steps, ramp_steps) / ramp_steps


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class CostPredictorSettings:
    """Hydra-configurable parameters of the legacy baseline/calibration layers."""

    cold_start_factor: float = COLD_START_FACTOR
    warmup_calls: int = WARMUP_CALLS
    calibration_window: int = CALIBRATION_WINDOW
    golden_ratio_max: float = GOLDEN_RATIO_MAX
    golden_ratio_ramp_steps: int = GOLDEN_RATIO_RAMP_STEPS
    sparse_interval_low_mult: float = 0.85
    sparse_interval_high_mult: float = 1.15
    calls_per_mutation: float = 94.0 / 50.0
    average_total_duration_s: float = 1397.5
    baseline_call_count: int = 94
    complexity_weight: float = 0.02
    depth_weight: float = 0.05
    growth_fast_mult: float = 1.4
    growth_medium_mult: float = 1.1
    growth_slow_mult: float = 0.9
    bottleneck_mult: float = 1.3
    chain_mult: float = 1.5
    base_tokens_per_call: int = 5000
    initial_duration_ci_low_mult: float = 0.6
    initial_duration_ci_high_mult: float = 1.5
    initial_token_ci_low_mult: float = 0.7
    initial_token_ci_high_mult: float = 1.3
    default_backpressure_utilization: float = 0.8
    low_backpressure_threshold: float = 0.5
    medium_backpressure_threshold: float = 0.8
    low_backpressure_mult: float = 1.5
    medium_backpressure_mult: float = 1.1
    calibrated_ci_low_mult: float = 0.7
    calibrated_ci_high_mult: float = 1.3
    smoothing_ci_low_mult: float = 0.8
    smoothing_ci_high_mult: float = 1.2
    smoothing_min_wall_s: float = 0.1
    smoothing_min_throughput: float = 0.001


@dataclass
class DagTopology:
    stage_names: list[str] = field(default_factory=list)
    llm_stages: set[str] = field(default_factory=set)
    non_llm_stages: set[str] = field(default_factory=set)
    seq_depth: int = 1
    fanout_depth: int = 1
    _BOTTLENECK_STAGES: ClassVar[set[str]] = {
        "CallProgramFunction",
        "CallValidatorFunction",
        "IntraMemoryStage",
    }

    @property
    def is_chain_task(self) -> bool:
        return any(
            kw in s.lower()
            for s in self.stage_names
            for kw in {"chain", "prompt", "eval"}
        )

    @classmethod
    def from_stage_names(cls, stage_names: list[str]) -> DagTopology:
        known_llm = {
            "MutationSuggestionStage",
            "MutationAgent",
            "LineageStage",
            "InsightsStage",
        }
        llm_s = {s for s in stage_names if s in known_llm}
        known_non_llm = {
            "ComputeComplexityStage",
            "ValidateCodeStage",
            "CallProgramFunction",
            "CallValidatorFunction",
            "FetchMetrics",
            "MergeMetricsStage",
            "EnsureMetricsStage",
            "MutationContextStage",
            "DescendantProgramIds",
            "AncestorProgramIds",
            "EvolutionaryStatisticsCollector",
            "MemoryContextStage",
            "LineagesFromAncestors",
            "LineagesToDescendants",
            "IntraMemoryStage",
            "ArchivePotentialGateStage",
            "FetchArtifact",
            "FormatterStage",
        }
        non_llm_s = {s for s in stage_names if s in known_non_llm and s not in llm_s}
        seq_parts = [
            "ComputeComplexityStage",
            "ValidateCodeStage",
            "CallProgramFunction",
            "CallValidatorFunction",
            "FetchMetrics",
            "MergeMetricsStage",
            "EnsureMetricsStage",
            "MutationContextStage",
            "MutationSuggestionStage",
        ]
        seq_depth = sum(1 for s in seq_parts if s in stage_names)
        return cls(
            stage_names=stage_names,
            llm_stages=llm_s,
            non_llm_stages=non_llm_s,
            seq_depth=max(seq_depth, 1),
            fanout_depth=len(stage_names),
        )

    @property
    def llm_stage_ratio(self) -> float:
        total = len(self.stage_names)
        return len(self.llm_stages) / total if total > 0 else 0.0

    @property
    def has_bottleneck(self) -> bool:
        return bool(self._BOTTLENECK_STAGES & set(self.stage_names))


@dataclass
class LlmCallSummary:
    total_calls: int = 0
    ok_calls: int = 0
    fail_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    latencies_9b: list[float] = field(default_factory=list)
    latencies_35b: list[float] = field(default_factory=list)
    tokens_per_call: list[int] = field(default_factory=list)

    @property
    def avg_latency_9b(self) -> float:
        return statistics.mean(self.latencies_9b) if self.latencies_9b else 0.0

    @property
    def avg_latency_35b(self) -> float:
        return statistics.mean(self.latencies_35b) if self.latencies_35b else 0.0

    @property
    def success_rate(self) -> float:
        total = self.ok_calls + self.fail_calls
        return self.ok_calls / total if total > 0 else 0.0

    @property
    def median_tokens(self) -> float:
        return _robust_rate([float(t) for t in self.tokens_per_call])

    @property
    def token_ci(self) -> tuple[float, float]:
        return _prediction_interval([float(t) for t in self.tokens_per_call])


@dataclass
class StageExecSummary:
    stage_durations: dict[str, list[float]] = field(default_factory=dict)

    def add(self, stage: str, duration_ms: float, decision: str = "") -> None:
        self.stage_durations.setdefault(stage, []).append(duration_ms)

    @property
    def call_program_avg_ms(self) -> float:
        return (
            statistics.mean(self.stage_durations.get("CallProgramFunction", [])) or 0.0
        )

    @property
    def call_validator_avg_ms(self) -> float:
        return (
            statistics.mean(self.stage_durations.get("CallValidatorFunction", []))
            or 0.0
        )

    @property
    def intra_memory_avg_ms(self) -> float:
        return statistics.mean(self.stage_durations.get("IntraMemoryStage", [])) or 0.0


@dataclass
class BackpressureSnapshot:
    producer_held: int = 0
    buffer_held: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    llm_active: int = 0

    @property
    def utilization(self) -> float:
        return self.in_flight / max(self.max_in_flight, 1)


@dataclass
class CostPrediction:
    """Live cost prediction with warmup, robust calibration, and LLM agent adjustments."""

    dag_topology: DagTopology | None = None
    prompt_tokens: int = 0
    code_length: int = 0
    complexity_score: float = 0.0
    llm_growth_rate: str = "fast"
    bottleneck_hint: str = ""
    max_mutants: int = 50
    max_in_flight: int = 8
    settings: CostPredictorSettings = field(default_factory=CostPredictorSettings)

    # LLM agent overrides (set by CostMonitorAgent)
    llm_cold_override: float = -1.0
    llm_golden_override: float = -1.0
    llm_growth_override: float = -1.0
    # Multiplier on the MEASURED achieved LLM concurrency. The agent's only
    # lever that acts on the divisor rather than on the work estimate — for
    # server-load regime changes the trailing-window measurement has not
    # caught up with yet.
    llm_concurrency_override: float = -1.0
    llm_outlier_indices: list[int] = field(default_factory=list)
    llm_skip_next_calib: bool = False
    llm_reasoning: str = ""

    warmup_completed: bool = False
    calibration_completed: bool = False
    calibration_mutants_done: int = 0
    wall_time_at_calibration: float = 0.0
    steps_since_calibration: int = 0

    predicted_duration_s: float = 0.0
    predicted_llm_calls: int = 0
    predicted_total_tokens: int = 0
    ci_low_s: float = 0.0
    ci_high_s: float = 0.0
    token_ci_low: int = 0
    token_ci_high: int = 0

    @property
    def is_ready(self) -> bool:
        return self.predicted_duration_s > 0

    def _log_estimate(self) -> str:
        s = self.predicted_duration_s
        h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
        return (
            f"Predicted: ~{h}h {m}m {sec}s "
            f"([{int(s // 60)} min] ± {self.ci_high_s - s:.0f}s)\n"
            f"  LLM calls: ~{self.predicted_llm_calls}\n"
            f"  Tokens: ~{self.predicted_total_tokens:,} "
            f"[{self.token_ci_low:,}–{self.token_ci_high:,}]\n"
            f"  CI: [{self.ci_low_s:.0f}s, {self.ci_high_s:.0f}s]\n"
            f"  LLM adj: {self.llm_reasoning or 'none'}"
        )


# --------------------------------------------------------------------------- #
# Layer 1: Static baseline
# --------------------------------------------------------------------------- #


def compute_static_baseline(
    *,
    dag: DagTopology,
    prompt_tokens: int,
    code_length: int,
    complexity_score: float,
    max_mutants: int,
    max_in_flight: int,
    llm_growth_rate: str = "fast",
    bottleneck_hint: str = "",
    settings: CostPredictorSettings | None = None,
) -> CostPrediction:
    settings = settings or CostPredictorSettings()
    calls_per_mutation = settings.calls_per_mutation
    avg_duration_per_call = (
        settings.average_total_duration_s / settings.baseline_call_count
    )

    complexity_factor = 1.0 + complexity_score * settings.complexity_weight
    depth_factor = 1.0 + (dag.seq_depth - 1) * settings.depth_weight
    growth_adj = {
        "fast": settings.growth_fast_mult,
        "medium": settings.growth_medium_mult,
        "slow": settings.growth_slow_mult,
    }.get(llm_growth_rate, settings.growth_medium_mult)
    bottleneck_adj = settings.bottleneck_mult if dag.has_bottleneck else 1.0
    chain_adj = settings.chain_mult if dag.is_chain_task else 1.0

    per_mutation = (
        avg_duration_per_call
        * calls_per_mutation
        * (complexity_factor * depth_factor * growth_adj * bottleneck_adj * chain_adj)
    )
    predicted = max_mutants * per_mutation / max(1, max_in_flight)

    base_tokens_per_call = settings.base_tokens_per_call
    cold_start_tokens = int(base_tokens_per_call * settings.cold_start_factor)

    return CostPrediction(
        dag_topology=dag,
        prompt_tokens=prompt_tokens,
        code_length=code_length,
        complexity_score=complexity_score,
        llm_growth_rate=llm_growth_rate,
        bottleneck_hint=bottleneck_hint,
        max_mutants=max_mutants,
        max_in_flight=max_in_flight,
        settings=settings,
        predicted_duration_s=predicted,
        predicted_llm_calls=int(max_mutants * calls_per_mutation),
        predicted_total_tokens=int(
            settings.warmup_calls * cold_start_tokens
            + (max_mutants * calls_per_mutation - settings.warmup_calls)
            * base_tokens_per_call
        ),
        ci_low_s=predicted * settings.initial_duration_ci_low_mult,
        ci_high_s=predicted * settings.initial_duration_ci_high_mult,
        token_ci_low=int(predicted * settings.initial_token_ci_low_mult),
        token_ci_high=int(predicted * settings.initial_token_ci_high_mult),
    )


# --------------------------------------------------------------------------- #
# Layer 2: Live calibration
# --------------------------------------------------------------------------- #


def calibrate_prediction(
    pred: CostPrediction,
    *,
    done_mutants: int,
    wall_time_s: float,
    llm_summary: LlmCallSummary,
    stage_summary: StageExecSummary,
    bp_snapshot: BackpressureSnapshot | None = None,
) -> CostPrediction:
    """Update prediction with live telemetry + LLM agent adjustments."""

    if done_mutants < 1:
        return pred

    # LLM agent says skip this calibration
    if pred.llm_skip_next_calib:
        pred.llm_skip_next_calib = False
        return pred

    # ── Warmup phase ──
    settings = pred.settings
    if done_mutants <= settings.warmup_calls and not pred.warmup_completed:
        alpha = done_mutants / settings.warmup_calls
        real_rate = wall_time_s / done_mutants
        cold_est = pred.predicted_duration_s / max(pred.max_mutants, 1)
        blended_rate = cold_est * (1 - alpha) + real_rate * alpha
        pred.predicted_duration_s = blended_rate * pred.max_mutants

        if llm_summary.tokens_per_call:
            real_tok = _robust_rate([float(t) for t in llm_summary.tokens_per_call])
            cold_tok = pred.predicted_total_tokens / max(pred.predicted_llm_calls, 1)
            blended_tok = cold_tok * (1 - alpha) + real_tok * alpha
            pred.predicted_total_tokens = int(blended_tok * pred.predicted_llm_calls)

        pred.wall_time_at_calibration = wall_time_s
        pred.calibration_mutants_done = done_mutants
        pred.steps_since_calibration = 0
        if done_mutants == settings.warmup_calls:
            pred.warmup_completed = True
        return pred

    # ── Robust calibration ──
    is_calib_point = done_mutants % settings.calibration_window == 0

    if is_calib_point or not pred.warmup_completed:
        real_throughput = wall_time_s / max(done_mutants, 1)
        util = (
            bp_snapshot.utilization
            if bp_snapshot
            else settings.default_backpressure_utilization
        )
        if util < settings.low_backpressure_threshold:
            bp_adj = settings.low_backpressure_mult
        elif util < settings.medium_backpressure_threshold:
            bp_adj = settings.medium_backpressure_mult
        else:
            bp_adj = 1.0

        golden = _ramped_golden(
            pred.steps_since_calibration,
            golden_ratio_max=settings.golden_ratio_max,
            ramp_steps=settings.golden_ratio_ramp_steps,
        )

        # Apply LLM agent overrides
        if 0.5 <= pred.llm_golden_override <= 2.0:
            golden = pred.llm_golden_override
        extra_growth = (
            pred.llm_growth_override if 0.3 <= pred.llm_growth_override <= 3.0 else 1.0
        )

        new_duration = (
            real_throughput * pred.max_mutants * bp_adj * golden * extra_growth
        )

        if llm_summary.tokens_per_call and len(llm_summary.tokens_per_call) >= 3:
            rate = _robust_rate([float(t) for t in llm_summary.tokens_per_call])
            lo, hi = _prediction_interval(
                [float(t) for t in llm_summary.tokens_per_call],
                sparse_low_mult=settings.sparse_interval_low_mult,
                sparse_high_mult=settings.sparse_interval_high_mult,
            )
            pred.predicted_total_tokens = int(rate * pred.predicted_llm_calls)
            pred.token_ci_low = int(lo * pred.predicted_llm_calls)
            pred.token_ci_high = int(hi * pred.predicted_llm_calls)

        if llm_summary.success_rate > 0:
            pred.predicted_llm_calls = int(
                pred.predicted_llm_calls * llm_summary.success_rate
            )

        pred.predicted_duration_s = new_duration
        pred.ci_low_s = new_duration * settings.calibrated_ci_low_mult
        pred.ci_high_s = new_duration * settings.calibrated_ci_high_mult
        pred.wall_time_at_calibration = wall_time_s
        pred.calibration_mutants_done = done_mutants
        pred.calibration_completed = True
        pred.steps_since_calibration = 0
    else:
        pred.steps_since_calibration += 1
        golden = _ramped_golden(
            pred.steps_since_calibration,
            golden_ratio_max=settings.golden_ratio_max,
            ramp_steps=settings.golden_ratio_ramp_steps,
        )
        if 0.5 <= pred.llm_golden_override <= 2.0:
            golden = pred.llm_golden_override
        if pred.calibration_completed:
            rate = pred.predicted_duration_s / max(pred.max_mutants, 1)
            extra = pred.max_mutants - pred.calibration_mutants_done
            pred.predicted_duration_s = (
                pred.wall_time_at_calibration + rate * golden * extra
            )

    return pred


# --------------------------------------------------------------------------- #
# Exponential smoothing
# --------------------------------------------------------------------------- #


def exponential_smoothing_estimate(
    predictions: list[CostPrediction],
    current_wall_s: float,
    current_mutants_done: int,
    alpha: float = 0.3,
) -> CostPrediction | None:
    if len(predictions) < 2:
        return None
    last = predictions[-1]
    if not last.is_ready:
        return None
    settings = last.settings
    current_throughput = current_mutants_done / max(
        current_wall_s, settings.smoothing_min_wall_s
    )
    prev_throughput = last.calibration_mutants_done / max(
        last.wall_time_at_calibration, settings.smoothing_min_wall_s
    )
    smoothed = alpha * current_throughput + (1 - alpha) * prev_throughput
    remaining = last.max_mutants - last.calibration_mutants_done
    golden = _ramped_golden(
        min(remaining, settings.golden_ratio_ramp_steps),
        golden_ratio_max=settings.golden_ratio_max,
        ramp_steps=settings.golden_ratio_ramp_steps,
    )
    if 0.5 <= last.llm_golden_override <= 2.0:
        golden = last.llm_golden_override
    additional_duration = (
        remaining / max(smoothed, settings.smoothing_min_throughput) * golden
    )
    result = CostPrediction(
        dag_topology=last.dag_topology,
        prompt_tokens=last.prompt_tokens,
        code_length=last.code_length,
        complexity_score=last.complexity_score,
        llm_growth_rate=last.llm_growth_rate,
        bottleneck_hint=last.bottleneck_hint,
        max_mutants=last.max_mutants,
        max_in_flight=last.max_in_flight,
        settings=settings,
        predicted_duration_s=last.wall_time_at_calibration + additional_duration,
        predicted_llm_calls=last.predicted_llm_calls,
        predicted_total_tokens=last.predicted_total_tokens,
        ci_low_s=(last.wall_time_at_calibration + additional_duration)
        * settings.smoothing_ci_low_mult,
        ci_high_s=(last.wall_time_at_calibration + additional_duration)
        * settings.smoothing_ci_high_mult,
        token_ci_low=last.token_ci_low,
        token_ci_high=last.token_ci_high,
        llm_cold_override=last.llm_cold_override,
        llm_golden_override=last.llm_golden_override,
        llm_growth_override=last.llm_growth_override,
        llm_outlier_indices=last.llm_outlier_indices,
        llm_skip_next_calib=last.llm_skip_next_calib,
        llm_reasoning=last.llm_reasoning,
    )
    result.calibration_completed = True
    result.calibration_mutants_done = current_mutants_done
    result.wall_time_at_calibration = current_wall_s
    result.warmup_completed = last.warmup_completed
    return result
