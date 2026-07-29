from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import statistics


# ── Constants (validated on 4 benchmarks) ─────────────────────
COLD_START_FACTOR = 0.57   # first call is ~43% cheaper than steady-state
WARMUP_CALLS = 3           # calls before switching to live calibration
CALIBRATION_WINDOW = 5     # recalibrate every N mutants
GOLDEN_RATIO_MAX = 1.1     # max GPU variance multiplier (ramps from 1.0)
GOLDEN_RATIO_RAMP_STEPS = 5  # steps to reach max golden_ratio


def _robust_rate(values: list[float]) -> float:
    """Median-based robust estimator — ignores outliers."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _prediction_interval(values: list[float]) -> tuple[float, float]:
    """Return [p10, p90] from a list of values."""
    if len(values) < 3:
        m = _robust_rate(values)
        return m * 0.85, m * 1.15
    s = sorted(values)
    lo = s[max(0, len(s) // 10)]
    hi = s[min(len(s) - 1, 9 * len(s) // 10)]
    return lo, hi


def _ramped_golden(steps_since_calibration: int) -> float:
    """Golden ratio ramps from 1.0 (next call) to GOLDEN_RATIO_MAX."""
    return 1.0 + (GOLDEN_RATIO_MAX - 1.0) * min(steps_since_calibration, GOLDEN_RATIO_RAMP_STEPS) / GOLDEN_RATIO_RAMP_STEPS


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

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
        chain_keywords = {"chain", "prompt", "eval"}
        return any(kw in s.lower() for s in self.stage_names for kw in chain_keywords)

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
            "ComputeComplexityStage", "ValidateCodeStage", "CallProgramFunction",
            "CallValidatorFunction", "FetchMetrics", "MergeMetricsStage",
            "EnsureMetricsStage", "MutationContextStage", "DescendantProgramIds",
            "AncestorProgramIds", "EvolutionaryStatisticsCollector",
            "MemoryContextStage", "LineagesFromAncestors", "LineagesToDescendants",
            "IntraMemoryStage", "ArchivePotentialGateStage", "FetchArtifact",
            "FormatterStage",
        }
        non_llm_s = {s for s in stage_names if s in known_non_llm and s not in llm_s}
        fanout = len(stage_names)
        seq_parts = [
            "ComputeComplexityStage", "ValidateCodeStage", "CallProgramFunction",
            "CallValidatorFunction", "FetchMetrics", "MergeMetricsStage",
            "EnsureMetricsStage", "MutationContextStage", "MutationSuggestionStage",
        ]
        seq_depth = sum(1 for s in seq_parts if s in stage_names)
        return cls(
            stage_names=stage_names, llm_stages=llm_s, non_llm_stages=non_llm_s,
            seq_depth=max(seq_depth, 1), fanout_depth=max(fanout, 1),
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
        return statistics.mean(self.stage_durations.get("CallProgramFunction", [])) or 0.0

    @property
    def call_validator_avg_ms(self) -> float:
        return statistics.mean(self.stage_durations.get("CallValidatorFunction", [])) or 0.0

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
        cap = max(self.max_in_flight, 1)
        return self.in_flight / cap


@dataclass
class CostPrediction:
    # Static layer
    dag_topology: DagTopology | None = None
    prompt_tokens: int = 0
    code_length: int = 0
    complexity_score: float = 0.0
    llm_growth_rate: str = "fast"
    bottleneck_hint: str = ""
    max_mutants: int = 50
    max_in_flight: int = 8

    # Warmup / calibration state
    warmup_completed: bool = False
    calibration_completed: bool = False
    calibration_mutants_done: int = 0
    wall_time_at_calibration: float = 0.0
    steps_since_calibration: int = 0

    # Predicted outputs
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
            f"([{int(s//60)} min] ± {self.ci_high_s - s:.0f}s)\n"
            f"  LLM calls: ~{self.predicted_llm_calls}\n"
            f"  Tokens: ~{self.predicted_total_tokens:,} "
            f"[{self.token_ci_low:,}–{self.token_ci_high:,}]\n"
            f"  CI: [{self.ci_low_s:.0f}s, {self.ci_high_s:.0f}s]"
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
) -> CostPrediction:
    """Build prediction from static features."""
    calls_per_mutation = 94.0 / 50.0
    avg_total_duration = 1397.5
    avg_duration_per_call = avg_total_duration / 94

    complexity_factor = 1.0 + complexity_score * 0.02
    depth_factor = 1.0 + (dag.seq_depth - 1) * 0.05
    growth_adj = {"fast": 1.4, "medium": 1.1, "slow": 0.9}.get(llm_growth_rate, 1.1)
    bottleneck_adj = 1.3 if dag.has_bottleneck else 1.0
    chain_adj = 1.5 if dag.is_chain_task else 1.0

    per_mutation = avg_duration_per_call * calls_per_mutation * (
        complexity_factor * depth_factor * growth_adj * bottleneck_adj * chain_adj
    )
    predicted = max_mutants * per_mutation / max(1, max_in_flight)

    # Apply cold start to token estimate
    base_tokens_per_call = 5000
    cold_start_tokens = int(base_tokens_per_call * COLD_START_FACTOR)

    return CostPrediction(
        dag_topology=dag, prompt_tokens=prompt_tokens, code_length=code_length,
        complexity_score=complexity_score, llm_growth_rate=llm_growth_rate,
        bottleneck_hint=bottleneck_hint, max_mutants=max_mutants,
        max_in_flight=max_in_flight,
        predicted_duration_s=predicted,
        predicted_llm_calls=int(max_mutants * calls_per_mutation),
        predicted_total_tokens=int(
            WARMUP_CALLS * cold_start_tokens +
            (max_mutants * calls_per_mutation - WARMUP_CALLS) * base_tokens_per_call
        ),
        ci_low_s=predicted * 0.6, ci_high_s=predicted * 1.5,
        token_ci_low=int(predicted * 0.7), token_ci_high=int(predicted * 1.3),
    )


# --------------------------------------------------------------------------- #
# Layer 2: Live calibration — warmup + robust median + ramped golden_ratio
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
    """Update prediction with live telemetry using robust estimators."""

    if done_mutants < 1:
        return pred

    # ── Warmup phase ──
    if done_mutants <= WARMUP_CALLS and not pred.warmup_completed:
        alpha = done_mutants / WARMUP_CALLS  # 0→1 blend
        real_rate = wall_time_s / done_mutants

        # Blend cold_start estimate with real data
        cold_est = pred.predicted_duration_s / pred.max_mutants  # per-mutant from static
        blended_rate = cold_est * (1 - alpha) + real_rate * alpha

        new_duration = blended_rate * pred.max_mutants

        # Token estimate with cold start
        if llm_summary.tokens_per_call:
            real_tok = _robust_rate([float(t) for t in llm_summary.tokens_per_call])
            cold_tok = pred.predicted_total_tokens / max(pred.predicted_llm_calls, 1)
            blended_tok = cold_tok * (1 - alpha) + real_tok * alpha
            pred.predicted_total_tokens = int(blended_tok * pred.predicted_llm_calls)

        pred.predicted_duration_s = new_duration
        pred.wall_time_at_calibration = wall_time_s
        pred.calibration_mutants_done = done_mutants
        pred.steps_since_calibration = 0

        if done_mutants == WARMUP_CALLS:
            pred.warmup_completed = True
        return pred

    # ── Robust calibration (every CALIBRATION_WINDOW mutants) ──
    is_calib_point = (done_mutants % CALIBRATION_WINDOW == 0)

    if is_calib_point or not pred.warmup_completed:
        real_throughput = wall_time_s / max(done_mutants, 1)

        # Backpressure
        util = bp_snapshot.utilization if bp_snapshot else 0.8
        if util < 0.5:
            bp_adj = 1.5
        elif util < 0.8:
            bp_adj = 1.1
        else:
            bp_adj = 1.0

        # Golden ratio: ramp from 1.0
        golden = _ramped_golden(pred.steps_since_calibration)

        new_duration = real_throughput * pred.max_mutants * bp_adj * golden

        # Token prediction with robust rate + intervals
        if llm_summary.tokens_per_call and len(llm_summary.tokens_per_call) >= 3:
            rate = _robust_rate([float(t) for t in llm_summary.tokens_per_call])
            lo, hi = _prediction_interval([float(t) for t in llm_summary.tokens_per_call])
            pred.predicted_total_tokens = int(rate * pred.predicted_llm_calls)
            pred.token_ci_low = int(lo * pred.predicted_llm_calls)
            pred.token_ci_high = int(hi * pred.predicted_llm_calls)

        # LLM calls adjusted by success rate
        if llm_summary.success_rate > 0:
            pred.predicted_llm_calls = int(pred.predicted_llm_calls * llm_summary.success_rate)

        pred.predicted_duration_s = new_duration
        pred.ci_low_s = new_duration * 0.7
        pred.ci_high_s = new_duration * 1.3
        pred.wall_time_at_calibration = wall_time_s
        pred.calibration_mutants_done = done_mutants
        pred.calibration_completed = True
        pred.steps_since_calibration = 0
    else:
        # Between calibrations: extrapolate with ramped golden
        pred.steps_since_calibration += 1
        golden = _ramped_golden(pred.steps_since_calibration)

        if pred.calibration_completed:
            rate = pred.predicted_duration_s / max(pred.max_mutants, 1)
            extra = pred.max_mutants - pred.calibration_mutants_done
            pred.predicted_duration_s = pred.wall_time_at_calibration + rate * golden * extra

    return pred


# --------------------------------------------------------------------------- #
# Exponential smoothing for online tracking
# --------------------------------------------------------------------------- #

def exponential_smoothing_estimate(
    predictions: list[CostPrediction],
    current_wall_s: float,
    current_mutants_done: int,
    alpha: float = 0.3,
) -> CostPrediction | None:
    """Smooth throughput with exponential moving average."""
    if len(predictions) < 2:
        return None

    last = predictions[-1]
    if not last.is_ready:
        return None

    current_throughput = current_mutants_done / max(current_wall_s, 0.1)
    prev_throughput = last.calibration_mutants_done / max(last.wall_time_at_calibration, 0.1)
    smoothed = alpha * current_throughput + (1 - alpha) * prev_throughput

    remaining = last.max_mutants - last.calibration_mutants_done
    # Ramped golden for remaining projection
    golden = _ramped_golden(min(remaining, GOLDEN_RATIO_RAMP_STEPS))
    additional_duration = remaining / max(smoothed, 0.001) * golden

    result = CostPrediction(
        dag_topology=last.dag_topology, prompt_tokens=last.prompt_tokens,
        code_length=last.code_length, complexity_score=last.complexity_score,
        llm_growth_rate=last.llm_growth_rate, bottleneck_hint=last.bottleneck_hint,
        max_mutants=last.max_mutants, max_in_flight=last.max_in_flight,
        predicted_duration_s=last.wall_time_at_calibration + additional_duration,
        predicted_llm_calls=last.predicted_llm_calls,
        predicted_total_tokens=last.predicted_total_tokens,
        ci_low_s=(last.wall_time_at_calibration + additional_duration) * 0.8,
        ci_high_s=(last.wall_time_at_calibration + additional_duration) * 1.2,
        token_ci_low=last.token_ci_low, token_ci_high=last.token_ci_high,
    )
    result.calibration_completed = True
    result.calibration_mutants_done = current_mutants_done
    result.wall_time_at_calibration = current_wall_s
    result.warmup_completed = last.warmup_completed
    return result