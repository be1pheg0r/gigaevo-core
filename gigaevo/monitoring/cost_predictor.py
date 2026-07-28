from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import statistics


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class DagTopology:
    stage_names: list[str] = field(default_factory=list)
    llm_stages: set[str] = field(default_factory=set)
    non_llm_stages: set[str] = field(default_factory=set)
    # Number of stages in the longest sequential chain.
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
        known_non_llm = known_dag = {
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
        fanout = len(stage_names)
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
            fanout_depth=max(fanout, 1),
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
        return (
            statistics.mean(self.stage_durations.get("IntraMemoryStage", [])) or 0.0
        )


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

    # Calibration layer (filled live)
    calibration_completed: bool = False
    calibration_mutants_done: int = 0
    wall_time_at_calibration: float = 0.0  # seconds

    # Predicted outputs
    predicted_duration_s: float = 0.0
    predicted_llm_calls: int = 0
    predicted_total_tokens: int = 0
    ci_low_s: float = 0.0
    ci_high_s: float = 0.0

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
            f"  Tokens: ~{self.predicted_total_tokens:,}\n"
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
    """Build a prediction purely from static features.

    Base costs derived from 4 benchmark observations (heilbron, hexagon_pack,
    spherical_codes, first_autocorr_ineq):
    - ~94 LLM calls per run (94 calls, ~740s wall-clock)
    - LLM calls dominate the wall-clock (1.5M ms total vs 739s wall-clock,
      due to parallelism, actual CPU time is dominated by LLM latency)
    """
    calls_per_mutation = 94.0 / 50.0

    avg_total_duration = 1397.5  # (739+658+1531+3062) / 4
    avg_duration_per_call = avg_total_duration / 94  # ~14.86s/call

    complexity_factor = 1.0 + complexity_score * 0.02
    depth_factor = 1.0 + (dag.seq_depth - 1) * 0.05
    growth_adj = {
        "fast": 1.4,
        "medium": 1.1,
        "slow": 0.9,
    }.get(llm_growth_rate, 1.1)
    bottleneck_adj = 1.3 if dag.has_bottleneck else 1.0
    chain_adj = 1.5 if dag.is_chain_task else 1.0

    per_mutation = avg_duration_per_call * calls_per_mutation * (
        complexity_factor * depth_factor * growth_adj * bottleneck_adj * chain_adj
    )
    predicted = max_mutants * per_mutation / max(1, max_in_flight)

    pred = CostPrediction(
        dag_topology=dag,
        prompt_tokens=prompt_tokens,
        code_length=code_length,
        complexity_score=complexity_score,
        llm_growth_rate=llm_growth_rate,
        bottleneck_hint=bottleneck_hint,
        max_mutants=max_mutants,
        max_in_flight=max_in_flight,
        predicted_duration_s=predicted,
        predicted_llm_calls=int(max_mutants * calls_per_mutation),
        predicted_total_tokens=int(max_mutants * calls_per_mutation * 5000),
        ci_low_s=predicted * 0.6,
        ci_high_s=predicted * 1.5,
    )
    return pred


# --------------------------------------------------------------------------- #
# Layer 2: Calibration — live updates after first k mutants
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
    """Update a static prediction with live telemetry.

    After `done_mutants` mutants complete:
    1. Compute actual throughput (wall_time / done_mutants).
    2. Replace LLM latency assumption with observed p95 latency.
    3. Adjust for backpressure utilization.
    4. Extrapolate remaining mutants linearly on log-scale.
    """
    if done_mutants < 2:
        return pred

    real_throughput = wall_time_s / done_mutants  # seconds per mutant

    # Backpressure utilization: lower utilization = more headroom
    util = bp_snapshot.utilization if bp_snapshot else 0.8  # default assumption

    # Calibration factor: if real throughput is faster than predicted, scale down
    if pred.is_ready:
        old_per_mutant = pred.predicted_duration_s / pred.max_mutants
        calibration_factor = min(real_throughput / max(old_per_mutant, 0.1), 3.0)
        # Don't over-correct: clamp to [0.3, 2.0]
        calibration_factor = max(0.3, min(2.0, calibration_factor))

        new_duration = real_throughput * pred.max_mutants
    else:
        # No previous prediction — use real throughput directly
        new_duration = real_throughput * pred.max_mutants

    # Apply utilization adjustment: underutilized pipeline means we're not
    # saturating max_in_flight, so actual duration may be higher.
    if util < 0.5:
        new_duration *= 1.5
    elif util < 0.8:
        new_duration *= 1.1

    # Adjust prediction based on observed LLM latency
    if llm_summary.ok_calls > 0:
        # Replace assumed LLM latency with observed
        observed_avg_llm_lat = (
            (llm_summary.avg_latency_9b + llm_summary.avg_latency_35b) / 2
        ) if llm_summary.avg_latency_9b > 0 or llm_summary.avg_latency_35b > 0 else 0.0
        if observed_avg_llm_lat > 0:
            # Real throughput already captured this, so no double adjustment.
            pass

    # LLM calls prediction: adjust by success rate
    success_ratio = llm_summary.success_rate
    pred.predicted_llm_calls = int(
        pred.predicted_llm_calls * success_ratio
    )

    pred.wall_time_at_calibration = wall_time_s
    pred.calibration_completed = True
    pred.calibration_mutants_done = done_mutants
    pred.predicted_duration_s = new_duration
    pred.ci_low_s = new_duration * 0.7
    pred.ci_high_s = new_duration * 1.3

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
    """Maintain running estimate using exponential smoothing of throughput.

    Smooths the throughput estimate (mutants per second) across time and
    returns the next updated prediction.

    This function does NOT modify the last prediction in-place — it creates
    a new `CostPrediction` instance with corrected values, keeping the
    history traceable.
    """
    if len(predictions) < 2:
        return None

    last = predictions[-1]
    if not last.is_ready:
        return None

    # Current throughput
    current_throughput = current_mutants_done / max(current_wall_s, 0.1)

    # Previous throughput (from the previous snapshot)
    prev_throughput = last.calibration_mutants_done / max(
        last.wall_time_at_calibration, 0.1
    )

    # Smoothed throughput
    smoothed = alpha * current_throughput + (1 - alpha) * prev_throughput
    remaining = last.max_mutants - last.calibration_mutants_done
    additional_duration = remaining / max(smoothed, 0.001)

    result = CostPrediction(
        dag_topology=last.dag_topology,
        prompt_tokens=last.prompt_tokens,
        code_length=last.code_length,
        complexity_score=last.complexity_score,
        llm_growth_rate=last.llm_growth_rate,
        bottleneck_hint=last.bottleneck_hint,
        max_mutants=last.max_mutants,
        max_in_flight=last.max_in_flight,
        # Time so far + smoothed estimate of remaining
        predicted_duration_s=last.wall_time_at_calibration + additional_duration,
        predicted_llm_calls=last.predicted_llm_calls,
        predicted_total_tokens=last.predicted_total_tokens,
        ci_low_s=(last.wall_time_at_calibration + additional_duration) * 0.8,
        ci_high_s=(last.wall_time_at_calibration + additional_duration) * 1.2,
    )
    result.calibration_completed = True
    result.calibration_mutants_done = current_mutants_done
    result.wall_time_at_calibration = current_wall_s
    return result
