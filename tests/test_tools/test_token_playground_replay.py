import json

from tools.cost_ablation.replay_from_log import hook_from_log


def test_infers_failed_mutation_attempts_from_llm_calls(tmp_path):
    lines = []
    for index, tokens in enumerate((100, 140), start=1):
        payload = {
            "event": "LLM_CALL",
            "run_label": None,
            "stage": "MutationSuggestionAgent",
            "program_id": None,
            "endpoint": "",
            "model": "test-model",
            "attempt": 1,
            "ok": True,
            "latency_ms": 1000,
            "tokens_in": tokens - 20,
            "tokens_out": 20,
            "error_type": None,
        }
        lines.append(
            f"2026-08-02 12:00:0{index}.000 | INFO | [LLM_CALL] "
            + json.dumps(payload)
        )
    log = tmp_path / "probe.log"
    log.write_text("\n".join(lines), encoding="utf-8")

    hook = hook_from_log(log, infer_attempts_from_llm=True)

    assert hook is not None
    assert hook._attempts == 2
    assert hook._tokens_by_stage == {"MutationSuggestionAgent": [100.0, 140.0]}
