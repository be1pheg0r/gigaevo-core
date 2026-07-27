#!/usr/bin/env bash
# Overnight pipeline: run all 15 tasks from the "разных доменов" experiment
# (AIRI Notion -> №29 -> Отчёты -> "Подбор 15 задач для эксперимента —
# 2026-07-27") at max_mutants=50 each.
#
# Deliberately sequential, not parallel: all single-population tasks share
# one Redis instance and the two AIRI summer-school GPU servers (16+8
# concurrent slots total across both) -- running 15 tasks at once would
# oversubscribe both and make per-task cost/timing numbers incomparable.
# One task at a time keeps the point of the experiment (feeding the #29
# cost-model work) intact: clean, isolated LLM_CALL/STAGE_EXEC logs per task.
#
# Prerequisites verified 2026-07-27 (see /gigaevo-core git log on branch
# feat/full-cost-logging-and-summer-school-llm):
#   - Token/time logging for BOTH the mutator and the chain/prompt validator
#     LLMClient confirmed end-to-end against the real summer-school servers
#     (see gigaevo.monitoring.subprocess_emit + the two LLMClient rewrites).
#   - Redis reachable, .venv-wsl has all extras this task list needs
#     ([chains,plotting,optimization,dev,test]).
#
# Usage:
#   bash tools/pipeline_15tasks/run_all.sh
#
# Env overrides:
#   MAX_MUTANTS          default 50 (applies to tasks 1-14; task 15 uses
#                         ADVERSARIAL_MAX_GEN instead, see below)
#   PER_TASK_TIMEOUT      default 7200s (2h) hard cap per task 1-14
#   ADVERSARIAL_MAX_GEN   default 10 (adversarial co-evolution is
#                         generation-paced, not mutant-paced -- see the
#                         dedicated block at the bottom)
#   VENV_PYTHON / LITELLM_BIN   override interpreter paths if not using
#                         .venv-wsl

set -uo pipefail  # NOT -e: one task's failure must not kill the whole night

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv-wsl/bin/python}"
LITELLM_BIN="${LITELLM_BIN:-$REPO_ROOT/.venv-wsl/bin/litellm}"
MAX_MUTANTS="${MAX_MUTANTS:-50}"
PER_TASK_TIMEOUT="${PER_TASK_TIMEOUT:-7200}"
ADVERSARIAL_MAX_GEN="${ADVERSARIAL_MAX_GEN:-10}"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "[FATAL] venv python not found at $VENV_PYTHON -- set VENV_PYTHON=..." >&2
    exit 1
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/15tasks_$RUN_TAG"
mkdir -p "$EXPERIMENT_DIR"
SUMMARY_FILE="$EXPERIMENT_DIR/summary.tsv"
printf 'task\tredis_db\texit_code\tduration_s\toutput_dir\n' > "$SUMMARY_FILE"

log()  { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------
# 1. Dataset prep -- idempotent, skipped if files already exist.
# ---------------------------------------------------------------------
log "checking task datasets..."

if [ ! -f problems/chains/ifbench/dataset/IFBench_train.jsonl ]; then
    log "generating IFBench dataset (chains/nlp/ifbench reuses this path)..."
    "$VENV_PYTHON" -m problems.chains.ifbench.dataset.load_dataset \
        || { log "[FATAL] IFBench dataset prep failed"; exit 1; }
fi

if [ ! -d problems/chains/hover/dataset/bm25s_index ]; then
    log "downloading HoVer wiki corpus + building BM25 index (~1.5GB, can take a while -- one-time cost)..."
    "$VENV_PYTHON" -m problems.chains.hover.dataset.download_corpus \
        || { log "[FATAL] HoVer corpus prep failed"; exit 1; }
fi

if [ ! -f problems/prompts/gsm8k/dataset/GSM8K_train.csv ]; then
    log "generating prompts/gsm8k dataset..."
    "$VENV_PYTHON" -m problems.prompts.gsm8k.dataset.load_dataset \
        || { log "[FATAL] prompts/gsm8k dataset prep failed"; exit 1; }
fi

# ---------------------------------------------------------------------
# 2. Redis.
# ---------------------------------------------------------------------
if ! redis-cli ping >/dev/null 2>&1; then
    log "redis not reachable, starting redis-server..."
    sudo service redis-server start
    sleep 1
fi
redis-cli ping >/dev/null 2>&1 || { log "[FATAL] redis still unreachable"; exit 1; }

# ---------------------------------------------------------------------
# 3. LiteLLM proxy on :8000 -- forwards to the summer-school servers.
#    Needed by tasks 11-15, whose configs hardcode/default to
#    http://localhost:8000/v1 (see litellm_config.yaml for why).
# ---------------------------------------------------------------------
log "starting litellm proxy on :8000..."
"$LITELLM_BIN" --config tools/pipeline_15tasks/litellm_config.yaml --port 8000 \
    > "$EXPERIMENT_DIR/litellm_proxy.log" 2>&1 &
LITELLM_PID=$!
cleanup() {
    log "stopping litellm proxy (pid $LITELLM_PID)..."
    kill "$LITELLM_PID" 2>/dev/null
}
trap cleanup EXIT

log "waiting for proxy health..."
proxy_up=0
for _ in $(seq 1 30); do
    if curl -sf http://localhost:8000/health/readiness >/dev/null 2>&1; then
        proxy_up=1
        break
    fi
    sleep 2
done
if [ "$proxy_up" -ne 1 ]; then
    log "[FATAL] litellm proxy never became healthy -- see $EXPERIMENT_DIR/litellm_proxy.log"
    exit 1
fi

log "smoke-testing proxy with a real completion..."
"$VENV_PYTHON" - <<'PYEOF'
import sys
from openai import OpenAI
try:
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-gigaevo")
    r = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": "Reply with exactly one word: hi"}],
        max_tokens=16,
    )
    print("proxy smoke test OK:", r.choices[0].message.content)
except Exception as exc:
    print(f"proxy smoke test FAILED: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then
    log "[FATAL] litellm proxy smoke test failed -- aborting before running any tasks"
    exit 1
fi

# ---------------------------------------------------------------------
# 4. Tasks 1-14: single-population, uniform max_mutants=50.
#    llm=summer_school_servers drives the mutator directly (bypasses the
#    proxy); tasks 1-10 have no LLM validator at all, tasks 11-14 also
#    make chain-eval/prompt-eval calls through the litellm proxy above.
# ---------------------------------------------------------------------
declare -a TASKS=(
    "heilbron"
    "hexagon_pack"
    "spherical_codes_baseline"
    "alphaevolve/matrix_multiplication/2_4_5"
    "alphaevolve/first_autocorr_ineq"
    "algotune/algotune_lqr"
    "algotune/algotune_markowitz"
    "algotune/algotune_pde_heat1d"
    "tabular_regression"
    "toy_kadane"
    "chains/nlp/gsm8k/static"
    "chains/nlp/hover/static"
    "chains/nlp/ifbench/static"
    "prompts/gsm8k"
)

db=1
for task in "${TASKS[@]}"; do
    safe_name="$(echo "$task" | tr '/' '_')"
    log "==================== [$task] (redis.db=$db) ===================="
    t0=$(date +%s)
    task_log="$EXPERIMENT_DIR/${safe_name}.log"

    timeout "$PER_TASK_TIMEOUT" "$VENV_PYTHON" run.py \
        problem.name="$task" \
        max_mutants="$MAX_MUTANTS" \
        llm=summer_school_servers \
        redis.db=$db \
        > "$task_log" 2>&1
    exit_code=$?

    t1=$(date +%s)
    duration=$((t1 - t0))
    out_dir="$(grep -oP 'Output dir: \K[^ ]+' "$task_log" | head -1)"
    printf '%s\t%s\t%s\t%s\t%s\n' "$task" "$db" "$exit_code" "$duration" "$out_dir" >> "$SUMMARY_FILE"

    if [ "$exit_code" -eq 0 ]; then
        log "[$task] done in ${duration}s -> ${out_dir:-<no output dir found>}"
    else
        log "[$task] FAILED (exit $exit_code) after ${duration}s -- see $task_log"
    fi

    db=$((db + 1))
done

# ---------------------------------------------------------------------
# 5. Task 15: adversarial/code -- two populations, generation-paced, not
#    mutant-paced. Structurally different from tasks 1-14: adapted from
#    problems/adversarial/launch_optimizer.sh (which pairs
#    adversarial/optimizer/pop_a + pop_b) for the code/pop_a
#    (expression evaluator) + code/pop_b (adversarial test generator)
#    pair instead. UNVERIFIED end-to-end in this environment as of
#    2026-07-27 -- everything above this point was smoke-tested for real
#    (see git log), this block was not. If it fails, tasks 1-14 above are
#    unaffected (this is the last stage).
# ---------------------------------------------------------------------
log "==================== [adversarial/code] (pop_a db=15, pop_b db=16) ===================="
t0=$(date +%s)
ADV_LOG_DIR="$EXPERIMENT_DIR/adversarial_code"
mkdir -p "$ADV_LOG_DIR"

POP_A_DB=15
POP_A_PREFIX="adversarial/code/pop_a"
POP_B_DB=16
POP_B_PREFIX="adversarial/code/pop_b"
ADV_LLM_URL="http://localhost:8000/v1"
ADV_MODEL="qwen3.5-9b"  # ignored by the proxy's wildcard routing -- cosmetic only

OPENAI_API_KEY=sk-gigaevo \
OPPONENT_REDIS_HOST=localhost \
OPPONENT_REDIS_PORT=6379 \
OPPONENT_REDIS_DB=$POP_B_DB \
OPPONENT_PREFIX=$POP_B_PREFIX \
"$VENV_PYTHON" run.py \
    problem.name=adversarial/code/pop_a \
    pipeline=adversarial \
    redis.db=$POP_A_DB \
    opponent_redis_db=$POP_B_DB \
    opponent_redis_prefix=$POP_B_PREFIX \
    max_generations=$ADVERSARIAL_MAX_GEN \
    llm_base_url=$ADV_LLM_URL \
    model_name=$ADV_MODEL \
    > "$ADV_LOG_DIR/pop_a.log" 2>&1 &
PID_A=$!

sleep 2

OPENAI_API_KEY=sk-gigaevo \
OPPONENT_REDIS_HOST=localhost \
OPPONENT_REDIS_PORT=6379 \
OPPONENT_REDIS_DB=$POP_A_DB \
OPPONENT_PREFIX=$POP_A_PREFIX \
"$VENV_PYTHON" run.py \
    problem.name=adversarial/code/pop_b \
    pipeline=adversarial \
    redis.db=$POP_B_DB \
    opponent_redis_db=$POP_A_DB \
    opponent_redis_prefix=$POP_A_PREFIX \
    max_generations=$ADVERSARIAL_MAX_GEN \
    llm_base_url=$ADV_LLM_URL \
    model_name=$ADV_MODEL \
    > "$ADV_LOG_DIR/pop_b.log" 2>&1 &
PID_B=$!

log "adversarial pop_a (pid $PID_A) + pop_b (pid $PID_B) launched, waiting..."
wait "$PID_A"; exit_a=$?
wait "$PID_B"; exit_b=$?

t1=$(date +%s)
duration=$((t1 - t0))
exit_code=$(( exit_a != 0 || exit_b != 0 ))
out_dir_a="$(grep -oP 'Output dir: \K[^ ]+' "$ADV_LOG_DIR/pop_a.log" | head -1)"
out_dir_b="$(grep -oP 'Output dir: \K[^ ]+' "$ADV_LOG_DIR/pop_b.log" | head -1)"
printf 'adversarial/code/pop_a\t%s\t%s\t%s\t%s\n' "$POP_A_DB" "$exit_a" "$duration" "$out_dir_a" >> "$SUMMARY_FILE"
printf 'adversarial/code/pop_b\t%s\t%s\t%s\t%s\n' "$POP_B_DB" "$exit_b" "$duration" "$out_dir_b" >> "$SUMMARY_FILE"

if [ "$exit_code" -eq 0 ]; then
    log "[adversarial/code] both populations done in ${duration}s"
else
    log "[adversarial/code] FAILED (pop_a=$exit_a pop_b=$exit_b) after ${duration}s -- see $ADV_LOG_DIR/"
fi

# ---------------------------------------------------------------------
# 6. Summary.
# ---------------------------------------------------------------------
log "==================== SUMMARY ===================="
column -t -s$'\t' "$SUMMARY_FILE"
log "full experiment artifacts: $EXPERIMENT_DIR"
