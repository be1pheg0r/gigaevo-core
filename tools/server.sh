#!/usr/bin/env bash
# Run things on a remote server without hand-typing ssh.
#
# Why this exists: `ssh host 'pkill -f cost_lab_web/app.py; ...'` looks fine and
# is a trap. pkill matches against full command lines, and the ssh session's own
# remote command line CONTAINS the pattern — so pkill kills its own shell, the
# rest of the chain never runs, and the service you meant to restart is simply
# gone. That took Cost Lab down on 2026-08-01.
#
# The `[c]ost_lab` bracket form is necessary but NOT sufficient: bracketing the
# pattern stops it matching the literal `[c]ost_lab`, while the launch half of
# the same one-liner still spells the path plainly, and pkill matches THAT. So
# the kill and the launch are two separate ssh calls here — the kill command's
# line contains nothing but the bracketed form. Belt and braces, because the
# failure mode is "the service is now down and nothing says why".
#
#   tools/server.sh status                 # what is up, from outside
#   tools/server.sh run 'df -h; free -g'   # arbitrary command
#   tools/server.sh restart costlab        # restart a service, verify it answers
#   tools/server.sh restart taskbuilder
#   tools/server.sh logs costlab [n]       # tail a service log
#   tools/server.sh ps                     # what of ours is running
#
# Auth: ~/.ssh/gigaevo_server (dedicated key). Override with GIGAEVO_SSH_KEY.
set -uo pipefail

: "${GIGAEVO_HOST:?Set GIGAEVO_HOST to user@host}"
: "${GIGAEVO_BASE_URL:?Set GIGAEVO_BASE_URL to the public service URL}"
HOST="$GIGAEVO_HOST"
KEY="${GIGAEVO_SSH_KEY:-$HOME/.ssh/gigaevo_server}"
BASE="$GIGAEVO_BASE_URL"
SECONDARY_LLM_BASE_URL="${GIGAEVO_SECONDARY_LLM_BASE_URL:-}"
REPO_DIR="~/gigaevo-core"

# service -> "script path|log file|url path"
svc_script() { case "$1" in
  costlab)     echo "tools/cost_lab_web/app.py" ;;
  taskbuilder) echo "tools/task_builder_web/app.py" ;;
  *) return 1 ;; esac; }
# Log names are what the services already write to — not derived from the
# service name, which silently pointed `logs` and `restart` at a file that
# never existed.
svc_log() { case "$1" in
  costlab)     echo "~/cost_lab_web.log" ;;
  taskbuilder) echo "~/task_builder_web.log" ;;
  *) return 1 ;; esac; }
svc_url() { case "$1" in
  costlab)     echo "$BASE/costlab/" ;;
  taskbuilder) echo "$BASE/taskbuilder/" ;;
  *) return 1 ;; esac; }

ssh_run() {
  ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=10 "$HOST" "$@"
}

http_code() { curl -s -o /dev/null --max-time 10 -w "%{http_code}" "$1" 2>/dev/null || echo "---"; }

cmd_status() {
  printf '%-14s %s\n' "nginx"       "$(http_code "$BASE/")"
  for s in costlab taskbuilder; do
    printf '%-14s %s\n' "$s" "$(http_code "$(svc_url "$s")")"
  done
  printf '%-14s %s\n' "grafana"     "$(http_code "$BASE/grafana/")"
  printf '%-14s %s\n' "vllm-35b"    "$(http_code "$BASE/v1/models")"
  if [ -n "$SECONDARY_LLM_BASE_URL" ]; then
    printf '%-14s %s\n' "vllm-secondary" "$(http_code "$SECONDARY_LLM_BASE_URL/v1/models")"
  fi
  echo "--- remote ---"
  ssh_run "cd $REPO_DIR && git log -1 --format='HEAD %h %s' && ps -o pid,etime,cmd -u \$USER | grep -E '[a]pp\.py' || echo 'no app.py running'"
}

cmd_ps() {
  ssh_run "ps -o pid,etime,rss,cmd -u \$USER | grep -E '[a]pp\.py|[r]un\.py|[r]un_ablation' || echo 'nothing of ours running'"
}

cmd_run() { ssh_run "$*"; }

cmd_logs() {
  local svc="${1:?service}" n="${2:-60}"
  ssh_run "tail -n $n $(svc_log "$svc")"
}

cmd_restart() {
  local svc="${1:?service: costlab | taskbuilder}"
  local script log url
  script="$(svc_script "$svc")" || { echo "unknown service: $svc" >&2; return 2; }
  log="$(svc_log "$svc")"; url="$(svc_url "$svc")"

  # Two calls, never one — see the header. The kill line must not contain the
  # unbracketed path anywhere, and the launch line obviously does.
  local pat="[${script:0:1}]${script:1}"
  ssh_run "pkill -f '$pat' ; exit 0"       # exit 0: nothing matched is a cold start
  sleep 1
  ssh_run "cd $REPO_DIR && setsid nohup python3 $script > $log 2>&1 < /dev/null & echo launched pid \$!"

  echo -n "waiting for $url "
  for _ in $(seq 1 20); do
    code="$(http_code "$url")"
    if [ "$code" = "200" ]; then echo "-> 200 OK"; return 0; fi
    echo -n "."
    sleep 2
  done
  echo " -> still $code"
  echo "--- last 30 lines of $log ---"
  cmd_logs "$svc" 30
  return 1
}

case "${1:-status}" in
  status)  cmd_status ;;
  ps)      cmd_ps ;;
  run)     shift; cmd_run "$@" ;;
  logs)    shift; cmd_logs "$@" ;;
  restart) shift; cmd_restart "$@" ;;
  *) sed -n '2,22p' "$0"; exit 2 ;;
esac
