#!/bin/sh
# Run from a server terminal. tmux keeps the job alive after SSH disconnects.
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
if [ -f configs/runtime.env ]; then . ./configs/runtime.env; fi
SESSION=rag-endterm-training
ACTION=${1:-help}

case "$ACTION" in
  smoke|train)
    command -v tmux >/dev/null 2>&1 || { echo 'Install tmux first (Ubuntu: apt-get install -y tmux).'; exit 1; }
    test -x .venv/bin/python || { echo 'Run sh scripts/bootstrap_gpu.sh first.'; exit 1; }
    # tmux rejects a duplicate session name, including concurrent starts.
    tmux new-session -d -s "$SESSION" -c "$TASK_ROOT" "sh scripts/remote_train.sh _worker $ACTION"
    echo "Started $ACTION. Status: sh scripts/remote_train.sh status"
    ;;
  _worker)
    MODE=${2:-}
    case "$MODE" in smoke|train) ;; *) exit 2 ;; esac
    mkdir -p outputs/remote
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    LOG="outputs/remote/$STAMP-$MODE.log"
    STATUS="outputs/remote/$STAMP-$MODE.status"
    printf '%s\n' "$LOG" > outputs/remote/latest-log.txt
    printf 'running\n' > "$STATUS"
    # Trap captures failures in validation, preflight, tokenization or training.
    trap 'RESULT=$?; printf "exit_code=%s\n" "$RESULT" > "$STATUS"; exit "$RESULT"' 0
    exec >> "$LOG" 2>&1
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONUNBUFFERED=1
    date -u
    .venv/bin/python scripts/validate_datasets.py
    .venv/bin/python scripts/preflight_gpu.py
    .venv/bin/python scripts/check_tokens.py
    if [ "$MODE" = smoke ]; then
      .venv/bin/python scripts/train_qlora.py --smoke-steps 20
    else
      .venv/bin/python scripts/train_qlora.py
    fi
    ;;
  status)
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
      echo 'Training session is running.'
    else
      echo 'No running training session. Read the log/status to distinguish completion from failure.'
    fi
    if [ -f outputs/remote/latest-log.txt ]; then
      LOG=$(cat outputs/remote/latest-log.txt)
      echo "$LOG"
      cat "${LOG%.log}.status"
      tail -n 25 "$LOG"
    fi
    ;;
  logs)
    test -f outputs/remote/latest-log.txt || { echo 'No log yet.'; exit 1; }
    tail -n 50 -f "$(cat outputs/remote/latest-log.txt)"
    ;;
  *)
    echo 'Usage: sh scripts/remote_train.sh smoke|train|status|logs'
    ;;
esac
