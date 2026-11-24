#!/bin/sh
# Execute inside tmux; one bounded campaign with an explicit final exit status.
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
. ./configs/runtime.env
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
mkdir -p outputs/remote
STATUS=outputs/remote/campaign-14b.exit
printf 'running\n' > "$STATUS"
trap 'RESULT=$?; printf "exit_code=%s\n" "$RESULT" > "$STATUS"; exit "$RESULT"' 0
exec >> outputs/remote/campaign-14b.log 2>&1
.venv/bin/python scripts/run_campaign.py --hours 3.75 --generation-limit 64
