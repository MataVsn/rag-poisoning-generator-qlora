#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Blackwell support requires a suitable CUDA wheel, not just a recent host driver.
.venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python scripts/preflight_gpu.py
