#!/usr/bin/env bash
# One-step setup and launch (Linux / macOS / Git Bash):   bash run.sh
# Creates a virtual environment, installs dependencies, downloads the TinyCD weights from the
# authors' repository (checksum verified), runs the sample pairs, then opens the live demo.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
[ -d .venv ] || "$PYTHON" -m venv .venv
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=.venv/Scripts/python.exe; fi
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt

"$PY" -m tinycd                 # downloads weights on first run, writes outputs/samples.png
echo
echo "Results image: outputs/samples.png"
if [ "${1:-}" != "--no-serve" ]; then
  echo "Opening the live demo (Ctrl+C to stop)..."
  "$PY" scripts/serve.py
fi
