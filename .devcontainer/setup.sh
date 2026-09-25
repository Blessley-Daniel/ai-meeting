#!/usr/bin/env bash
# Install the system packages and Python dependencies the pipeline needs.
#
# Runs once when the Codespace is created. ffmpeg is a *system* package, so it
# cannot come from requirements.txt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Installing ffmpeg"
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg

echo "==> Creating virtual environment"
python -m venv "${REPO_ROOT}/backend/.venv"

echo "==> Installing Python dependencies"
"${REPO_ROOT}/backend/.venv/bin/pip" install --quiet --upgrade pip
"${REPO_ROOT}/backend/.venv/bin/pip" install --quiet -r "${REPO_ROOT}/backend/requirements.txt"

echo
echo "Setup complete. Start the app with:"
echo "    cd ${REPO_ROOT}/backend && .venv/bin/uvicorn app.main:app --reload --port 8000"
