#!/usr/bin/env bash
#
# Set up (or repair) the development environment for this project.
#
#   bash backend/scripts/setup.sh
#
# Safe to re-run. It exists because the two system binaries the pipeline needs
# (ffmpeg, espeak-ng) are installed with apt and therefore do not survive a
# container rebuild, while the Python virtualenv does. Running this script
# after a reset puts everything back.
#
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"
VENV="$BACKEND_DIR/.venv"

log()  { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- system deps
log "System packages (apt)"

# sudo is not always present (e.g. already root in a container).
if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi

APT_PACKAGES="ffmpeg espeak-ng"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v espeak-ng >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO apt-get update -qq
        # shellcheck disable=SC2086
        $SUDO apt-get install -y -qq $APT_PACKAGES
    else
        warn "apt-get not found. Install ffmpeg and espeak-ng with your platform's package manager."
    fi
else
    echo "ffmpeg and espeak-ng already present."
fi

# ------------------------------------------------------------------- python
log "Python virtualenv"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "created $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

log "Base dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r "$BACKEND_DIR/requirements.txt"

# ---------------------------------------------------------------- AI stack
log "AI stack (CPU-only torch)"
# torch must come from the CPU wheel index first. Installing it from PyPI
# would pull ~2 GB of CUDA libraries that are useless without a GPU.
pip install --quiet --extra-index-url https://download.pytorch.org/whl/cpu \
    torch --upgrade
pip install --quiet -r "$REPO_ROOT/requirements-ai.txt"

# ------------------------------------------------------------------ verify
log "Verifying environment"
python "$BACKEND_DIR/scripts/verify_environment.py"

log "Done"
echo "Activate the environment with:  source backend/.venv/bin/activate"
echo "Run the tests with:             cd backend && python -m pytest -q"
echo "Start the server with:          cd backend && uvicorn app.main:app --reload"
