#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APP_ROOT="${APP_ROOT:-$APP_DIR}"
export SAM3D_REPO="${SAM3D_REPO:-/workspace/sam-3d-objects}"
export SAM3D_TAG="${SAM3D_TAG:-hf}"

uvicorn app.main:app --host 0.0.0.0 --port 8000
