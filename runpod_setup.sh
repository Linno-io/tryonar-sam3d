#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM3D_REPO="${SAM3D_REPO:-/workspace/sam-3d-objects}"
TAG="${SAM3D_TAG:-hf}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for downloading gated weights."
  echo "Example: export HF_TOKEN=hf_your_token_here"
  exit 1
fi

echo "--- System dependencies ---"
apt-get update && apt-get install -y \
  git \
  ninja-build \
  libgl1-mesa-glx \
  libglu1-mesa \
  && rm -rf /var/lib/apt/lists/*

echo "--- Clone SAM3D repo ---"
if [[ -d "$SAM3D_REPO" ]]; then
  echo "Using existing repo at $SAM3D_REPO"
  git -C "$SAM3D_REPO" pull
else
  git clone https://github.com/facebookresearch/sam-3d-objects.git "$SAM3D_REPO"
fi

echo "--- Python deps ---"
python -m pip install -U pip
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

python -m pip install -e "$SAM3D_REPO[dev,p3d,inference]"

if [[ -f "$SAM3D_REPO/patching/hydra" ]]; then
  chmod +x "$SAM3D_REPO/patching/hydra"
  "$SAM3D_REPO/patching/hydra"
fi

python -m pip install -r "$APP_DIR/requirements.txt"

echo "--- Download checkpoints ---"
python -m pip install 'huggingface-hub[cli]<1.0'

if [[ ! -d "$SAM3D_REPO/checkpoints/$TAG" ]]; then
  mkdir -p "$SAM3D_REPO/checkpoints"
  huggingface-cli login --token "$HF_TOKEN"
  huggingface-cli download \
    --repo-type model \
    --local-dir "$SAM3D_REPO/checkpoints/${TAG}-download" \
    --max-workers 1 \
    facebook/sam-3d-objects

  mv "$SAM3D_REPO/checkpoints/${TAG}-download/checkpoints" "$SAM3D_REPO/checkpoints/$TAG"
  rm -rf "$SAM3D_REPO/checkpoints/${TAG}-download"
else
  echo "Checkpoints already present at $SAM3D_REPO/checkpoints/$TAG"
fi

echo "--- Done ---"
echo "Export env vars before starting the API:"
echo "  export APP_ROOT=\"$APP_DIR\""
echo "  export SAM3D_REPO=\"$SAM3D_REPO\""
echo "  export SAM3D_TAG=\"$TAG\""
