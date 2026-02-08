# Base: CUDA 12.1 + PyTorch 2.2.1 (Matches official requirements)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Build Argument for HF Gated Model Access
ARG HF_TOKEN

# 1. System Dependencies
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# 2. NVIDIA & PyTorch3D Index URLs
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# 3. Clone & Install Meta SAM3D
# Cloning into /app (current dir)
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git .

# Install dependencies
RUN pip install -e '.[dev]' && \
    pip install -e '.[p3d]' && \
    pip install -e '.[inference]'

# 4. Apply Required Config Patches
# Assumes the repo contains patching/hydra script as clarified by user
RUN chmod +x ./patching/hydra && ./patching/hydra

# 5. Weight Acquisition (Direct-to-Image)
RUN pip install 'huggingface-hub[cli]<1.0' && \
    huggingface-cli login --token $HF_TOKEN && \
    mkdir -p checkpoints/hf && \
    huggingface-cli download \
    --repo-type model \
    --local-dir checkpoints/hf-download \
    --max-workers 4 \
    facebook/sam-3d-objects && \
    mv checkpoints/hf-download/checkpoints checkpoints/hf && \
    rm -rf checkpoints/hf-download

# 6. API Service Stack
RUN pip install fastapi uvicorn python-multipart rembg trimesh

# 7. Application Code
# Copy our app package into /app/app
COPY ./app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
