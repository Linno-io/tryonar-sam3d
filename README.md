# SAM3D All-In-One Production Backend

A high-performance, containerized API service wrapping **Meta's SAM3D-Objects**. This system provides a single-entry pipeline for generating high-fidelity 3D assets (Geometry + Texture) from a single image using a robust REST API.

## 🚀 Features

- **Single-Image to 3D**: Converts 2D images into textured 3D models (.glb).
- **Automated Preprocessing**: Integrated background removal using `rembg`.
- **GPU Accelerated**: Optimized for CUDA 12.1 environments (specifically RTX 4090/RunPod).
- **Production Ready**: Fully containerized with Docker, featuring automated model caching and file cleanup.
- **Simple API**: Easy-to-use FastAPI endpoints for integration.

## 🛠 Prerequisites

- **NVIDIA GPU**: Minimum 24GB VRAM recommended (e.g., RTX 3090/4090).
- **Docker**: With NVIDIA Container Toolkit installed.
- **Hugging Face Token**: Access to `facebook/sam-3d-objects` (gated model).

## 🐳 Docker Deployment

The system is designed to run in a container. No local Python environment setup is required for production.

### 1. Build the Image
You must provide your Hugging Face token during the build process to download the model weights.

```bash
docker build -t sam3d-backend \
  --build-arg HF_TOKEN=hf_your_token_here \
  .
```

### 2. Run the Container
Map port 8000 and enable GPU access.

```bash
docker run --gpus all -p 8000:8000 sam3d-backend
```

## 🔌 API Usage

### `POST /api/inference`

Generate a 3D model from an image.

**Request:** `multipart/form-data`
- `file`: The input image (PNG/JPG).

**Response:** JSON
```json
{
  "status": "success",
  "download_url": "/outputs/timestamp_image.glb",
  "inference_time": 45.23
}
```

### `GET /health`

Check system readiness. Returns 200 OK only when the model is fully loaded in VRAM.

## 📦 Output

Generated 3D models can be downloaded via the URL returned in the inference response. The server automatically cleans up files older than 1 hour.

## 💻 Local Development

For testing or development without Docker (requires complex local environment setup):

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Tests:**
    ```bash
    # Set env vars for local test paths
    export OUTPUT_DIR=./outputs
    export UPLOAD_DIR=./uploads
    pytest tests/
    ```

## 📄 License
MIT
