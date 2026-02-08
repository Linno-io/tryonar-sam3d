PRD: SAM3D All-In-One Production Backend
========================================

1\. System Overview
-------------------

A containerized API service wrapping **Meta’s SAM3D-Objects**.The system provides a **single-entry pipeline** for high-fidelity 3D asset generation (Geometry + Texture) through a **FastAPI** interface.

2\. Core Functional Requirements
--------------------------------

### 2.1 API Layer

*   Expose a POST /api/inference endpoint
    
*   Accept an image as input
    
*   Return a downloadable .glb file link
    

### 2.2 Inference Wrapper

*   Initialize the Inference class from the SAM3D notebook module
    
*   Manage GPU context and lifecycle
    
*   Ensure models are loaded once and reused across requests
    

### 2.3 Preprocessing

*   Automatically generate an object mask using rembg
    
*   Isolate the foreground object before inference
    

### 2.4 Post-processing

*   Convert the internal 3D representation into a standard binary glTF format (.glb)
    
*   Ensure geometry and texture data are preserved
    

### 2.5 Cleanup

*   Periodically remove temporary files from:
    
    *   /tmp
        
    *   /app/outputs
        
*   Prevent disk exhaustion in long-running deployments
    

3\. Deployment Architecture
---------------------------

### 3.1 Container Strategy

*   Fully containerized deployment using Docker
    
*   Model weights baked into the image for zero-setup runtime
    
*   Optimized for **RunPod GPU instances**
    

### 3.2 Dockerfile

```dockerfile
# Base: CUDA 12.1 + PyTorch 2.2.1 (Matches official requirements)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel

WORKDIR /app

# Build Argument for HF Gated Model Access
ARG HF_TOKEN

# 1. System Dependencies
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libglu1-mesa

# 2. NVIDIA & PyTorch3D Index URLs
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# 3. Clone & Install Meta SAM3D
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git .
RUN pip install -e '.[dev]' && \
    pip install -e '.[p3d]' && \
    pip install -e '.[inference]'

# 4. Apply Required Config Patches
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
RUN pip install fastapi uvicorn python-multipart rembg

# 7. Application Code
COPY ./app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


```


4\. Implementation Logic for AI Agent
-------------------------------------

### 4.1 app/engine.py

#### Responsibilities

*   **Initialization**
    
    *   Load pipeline.yaml from /app/checkpoints/hf/
        
    *   Initialize the SAM3D model in GPU memory
        
*   **Preprocessing**
    
    *   Generate a foreground mask using rembg.remove()
        
    *   Apply the mask to the input image
        
*   **Inference**
    
    *   Invoke generate\_single\_object() on the SAM3D model
        
*   **Export**
    
    *   Convert the inference output dictionary into a .glb file
        
    *   Store the output in /app/outputs
        

### 4.2 app/main.py

#### Responsibilities

*   **Inference Endpoint**
    
    *   Implement POST /api/inference
        
    *   Define async def inference(file: UploadFile)
        
*   **Temporary Storage**
    
    *   Persist uploaded image bytes to a temporary file
        
*   **Health Check**
    
    *   Implement GET /health
        
    *   Return 200 OK only after:
        
        *   Models are loaded
            
        *   VRAM allocation is successful
            

5\. Non-Functional Requirements (NFRs)
--------------------------------------

### 5.1 Performance

*   End-to-end inference time must be **≤ 120 seconds** for a 1024px image
    
*   Model initialization should occur **once per container lifecycle**
    

### 5.2 Scalability

*   Single-GPU, single-request execution model
    
*   Designed for horizontal scaling via multiple GPU pods
    
*   No shared state across containers
    

### 5.3 Reliability

*   API must fail gracefully on:
    
    *   Out-of-memory errors
        
    *   Invalid image inputs
        
*   Temporary files must be cleaned up even after failed requests
    

### 5.4 Observability

*   Log:
    
    *   Request start and completion time
        
    *   GPU memory usage at inference start
        
*   Health endpoint must reflect **true readiness**, not just process liveness
    

### 5.5 Security

*   Hugging Face token passed only as a build-time argument
    
*   No credentials stored in runtime environment variables
    
*   Uploaded files must not be persisted beyond processing lifecycle
    

### 5.6 Portability

*   Docker image must run without modification on:
    
    *   RunPod
        
    *   Any CUDA 12.1 compatible GPU environment
        

6\. Verification & Validation
-----------------------------

### 6.1 VRAM Validation

*   Confirm Stage-1 and Stage-2 models fit within **24GB VRAM**
    
*   Target hardware: **RTX 4090**
    

### 6.2 Output Integrity

*   Use trimesh to validate:
    
    *   Mesh exists and is non-empty
        
    *   Texture count is greater than zero
        

### 6.3 Latency Validation

*   Measure total inference time
    
*   Ensure SLA compliance under normal load