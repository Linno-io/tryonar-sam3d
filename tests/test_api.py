import sys
from unittest.mock import MagicMock

# Mock heavy dependencies BEFORE importing app text
sys.modules["torch"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["rembg"] = MagicMock()
sys.modules["trimesh"] = MagicMock()
sys.modules["inference"] = MagicMock()

# Mock PIL
mock_pil = MagicMock()
sys.modules["PIL"] = mock_pil

import os
# Set env vars for testing before import
os.environ["OUTPUT_DIR"] = "./outputs"
os.environ["UPLOAD_DIR"] = "./uploads"

import pytest
from fastapi.testclient import TestClient
from app.main import app
# Do not import engine directly as it is re-bound

@pytest.fixture
def mock_inference_engine(monkeypatch):
    """Patch the InferenceEngine class to return a mock instance."""
    mock_instance = MagicMock()
    mock_instance.process.return_value = "/app/outputs/test_job.glb"
    
    # We need to splice the mock instance into what app.main uses.
    # app.main imports InferenceEngine from app.engine
    
    # Create a mock Class that returns our mock_instance
    mock_class = MagicMock(return_value=mock_instance)
    
    # Patch where app.main imports it specifically if possible, 
    # OR patch app.engine.InferenceEngine (the definition)
    import app.engine
    monkeypatch.setattr(app.engine, "InferenceEngine", mock_class)
    
    return mock_instance

@pytest.fixture
def client(mock_inference_engine):
    """Create client after mocking is set up."""
    # TestClient will trigger lifespan startup, which calls InferenceEngine()
    # Our mock_inference_engine fixture ensures InferenceEngine() returns a mock.
    with TestClient(app) as c:
        yield c

def test_health_check_ready(client, mock_inference_engine):
    """Test health endpoint when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model": "loaded"}

def test_inference_endpoint(client, mock_inference_engine):
    """Test the inference flow with a mock image."""
    files = {"file": ("test.png", b"fakeimagebytes", "image/png")}
    
    response = client.post("/api/inference", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # The return_value of process should influence this
    # Note: process returns absolute path /app/outputs/..., API converts to /outputs/...
    assert "download_url" in data
    assert "inference_time" in data
    # Check that process was called
    mock_inference_engine.process.assert_called()

def test_invalid_file_type(client):
    """Test uploading a text file instead of an image."""
    files = {"file": ("test.txt", b"textcontent", "text/plain")}
    response = client.post("/api/inference", files=files)
    assert response.status_code == 400
