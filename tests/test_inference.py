import pytest
from unittest.mock import MagicMock, patch
import os
from backend.inference import model_loader, router

def test_load_models_existing_files():
    """Test load_models when model files supposedly exist."""
    # Patch os.path.exists to return True so we skip download
    with patch("os.path.exists", return_value=True):
        yolo, insight = model_loader.load_models()

        # Verify YOLO was "loaded" (instantiated from Mock)
        assert yolo is not None
        # Verify InsightFace was "loaded"
        assert insight is not None

def test_ensure_models_loaded_raises():
    """Test ensure_models_loaded raises 503 if models not loaded."""
    from fastapi import HTTPException

    # Ensure global models are None (default state)
    router.MODELS["yolo"] = None
    router.MODELS["insightface"] = None

    with pytest.raises(HTTPException) as excinfo:
        router.ensure_models_loaded()
    assert excinfo.value.status_code == 503
