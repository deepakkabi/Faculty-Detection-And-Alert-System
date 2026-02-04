from fastapi.testclient import TestClient
import pytest
from backend.main import app

client = TestClient(app)

def test_read_root():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "Faculty Presence Backend Running",
        "services": [
            "/inference",
            "/recognition",
            "/attendance",
            "/config",
            "/notify"
        ]
    }

def test_import_modules():
    """Ensure all submodules can be imported."""
    import backend.inference
    import backend.recognition
    import backend.attendance
    import backend.config
    import backend.notification
    assert True
