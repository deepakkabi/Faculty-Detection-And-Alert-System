import sys
import os
import pytest
from unittest.mock import MagicMock

# --- GLOBAL MOCKS ---
# These must be set before any backend modules are imported.

# Mock heavy/external libraries
# We use MagicMock so that any attribute access or method call returns another MagicMock,
# preventing ImportErrors and runtime crashes during imports or basic execution.
sys.modules["cv2"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()
sys.modules["insightface"] = MagicMock()
sys.modules["insightface.app"] = MagicMock()
sys.modules["faiss"] = MagicMock()
sys.modules["requests"] = MagicMock()

# Mock smtplib to avoid sending emails or needing internet
sys.modules["smtplib"] = MagicMock()

# --- PATH SETUP ---
# Add backend directory to sys.path so we can import 'backend' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- FIXTURES ---
@pytest.fixture(autouse=True)
def setup_environment():
    """
    Common setup for tests.
    """
    pass
