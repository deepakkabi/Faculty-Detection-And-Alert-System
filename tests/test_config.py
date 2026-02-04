import pytest
from unittest.mock import MagicMock, patch, mock_open
import json
from backend.config import config_store

def test_load_config_default():
    """Test loading config when file does not exist."""
    with patch("os.path.exists", return_value=False):
        config = config_store.load_config()
        assert config == config_store.DEFAULT_CONFIG

def test_load_config_exists():
    """Test loading existing config."""
    dummy_config = {"detection_time": 99}

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json.dumps(dummy_config))):

        config = config_store.load_config()
        assert config["detection_time"] == 99

def test_save_config():
    """Test saving config."""
    with patch("builtins.open", mock_open()) as mock_file:
        success = config_store.save_config({"key": "val"})
        assert success is True

        # Check if json was written (at least something was written)
        mock_file().write.assert_called()
