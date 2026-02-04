import pytest
from unittest.mock import MagicMock, patch, mock_open
from backend.recognition import faiss_store, faculty_manager

def test_load_faculty_database_empty():
    """Test loading database when files don't exist."""
    with patch("os.path.exists", return_value=False):
        data, index = faiss_store.load_faculty_database()
        assert data['names'] == []
        assert index is None

def test_load_faculty_database_exists():
    """Test loading database when files exist."""
    mock_data = {'names': ['Alice'], 'embeddings': [[0.1]*512], 'image_files': ['path/to/img']}

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=b"data")), \
         patch("pickle.load", return_value=mock_data):

        data, index = faiss_store.load_faculty_database()

        assert data['names'] == ['Alice']
        assert isinstance(index, MagicMock)

def test_add_faculty_member():
    """Test adding faculty member logic."""
    yolo_mock = MagicMock()
    insight_mock = MagicMock()

    # Patch the functions where they are USED, because they were imported using 'from ... import ...'
    with patch("backend.recognition.faculty_manager.detect_faces_yolo") as mock_detect, \
         patch("backend.recognition.faculty_manager.get_face_embedding") as mock_embed, \
         patch("backend.recognition.faiss_store.load_faculty_database") as mock_load_db, \
         patch("backend.recognition.faiss_store.save_faculty_database") as mock_save_db, \
         patch("backend.recognition.faiss_store.build_faiss_index") as mock_build_index:

        # Mock detections
        mock_detect.return_value = [{'bbox': [0,0,10,10], 'confidence': 0.9}]

        # Mock embedding
        embed_mock = MagicMock()
        embed_mock.tolist.return_value = [0.1, 0.2]
        mock_embed.return_value = embed_mock

        # Mock load db
        mock_load_db.return_value = ({'names': [], 'embeddings': [], 'image_files': []}, None)

        # Mock save db
        mock_save_db.return_value = True

        success, message = faculty_manager.add_faculty_member(
            yolo_mock, insight_mock, "dummy_path.jpg", "Bob", "bob.jpg"
        )

        assert success is True, f"Failed with message: {message}"
        assert "successfully" in message.lower()
