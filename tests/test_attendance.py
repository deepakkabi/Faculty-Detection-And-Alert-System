import pytest
from unittest.mock import MagicMock, patch
from backend.attendance import attendance_engine

def test_perform_attendance_check_no_match():
    """Test attendance check with no faces/match (timeout)."""
    yolo_mock = MagicMock()
    insight_mock = MagicMock()

    # Short detection time to make test fast
    config = {'detection_time': 0.1, 'threshold': 0.6}

    # Mock cap.read to return valid unpackable values
    cap_mock = MagicMock()
    cap_mock.isOpened.return_value = True
    cap_mock.read.return_value = (True, MagicMock())

    with patch("backend.attendance.attendance_engine.detect_faces_yolo", return_value=[]), \
         patch("backend.attendance.attendance_engine.load_faculty_database", return_value=({}, None)), \
         patch("backend.attendance.attendance_engine.log_attendance") as mock_log, \
         patch("cv2.VideoCapture", return_value=cap_mock):

        matched, name, confidence = attendance_engine.perform_attendance_check(
             yolo_mock, insight_mock, config, None, "Class A", "auto"
        )

        assert matched is False
        assert name is None
        mock_log.assert_called_with("Absent", "Unknown", 0.0, "Class A", "auto")

def test_perform_attendance_check_match():
    """Test attendance check with match."""
    yolo_mock = MagicMock()
    insight_mock = MagicMock()
    config = {'detection_time': 5, 'threshold': 0.6}

    # Mock cap.read to return True, frame
    cap_mock = MagicMock()
    cap_mock.isOpened.return_value = True
    cap_mock.read.return_value = (True, MagicMock())

    with patch("backend.attendance.attendance_engine.detect_faces_yolo") as mock_detect, \
         patch("backend.attendance.attendance_engine.get_face_embedding") as mock_embed, \
         patch("backend.attendance.attendance_engine.load_faculty_database") as mock_load_db, \
         patch("backend.attendance.attendance_engine.search_faculty") as mock_search, \
         patch("backend.attendance.attendance_engine.log_attendance") as mock_log, \
         patch("cv2.VideoCapture", return_value=cap_mock):

         # 1. Detect face
         mock_detect.return_value = [{'bbox': [0,0,10,10], 'confidence': 0.9}]
         # 2. Extract embedding
         mock_embed.return_value = MagicMock()
         # 3. Load DB
         mock_load_db.return_value = ({'names':['Bob']}, MagicMock())
         # 4. Search returns match
         mock_search.return_value = (True, "Bob", 0.85)

         matched, name, confidence = attendance_engine.perform_attendance_check(
             yolo_mock, insight_mock, config, None, "Class A", "auto"
         )

         assert matched is True
         assert name == "Bob"
         assert confidence == 0.85
         mock_log.assert_called_with("Present", "Bob", 0.85, "Class A", "auto")
