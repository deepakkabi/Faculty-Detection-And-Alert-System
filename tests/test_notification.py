import pytest
from unittest.mock import MagicMock, patch
from backend.notification import emailer

def test_send_email_success():
    """Test successful email sending."""
    # Patch smtplib.SMTP_SSL which is imported by emailer
    with patch("smtplib.SMTP_SSL") as mock_ssl:
        instance = mock_ssl.return_value
        instance.__enter__.return_value = instance # Context manager

        result = emailer.send_email("sender@test.com", "pass", "Subject", "Body", "receiver@test.com")

        assert result is True
        instance.login.assert_called_with("sender@test.com", "pass")
        instance.send_message.assert_called_once()

def test_send_email_fail():
    """Test failure in email sending."""
    with patch("smtplib.SMTP_SSL", side_effect=Exception("Connection failed")):
        result = emailer.send_email("sender@test.com", "pass", "Subject", "Body", "receiver@test.com")
        assert result is False

def test_send_email_missing_credentials():
    result = emailer.send_email("", "", "S", "B", "R")
    assert result is False
