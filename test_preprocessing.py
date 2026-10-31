"""
SMS GUARD — Unit Tests for Preprocessing Module
Validates input sanitization, error boundaries, and linguistic feature extraction.
"""

import pytest
from src.preprocessing import validate_input, clean_text, extract_message_features


def test_validate_valid_text():
    """Verifies that legitimate text passes validation and is stripped."""
    raw = "   Hello, are you coming to college tomorrow?   "
    cleaned = validate_input(raw)
    assert cleaned == "Hello, are you coming to college tomorrow?"


def test_validate_empty_input_raises_error():
    """Verifies that an empty string raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_input("")


def test_validate_whitespace_only_raises_error():
    """Verifies that whitespace-only input raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_input("     \n\t   ")


def test_validate_none_input_raises_error():
    """Verifies that None input raises ValueError."""
    with pytest.raises(ValueError):
        validate_input(None)


def test_validate_excessive_length_raises_error():
    """Verifies that text exceeding max_length bound raises ValueError."""
    long_text = "A" * 6000
    with pytest.raises(ValueError, match="exceeds the maximum permitted length"):
        validate_input(long_text, max_length=5000)


def test_clean_text_normalizes_whitespace():
    """Verifies that consecutive spaces, tabs, and line breaks are collapsed."""
    raw = "Urgent!   Claim   your   prize\n\nnow at http://test.com"
    cleaned = clean_text(raw)
    assert cleaned == "Urgent! Claim your prize now at http://test.com"


def test_clean_text_preserves_spam_signals():
    """Verifies that currency symbols and exclamation marks are preserved."""
    raw = "WIN $1000 CASH NOW! CALL 0800123456!"
    cleaned = clean_text(raw)
    assert "$" in cleaned
    assert "!" in cleaned
    assert "0800123456" in cleaned


def test_extract_features_with_url_and_currency():
    """Verifies detection of URLs, currencies, and urgency keywords."""
    spam_sample = "CONGRATULATIONS! You won a $500 cash prize! Claim at https://win-now.xyz"
    features = extract_message_features(spam_sample)

    assert features["has_url"] is True
    assert features["has_currency"] is True
    assert features["has_urgency"] is True
    assert features["char_count"] > 0
    assert features["word_count"] > 0
    assert features["uppercase_ratio"] > 0.1


def test_extract_features_with_normal_message():
    """Verifies that standard friendly messages do not falsely trigger spam indicators."""
    normal_sample = "Hi mom, I will be home by dinner. Love you."
    features = extract_message_features(normal_sample)

    assert features["has_url"] is False
    assert features["has_currency"] is False
    assert features["has_urgency"] is False
    assert features["has_phone"] is False
