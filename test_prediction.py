"""
SMS GUARD — Unit Tests for Prediction Service Module
Validates end-to-end inference, probability bounds, and classification accuracy on benchmark examples.
"""

from pathlib import Path
import pytest
from src.predictor import SMSPredictor, get_predictor, predict_message


def test_predictor_initialization_and_loading():
    """Verifies that the predictor loads the saved model pipeline successfully."""
    predictor = get_predictor()
    assert predictor.is_loaded is True


def test_prediction_output_schema():
    """Verifies that the prediction dictionary returns all required fields and correct types."""
    result = predict_message("Hey, can you send the assignment notes?")

    assert isinstance(result, dict)
    assert "message" in result
    assert "label" in result
    assert "is_spam" in result
    assert "confidence" in result
    assert "prob_spam" in result
    assert "prob_ham" in result
    assert "features" in result
    assert "timestamp" in result

    assert result["label"] in ["HAM", "SPAM"]
    assert isinstance(result["is_spam"], bool)
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["prob_spam"] <= 1.0
    assert 0.0 <= result["prob_ham"] <= 1.0

    # Probabilities should sum to approximately 1.0
    assert pytest.approx(result["prob_spam"] + result["prob_ham"], abs=1e-3) == 1.0


def test_prediction_known_spam_message():
    """Verifies that an overt spam prize message is classified as SPAM."""
    spam_sample = "URGENT! You have won a guaranteed 1000 cash prize. Call 0800-999-8888 now to claim your free reward."
    result = predict_message(spam_sample)

    assert result["label"] == "SPAM"
    assert result["is_spam"] is True
    assert result["prob_spam"] > 0.5


def test_prediction_known_ham_message():
    """Verifies that normal everyday communication is classified as HAM."""
    ham_sample = "Hi Sarah, are we meeting at the college library at 4 PM for the project discussion?"
    result = predict_message(ham_sample)

    assert result["label"] == "HAM"
    assert result["is_spam"] is False
    assert result["prob_ham"] > 0.5


def test_prediction_empty_input_raises_error():
    """Verifies that predicting on empty or blank input raises a ValueError."""
    predictor = get_predictor()
    with pytest.raises(ValueError):
        predictor.predict("   ")


def test_predictor_missing_model_file():
    """Verifies that a FileNotFoundError is raised when initializing with a non-existent model path."""
    with pytest.raises(FileNotFoundError):
        SMSPredictor(model_path=Path("non_existent_model_dir/fake_model.pkl"))
