"""
SMS GUARD — Prediction Service Module
Loads the pre-trained Scikit-learn Pipeline and executes inference on user messages.

Academic Context (B.Tech CSE):
During the inference (prediction) phase, the model must NOT retrain.
The pipeline directly reuses:
1. The fitted vocabulary and IDF weights learned during training.
2. The conditional probability tables P(w_i | class) of the Multinomial Naive Bayes model.
The prediction service transforms the incoming message into an identical vector space
and computes class probabilities using Bayes' rule.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np

import config
from src.preprocessing import validate_input, extract_message_features


class SMSPredictor:
    """
    Inference service encapsulating the trained SMS classification pipeline.
    Implements model caching to prevent redundant disk reads.
    """

    _instance = None
    _pipeline = None

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initializes the predictor.
        """
        self.model_path = model_path or config.MODEL_PKL_PATH
        self.load_model()

    def load_model(self, force_reload: bool = False) -> None:
        """
        Loads the joblib serialized pipeline if not already cached in memory.

        Raises:
            FileNotFoundError: If the trained model file does not exist.
        """
        if self._pipeline is not None and not force_reload:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model pipeline not found at '{self.model_path}'.\n"
                f"Please train the model first by executing: python -m src.train_model"
            )

        try:
            self._pipeline = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize model pipeline from {self.model_path}: {str(e)}")

    @property
    def is_loaded(self) -> bool:
        """Checks whether the model pipeline is loaded and ready for inference."""
        return self._pipeline is not None

    def predict(self, raw_message: str) -> Dict[str, Any]:
        """
        Performs end-to-end spam classification on a user-submitted SMS message.

        Parameters:
            raw_message (str): The raw text message.

        Returns:
            dict: Structured prediction response containing:
                - label: "SPAM" or "HAM"
                - is_spam: bool (True for spam, False for ham)
                - confidence: float (0.0 to 1.0)
                - prob_spam: float (0.0 to 1.0)
                - prob_ham: float (0.0 to 1.0)
                - features: dict of text structural attributes
                - timestamp: ISO 8601 string
        """
        # Step 1: Input Validation
        validated_text = validate_input(raw_message)

        # Step 2: Ensure model is available
        if self._pipeline is None:
            self.load_model()

        # Step 3: Run pipeline prediction & probability estimation
        # Note: Scikit-learn expects an iterable/list of messages
        pred_label_id = int(self._pipeline.predict([validated_text])[0])

        prob_spam = 0.5
        prob_ham = 0.5
        confidence = 0.5

        if hasattr(self._pipeline, "predict_proba"):
            probabilities = self._pipeline.predict_proba([validated_text])[0]
            # Classes in pipeline: 0 = HAM, 1 = SPAM
            prob_ham = float(probabilities[0])
            prob_spam = float(probabilities[1])
            confidence = prob_spam if pred_label_id == config.LABEL_SPAM else prob_ham

        label_name = config.LABEL_NAMES.get(pred_label_id, "UNKNOWN")
        is_spam = (pred_label_id == config.LABEL_SPAM)

        # Step 4: Extract structural indicators for educational transparency
        features = extract_message_features(validated_text)

        return {
            "message": validated_text,
            "label": label_name,
            "is_spam": is_spam,
            "confidence": round(confidence, 4),
            "prob_spam": round(prob_spam, 4),
            "prob_ham": round(prob_ham, 4),
            "features": features,
            "timestamp": datetime.now().isoformat()
        }


# Global singleton instance cache
_GLOBAL_PREDICTOR: Optional[SMSPredictor] = None


def get_predictor() -> SMSPredictor:
    """
    Retrieves or instantiates the global SMSPredictor singleton.
    """
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = SMSPredictor()
    return _GLOBAL_PREDICTOR


def predict_message(text: str) -> Dict[str, Any]:
    """
    Convenience wrapper to classify an SMS message using the global predictor.
    """
    predictor = get_predictor()
    return predictor.predict(text)
