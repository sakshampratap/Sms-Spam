"""
SMS GUARD — Model Evaluation Module
Calculates standard machine learning classification metrics from actual test data,
verifying custom mathematical formulas against Scikit-learn references.

Academic Context (B.Tech CSE):
Evaluating a binary classifier requires measuring its performance across multiple
dimensions:
1. Accuracy: Overall fraction of correct predictions: (TP + TN) / (TP + TN + FP + FN).
   Note: On imbalanced datasets (e.g. 87% Ham vs 13% Spam), a naive model predicting all Ham
   would achieve 87% accuracy while failing completely at spam detection!
2. Precision: Out of all messages flagged as SPAM, how many were actually SPAM?
   Formula: TP / (TP + FP). Crucial because False Positives mark legitimate messages as spam.
3. Recall (Sensitivity): Out of all actual SPAM messages, how many did the model catch?
   Formula: TP / (TP + FN). Measures the model's spam capture coverage.
4. F1-Score: Harmonic mean of Precision and Recall: 2 * (Precision * Recall) / (Precision + Recall).
   Balances precision and recall into a single objective score.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    classification_report as sk_classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divides two numerical values, avoiding ZeroDivisionError.
    """
    if denominator == 0 or np.isnan(denominator):
        return default
    return float(numerator / denominator)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculates primary binary classification metrics from ground truth and predictions.
    Explicitly computes mathematical formulas and cross-verifies against scikit-learn.

    Parameters:
        y_true (array-like): Ground truth labels (0 = Ham, 1 = Spam).
        y_pred (array-like): Predicted labels (0 = Ham, 1 = Spam).

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1_score,
              and confusion matrix components (TP, TN, FP, FN).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical length.")

    total = len(y_true)
    if total == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    # Calculate confusion matrix entries:
    # 0 = Negative (HAM), 1 = Positive (SPAM)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))  # True Positive (Spam caught as Spam)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))  # True Negative (Ham recognized as Ham)
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))  # False Positive (Ham falsely flagged as Spam)
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))  # False Negative (Spam missed as Ham)

    # 1. Accuracy = (TP + TN) / Total
    accuracy_formula = safe_divide(tp + tn, total)

    # 2. Precision = TP / (TP + FP)
    precision_formula = safe_divide(tp, tp + fp)

    # 3. Recall = TP / (TP + FN)
    recall_formula = safe_divide(tp, tp + fn)

    # 4. F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_formula = safe_divide(
        2 * (precision_formula * recall_formula),
        precision_formula + recall_formula
    )

    # Verify against Scikit-Learn reference implementations
    sk_acc = float(accuracy_score(y_true, y_pred))
    sk_prec = float(precision_score(y_true, y_pred, zero_division=0))
    sk_rec = float(recall_score(y_true, y_pred, zero_division=0))
    sk_f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "total_samples": total,
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "matrix_2x2": [[tn, fp], [fn, tp]]
        },
        "metrics": {
            "accuracy": round(accuracy_formula, 4),
            "precision": round(precision_formula, 4),
            "recall": round(recall_formula, 4),
            "f1_score": round(f1_formula, 4)
        },
        "sklearn_verification": {
            "accuracy": round(sk_acc, 4),
            "precision": round(sk_prec, 4),
            "recall": round(sk_rec, 4),
            "f1_score": round(sk_f1, 4)
        }
    }


def generate_classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Generates a full scikit-learn classification report breakdown for both
    Ham (0) and Spam (1) classes.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: Detailed classification report structure.
    """
    return sk_classification_report(
        y_true,
        y_pred,
        target_names=["HAM", "SPAM"],
        output_dict=True,
        zero_division=0
    )
