"""
SMS GUARD — Model Training Pipeline
Loads the SMS Spam Collection dataset, trains a Multinomial Naive Bayes classifier
using TF-IDF feature extraction inside a Scikit-learn Pipeline, evaluates performance
on unseen test data, and exports the fitted pipeline and metadata.

================================================================================
CONCEPTUAL EXPLANATIONS FOR STUDENTS & RESEARCHERS (B.Tech CSE Level):
================================================================================

1. WHAT TF-IDF DOES:
   Computers cannot directly compute mathematical distances or probabilities on
   raw string sentences like "Win a free cash prize!".
   TF-IDF (Term Frequency - Inverse Document Frequency) converts text into numerical vectors:
   - Term Frequency (TF): Measures how frequently a word appears in a specific SMS message.
     If the word "cash" appears 3 times in a short message, its local importance is high.
   - Inverse Document Frequency (IDF): Measures how rare or informative a word is across the
     entire collection of messages. Common words like "the", "is", "at" appear in almost every
     message, so their IDF weight is suppressed towards zero. Rare, informative words like
     "guaranteed", "smishing", or "unclaimed" get a high IDF weight.
   - Combined Formula: TF-IDF(t, d, D) = TF(t, d) * log(N / (DF(t) + 1))
     Result: High scores for distinctive words that strongly characterize specific messages.

2. WHAT TRAINING MEANS:
   Training is the optimization process where the Machine Learning algorithm analyzes
   labeled examples (SMS messages paired with known answers: 0 for HAM, 1 for SPAM).
   During training, the Naive Bayes algorithm counts how often each token and word combination
   occurs in spam messages versus ham messages to learn their conditional probabilities:
   P(word | SPAM) and P(word | HAM).

3. WHAT TESTING MEANS:
   Testing evaluates whether the model has genuinely learned patterns rather than simply
   memorizing the training examples (overfitting). A separate portion of the dataset (20%)
   is held back and kept completely invisible during training. We evaluate the model's
   ability to generalize by asking it to classify these unseen messages.

4. WHAT NAIVE BAYES DOES:
   Multinomial Naive Bayes applies Bayes' Theorem with the "naive" conditional independence
   assumption that words in a message occur independently of one another given the class:
   P(SPAM | words) = P(SPAM) * Product(P(w_i | SPAM)) / P(words)
   Despite this simplifying assumption, Naive Bayes is extraordinarily effective, fast, and
   robust for text classification because word frequency distributions differ drastically
   between legitimate human conversations and unsolicited promotional spam.
   Laplace smoothing (alpha=0.1) prevents zero probabilities when a message contains words
   not observed during training.

5. WHAT MODEL PREDICTION MEANS:
   When given a new, unseen SMS, the system computes the posterior probability for both
   classes: P(SPAM | message) and P(HAM | message). The class with the higher probability
   becomes the predicted label, and the calculated probability serves as the confidence score.
================================================================================
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import config
from src.preprocessing import clean_text
from src.evaluation import compute_binary_metrics, generate_classification_summary

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SMS_GUARD_TRAINER")


def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Loads the SMS dataset, validates columns, maps text labels to binary integers,
    and removes true duplicate records.

    Parameters:
        csv_path (Path): Path to the CSV dataset file.

    Returns:
        pd.DataFrame: Cleaned dataframe with columns ['label', 'message', 'label_id'].
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {csv_path}\n"
            f"Please ensure the SMS Spam Collection dataset is placed in the data/ directory."
        )

    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize column names to lower case
    df.columns = [c.strip().lower() for c in df.columns]

    if "label" not in df.columns or "message" not in df.columns:
        raise ValueError(
            f"Dataset must contain 'label' and 'message' columns. Found columns: {list(df.columns)}"
        )

    # Drop null or empty rows
    initial_count = len(df)
    df = df.dropna(subset=["label", "message"]).copy()
    df["message"] = df["message"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Map labels: ham -> 0, spam -> 1
    label_map = {"ham": config.LABEL_HAM, "spam": config.LABEL_SPAM}
    valid_labels = df["label"].isin(label_map.keys())
    df = df[valid_labels].copy()
    df["label_id"] = df["label"].map(label_map).astype(int)

    # Remove exact duplicate messages to prevent train-test contamination
    dedup_count = len(df)
    df = df.drop_duplicates(subset=["message"]).copy()
    final_count = len(df)

    logger.info(
        f"Dataset loaded successfully: {initial_count} raw rows -> "
        f"{dedup_count} valid rows -> {final_count} unique messages after deduplication."
    )

    ham_count = (df["label_id"] == config.LABEL_HAM).sum()
    spam_count = (df["label_id"] == config.LABEL_SPAM).sum()
    logger.info(
        f"Class Distribution: HAM={ham_count} ({ham_count / final_count * 100:.1f}%), "
        f"SPAM={spam_count} ({spam_count / final_count * 100:.1f}%)"
    )

    return df


def build_pipeline() -> Pipeline:
    """
    Constructs an end-to-end Scikit-learn Pipeline combining:
    1. TfidfVectorizer: Converts cleaned text to unigram and bigram TF-IDF features.
    2. MultinomialNB: Probabilistic Naive Bayes classifier with Laplace smoothing.

    Using a Pipeline guarantees that all feature extraction transforms are learned
    strictly from training data and applied consistently during inference.

    Returns:
        Pipeline: Unfitted Scikit-learn Pipeline object.
    """
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_text,
                ngram_range=config.TFIDF_NGRAM_RANGE,
                max_features=config.TFIDF_MAX_FEATURES,
                sublinear_tf=True,
                stop_words="english",
                strip_accents="unicode"
            )
        ),
        (
            "classifier",
            MultinomialNB(
                alpha=config.MULTINOMIAL_NB_ALPHA,
                fit_prior=True
            )
        )
    ])
    return pipeline


def train_and_evaluate() -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Executes the complete machine learning workflow:
    - Loads and preprocesses dataset.
    - Performs stratified 80/20 train/test split.
    - Fits the ML pipeline on training messages only.
    - Evaluates predictions on holdout test set.
    - Saves model artifact (.pkl) and evaluation metadata (.json).

    Returns:
        tuple: (fitted_pipeline, evaluation_metadata_dict)
    """
    df = load_and_validate_dataset(config.DATASET_CSV_PATH)

    # Convert pandas series to standard python list and numpy array
    # to ensure full compatibility across all Pandas and PyArrow versions
    X = list(df["message"].astype(str))
    y = np.array(df["label_id"].tolist(), dtype=int)

    # Stratified split ensures exact class proportions are preserved in both train and test splits
    logger.info(f"Performing stratified split (test_size={config.TEST_SIZE}, random_state={config.RANDOM_STATE})")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    train_ham = int(np.sum(y_train == config.LABEL_HAM))
    train_spam = int(np.sum(y_train == config.LABEL_SPAM))
    test_ham = int(np.sum(y_test == config.LABEL_HAM))
    test_spam = int(np.sum(y_test == config.LABEL_SPAM))

    logger.info(f"Training split: {len(X_train)} samples (HAM: {train_ham}, SPAM: {train_spam})")
    logger.info(f"Testing split:  {len(X_test)} samples (HAM: {test_ham}, SPAM: {test_spam})")

    # Fit pipeline
    logger.info("Fitting TF-IDF Vectorizer and Multinomial Naive Bayes pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate on unseen test data
    logger.info("Generating predictions on holdout test set...")
    y_pred = pipeline.predict(X_test)

    # Calculate actual metrics
    eval_metrics = compute_binary_metrics(y_test, y_pred)
    class_report = generate_classification_summary(y_test, y_pred)

    tfidf_step = pipeline.named_steps["tfidf"]
    vocab_size = len(tfidf_step.vocabulary_)

    metadata = {
        "model_name": "SMS Guard Multinomial Naive Bayes",
        "algorithm": "MultinomialNB",
        "vectorizer": "TfidfVectorizer",
        "training_timestamp": datetime.now().isoformat(),
        "random_state": config.RANDOM_STATE,
        "hyperparameters": {
            "alpha": config.MULTINOMIAL_NB_ALPHA,
            "ngram_range": list(config.TFIDF_NGRAM_RANGE),
            "max_features": config.TFIDF_MAX_FEATURES,
            "vocabulary_size": vocab_size
        },
        "dataset_statistics": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_ham": train_ham,
            "train_spam": train_spam,
            "test_ham": test_ham,
            "test_spam": test_spam
        },
        "evaluation_metrics": eval_metrics["metrics"],
        "confusion_matrix": eval_metrics["confusion_matrix"],
        "classification_report": class_report
    }

    # Print summary to terminal
    print("\n" + "=" * 65)
    print("           SMS GUARD — MODEL EVALUATION REPORT           ")
    print("=" * 65)
    print(f" Test Accuracy:  {metadata['evaluation_metrics']['accuracy'] * 100:.2f}%")
    print(f" Test Precision: {metadata['evaluation_metrics']['precision'] * 100:.2f}% (Spam Precision)")
    print(f" Test Recall:    {metadata['evaluation_metrics']['recall'] * 100:.2f}% (Spam Catch Rate)")
    print(f" Test F1-Score:  {metadata['evaluation_metrics']['f1_score'] * 100:.2f}%")
    print("-" * 65)
    cm = metadata["confusion_matrix"]
    print(f" True Negatives  (Ham correctly identified):   {cm['true_negative']}")
    print(f" False Positives (Ham incorrectly flagged):    {cm['false_positive']}")
    print(f" False Negatives (Spam missed as Ham):         {cm['false_negative']}")
    print(f" True Positives  (Spam correctly identified):  {cm['true_positive']}")
    print("=" * 65 + "\n")

    # Persist model pipeline
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving fitted pipeline to: {config.MODEL_PKL_PATH}")
    joblib.dump(pipeline, config.MODEL_PKL_PATH)

    # Persist metadata JSON
    logger.info(f"Saving evaluation metadata to: {config.MODEL_METADATA_PATH}")
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Training and persistence completed successfully!")
    return pipeline, metadata


if __name__ == "__main__":
    train_and_evaluate()
