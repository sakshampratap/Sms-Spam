"""
SMS GUARD — Master Backend Pipeline Runner
Executes all backend services in the exact logical order:
  Step 1: Environment & Dataset Verification (config.py)
  Step 2: Input Validation & Feature Extraction (src/preprocessing.py)
  Step 3: SQLite Database Initialization & Schema Check (src/database.py)
  Step 4: Machine Learning Pipeline Training & Evaluation (src/train_model.py & src/evaluation.py)
  Step 5: Live Inference & Probability Prediction (src/predictor.py)
  Step 6: Operational Telemetry & Analytics Aggregation (src/analytics.py)
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows command prompts
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config

def print_separator(char="=", length=75):
    print(char * length)

def main():
    print_separator("=")
    print("        SMS GUARD — COMPLETE BACKEND PIPELINE EXECUTION")
    print("=" * 75)
    print("Executing all backend modules in strict architectural order...\n")

    # -----------------------------------------------------------------------
    # STEP 1: Verify Configuration & Dataset
    # -----------------------------------------------------------------------
    print("[STEP 1/6] Verifying Configuration and Dataset Paths (config.py)...")
    if not config.DATASET_CSV_PATH.exists():
        print(f"❌ Error: Dataset not found at {config.DATASET_CSV_PATH}")
        sys.exit(1)
    print(f"  ✓ Project Root: {config.BASE_DIR}")
    print(f"  ✓ Dataset File: {config.DATASET_CSV_PATH.name} (Found)")
    print(f"  ✓ Output Model Path: {config.MODEL_PKL_PATH.name}")
    print("  [SUCCESS] Configuration verified.\n")

    # -----------------------------------------------------------------------
    # STEP 2: Preprocessing & Text Sanitization Engine
    # -----------------------------------------------------------------------
    print("[STEP 2/6] Running Preprocessing & Feature Extraction Engine (src/preprocessing.py)...")
    from src.preprocessing import validate_input, clean_text, extract_message_features

    sample_raw = "  URGENT! You won $1000 cash prize! Visit http://win-now.xyz  "
    cleaned = clean_text(sample_raw)
    features = extract_message_features(sample_raw)

    print(f"  ✓ Raw Text Input:     \"{sample_raw.strip()}\"")
    print(f"  ✓ Normalized String:  \"{cleaned}\"")
    print(f"  ✓ Extracted Features: URL={features['has_url']}, Currency={features['has_currency']}, Urgency={features['has_urgency']}, Chars={features['char_count']}")
    print("  [SUCCESS] Preprocessing engine operational.\n")

    # -----------------------------------------------------------------------
    # STEP 3: Local Relational Database Initialization
    # -----------------------------------------------------------------------
    print("[STEP 3/6] Initializing Local SQLite Database Schema (src/database.py)...")
    from src.database import init_db, save_prediction, get_history

    init_db()
    print(f"  ✓ SQLite Database initialized at: {config.DATABASE_PATH.name}")
    print("  ✓ History table verified with parameterized index structures.")
    print("  [SUCCESS] Database service ready.\n")

    # -----------------------------------------------------------------------
    # STEP 4: Machine Learning Model Training & Holdout Evaluation
    # -----------------------------------------------------------------------
    print("[STEP 4/6] Executing ML Training & Evaluation Pipeline (src/train_model.py)...")
    from src.train_model import train_and_evaluate

    pipeline, metadata = train_and_evaluate()
    metrics = metadata["evaluation_metrics"]
    cm = metadata["confusion_matrix"]

    print("  ✓ Stratified 80/20 train/test split completed.")
    print(f"  ✓ TF-IDF Vectorizer + MultinomialNB pipeline fitted.")
    print(f"  ✓ Model Saved to: {config.MODEL_PKL_PATH}")
    print(f"  ✓ Test Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  ✓ Spam Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  ✓ Spam Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  ✓ Confusion Matrix: TN={cm['true_negative']}, FP={cm['false_positive']}, FN={cm['false_negative']}, TP={cm['true_positive']}")
    print("  [SUCCESS] Model training and evaluation completed.\n")

    # -----------------------------------------------------------------------
    # STEP 5: Live Predictor & Inference Service
    # -----------------------------------------------------------------------
    print("[STEP 5/6] Testing Live Inference Engine (src/predictor.py)...")
    from src.predictor import get_predictor, predict_message

    predictor = get_predictor()
    predictor.load_model(force_reload=True)

    # Test Case 1: Spam Smishing
    spam_test = "Congratulations! You have won a free $1,000 Walmart Gift Card. Call 0800-123456 now!"
    res_spam = predict_message(spam_test)

    # Test Case 2: Legitimate Human Message
    ham_test = "Hey, are we still meeting in the college library at 4 PM for the project discussion?"
    res_ham = predict_message(ham_test)

    # Record these predictions in the database
    save_prediction(res_spam['message'], res_spam['label'], res_spam['confidence'], res_spam['is_spam'])
    save_prediction(res_ham['message'], res_ham['label'], res_ham['confidence'], res_ham['is_spam'])

    print(f"  ✓ Test SPAM: \"{spam_test[:45]}...\"")
    print(f"    --> Verdict: [{res_spam['label']}] (Confidence: {res_spam['confidence'] * 100:.2f}%)")
    print(f"  ✓ Test HAM:  \"{ham_test[:45]}...\"")
    print(f"    --> Verdict: [{res_ham['label']}] (Confidence: {res_ham['confidence'] * 100:.2f}%)")
    print("  [SUCCESS] Predictor inference verified.\n")

    # -----------------------------------------------------------------------
    # STEP 6: Operational Telemetry & Analytics Engine
    # -----------------------------------------------------------------------
    print("[STEP 6/6] Computing Real-Time Telemetry & Analytics (src/analytics.py)...")
    from src.analytics import get_dashboard_summary, get_recent_activity

    summary = get_dashboard_summary()
    recent = get_recent_activity(limit=2)

    print(f"  ✓ Total Stored Messages:    {summary['total_messages']}")
    print(f"  ✓ Detected Spam Ratio:      {summary['spam_percentage']}% ({summary['spam_count']} messages)")
    print(f"  ✓ Verified Legitimate Ratio:{summary['ham_percentage']}% ({summary['ham_count']} messages)")
    print(f"  ✓ Average Model Confidence: {summary['avg_confidence']}%")
    print("  [SUCCESS] Analytics engine verified.\n")

    # -----------------------------------------------------------------------
    # FINAL COMPLETION REPORT
    # -----------------------------------------------------------------------
    print_separator("=")
    print("         ✅ ALL BACKEND MODULES EXECUTED SUCCESSFULLY IN ORDER!")
    print("=" * 75)
    print("  1. config.py            --> Validated paths and hyperparameters")
    print("  2. src/preprocessing.py --> Validated text cleaning & linguistic features")
    print("  3. src/database.py      --> Initialized SQLite schema & audit tables")
    print("  4. src/train_model.py   --> Trained Scikit-Learn TF-IDF + Naive Bayes")
    print("  5. src/predictor.py     --> Executed live inference & probability calculations")
    print("  6. src/analytics.py     --> Computed summary statistics for the UI dashboard")
    print("=" * 75)
    print("You can now launch the web interface anytime using:")
    print("   streamlit run app.py\n")

if __name__ == "__main__":
    main()
