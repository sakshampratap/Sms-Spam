"""
SMS GUARD — Configuration Module
Defines system-wide constants, file paths, visual theme colors,
and default hyperparameters for the SMS Spam Classification System.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Information & Branding
# ---------------------------------------------------------------------------
APP_NAME = "SMS GUARD"
TAGLINE = "Smart Detection. Safer Messaging."
APP_VERSION = "1.0.0"
AUTHOR = "Department of Computer Science & Engineering"
PROJECT_TYPE = "B.Tech CSE First-Year AI Application Project"

# ---------------------------------------------------------------------------
# Directory & File Paths (Auto-resolved relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
PAGES_DIR = BASE_DIR / "pages"
ASSETS_DIR = BASE_DIR / "assets"
DOCS_DIR = BASE_DIR / "docs"
TESTS_DIR = BASE_DIR / "tests"

# Core File Paths
DATASET_CSV_PATH = DATA_DIR / "sms_dataset.csv"
MODEL_PKL_PATH = MODELS_DIR / "spam_classifier.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
DATABASE_PATH = DATA_DIR / "sms_guard.db"
STYLES_CSS_PATH = ASSETS_DIR / "styles.css"
LOGO_PNG_PATH = ASSETS_DIR / "logo.png"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Visual Theme & Color Palette (Cybersecurity & Modern Tech Theme)
# ---------------------------------------------------------------------------
THEME_COLORS = {
    "primary": "#14213D",       # Dark Navy
    "secondary": "#2563EB",     # Electric Blue
    "accent": "#14B8A6",        # Teal
    "spam_danger": "#EF4444",   # Warning Red
    "ham_safe": "#16A34A",      # Verified Green
    "background": "#F4F7FB",    # Slate Soft Background
    "card_bg": "#FFFFFF",       # Pure White Card
    "text_main": "#1E293B",     # Deep Slate Text
    "text_secondary": "#64748B",# Muted Slate Gray
    "border": "#E2E8F0"         # Light Gray Border
}

# ---------------------------------------------------------------------------
# Machine Learning Hyperparameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
MULTINOMIAL_NB_ALPHA = 0.1

# Label Mappings
LABEL_HAM = 0
LABEL_SPAM = 1
LABEL_NAMES = {0: "HAM", 1: "SPAM"}

# ---------------------------------------------------------------------------
# Preset Sample Messages for Testing & Interactive Demos
# ---------------------------------------------------------------------------
SAMPLE_MESSAGES = [
    {
        "text": "Congratulations! You have won a free prize of $1,000 cash. Call 0800-123-4567 immediately to claim now!",
        "expected": "SPAM",
        "description": "Promotional Prize Smishing"
    },
    {
        "text": "Hey are you coming to college tomorrow? Let me know if you need the notes from today's lecture.",
        "expected": "HAM",
        "description": "Legitimate Academic Query"
    },
    {
        "text": "URGENT! Your mobile account has been credited with 500 bonus points. Click http://claim-reward.xyz to redeem before midnight.",
        "expected": "SPAM",
        "description": "Urgent Financial Phishing"
    },
    {
        "text": "Hi Dad, I will be reaching home by 8 PM. Please keep the dinner ready.",
        "expected": "HAM",
        "description": "Normal Family Message"
    }
]
