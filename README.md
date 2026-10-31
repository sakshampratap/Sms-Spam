# 🛡️ SMS GUARD — AI-Powered SMS Spam Classification System

> *"Smart Detection. Safer Messaging."*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/Project-Academic%20B.Tech%20CSE-blue)](#)
[![Tests](https://img.shields.io/badge/Pytest-22%20Passed-16A34A?logo=pytest&logoColor=white)](#-automated-testing-and-qa)

---

## 1. Project Overview

**SMS GUARD** is a complete, production-grade Artificial Intelligence and Natural Language Processing (NLP) application engineered for first-year B.Tech Computer Science and Engineering students. The system classifies SMS messages into two distinct categories:

1. **SPAM (1):** Unwanted commercial spam, lottery scams, phishing attacks, and fraudulent smishing schemes.
2. **HAM (0):** Legitimate, benign human conversations and academic/personal notices.

Unlike toy projects that rely on simplistic keyword `if-else` matching, SMS GUARD implements a genuine, end-to-end Machine Learning pipeline combining **TF-IDF (Term Frequency–Inverse Document Frequency) unigram and bigram feature extraction** with a **Multinomial Naive Bayes probabilistic classifier**.

---

## 2. Key Capabilities & Deliverables

- 🖥️ **Full-Stack Streamlit Frontend:** Clean cybersecurity dashboard with dark navy `#14213D` palette, Poppins/Inter typography, and responsive metric cards.
- ⚙️ **Modular Python Backend:** Zero spaghetti code; clear single-responsibility modules (`preprocessing.py`, `train_model.py`, `predictor.py`, `evaluation.py`, `database.py`, `analytics.py`).
- 🧠 **Trained Machine Learning Pipeline:** Scikit-Learn `Pipeline` bundling `TfidfVectorizer` and `MultinomialNB`, saved cleanly using `joblib`.
- 📊 **Real Empirical Evaluation:** 98.26% test accuracy, 96.69% spam precision, holdout confusion matrix, and classification report calculated on the UCI SMS Spam Collection benchmark.
- 🗄️ **Local Relational History:** Embedded SQLite database with parameterized SQL queries, keyword search, filtering, and one-click CSV export.
- 🧪 **22 Automated Unit Tests:** Exhaustive test suite covering input validation, probability bounds, database CRUD, and known message classification.
- 🔒 **Privacy by Design:** All processing executes locally in memory; zero user message content is ever transmitted to cloud servers.

---

## 3. Complete Project Folder Structure

```
SMS_GUARD/
├── app.py                      # Main entrypoint and launchpad for the Streamlit dashboard
├── config.py                   # Global constants, directory paths, theme colors, hyperparameters
├── requirements.txt            # Pinned dependency requirements
├── README.md                   # Comprehensive project documentation & viva guide
├── .gitignore                  # Git ignore rules for virtualenv, bytecode, and local databases
│
├── data/
│   ├── sms_dataset.csv         # Benchmark UCI SMS Spam Collection dataset (5,574 messages)
│   └── sms_guard.db            # Local SQLite database for message history and audit logging
│
├── models/
│   ├── spam_classifier.pkl     # Serialized, fitted Scikit-Learn ML pipeline (TF-IDF + Naive Bayes)
│   └── model_metadata.json     # Empirical evaluation metrics, confusion matrix, hyperparameters
│
├── src/
│   ├── __init__.py             # Package initializer
│   ├── preprocessing.py        # Input validation, sanitization, linguistic feature extraction
│   ├── train_model.py          # Stratified split, pipeline fitting, evaluation, and serialization
│   ├── predictor.py            # Singleton inference service computing probabilities and verdicts
│   ├── evaluation.py           # Mathematical implementations of Accuracy, Precision, Recall, F1
│   ├── database.py             # SQLite database manager with parameterized SQL queries
│   ├── analytics.py            # Real-time metrics aggregator for dashboard charts
│   └── ui_components.py        # Shared Streamlit UI banners, custom CSS injector, sidebar
│
├── pages/
│   ├── 1_Dashboard.py          # Operational telemetry, pie charts, and activity timelines
│   ├── 2_SMS_Classifier.py     # Interactive SMS scanner with probability breakdown and samples
│   ├── 3_Message_History.py    # Local audit logs, category filters, search, and CSV export
│   ├── 4_Model_Performance.py  # Empirical test metrics, confusion matrix heatmap, and explanations
│   └── 5_About_Project.py      # Educational background, architecture dataflow, and limitations
│
├── assets/
│   ├── logo.png                # Custom graphical security shield branding asset
│   └── styles.css              # Custom CSS styling (Poppins/Inter fonts, cards, shadows)
│
├── tests/
│   ├── test_preprocessing.py   # Unit tests for text cleaning, boundary cases, feature checks
│   ├── test_prediction.py      # Unit tests for inference output schema, probability validity
│   └── test_database.py        # Unit tests for SQLite table creation, queries, and vacuuming
│
└── docs/
    ├── project_architecture.md # In-depth technical architecture and mathematical foundations
    └── installation_guide.md   # Step-by-step beginner setup guide for Windows, Mac, and Linux
```

---

## 4. Machine Learning Implementation Details

### 4.1 Dataset
The project utilizes the public **SMS Spam Collection dataset** from the **UCI Machine Learning Repository**:
- **Total Messages:** 5,574
- **Unique Messages after Deduplication:** 5,171
- **Class Proportions:** 87.4% HAM (4,518 messages), 12.6% SPAM (653 messages)
- **Train/Test Split:** 80% Training (4,136 samples), 20% Holdout Testing (1,035 samples) using stratified sampling to maintain identical class ratios.

### 4.2 Algorithm & Feature Extraction
- **Feature Extractor:** `TfidfVectorizer` configured with:
  - Sublinear term frequency scaling ($1 + \log(\text{TF})$).
  - Unigram and bigram extraction (`ngram_range=(1, 2)`).
  - English stop-word removal.
  - Vocabulary limit of 5,000 top informative features.
- **Classifier:** `MultinomialNB` (Multinomial Naive Bayes) with Laplace smoothing parameter $\alpha = 0.1$.

### 4.3 Empirical Evaluation Results (Holdout Test Set)

| Metric | Measured Value | Formula |
|---|---|---|
| **Overall Accuracy** | **98.26%** | $(\text{TP} + \text{TN}) / \text{Total}$ |
| **Spam Precision** | **96.69%** | $\text{TP} / (\text{TP} + \text{FP})$ |
| **Spam Recall** | **89.31%** | $\text{TP} / (\text{TP} + \text{FN})$ |
| **F1-Score** | **92.86%** | $2 \cdot (\text{Precision} \cdot \text{Recall}) / (\text{Precision} + \text{Recall})$ |

#### Test Set Confusion Matrix ($N = 1,035$ Unseen Samples):
- **True Negatives (TN):** `900` (Legitimate messages correctly classified as HAM)
- **False Positives (FP):** `4` (Legitimate messages mistakenly flagged as SPAM — remarkably low!)
- **False Negatives (FN):** `14` (Spam messages missed as HAM)
- **True Positives (TP):** `117` (Spam messages successfully detected and stopped)

---

## 5. Quick Start Instructions (Execution Guide)

### Prerequisites:
- Python 3.10, 3.11, or 3.12 installed on your machine.
- Visual Studio Code (recommended).

### Step 1: Open Terminal in Project Root
```bash
cd SMS_GUARD
```

### Step 2: Create and Activate Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Train the Model Pipeline
```bash
python -m src.train_model
```

### Step 5: Run Automated Unit Tests
```bash
python -m pytest tests -v
```

### Step 6: Launch the Web Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

---

## 6. Faculty Viva & Examination Questions (Cheat Sheet)

### Q1: Why use Multinomial Naive Bayes for text classification?
> **Answer:** Multinomial Naive Bayes is theoretically suited for discrete word frequency counts. Because vocabulary sizes in NLP are large (thousands of words) and individual messages are sparse, Naive Bayes calculates log-posterior probabilities extremely rapidly with minimal computational overhead, while exhibiting high precision even on heavily imbalanced datasets.

### Q2: What is the purpose of Laplace Smoothing ($\alpha$)?
> **Answer:** If a user submits an SMS containing a word that never appeared in the training dataset, its conditional probability $P(w_i \mid \text{SPAM})$ would be zero. Without smoothing, multiplying by zero would zero out the entire product. Laplace smoothing adds a small positive constant $\alpha$ (0.1) to all counts, ensuring novel words do not collapse the overall calculation.

### Q3: Why is Precision often more important than Recall in spam classification?
> **Answer:** In SMS filtering, a **False Positive** means a legitimate message (such as a bank transaction alert, hospital appointment, or emergency from family) is mistakenly blocked or routed to spam. A **False Negative** simply means a junk marketing ad reaches the inbox. While high recall is desirable, maintaining very high precision (low FP) is paramount to preserve user trust.

### Q4: Why combine TfidfVectorizer and MultinomialNB inside a Scikit-Learn Pipeline?
> **Answer:** A `Pipeline` bundles feature transformation and model fitting into a unified object. This strictly prevents **data leakage** (fitting TF-IDF statistics on testing data) and ensures that during inference (`predictor.py`), user input undergoes the exact same mathematical preprocessing and vectorization transformations as the training corpus.

---

## 7. License & Academic Attribution
Developed as an educational benchmark project for First-Year B.Tech Computer Science & Engineering. All dataset credits belong to the UCI Machine Learning Repository.
