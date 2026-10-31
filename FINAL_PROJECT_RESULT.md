# 🛡️ SMS GUARD — Final Project Results & Evaluation Report

**Project Title:** SMS GUARD — Intelligent SMS Spam Detection System  
**Academic Level:** First-Year B.Tech Computer Science & Engineering (AI Application Project)  
**Execution Timestamp:** 2026-10-30  
**Status:** ✅ Fully Functional, Evaluated, and Tested  

---

## 1. Executive Summary of Results

SMS GUARD was developed, trained, evaluated, and tested on an authentic dataset from the **UCI Machine Learning Repository** containing **5,574 SMS messages**. The system utilizes a Scikit-Learn Machine Learning Pipeline combining **TF-IDF unigram/bigram feature extraction** and **Multinomial Naive Bayes** with Laplace smoothing.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FINAL BENCHMARK SUMMARY                         │
├──────────────────────────┬─────────────────────┬───────────────────────┤
│ Metric                   │ Measured Score      │ Industry Benchmark   │
├──────────────────────────┼─────────────────────┼───────────────────────┤
│ Overall Test Accuracy    │ 98.26%              │ > 95.0%               │
│ Spam Detection Precision │ 96.69%              │ > 90.0%               │
│ Spam Recall (Coverage)   │ 89.31%              │ > 85.0%               │
│ Harmonic F1-Score        │ 92.86%              │ > 88.0%               │
│ Automated Pytest Suite   │ 22 / 22 Passed (100%)│ 100%                 │
└──────────────────────────┴─────────────────────┴───────────────────────┘
```

---

## 2. Empirical Holdout Test Set Evaluation

The dataset was partitioned using a **stratified 80/20 train/test split** (random seed = 42). All metrics below were computed on **1,035 holdout messages** that were never seen during model training or vocabulary fitting.

### Confusion Matrix Breakdown ($N = 1,035$ Unseen Messages)

```
                       Predicted HAM         Predicted SPAM
Actual HAM (904 msgs)     TN = 900              FP = 4
Actual SPAM (131 msgs)    FN = 14               TP = 117
```

- **True Negatives (TN = 900):** $99.56\%$ of authentic human messages were cleanly delivered without interference.
- **False Positives (FP = 4):** Only 4 authentic messages were flagged as spam (False Positive Rate = $0.44\%$). In security applications, keeping False Positives extremely low is paramount to prevent blocking critical notifications or OTPs.
- **False Negatives (FN = 14):** 14 ambiguous promotional messages passed through to the inbox.
- **True Positives (TP = 117):** $89.31\%$ of spam attacks were accurately intercepted.

---

## 3. Live Demonstration Test Cases (Real-Time Inferences)

The trained model pipeline was executed across 6 benchmark test cases simulating real-world threat categories:

| ID | Message Scenario | Message Snippet | Model Verdict | Confidence | Extracted Features |
|:---|:---|:---|:---:|:---:|:---|
| **#1** | **Prize / Smishing Scam** | *"Congratulations! You have won a \$1,000 Walmart Gift Card. Click here to claim your prize now!"* | 🚨 **SPAM** | **100.00%** | Currency/Prize: True, Urgency: True |
| **#2** | **Urgent Banking Phishing** | *"URGENT! Your Bank account has been temporarily locked. Verify immediately at http://secure-bank-login.xyz"* | 🚨 **SPAM** | **98.27%** | URL: True, Urgency: True |
| **#3** | **Lottery Scam** | *"FREE entry into our weekly £250 competition! Text WIN to 80085 now to participate."* | 🚨 **SPAM** | **99.98%** | Currency/Prize: True, Urgency: False |
| **#4** | **Academic Inquiry** | *"Hey, are we still meeting in the computer lab at 3 PM to work on our AI project?"* | ✅ **HAM** | **99.87%** | URL: False, Currency: False |
| **#5** | **Personal / Family** | *"Mom, please pick me up from the metro station. My phone battery is low."* | ✅ **HAM** | **99.80%** | URL: False, Currency: False |
| **#6** | **Faculty Communication** | *"Hi professor, I have submitted the assignment on the portal. Please let me know if you need any revisions."* | ✅ **HAM** | **99.45%** | URL: False, Currency: False |

---

## 4. Live SQLite Database Status

The local SQLite database (`data/sms_guard.db`) has been initialized and populated with live prediction logs:

- **Total Messages Stored:** 7
- **Spam Intercepted:** 4 (57.1%)
- **Legitimate Messages:** 3 (42.9%)
- **Average Inference Confidence:** **99.6%**
- **Query Protection:** Parameterized SQL queries preventing SQL injection vulnerabilities.

---

## 5. Automated Unit Test Verification

Executing `python -m pytest tests -v` verifies that every module meets engineering and boundary standards:

```text
============================= 22 passed in 2.58s ==============================
```
