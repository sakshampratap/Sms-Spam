# SMS GUARD — Technical Architecture Documentation

## 1. System Overview

**SMS GUARD** is a complete, modular, privacy-preserving AI application developed for first-year B.Tech Computer Science and Engineering students. The system classifies incoming Short Message Service (SMS) text messages into two discrete categories:

- **SPAM:** Unsolicited marketing, lottery promotions, fraudulent financial traps, or smishing attacks.
- **HAM:** Authentic, legitimate personal or academic human messages.

The system emphasizes **real machine learning** using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization combined with a Multinomial Naive Bayes probabilistic classifier, avoiding trivial hardcoded `if-else` heuristic rules.

---

## 2. End-to-End Architectural Dataflow

```
               ┌───────────────────────────────┐
               │         User / Client         │
               └───────────────┬───────────────┘
                               │ (Submits SMS Text via Browser)
                               ▼
               ┌───────────────────────────────┐
               │    Streamlit Web Frontend     │
               │  (app.py & pages/*.py)        │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │   Input Sanitization Engine   │
               │    (src/preprocessing.py)     │
               │  - Length & Type Validation   │
               │  - Whitespace Normalization   │
               │  - Feature Indicator Probe    │
               └───────────────┬───────────────┘
                               │ Cleaned String
                               ▼
               ┌───────────────────────────────┐
               │  Fitted Scikit-Learn Pipeline │
               │   (models/spam_classifier.pkl)│
               ├───────────────────────────────┤
               │ 1. TfidfVectorizer:           │
               │    - Tokenizes unigrams/bigrams│
               │    - Computes TF-IDF Weights  │
               │ 2. MultinomialNB:             │
               │    - Evaluates log-posteriors │
               │    - Computes class probs     │
               └───────────────┬───────────────┘
                               │ Prediction & Confidence
                               ▼
               ┌───────────────────────────────┐
               │      Relational Storage       │
               │     (data/sms_guard.db)       │
               │   - SQLite History & Audit    │
               │   - User Opt-In Logging       │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │     Streamlit Visual Feed     │
               │  - Color-coded Result Cards   │
               │  - Plotly Telemetry Charts    │
               │  - Linguistic Indicators      │
               └───────────────────────────────┘
```

---

## 3. Mathematical Foundations of the ML Pipeline

### 3.1 TF-IDF Vectorization

Computers cannot perform arithmetic on raw character strings. Text must be transformed into a continuous vector space where dimensions correspond to distinctive vocabulary terms.

#### Term Frequency (TF):
Measures the frequency of term $t$ in a specific SMS message $d$:

$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

To prevent excessively long messages from skewing importance, our pipeline employs sublinear term frequency scaling:

$$\text{Sublinear TF} = 1 + \log(\text{TF}(t, d)) \quad \text{for } \text{TF} > 0$$

#### Inverse Document Frequency (IDF):
Evaluates how rare or informative a term is across the global corpus $D$ consisting of $N$ training messages:

$$\text{IDF}(t, D) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$

Where $\text{DF}(t)$ is the number of documents in which term $t$ appears. Ubiquitous words (e.g., "the", "and") appear in nearly every document, driving their IDF weight toward zero. Distinctive words (e.g., "claim", "prize", "urgent") achieve high IDF scores.

#### Composite TF-IDF Weight:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

---

### 3.2 Multinomial Naive Bayes Classification

Multinomial Naive Bayes models the probability of observing a bag-of-words / TF-IDF vector given class $c \in \{\text{HAM}, \text{SPAM}\}$.

#### Bayes' Theorem:
$$P(c \mid \mathbf{x}) = \frac{P(c) \cdot P(\mathbf{x} \mid c)}{P(\mathbf{x})}$$

#### The "Naive" Conditional Independence Assumption:
Assuming each feature $x_i$ appears independently given class $c$:

$$P(\mathbf{x} \mid c) = \prod_{i=1}^{n} P(x_i \mid c)$$

In practice, computations are performed in log-space to prevent floating-point underflow:

$$\log P(c \mid \mathbf{x}) \propto \log P(c) + \sum_{i=1}^{n} x_i \cdot \log \theta_{c, i}$$

#### Laplace Smoothing:
To handle words not seen during training and prevent multiplying by zero probability:

$$\hat{\theta}_{c, i} = \frac{N_{c, i} + \alpha}{N_c + \alpha \cdot |V|}$$

Where $\alpha = 0.1$ is the Laplace smoothing parameter, and $|V|$ is the vocabulary size (5,000 features).

---

## 4. Evaluation Metrics Formulation

| Metric | Mathematical Formula | Meaning in SMS Spam Filtering |
|---|---|---|
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ | Percentage of total messages correctly classified. |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | When the model sounds an alert for SPAM, how often is it right? High precision avoids blocking legitimate human messages. |
| **Recall** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Out of all real spam attacks, what fraction did the model successfully catch? |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic balance reconciling precision and recall into a single objective score. |

---

## 5. Relational Database Schema (SQLite)

The local SQLite database (`data/sms_guard.db`) maintains user audit history:

```sql
CREATE TABLE IF NOT EXISTS message_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_text TEXT NOT NULL,
    predicted_category TEXT NOT NULL,
    confidence REAL NOT NULL,
    is_spam INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_history_category ON message_history (predicted_category);
CREATE INDEX IF NOT EXISTS idx_history_created ON message_history (created_at);
```

### Security & Privacy Protections:
1. **Zero External Transmission:** All processing occurs in local RAM; zero telemetry is broadcast to external cloud endpoints.
2. **Parameterized Queries:** Every SQL interaction uses parameterized bind parameters (`?`), eliminating SQL injection.
3. **User Opt-In:** History logging is toggleable per classification.
