"""
SMS GUARD — Page 5: About Project
Comprehensive educational documentation, problem statement, architecture flow,
technology stack details, and system limitations.
"""

import streamlit as st
import config
from src.ui_components import load_css, render_sidebar, render_header

st.set_page_config(
    page_title="SMS GUARD — About Project",
    page_icon="📖",
    layout="wide"
)

load_css()
render_sidebar()

# Page Banner
render_header(
    title="About SMS GUARD",
    subtitle="AI-Powered SMS Spam Classification System",
    description="Undergraduate Computer Science & Engineering (B.Tech CSE) Artificial Intelligence Application Project."
)

# ---------------------------------------------------------------------------
# Section 1: Introduction & Problem Statement
# ---------------------------------------------------------------------------
st.markdown("### 📌 1. Introduction & Background")
st.markdown("""
Short Message Service (SMS) remains one of the most ubiquitous forms of direct digital communication globally, 
boasting open rates exceeding 95%. However, its direct reach and inherent trust make it a prime target for malicious actors.

SMS spam encompasses unsolicited commercial advertising, fraudulent financial offers, lottery scams, and highly dangerous 
**smishing** (SMS phishing) attacks intended to steal credentials, bank OTPs, or install malicious mobile software.
""")

st.markdown("### 🚨 2. Problem Statement")
st.markdown("""
Unlike modern email platforms that utilize sophisticated multi-layered server-side spam filtering, SMS delivery is handled 
by telecom cellular protocols that traditionally lack intelligent content analysis at the subscriber endpoint. Consequently:
- **Financial Fraud:** Unsuspecting mobile subscribers lose millions to fraudulent investment, loan, and lottery SMS schemes.
- **Privacy Invasion & Smishing:** Malicious links disguised as bank updates compromise sensitive user accounts.
- **Cognitive Fatigue:** Excessive marketing SMS messages cause notification fatigue, obscuring critical genuine alerts.
- **Need for Local AI:** Privacy concerns require that personal messages are filtered on-device rather than exposed to third-party cloud servers.
""")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2: Project Objectives
# ---------------------------------------------------------------------------
st.markdown("### 🎯 3. Project Objectives")
obj_col1, obj_col2 = st.columns(2)

with obj_col1:
    st.markdown("""
    <div class="info-card">
        <h4>1. Automatic Content Classification</h4>
        <p>Accurately categorize incoming messages into <strong>SPAM</strong> and <strong>HAM</strong> using empirical Natural Language Processing and probabilistic Machine Learning.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>2. Eliminate Manual Checking</h4>
        <p>Automate threat identification so users receive immediate, objective confidence scores before clicking links or dialing numbers.</p>
    </div>
    """, unsafe_allow_html=True)

with obj_col2:
    st.markdown("""
    <div class="info-card">
        <h4>3. Apply NLP & Machine Learning to Text</h4>
        <p>Demonstrate the practical application of <strong>TF-IDF vectorization</strong> and <strong>Multinomial Naive Bayes</strong> on real-world unstructured text datasets.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>4. Production-Ready Software Product</h4>
        <p>Develop a functional full-stack architecture combining a modular Python backend, SQLite local database, and a responsive Streamlit dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: Technology Stack
# ---------------------------------------------------------------------------
st.markdown("### 💻 4. Technology Stack & Role of Each Library")

tcol1, tcol2 = st.columns(2)

with tcol1:
    st.markdown("""
    - **Python 3:** The core programming language powering the application logic, data transformations, and mathematical modeling.
    - **Scikit-learn:** Provides industrial-strength implementations of the `TfidfVectorizer` feature extractor, `MultinomialNB` classifier, and metric evaluation routines (`accuracy_score`, `confusion_matrix`).
    - **Pandas & NumPy:** Handle high-performance tabular data manipulation, label encoding, array indexing, and summary metrics computation.
    - **Streamlit:** Powers the reactive, browser-based user interface without requiring complex JavaScript frontend tooling.
    """)

with tcol2:
    st.markdown("""
    - **Plotly:** Renders high-resolution interactive charts (pie charts, bar timelines, and confusion matrix heatmaps).
    - **SQLite:** Lightweight, embedded SQL relational database providing zero-configuration local persistence for message history and audit logs.
    - **Joblib:** Provides efficient serialization and disk caching for the fitted Scikit-learn Pipeline object.
    - **Pytest:** Drives automated unit testing to verify text normalization, database queries, and inference reliability.
    """)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4: Architecture Flowchart
# ---------------------------------------------------------------------------
st.markdown("### 🔄 5. How the Application Works")
st.markdown("""
The classification pipeline executes deterministically following this end-to-end dataflow:
""")

st.markdown("""
```
                   ┌─────────────────────────────┐
                   │       SMS Text Input        │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │     Text Preprocessing      │
                   │ (Validation, Normalization) │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │    TF-IDF Transformation    │
                   │  (Unigram & Bigram Weights) │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │ Trained Naive Bayes Model   │
                   │ (Multinomial Bayes Rule)    │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │  Posterior Class Probabilities│
                   │  (Confidence Estimation)    │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │       Final Verdict         │
                   │      [ SPAM or HAM ]        │
                   └─────────────────────────────┘
```
""")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 5: Real-World Limitations & Responsible AI
# ---------------------------------------------------------------------------
st.markdown("### ⚠️ 6. Limitations & Responsible AI Considerations")
st.markdown("""
While the model achieves over **98% empirical accuracy** on benchmark testing data, practical deployment requires understanding its theoretical and operational boundaries:
1. **Adversarial Evasion:** Sophisticated spammers deliberately alter spellings (e.g., *w1nner*, *fr33 c@sh*, *cl!ck*) to bypass vocabulary matching.
2. **Context Blindness:** The model analyzes statistical word co-occurrence and n-grams; it does not possess full conversational or situational context.
3. **No Absolute Guarantee:** Automated machine learning models output statistical probabilities rather than definitive guarantees of malicious intent or safety.
4. **Data Drift:** Over time, spammers shift to new promotional topics and phone patterns, requiring periodic retraining on updated datasets.
""")

st.caption("SMS GUARD System • B.Tech CSE AI Project • Academic Use")
