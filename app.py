"""
SMS GUARD — AI-Powered SMS Spam Classification System
Main Application Entry Point

Project Type: B.Tech Computer Science & Engineering AI Application Project
Developed using Python, Streamlit, Scikit-learn, TF-IDF, Multinomial Naive Bayes, and SQLite.
"""

import streamlit as st
import config
from src.ui_components import load_css, render_sidebar, render_header, get_model_metadata
from src.analytics import get_dashboard_summary

# Configure page layout and title
st.set_page_config(
    page_title="SMS GUARD — Intelligent SMS Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles and sidebar
load_css()
render_sidebar()

# Render Main Hero Banner
render_header(
    title=config.APP_NAME,
    subtitle=f'"{config.TAGLINE}" — AI-Powered SMS Spam Detection System',
    description=(
        "SMS GUARD is an intelligent security application designed to analyze SMS messages in real time "
        "and classify them into SPAM (unwanted, promotional, or smishing attacks) and HAM (legitimate human messages). "
        "Powered by TF-IDF feature extraction and a trained Multinomial Naive Bayes machine learning pipeline."
    )
)

# ---------------------------------------------------------------------------
# Key System Status & Quick Summary Cards
# ---------------------------------------------------------------------------
summary = get_dashboard_summary()
metadata = get_model_metadata()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Messages Analysed</div>
        <div class="metric-value">{summary['total_messages']:,}</div>
        <div class="metric-footer">📊 Stored in local database</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Spam Detected</div>
        <div class="metric-value" style="color: {config.THEME_COLORS['spam_danger']};">{summary['spam_count']:,}</div>
        <div class="metric-footer">⚠️ {summary['spam_percentage']}% of total traffic</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Legitimate Messages</div>
        <div class="metric-value" style="color: {config.THEME_COLORS['ham_safe']};">{summary['ham_count']:,}</div>
        <div class="metric-footer">✅ {summary['ham_percentage']}% verified safe</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    if metadata and "evaluation_metrics" in metadata:
        acc_val = f"{metadata['evaluation_metrics']['accuracy'] * 100:.2f}%"
        acc_footer = f"🎯 Precision: {metadata['evaluation_metrics']['precision'] * 100:.1f}%"
    else:
        acc_val = "Not evaluated"
        acc_footer = "⚠️ Run model training"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Model Test Accuracy</div>
        <div class="metric-value" style="color: {config.THEME_COLORS['secondary']};">{acc_val}</div>
        <div class="metric-footer">{acc_footer}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------------------------------------------------------------------
# Quick Navigation Grid
# ---------------------------------------------------------------------------
st.markdown("### 🚀 Fast Navigation")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    <div class="info-card">
        <h4>🔍 SMS Classifier</h4>
        <p>Test real SMS text against the trained AI model. View instant probability breakdown and feature analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open SMS Classifier ➔", use_container_width=True, type="primary"):
        st.switch_page("pages/2_SMS_Classifier.py")

with nav_col2:
    st.markdown("""
    <div class="info-card">
        <h4>📊 Main Analytics</h4>
        <p>Explore real-time activity timelines, spam vs. ham proportion charts, and recent classification records.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Dashboard ➔", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")

with nav_col3:
    st.markdown("""
    <div class="info-card">
        <h4>📈 Model Performance</h4>
        <p>Inspect the confusion matrix, precision-recall metrics, classification report, and train/test splits.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Inspect Performance ➔", use_container_width=True):
        st.switch_page("pages/4_Model_Performance.py")

st.markdown("---")

# ---------------------------------------------------------------------------
# Architecture Flow Preview
# ---------------------------------------------------------------------------
st.markdown("### ⚙️ System Processing Pipeline")
st.markdown("""
```
 [ User SMS Input ] 
        │
        ▼
 [ Text Validation & Cleaning (src/preprocessing.py) ]
        │
        ▼
 [ TF-IDF Feature Extraction (Unigrams & Bigrams) ]
        │
        ▼
 [ Trained Multinomial Naive Bayes Classifier (src/predictor.py) ]
        │
        ▼
 [ Posterior Probability & Label Assignment: SPAM or HAM ]
        │
        ▼
 [ Local Persistence & Streamlit UI Result Cards ]
```
""")

st.caption("Developed for B.Tech Computer Science & Engineering • Powered by Scikit-Learn and Streamlit")
