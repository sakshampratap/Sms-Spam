"""
SMS GUARD — Page 2: SMS Classifier
Interactive SMS spam classification interface powered by the trained ML pipeline.
"""

import streamlit as st
import config
from src.ui_components import load_css, render_sidebar, render_header
from src.predictor import get_predictor
from src.database import save_prediction

st.set_page_config(
    page_title="SMS GUARD — SMS Classifier",
    page_icon="🔍",
    layout="wide"
)

load_css()
render_sidebar()

# Page Banner
render_header(
    title="SMS Security Scanner",
    subtitle="Instant AI-Powered SMS Spam & Smishing Classifier",
    description="Analyze text messages against our trained Machine Learning pipeline to identify unsolicited spam, financial scams, and phishing threats."
)

st.markdown("### 💬 Check Your SMS")
st.markdown("Paste a message below to identify whether it is spam or legitimate.")

# Initialize session state for text input if needed
if "sms_input_text" not in st.session_state:
    st.session_state["sms_input_text"] = ""

# ---------------------------------------------------------------------------
# Interactive Sample Messages
# ---------------------------------------------------------------------------
st.markdown("##### 💡 Try Sample Messages:")
sample_cols = st.columns(4)

sample_messages = [
    ("🏆 Free Prize Scam", "Congratulations! You have won a free prize of $1,000. Claim now by calling 0800-123456!"),
    ("🎓 College Lecture", "Are you coming to college tomorrow? Let me know if you need today's lecture notes."),
    ("🚨 Urgent Reward", "URGENT! Claim your cash reward now before it expires at midnight. Click http://claim.xyz"),
    ("📚 Assignment Request", "Please send me today's assignment question sheet when you reach home.")
]

for idx, (label, sample_text) in enumerate(sample_messages):
    with sample_cols[idx]:
        if st.button(label, key=f"sample_{idx}", use_container_width=True):
            st.session_state["sms_input_text"] = sample_text

# ---------------------------------------------------------------------------
# Message Input Form
# ---------------------------------------------------------------------------
with st.form("sms_classification_form", clear_on_submit=False):
    user_sms = st.text_area(
        label="SMS Message Content",
        value=st.session_state["sms_input_text"],
        placeholder="Enter or paste your SMS message here...",
        height=140,
        help="Type or paste the full SMS text you wish to evaluate."
    )

    opt_col1, opt_col2 = st.columns([1.5, 2])
    with opt_col1:
        store_history = st.checkbox(
            "🔒 Save this check in local message history",
            value=True,
            help="When checked, your prediction is saved locally to your device's SQLite database."
        )

    btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1, 4])
    with btn_col1:
        submit_btn = st.form_submit_button("🛡️ Classify Message", type="primary", use_container_width=True)
    with btn_col2:
        clear_btn = st.form_submit_button("🗑️ Clear", use_container_width=True)

# Handle clear action
if clear_btn:
    st.session_state["sms_input_text"] = ""
    st.rerun()

# ---------------------------------------------------------------------------
# Prediction Execution & Results Rendering
# ---------------------------------------------------------------------------
if submit_btn:
    stripped_text = user_sms.strip() if user_sms else ""

    if not stripped_text:
        st.error("⚠️ Please enter an SMS message before clicking Classify.")
    else:
        try:
            predictor = get_predictor()
            with st.spinner("Analyzing message text through TF-IDF and Naive Bayes pipeline..."):
                result = predictor.predict(stripped_text)

            # Store in local SQLite database if enabled by user
            if store_history:
                save_prediction(
                    message_text=result["message"],
                    predicted_category=result["label"],
                    confidence=result["confidence"],
                    is_spam=result["is_spam"]
                )

            # Display Prominent Result Card
            is_spam = result["is_spam"]
            conf_percent = result["confidence"] * 100

            if is_spam:
                st.markdown(f"""
                <div class="result-card-spam">
                    <span class="result-badge-spam">⚠️ SPAM DETECTED</span>
                    <h2 class="result-title-spam">SPAM / SUSPICIOUS MESSAGE</h2>
                    <div class="result-desc-spam">
                        This message has been classified as potentially unwanted, promotional, or suspicious smishing activity.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-ham">
                    <span class="result-badge-ham">✅ NORMAL MESSAGE</span>
                    <h2 class="result-title-ham">LEGITIMATE / HAM MESSAGE</h2>
                    <div class="result-desc-ham">
                        This message has been classified as a normal, legitimate communication.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Detailed Probability & Structural Breakdown
            st.markdown("#### 🔬 Detailed Analysis Breakdown")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Model Verdict", result["label"])
            with m_col2:
                st.metric("Model Confidence", f"{conf_percent:.2f}%")
            with m_col3:
                st.metric("Spam Probability", f"{result['prob_spam'] * 100:.2f}%")

            # Confidence Progress Bar
            st.write("**Classification Confidence Level:**")
            bar_color = config.THEME_COLORS["spam_danger"] if is_spam else config.THEME_COLORS["ham_safe"]
            st.progress(min(result["confidence"], 1.0))

            # Feature inspection details
            feat = result["features"]
            with st.expander("🔍 View Extracted Linguistic & Structural Features"):
                fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                fcol1.metric("Character Count", feat["char_count"])
                fcol2.metric("Word Count", feat["word_count"])
                fcol3.metric("Contains URLs", "Yes ⚠️" if feat["has_url"] else "No")
                fcol4.metric("Contains Currency/Prize", "Yes ⚠️" if feat["has_currency"] else "No")

                fcol5, fcol6 = st.columns(2)
                fcol5.metric("Contains Urgency Terms", "Yes ⚠️" if feat["has_urgency"] else "No")
                fcol6.metric("Uppercase Ratio", f"{feat['uppercase_ratio'] * 100:.1f}%")

            # Mandatory Disclaimer
            st.markdown("""
            <div class="privacy-notice">
                ℹ️ <strong>Academic Disclaimer:</strong> This classification is generated by an automated probabilistic Machine Learning model (Multinomial Naive Bayes). While highly accurate, automated predictions do not guarantee the absolute legitimacy or maliciousness of any message. Always exercise caution before sharing credentials or clicking unknown links.
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError as fnf:
            st.error(f"❌ {str(fnf)}")
            st.info("💡 You can train the model from the **Model Performance** page or by running `python -m src.train_model` in your terminal.")
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

# Privacy notice at bottom
st.markdown("""
<div class="privacy-notice">
    🔒 <strong>Data Privacy Guarantee:</strong> All SMS messages are processed strictly in memory on your local machine. No message data is transmitted to external servers or cloud services.
</div>
""", unsafe_allow_html=True)
