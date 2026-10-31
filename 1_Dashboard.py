"""
SMS GUARD — Page 1: Main Dashboard
Displays overall operational statistics, spam vs. ham distributions,
classification activity trends, and recent message logs.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from src.ui_components import load_css, render_sidebar, render_header, get_model_metadata
from src.analytics import get_dashboard_summary, get_activity_timeline, get_recent_activity

# Page configuration
st.set_page_config(
    page_title="SMS GUARD — Dashboard",
    page_icon="📊",
    layout="wide"
)

load_css()
render_sidebar()

# Page Banner
render_header(
    title="Security Intelligence Dashboard",
    subtitle="AI-Powered SMS Spam Detection & Live Traffic Analytics",
    description="Live operational telemetry of classified SMS messages, detection rates, and Machine Learning performance."
)

# ---------------------------------------------------------------------------
# Metric Cards
# ---------------------------------------------------------------------------
summary = get_dashboard_summary()
metadata = get_model_metadata()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Messages Analysed</div>
        <div class="metric-value">{summary['total_messages']:,}</div>
        <div class="metric-footer">📁 Local SQLite Database</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Spam Detected</div>
        <div class="metric-value" style="color: {config.THEME_COLORS['spam_danger']};">{summary['spam_count']:,}</div>
        <div class="metric-footer">🚨 {summary['spam_percentage']}% Threat Ratio</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Normal Messages</div>
        <div class="metric-value" style="color: {config.THEME_COLORS['ham_safe']};">{summary['ham_count']:,}</div>
        <div class="metric-footer">✅ {summary['ham_percentage']}% Clean Traffic</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    if metadata and "evaluation_metrics" in metadata:
        acc_text = f"{metadata['evaluation_metrics']['accuracy'] * 100:.2f}%"
        footer_text = f"F1-Score: {metadata['evaluation_metrics']['f1_score'] * 100:.2f}%"
    else:
        acc_text = "Not evaluated yet"
        footer_text = "Execute model training"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Model Test Accuracy</div>
        <div class="metric-value" style="color: {config.THEME_COLORS['secondary']};">{acc_text}</div>
        <div class="metric-footer">🎯 {footer_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------------------------------------------------------------------
# Interactive Charts Row
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns([1, 1.4])

with chart_col1:
    st.markdown("#### 🍩 Message Classification Distribution")
    if summary["total_messages"] > 0:
        fig_pie = px.pie(
            values=[summary["ham_count"], summary["spam_count"]],
            names=["Legitimate (HAM)", "Spam / Smishing (SPAM)"],
            color=["Legitimate (HAM)", "Spam / Smishing (SPAM)"],
            color_discrete_map={
                "Legitimate (HAM)": config.THEME_COLORS["ham_safe"],
                "Spam / Smishing (SPAM)": config.THEME_COLORS["spam_danger"]
            },
            hole=0.45
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#FFFFFF", width=2))
        )
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("ℹ️ No messages stored yet. Classify an SMS to populate real-time charts!")

with chart_col2:
    st.markdown("#### 📈 Classification Activity Timeline")
    timeline_data = get_activity_timeline()
    if timeline_data:
        df_time = pd.DataFrame(timeline_data)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_time["time_bucket"],
            y=df_time["ham"],
            name="HAM",
            marker_color=config.THEME_COLORS["ham_safe"]
        ))
        fig_bar.add_trace(go.Bar(
            x=df_time["time_bucket"],
            y=df_time["spam"],
            name="SPAM",
            marker_color=config.THEME_COLORS["spam_danger"]
        ))
        fig_bar.update_layout(
            barmode="stack",
            xaxis_title="Timestamp",
            yaxis_title="Count",
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("ℹ️ Activity history is currently empty. Run predictions to display timeline trends.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Recent Activity Feed & Quick Action
# ---------------------------------------------------------------------------
st.markdown("#### 🕒 Recent Classified Messages")
recent_records = get_recent_activity(limit=5)

if recent_records:
    df_recent = pd.DataFrame(recent_records)
    df_recent["Confidence"] = df_recent["confidence"].apply(lambda c: f"{float(c)*100:.1f}%")
    df_recent = df_recent.rename(columns={
        "id": "ID",
        "message_text": "SMS Text",
        "predicted_category": "Verdict",
        "created_at": "Timestamp"
    })
    st.dataframe(
        df_recent[["ID", "Verdict", "Confidence", "SMS Text", "Timestamp"]],
        use_container_width=True,
        hide_index=True
    )
else:
    st.caption("No recent classifications logged.")

st.write("")
quick_col1, quick_col2 = st.columns([1, 3])
with quick_col1:
    if st.button("🛡️ Classify New SMS Now", type="primary", use_container_width=True):
        st.switch_page("pages/2_SMS_Classifier.py")
