"""
SMS GUARD — Reusable UI Components Module
Provides shared Streamlit components, custom CSS injectors, header banners,
sidebar branding, and status indicators.
"""

import json
from pathlib import Path
import streamlit as st

import config


def load_css() -> None:
    """
    Injects custom CSS styling from assets/styles.css into the Streamlit app.
    """
    css_path = config.STYLES_CSS_PATH
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def get_model_metadata() -> dict:
    """
    Loads saved model metadata JSON if available, otherwise returns empty dict.
    """
    meta_path = config.MODEL_METADATA_PATH
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def render_sidebar() -> None:
    """
    Renders the consistent cybersecurity sidebar branding and model status indicator.
    """
    with st.sidebar:
        # Display Logo
        if config.LOGO_PNG_PATH.exists():
            st.image(str(config.LOGO_PNG_PATH), width=120)

        st.markdown(f"### {config.APP_NAME}")
        st.markdown(f"*{config.TAGLINE}*")
        st.markdown("---")

        # Model Status Badge
        metadata = get_model_metadata()
        if metadata and config.MODEL_PKL_PATH.exists():
            acc = metadata.get("evaluation_metrics", {}).get("accuracy", 0.0) * 100
            st.success(f"🟢 **Model Active**\n\nAlgo: `{metadata.get('algorithm', 'Naive Bayes')}`\n\nAccuracy: **{acc:.1f}%**")
        else:
            st.warning("🟡 **Model Not Trained**\n\nRun training from Model Performance page or terminal.")

        st.markdown("---")
        st.caption(f"**Project:** {config.PROJECT_TYPE}")
        st.caption(f"**Version:** {config.APP_VERSION}")


def render_header(title: str, subtitle: str, description: str = "") -> None:
    """
    Renders the unified cybersecurity dark-navy banner header.
    """
    desc_html = f"<div class='guard-header-desc'>{description}</div>" if description else ""
    header_html = f"""
    <div class="guard-header-card">
        <h1 class="guard-header-title">🛡️ {title}</h1>
        <div class="guard-header-subtitle">{subtitle}</div>
        {desc_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
