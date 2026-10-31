"""
SMS GUARD — Page 3: Message History
Inspect, filter, search, export, and manage locally recorded classification logs.
"""

import streamlit as st
import pandas as pd

import config
from src.ui_components import load_css, render_sidebar, render_header
from src.database import get_history, clear_history, export_history_to_csv

st.set_page_config(
    page_title="SMS GUARD — Message History",
    page_icon="📜",
    layout="wide"
)

load_css()
render_sidebar()

# Page Banner
render_header(
    title="Message Classification History",
    subtitle="Audit Logs & Local Relational Storage",
    description="Inspect past evaluations, filter by verdict, search keywords, export audit logs to CSV, or securely clear stored data."
)

# ---------------------------------------------------------------------------
# Filter and Search Controls
# ---------------------------------------------------------------------------
filter_col1, filter_col2, filter_col3 = st.columns([1.5, 2.5, 1.5])

with filter_col1:
    category_option = st.selectbox(
        "Filter by Verdict",
        options=["ALL", "SPAM", "HAM"],
        index=0,
        help="Select to isolate spam or legitimate messages."
    )

with filter_col2:
    search_keyword = st.text_input(
        "Search Message Content",
        placeholder="Type keyword (e.g. prize, urgent, college)...",
        help="Filters messages containing the specified keyword."
    )

with filter_col3:
    record_limit = st.selectbox(
        "Max Records",
        options=[25, 50, 100, 250],
        index=1
    )

# ---------------------------------------------------------------------------
# Query Database Records
# ---------------------------------------------------------------------------
filter_arg = None if category_option == "ALL" else category_option
records = get_history(
    category_filter=filter_arg,
    search_term=search_keyword,
    limit=record_limit
)

st.markdown("---")

if records:
    df_history = pd.DataFrame(records)

    # Format confidence as readable percentage
    df_history["confidence_pct"] = df_history["confidence"].apply(lambda c: f"{float(c) * 100:.1f}%")

    # Rename columns for presentation
    display_df = df_history[["id", "predicted_category", "confidence_pct", "message_text", "created_at"]].rename(
        columns={
            "id": "Record ID",
            "predicted_category": "Classification",
            "confidence_pct": "Confidence",
            "message_text": "SMS Message",
            "created_at": "Timestamp"
        }
    )

    st.markdown(f"**Found {len(records)} recorded messages:**")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------------------------
    # Export and Clear History Actions
    # -----------------------------------------------------------------------
    st.markdown("#### ⚙️ Data Management Actions")
    action_col1, action_col2 = st.columns([1.5, 2.5])

    with action_col1:
        csv_data = export_history_to_csv()
        st.download_button(
            label="📥 Download History as CSV",
            data=csv_data,
            file_name="sms_guard_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    with action_col2:
        with st.expander("⚠️ Danger Zone: Clear Message History"):
            st.warning("This action permanently erases all saved prediction logs from your local database.")
            confirm_clear = st.checkbox("I confirm that I wish to erase all stored history.")
            if st.button("🗑️ Erase All Records", type="secondary", disabled=not confirm_clear):
                deleted_count = clear_history()
                st.success(f"Successfully removed {deleted_count} records from database.")
                st.rerun()

else:
    st.info("ℹ️ No matching records found in local message history. Classify a message in the SMS Classifier page with history logging enabled.")

# Privacy notice
st.markdown("""
<div class="privacy-notice">
    🔒 <strong>User Privacy & Local Storage:</strong> Message records are stored strictly in your local SQLite database (<code>data/sms_guard.db</code>). No personal communications are transmitted across the internet.
</div>
""", unsafe_allow_html=True)
