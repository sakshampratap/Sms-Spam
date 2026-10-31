"""
SMS GUARD — Analytics & Metrics Engine
Aggregates message history statistics from the SQLite database to power
the interactive dashboard visualizations and performance monitoring.

Academic Context (B.Tech CSE):
Real-time operational metrics provide situational awareness regarding
incoming traffic distribution, model confidence distributions, and
temporal activity spikes (such as phishing campaigns).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

import config
from src.database import get_db_connection, init_db


def get_dashboard_summary(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Computes summary classification statistics across all stored messages.

    Parameters:
        db_path (Path, optional): Path to database file.

    Returns:
        dict: Summary metrics including total, spam count, ham count,
              percentages, and average confidence.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM message_history")
        total_messages = cursor.fetchone()[0]

        if total_messages == 0:
            return {
                "total_messages": 0,
                "spam_count": 0,
                "ham_count": 0,
                "spam_percentage": 0.0,
                "ham_percentage": 0.0,
                "avg_confidence": 0.0
            }

        # Spam count
        cursor.execute("SELECT COUNT(*) FROM message_history WHERE predicted_category = 'SPAM'")
        spam_count = cursor.fetchone()[0]

        # Ham count
        cursor.execute("SELECT COUNT(*) FROM message_history WHERE predicted_category = 'HAM'")
        ham_count = cursor.fetchone()[0]

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM message_history")
        avg_confidence = cursor.fetchone()[0] or 0.0

        spam_pct = round((spam_count / total_messages) * 100, 1)
        ham_pct = round((ham_count / total_messages) * 100, 1)

        return {
            "total_messages": total_messages,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "spam_percentage": spam_pct,
            "ham_percentage": ham_pct,
            "avg_confidence": round(float(avg_confidence) * 100, 1)
        }
    finally:
        conn.close()


def get_activity_timeline(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Aggregates classification counts grouped by date/hour for timeline visualizations.

    Returns:
        list of dict: [{'time_bucket': str, 'total': int, 'spam': int, 'ham': int}]
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                SUBSTR(created_at, 1, 16) as time_bucket,
                COUNT(*) as total,
                SUM(CASE WHEN predicted_category = 'SPAM' THEN 1 ELSE 0 END) as spam,
                SUM(CASE WHEN predicted_category = 'HAM' THEN 1 ELSE 0 END) as ham
            FROM message_history
            GROUP BY time_bucket
            ORDER BY time_bucket ASC
            LIMIT 50
        """)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_recent_activity(limit: int = 5, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Fetches the most recent classifications for the main dashboard feed.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, message_text, predicted_category, confidence, created_at
            FROM message_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
