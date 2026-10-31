"""
SMS GUARD — Unit Tests for SQLite Database Module
Validates schema creation, CRUD operations, parameterized queries, and CSV export.
"""

from pathlib import Path
import pytest
from src.database import (
    init_db,
    save_prediction,
    get_history,
    clear_history,
    export_history_to_csv
)
from src.analytics import get_dashboard_summary


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provides a temporary, isolated SQLite database file for testing."""
    test_db_path = tmp_path / "test_sms_guard.db"
    init_db(test_db_path)
    return test_db_path


def test_init_db(temp_db: Path):
    """Verifies database initialization creates the database file."""
    assert temp_db.exists()


def test_save_and_retrieve_prediction(temp_db: Path):
    """Verifies that inserting a record saves and retrieves exact data."""
    rec_id = save_prediction(
        message_text="Test phishing link http://phish.xyz",
        predicted_category="SPAM",
        confidence=0.985,
        is_spam=True,
        db_path=temp_db
    )
    assert rec_id == 1

    records = get_history(db_path=temp_db)
    assert len(records) == 1
    assert records[0]["message_text"] == "Test phishing link http://phish.xyz"
    assert records[0]["predicted_category"] == "SPAM"
    assert records[0]["confidence"] == 0.985
    assert records[0]["is_spam"] == 1


def test_filter_by_category(temp_db: Path):
    """Verifies category filtering between SPAM and HAM."""
    save_prediction("Spam 1", "SPAM", 0.92, True, db_path=temp_db)
    save_prediction("Ham 1", "HAM", 0.97, False, db_path=temp_db)
    save_prediction("Spam 2", "SPAM", 0.88, True, db_path=temp_db)

    spam_only = get_history(category_filter="SPAM", db_path=temp_db)
    assert len(spam_only) == 2
    assert all(r["predicted_category"] == "SPAM" for r in spam_only)

    ham_only = get_history(category_filter="HAM", db_path=temp_db)
    assert len(ham_only) == 1
    assert ham_only[0]["predicted_category"] == "HAM"


def test_search_by_keyword(temp_db: Path):
    """Verifies keyword search within message contents."""
    save_prediction("Urgent: your account is locked", "SPAM", 0.95, True, db_path=temp_db)
    save_prediction("Meeting at cafeteria at 2pm", "HAM", 0.99, False, db_path=temp_db)

    results = get_history(search_term="locked", db_path=temp_db)
    assert len(results) == 1
    assert "locked" in results[0]["message_text"]


def test_clear_history(temp_db: Path):
    """Verifies clearing stored history returns total deleted count and empties table."""
    save_prediction("Msg 1", "HAM", 0.9, False, db_path=temp_db)
    save_prediction("Msg 2", "SPAM", 0.9, True, db_path=temp_db)

    deleted = clear_history(db_path=temp_db)
    assert deleted == 2

    remaining = get_history(db_path=temp_db)
    assert len(remaining) == 0


def test_export_history_to_csv(temp_db: Path):
    """Verifies CSV string export contains header and data rows."""
    save_prediction("Claim reward now", "SPAM", 0.94, True, db_path=temp_db)

    csv_text = export_history_to_csv(db_path=temp_db)
    assert "id,message_text,predicted_category" in csv_text
    assert "Claim reward now" in csv_text


def test_analytics_summary_empty_and_populated(temp_db: Path):
    """Verifies summary statistics compute correct percentages and handle empty databases."""
    empty_summary = get_dashboard_summary(db_path=temp_db)
    assert empty_summary["total_messages"] == 0
    assert empty_summary["spam_percentage"] == 0.0

    save_prediction("Spam 1", "SPAM", 0.9, True, db_path=temp_db)
    save_prediction("Ham 1", "HAM", 0.9, False, db_path=temp_db)
    save_prediction("Ham 2", "HAM", 0.9, False, db_path=temp_db)

    summary = get_dashboard_summary(db_path=temp_db)
    assert summary["total_messages"] == 3
    assert summary["spam_count"] == 1
    assert summary["ham_count"] == 2
    assert summary["spam_percentage"] == 33.3
    assert summary["ham_percentage"] == 66.7
