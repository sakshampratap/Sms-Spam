"""
SMS GUARD — SQLite Database Service Module
Manages local persistence of user-classified SMS messages, history filtering,
and reporting analytics using safe parameterized queries.

Academic Context (B.Tech CSE):
A production AI system often stores user interactions for auditing, model drift
monitoring, and user review. Using SQLite provides a zero-configuration, ACID-compliant,
serverless local relational database. Parameterized SQL queries (using '?') ensure
absolute immunity against SQL injection vulnerabilities.
"""

import io
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import config


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Establishes and returns a SQLite database connection with row factory enabled.

    Parameters:
        db_path (Path, optional): Path to database file. Defaults to config.DATABASE_PATH.

    Returns:
        sqlite3.Connection: Active SQLite database connection.
    """
    target_path = db_path or config.DATABASE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initializes database tables if they do not exist.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_text TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_spam INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_category
                ON message_history (predicted_category);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_created
                ON message_history (created_at);
            """)
    finally:
        conn.close()


def save_prediction(
    message_text: str,
    predicted_category: str,
    confidence: float,
    is_spam: bool,
    db_path: Optional[Path] = None
) -> int:
    """
    Inserts a newly classified message record into the database history.

    Parameters:
        message_text (str): The SMS text.
        predicted_category (str): "SPAM" or "HAM".
        confidence (float): Model confidence score (0.0 to 1.0).
        is_spam (bool): True if classified as spam, False if ham.
        db_path (Path, optional): Database path.

    Returns:
        int: The primary key ID of the inserted record.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO message_history (message_text, predicted_category, confidence, is_spam, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message_text,
                    predicted_category.upper(),
                    float(confidence),
                    1 if is_spam else 0,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_history(
    category_filter: Optional[str] = None,
    search_term: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves message classification records with optional category and keyword filtering.

    Parameters:
        category_filter (str, optional): "SPAM", "HAM", or None/ALL.
        search_term (str, optional): Substring search inside message text.
        limit (int): Maximum records to retrieve.
        offset (int): Pagination offset.
        db_path (Path, optional): Database path.

    Returns:
        list of dict: List of matching message history rows.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        query = "SELECT id, message_text, predicted_category, confidence, is_spam, created_at FROM message_history"
        conditions = []
        params: List[Any] = []

        if category_filter and category_filter.upper() in ["SPAM", "HAM"]:
            conditions.append("predicted_category = ?")
            params.append(category_filter.upper())

        if search_term and search_term.strip():
            conditions.append("message_text LIKE ?")
            params.append(f"%{search_term.strip()}%")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def clear_history(db_path: Optional[Path] = None) -> int:
    """
    Clears all stored message records from the history table.

    Returns:
        int: Total number of deleted records.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM message_history")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM message_history")
        conn.commit()
        # Set isolation level to None to execute VACUUM outside a transaction
        conn.isolation_level = None
        cursor.execute("VACUUM")
        return count
    finally:
        conn.close()


def export_history_to_csv(db_path: Optional[Path] = None) -> str:
    """
    Exports the entire message history table to a formatted CSV string.

    Returns:
        str: CSV formatted string representation of message history.
    """
    records = get_history(limit=10000, db_path=db_path)
    if not records:
        return "id,message_text,predicted_category,confidence,created_at\n"

    df = pd.DataFrame(records)
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()
