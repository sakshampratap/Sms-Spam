"""
SMS GUARD — Preprocessing Module
Handles text validation, normalization, and feature extraction inspection.

Academic Context (B.Tech CSE):
Text preprocessing in Natural Language Processing (NLP) prepares raw, unformatted
human language strings for mathematical vectorization. In SMS spam classification,
aggressive stripping of punctuation or symbols (such as '$', '!', or capitalized words)
is detrimental because promotional and smishing messages deliberately rely on currency
symbols, urgency markers, and contact phone numbers. Therefore, this module applies
targeted normalization while safeguarding essential spam discriminators.
"""

import re
from typing import Dict, Any


def validate_input(text: str, max_length: int = 5000) -> str:
    """
    Validates user-submitted SMS text input.

    Parameters:
        text (str): The raw SMS message string from user input or dataset.
        max_length (int): Upper safety bound for message character length.

    Returns:
        str: Stripped, valid message string.

    Raises:
        ValueError: If input is None, empty, whitespace-only, or exceeds max_length.
    """
    if text is None:
        raise ValueError("Message input cannot be None.")

    if not isinstance(text, str):
        text = str(text)

    stripped = text.strip()

    if len(stripped) == 0:
        raise ValueError("Message cannot be empty or contain only whitespace.")

    if len(stripped) > max_length:
        raise ValueError(f"Message exceeds the maximum permitted length of {max_length} characters.")

    return stripped


def clean_text(text: str) -> str:
    """
    Normalizes SMS text for consistent representation without removing crucial
    spam indicators such as currency symbols, exclamation marks, or digits.

    Steps:
        1. Validates string integrity.
        2. Normalizes non-standard Unicode spaces and newlines.
        3. Collapses consecutive whitespace characters into a single space.
        4. Trims leading and trailing whitespace.

    Parameters:
        text (str): Raw input SMS message.

    Returns:
        str: Cleaned and normalized SMS string.
    """
    validated = validate_input(text)

    # Normalize excessive newlines, tabs, and spaces to a single space
    cleaned = re.sub(r"\s+", " ", validated)

    return cleaned.strip()


def extract_message_features(text: str) -> Dict[str, Any]:
    """
    Extracts descriptive structural indicators from an SMS message.
    These features provide educational insight into characteristics that
    often distinguish promotional/phishing spam from legitimate human messages.

    Parameters:
        text (str): The SMS text.

    Returns:
        dict: Breakdown containing character count, word count, presence of URLs,
              phone numbers, currency symbols, and uppercase character ratio.
    """
    cleaned = clean_text(text)
    total_chars = len(cleaned)
    words = cleaned.split()
    total_words = len(words)

    # Detect URLs (http, https, www, or typical domain patterns)
    url_pattern = re.compile(r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|org|net|xyz|info|biz|co|uk)\b)", re.IGNORECASE)
    has_url = bool(url_pattern.search(cleaned))

    # Detect currency symbols and money terms ($ £ € Rs INR win prize cash reward)
    currency_pattern = re.compile(r"([$£€¥]|₹|\b(?:rs|inr|cash|dollar|dollars|pounds|euro|free|prize|win|reward)\b)", re.IGNORECASE)
    has_currency = bool(currency_pattern.search(cleaned))

    # Detect urgency keywords (urgent, claim, immediately, expire, hurry)
    urgency_pattern = re.compile(r"\b(urgent|claim|hurry|call now|expires?|won|winner|congratulations)\b", re.IGNORECASE)
    has_urgency = bool(urgency_pattern.search(cleaned))

    # Detect explicit phone numbers or dial codes (e.g., 0800, 10-12 digit numbers)
    phone_pattern = re.compile(r"(\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b|\b0800\d{5,7}\b|\b\d{10,12}\b)")
    has_phone = bool(phone_pattern.search(cleaned))

    # Calculate uppercase letter ratio (spammers frequently use ALL CAPS for urgency)
    alpha_chars = [c for c in cleaned if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    uppercase_ratio = (len(upper_chars) / len(alpha_chars)) if alpha_chars else 0.0

    return {
        "char_count": total_chars,
        "word_count": total_words,
        "has_url": has_url,
        "has_currency": has_currency,
        "has_urgency": has_urgency,
        "has_phone": has_phone,
        "uppercase_ratio": round(uppercase_ratio, 3)
    }
