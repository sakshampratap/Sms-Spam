"""
SMS GUARD — Live Final Result & Demonstration Script
Executes real-time predictions across multiple benchmark SMS scenarios,
populates the local SQLite database, and prints the comprehensive final report.
"""

import sys
from pathlib import Path

# Configure utf-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
from src.predictor import predict_message
from src.database import save_prediction, get_history
from src.analytics import get_dashboard_summary

def run_demo():
    test_cases = [
        ("Congratulations! You have won a $1,000 Walmart Gift Card. Click here to claim your prize now!", "Promotional Prize Scam"),
        ("URGENT! Your Bank account has been temporarily locked. Verify immediately at http://secure-bank-login.xyz", "Urgent Banking Phishing"),
        ("FREE entry into our weekly £250 competition! Text WIN to 80085 now to participate.", "Lottery Smishing"),
        ("Hey, are we still meeting in the computer lab at 3 PM to work on our AI project?", "Academic Discussion"),
        ("Mom, please pick me up from the metro station. My phone battery is low.", "Family Message"),
        ("Hi professor, I have submitted the assignment on the portal. Please let me know if you need any revisions.", "Student-Faculty Communication")
    ]

    print("\n" + "=" * 76)
    print("           SMS GUARD — LIVE DEMONSTRATION & PREDICTION TEST")
    print("=" * 76 + "\n")

    for idx, (msg, desc) in enumerate(test_cases, 1):
        res = predict_message(msg)
        
        # Save to local SQLite database so the dashboard is live populated
        save_prediction(
            message_text=res['message'],
            predicted_category=res['label'],
            confidence=res['confidence'],
            is_spam=res['is_spam']
        )
        
        verdict_badge = "[🚨 SPAM DETECTED]" if res['is_spam'] else "[✅ NORMAL MESSAGE]"
        
        print(f"[{idx}] Scenario:   {desc}")
        print(f"    Message:    \"{msg}\"")
        print(f"    Verdict:    {verdict_badge} {res['label']}")
        print(f"    Confidence: {res['confidence'] * 100:.2f}% (P(Spam)={res['prob_spam']*100:.2f}%, P(Ham)={res['prob_ham']*100:.2f}%)")
        feat = res['features']
        print(f"    Features:   Chars: {feat['char_count']} | Words: {feat['word_count']} | URL: {feat['has_url']} | Currency/Prize: {feat['has_currency']} | Urgency: {feat['has_urgency']}")
        print("-" * 76)

    # Fetch updated analytics from database
    summary = get_dashboard_summary()
    print("\n" + "=" * 76)
    print("                LIVE DATABASE TELEMETRY (DASHBOARD METRICS)")
    print("=" * 76)
    print(f"  Total Messages Analysed: {summary['total_messages']}")
    print(f"  Spam Messages Detected:  {summary['spam_count']} ({summary['spam_percentage']}%)")
    print(f"  Normal Messages (Ham):   {summary['ham_count']} ({summary['ham_percentage']}%)")
    print(f"  Average Model Confidence:{summary['avg_confidence']}%")
    print("=" * 76 + "\n")

if __name__ == "__main__":
    run_demo()
