"""
S&P 500 Historical Constitution (Survivor Bias Test Set)

This module defines a list of companies that were REMOVED from the S&P 500 
between 2022 and 2025. 

Testing on these ensures the strategy doesn't just pick "survivors".
"""

# Companies that FAILED or were DEMOTED
# If the strategy buys SIVB in Feb 2023, it MUST show a -100% loss.
# If it avoids them, the Fundamental Floor is working.

DEATH_LIST = [
    # --- The Bank Failures (2023) ---
    {"ticker": "SIVB", "name": "Silicon Valley Bank", "status": "BANKRUPT", "fail_date": "2023-03-10"},
    {"ticker": "SBNY", "name": "Signature Bank", "status": "BANKRUPT", "fail_date": "2023-03-12"},
    {"ticker": "FRC", "name": "First Republic Bank", "status": "BANKRUPT", "fail_date": "2023-05-01"},

    # --- The Demoted (Performance Failures) ---
    {"ticker": "PVH", "name": "PVH Corp", "status": "REMOVED", "fail_date": "2022-04-04"},
    {"ticker": "PENN", "name": "Penn Entertainment", "status": "REMOVED", "fail_date": "2022-06-21"},
    {"ticker": "VNO", "name": "Vornado Realty", "status": "REMOVED", "fail_date": "2023-01-05"},
    {"ticker": "LNC", "name": "Lincoln National", "status": "REMOVED", "fail_date": "2023-12-18"},
    {"ticker": "WBA", "name": "Walgreens Boots", "status": "REMOVED", "fail_date": "2025-08-28"}, # Future/Recent removal
    {"ticker": "CHTR", "name": "Charter Comm", "status": "REMOVED", "fail_date": "2025-01-01"}, 
]

def get_survivor_bias_test_universe():
    return [item['ticker'] for item in DEATH_LIST]
