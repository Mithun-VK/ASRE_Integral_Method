from services.alert_service import AlertService, CrashSeverity, AlertSeverity

alert_service = AlertService()

print("=" * 80)
print(" ALERT SERVICE TEST")
print("=" * 80)

# Test 1: Crash simulation
print("\n[1] Simulating market crash...")
crash_results = alert_service.simulate_market_crash(
    tickers=["NVDA", "MSFT"],
    original_ratings={"NVDA": 88.5, "MSFT": 75.0},
    severity=CrashSeverity.MODERATE,
    notify=False
)

for result in crash_results:
    print(f"  {result['ticker']}: {result['original_rating']:.1f}  {result['crashed_rating']:.1f} ({result['drop_percentage']:.1f}%)")

# Test 2: Format alerts
print("\n[2] Testing alert formatting...")
message = alert_service.formatter.format_crash_warning(
    "NVDA", 88.5, 61.95, CrashSeverity.MODERATE.value
)
print(f"  Alert: {message[:80]}...")

# Test 3: Send mock WhatsApp alert
print("\n[3] Sending test WhatsApp alert...")
result = alert_service.send_alert(
    "whatsapp",
    "+1234567890",
    message,
    AlertSeverity.CRITICAL
)
print(f"  Status: {result['status']}")
print(f"  Message ID: {result['message_id']}")

print("\n[OK] Alert service works!")
