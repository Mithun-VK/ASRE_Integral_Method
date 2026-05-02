"""
Alert Service
Handles market alerts via WhatsApp, SMS, and Email with crash simulation.

Features:
- Market crash simulation with configurable severity
- WhatsApp alerts via Twilio
- SMS alerts via Twilio
- Email alerts via SMTP
- Alert templates and formatting
- Alert history tracking
- Rate limiting for alerts
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

from api.config import settings

logger = logging.getLogger(__name__)

# Twilio imports (optional - will use mock if not available)
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio not installed. Install with: pip install twilio")

# Email imports (optional)
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logger.warning("Email libraries not available")


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CrashSeverity(str, Enum):
    """Market crash severity levels"""
    MILD = "mild"           # -10% to -20%
    MODERATE = "moderate"   # -20% to -35%
    SEVERE = "severe"       # -35% to -50%
    CATASTROPHIC = "catastrophic"  # -50%+


# Crash impact multipliers
CRASH_MULTIPLIERS = {
    CrashSeverity.MILD: 0.85,          # 15% drop
    CrashSeverity.MODERATE: 0.70,      # 30% drop
    CrashSeverity.SEVERE: 0.55,        # 45% drop
    CrashSeverity.CATASTROPHIC: 0.40,  # 60% drop
}

# Alert templates
ALERT_TEMPLATES = {
    "rating_change": "🚨 {ticker} Alert\nRating: {old_rating:.1f} → {new_rating:.1f}\nSignal: {signal}\n{message}",
    "crash_warning": "⚠️ MARKET CRASH DETECTED\n{ticker}: {rating:.1f} → {new_rating:.1f}\nDrop: {drop_pct:.1f}%\n{severity} crash in progress!",
    "momentum_trap": "⚠️ MOMENTUM TRAP: {ticker}\nRating: {rating:.1f}\nWarning: {warning}\nAction: {action}",
    "buy_opportunity": "💎 BUY OPPORTUNITY\n{ticker}: {rating:.1f}/100\nSignal: {signal}\nCategory: {category}\n{context}",
    "portfolio_alert": "📊 Portfolio Alert\nHealth: {health}\nScore: {score:.1f}/100\nHigh-risk stocks: {count}\n{message}",
}


# ============================================================================
# MARKET CRASH SIMULATOR
# ============================================================================

class MarketCrashSimulator:
    """Simulate market crash scenarios for testing alerts"""
    
    @staticmethod
    def simulate_crash(
        ticker: str,
        original_rating: float,
        severity: CrashSeverity = CrashSeverity.MODERATE,
        sector_wide: bool = False
    ) -> Dict:
        """
        Simulate market crash impact on stock rating.
        
        Args:
            ticker: Stock ticker
            original_rating: Original ASRE rating
            severity: Crash severity level
            sector_wide: If True, simulate sector-wide crash
            
        Returns:
            Simulated crash data
        """
        logger.info(f"Simulating {severity.value} crash for {ticker}")
        
        # Get crash multiplier
        multiplier = CRASH_MULTIPLIERS[severity]
        
        # Apply crash to rating
        crashed_rating = original_rating * multiplier
        
        # Calculate percentage drop
        drop_pct = ((crashed_rating - original_rating) / original_rating) * 100
        
        # Determine new signal
        if crashed_rating < 35:
            new_signal = "STRONG SELL"
        elif crashed_rating < 50:
            new_signal = "CAUTION"
        else:
            new_signal = "HOLD"
        
        # Calculate recovery time (days)
        recovery_days = {
            CrashSeverity.MILD: 7,
            CrashSeverity.MODERATE: 30,
            CrashSeverity.SEVERE: 90,
            CrashSeverity.CATASTROPHIC: 180,
        }[severity]
        
        # Generate crash context
        context = MarketCrashSimulator._generate_crash_context(
            severity, 
            sector_wide
        )
        
        return {
            "ticker": ticker,
            "original_rating": round(original_rating, 2),
            "crashed_rating": round(crashed_rating, 2),
            "drop_percentage": round(drop_pct, 2),
            "severity": severity.value,
            "new_signal": new_signal,
            "sector_wide": sector_wide,
            "recovery_days": recovery_days,
            "context": context,
            "simulated_at": datetime.now().isoformat(),
            "is_simulation": True
        }
    
    @staticmethod
    def simulate_sector_crash(
        tickers: List[str],
        original_ratings: Dict[str, float],
        severity: CrashSeverity = CrashSeverity.MODERATE
    ) -> List[Dict]:
        """Simulate crash across multiple stocks"""
        results = []
        
        for ticker in tickers:
            original_rating = original_ratings.get(ticker, 50.0)
            crash_data = MarketCrashSimulator.simulate_crash(
                ticker,
                original_rating,
                severity,
                sector_wide=True
            )
            results.append(crash_data)
        
        logger.info(f"Simulated {severity.value} crash for {len(tickers)} stocks")
        return results
    
    @staticmethod
    def _generate_crash_context(
        severity: CrashSeverity,
        sector_wide: bool
    ) -> str:
        """Generate crash context message"""
        contexts = {
            CrashSeverity.MILD: "Minor market correction",
            CrashSeverity.MODERATE: "Significant market downturn",
            CrashSeverity.SEVERE: "Major market crash",
            CrashSeverity.CATASTROPHIC: "Black swan event"
        }
        
        base = contexts[severity]
        if sector_wide:
            base += " affecting entire sector"
        
        return base


# ============================================================================
# TWILIO WHATSAPP SERVICE
# ============================================================================

class WhatsAppService:
    """Send WhatsApp messages via Twilio"""
    
    def __init__(self):
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio not available - alerts will be mocked")
            self.client = None
            return
        
        if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
            logger.warning("Twilio credentials not configured")
            self.client = None
            return
        
        try:
            self.client = TwilioClient(
                settings.TWILIO_ACCOUNT_SID,
                settings.TWILIO_AUTH_TOKEN
            )
            self.from_number = f"whatsapp:{settings.TWILIO_PHONE_NUMBER}"
            logger.info("WhatsApp service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio: {e}")
            self.client = None
    
    def send_alert(
        self,
        to_number: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO
    ) -> Dict:
        """
        Send WhatsApp alert.
        
        Args:
            to_number: Recipient phone number (with country code)
            message: Alert message
            severity: Alert severity
            
        Returns:
            Send status
        """
        # Format phone number
        if not to_number.startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}"
        
        # Add severity emoji
        severity_emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🆘"
        }
        
        formatted_message = f"{severity_emojis[severity]} {message}"
        
        # Send via Twilio
        if not self.client:
            logger.warning(f"[MOCK] WhatsApp to {to_number}: {formatted_message}")
            return {
                "success": True,
                "message_id": "mock_" + datetime.now().strftime("%Y%m%d%H%M%S"),
                "to": to_number,
                "status": "mocked",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            message_obj = self.client.messages.create(
                body=formatted_message,
                from_=self.from_number,
                to=to_number
            )
            
            logger.info(f"WhatsApp sent to {to_number}: {message_obj.sid}")
            
            return {
                "success": True,
                "message_id": message_obj.sid,
                "to": to_number,
                "status": message_obj.status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp to {to_number}: {e}")
            return {
                "success": False,
                "error": str(e),
                "to": to_number,
                "timestamp": datetime.now().isoformat()
            }


# ============================================================================
# SMS SERVICE
# ============================================================================

class SMSService:
    """Send SMS via Twilio"""
    
    def __init__(self):
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio not available - SMS will be mocked")
            self.client = None
            return
        
        if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
            logger.warning("Twilio credentials not configured")
            self.client = None
            return
        
        try:
            self.client = TwilioClient(
                settings.TWILIO_ACCOUNT_SID,
                settings.TWILIO_AUTH_TOKEN
            )
            self.from_number = settings.TWILIO_PHONE_NUMBER
            logger.info("SMS service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio: {e}")
            self.client = None
    
    def send_alert(
        self,
        to_number: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO
    ) -> Dict:
        """Send SMS alert"""
        # Truncate message to 160 chars for SMS
        if len(message) > 160:
            message = message[:157] + "..."
        
        if not self.client:
            logger.warning(f"[MOCK] SMS to {to_number}: {message}")
            return {
                "success": True,
                "message_id": "mock_sms_" + datetime.now().strftime("%Y%m%d%H%M%S"),
                "to": to_number,
                "status": "mocked",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            message_obj = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            
            logger.info(f"SMS sent to {to_number}: {message_obj.sid}")
            
            return {
                "success": True,
                "message_id": message_obj.sid,
                "to": to_number,
                "status": message_obj.status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send SMS to {to_number}: {e}")
            return {
                "success": False,
                "error": str(e),
                "to": to_number,
                "timestamp": datetime.now().isoformat()
            }


# ============================================================================
# EMAIL SERVICE
# ============================================================================

class EmailService:
    """Send email alerts via SMTP"""
    
    def __init__(self):
        self.smtp_enabled = EMAIL_AVAILABLE and hasattr(settings, 'SMTP_HOST')
        
        if not self.smtp_enabled:
            logger.warning("Email service not configured")
    
    def send_alert(
        self,
        to_email: str,
        subject: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO
    ) -> Dict:
        """Send email alert"""
        
        if not self.smtp_enabled:
            logger.warning(f"[MOCK] Email to {to_email}: {subject}")
            return {
                "success": True,
                "to": to_email,
                "status": "mocked",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = getattr(settings, 'SMTP_FROM', 'alerts@asre.app')
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add severity to subject
            severity_prefix = {
                AlertSeverity.INFO: "[INFO]",
                AlertSeverity.WARNING: "[WARNING]",
                AlertSeverity.CRITICAL: "[CRITICAL]",
                AlertSeverity.EMERGENCY: "[EMERGENCY]"
            }
            msg['Subject'] = f"{severity_prefix[severity]} {subject}"
            
            # Attach message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Send via SMTP
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                if hasattr(settings, 'SMTP_USERNAME'):
                    server.starttls()
                    server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}")
            
            return {
                "success": True,
                "to": to_email,
                "subject": msg['Subject'],
                "status": "sent",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return {
                "success": False,
                "error": str(e),
                "to": to_email,
                "timestamp": datetime.now().isoformat()
            }


# ============================================================================
# ALERT FORMATTER
# ============================================================================

class AlertFormatter:
    """Format alert messages from templates"""
    
    @staticmethod
    def format_rating_change(
        ticker: str,
        old_rating: float,
        new_rating: float,
        signal: str,
        context: str = ""
    ) -> str:
        """Format rating change alert"""
        change = new_rating - old_rating
        change_icon = "📈" if change > 0 else "📉"
        
        message = context if context else "Rating updated"
        
        return ALERT_TEMPLATES["rating_change"].format(
            ticker=ticker,
            old_rating=old_rating,
            new_rating=new_rating,
            signal=signal,
            message=f"{change_icon} {message}"
        )
    
    @staticmethod
    def format_crash_warning(
        ticker: str,
        rating: float,
        new_rating: float,
        severity: str
    ) -> str:
        """Format crash warning"""
        drop_pct = ((new_rating - rating) / rating) * 100
        
        return ALERT_TEMPLATES["crash_warning"].format(
            ticker=ticker,
            rating=rating,
            new_rating=new_rating,
            drop_pct=drop_pct,
            severity=severity.upper()
        )
    
    @staticmethod
    def format_momentum_trap(
        ticker: str,
        rating: float,
        warning: str,
        action: str
    ) -> str:
        """Format momentum trap alert"""
        return ALERT_TEMPLATES["momentum_trap"].format(
            ticker=ticker,
            rating=rating,
            warning=warning,
            action=action
        )
    
    @staticmethod
    def format_buy_opportunity(
        ticker: str,
        rating: float,
        signal: str,
        category: str,
        context: str
    ) -> str:
        """Format buy opportunity alert"""
        return ALERT_TEMPLATES["buy_opportunity"].format(
            ticker=ticker,
            rating=rating,
            signal=signal,
            category=category,
            context=context
        )
    
    @staticmethod
    def format_portfolio_alert(
        health: str,
        score: float,
        high_risk_count: int,
        message: str
    ) -> str:
        """Format portfolio alert"""
        return ALERT_TEMPLATES["portfolio_alert"].format(
            health=health,
            score=score,
            count=high_risk_count,
            message=message
        )


# ============================================================================
# UNIFIED ALERT SERVICE
# ============================================================================

class AlertService:
    """
    Unified alert service for all notification channels.
    
    Handles WhatsApp, SMS, and Email with crash simulation.
    """
    
    def __init__(self):
        self.whatsapp = WhatsAppService()
        self.sms = SMSService()
        self.email = EmailService()
        self.formatter = AlertFormatter()
        self.crash_simulator = MarketCrashSimulator()
        
        # Alert history (in-memory for now)
        self._alert_history: List[Dict] = []
        self._rate_limits: Dict[str, datetime] = {}
    
    def send_alert(
        self,
        channel: str,
        recipient: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        subject: Optional[str] = None
    ) -> Dict:
        """
        Send alert via specified channel.
        
        Args:
            channel: 'whatsapp', 'sms', or 'email'
            recipient: Phone number or email
            message: Alert message
            severity: Alert severity
            subject: Email subject (for email channel)
            
        Returns:
            Send status
        """
        # Check rate limit
        if not self._check_rate_limit(recipient):
            logger.warning(f"Rate limit exceeded for {recipient}")
            return {
                "success": False,
                "error": "Rate limit exceeded. Try again later.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Send via appropriate channel
        if channel == "whatsapp":
            result = self.whatsapp.send_alert(recipient, message, severity)
        elif channel == "sms":
            result = self.sms.send_alert(recipient, message, severity)
        elif channel == "email":
            if not subject:
                subject = "ASRE Alert"
            result = self.email.send_alert(recipient, subject, message, severity)
        else:
            raise ValueError(f"Unknown channel: {channel}")
        
        # Log to history
        self._log_alert(channel, recipient, message, severity, result)
        
        return result
    
    def simulate_market_crash(
        self,
        tickers: List[str],
        original_ratings: Dict[str, float],
        severity: CrashSeverity = CrashSeverity.MODERATE,
        notify: bool = True,
        notification_channel: str = "whatsapp",
        recipient: Optional[str] = None
    ) -> List[Dict]:
        """
        Simulate market crash and optionally send alerts.
        
        Args:
            tickers: List of stock tickers
            original_ratings: Current ratings
            severity: Crash severity
            notify: Send notifications
            notification_channel: Alert channel
            recipient: Notification recipient
            
        Returns:
            List of crash simulation results
        """
        logger.info(f"Simulating {severity.value} crash for {len(tickers)} stocks")
        
        results = self.crash_simulator.simulate_sector_crash(
            tickers,
            original_ratings,
            severity
        )
        
        # Send notifications
        if notify and recipient:
            for crash_data in results:
                message = self.formatter.format_crash_warning(
                    crash_data['ticker'],
                    crash_data['original_rating'],
                    crash_data['crashed_rating'],
                    crash_data['severity']
                )
                
                self.send_alert(
                    notification_channel,
                    recipient,
                    message,
                    AlertSeverity.CRITICAL
                )
        
        return results
    
    def _check_rate_limit(self, recipient: str, limit_minutes: int = 5) -> bool:
        """Check if recipient is rate limited"""
        if recipient in self._rate_limits:
            last_alert = self._rate_limits[recipient]
            if datetime.now() - last_alert < timedelta(minutes=limit_minutes):
                return False
        
        self._rate_limits[recipient] = datetime.now()
        return True
    
    def _log_alert(
        self,
        channel: str,
        recipient: str,
        message: str,
        severity: AlertSeverity,
        result: Dict
    ):
        """Log alert to history"""
        self._alert_history.append({
            "channel": channel,
            "recipient": recipient,
            "message": message[:100],  # Truncate for storage
            "severity": severity.value,
            "success": result.get("success", False),
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 1000 alerts
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]
    
    def get_alert_history(
        self,
        recipient: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get alert history"""
        if recipient:
            history = [
                a for a in self._alert_history 
                if a['recipient'] == recipient
            ]
        else:
            history = self._alert_history
        
        return history[-limit:]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global instance
_alert_service = None

def get_alert_service() -> AlertService:
    """Get or create AlertService instance"""
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service


def send_whatsapp_alert(phone: str, message: str, severity: AlertSeverity = AlertSeverity.INFO) -> Dict:
    """Quick WhatsApp alert"""
    return get_alert_service().send_alert("whatsapp", phone, message, severity)


def send_sms_alert(phone: str, message: str, severity: AlertSeverity = AlertSeverity.INFO) -> Dict:
    """Quick SMS alert"""
    return get_alert_service().send_alert("sms", phone, message, severity)


def simulate_crash(
    tickers: List[str],
    ratings: Dict[str, float],
    severity: CrashSeverity = CrashSeverity.MODERATE
) -> List[Dict]:
    """Quick crash simulation"""
    return get_alert_service().simulate_market_crash(tickers, ratings, severity, notify=False)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'AlertService',
    'WhatsAppService',
    'SMSService',
    'EmailService',
    'AlertFormatter',
    'MarketCrashSimulator',
    'AlertSeverity',
    'CrashSeverity',
    'get_alert_service',
    'send_whatsapp_alert',
    'send_sms_alert',
    'simulate_crash',
]
