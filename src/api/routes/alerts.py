"""
Alert Routes
============
Market alerts via WhatsApp, SMS, and Email with crash simulation.

Directly imports from existing alert_service.py:
- WhatsApp alerts via Twilio
- SMS alerts via Twilio  
- Email alerts via SMTP
- Market crash simulation
- Alert history tracking
- Rate limiting

Author: ASRE Team
Version: 2.0.0
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import existing alert service
try:
    from api.services.alert_service import (
        AlertService,
        AlertSeverity,
        CrashSeverity,
        AlertFormatter,
        MarketCrashSimulator,
        get_alert_service,
        send_whatsapp_alert,
        send_sms_alert
    )
    ALERT_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Alert service not available: {e}")
    ALERT_SERVICE_AVAILABLE = False

# Import ASRE service for ratings
try:
    from api.services.asre_service import ASREService
    ASRE_SERVICE_AVAILABLE = True
except ImportError:
    ASRE_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SendAlertRequest(BaseModel):
    """Request to send an alert"""
    channel: str = Field(..., description="Alert channel: whatsapp, sms, email")
    recipient: str = Field(..., description="Phone number (with country code) or email")
    message: str = Field(..., description="Alert message")
    severity: str = Field("info", description="Severity: info, warning, critical, emergency")
    subject: Optional[str] = Field(None, description="Email subject (for email channel)")
    
    @validator('channel')
    def validate_channel(cls, v):
        allowed = ['whatsapp', 'sms', 'email']
        if v.lower() not in allowed:
            raise ValueError(f'Channel must be one of: {allowed}')
        return v.lower()
    
    @validator('severity')
    def validate_severity(cls, v):
        allowed = ['info', 'warning', 'critical', 'emergency']
        if v.lower() not in allowed:
            raise ValueError(f'Severity must be one of: {allowed}')
        return v.lower()


class SendAlertResponse(BaseModel):
    """Alert send result"""
    success: bool
    message_id: Optional[str]
    channel: str
    recipient: str
    status: str
    timestamp: str
    error: Optional[str] = None


class SimulateCrashRequest(BaseModel):
    """Request to simulate market crash"""
    tickers: List[str] = Field(..., description="List of stock tickers")
    severity: str = Field("moderate", description="Crash severity: mild, moderate, severe, catastrophic")
    notify: bool = Field(False, description="Send alert notifications")
    notification_channel: str = Field("whatsapp", description="Notification channel")
    recipient: Optional[str] = Field(None, description="Notification recipient")
    
    @validator('severity')
    def validate_severity(cls, v):
        allowed = ['mild', 'moderate', 'severe', 'catastrophic']
        if v.lower() not in allowed:
            raise ValueError(f'Severity must be one of: {allowed}')
        return v.lower()


class CrashSimulationResult(BaseModel):
    """Individual crash simulation result"""
    ticker: str
    original_rating: float
    crashed_rating: float
    drop_percentage: float
    severity: str
    new_signal: str
    sector_wide: bool
    recovery_days: int
    context: str
    simulated_at: str
    is_simulation: bool


class SimulateCrashResponse(BaseModel):
    """Crash simulation results"""
    simulations: List[CrashSimulationResult]
    notifications_sent: int
    severity: str
    timestamp: str
    message: str


class RatingAlertRequest(BaseModel):
    """Request for rating change alert"""
    ticker: str
    old_rating: float
    new_rating: float
    signal: str
    context: Optional[str] = Field(None, description="Additional context")
    channel: str = Field("whatsapp", description="Alert channel")
    recipient: str = Field(..., description="Phone number or email")


class MomentumTrapAlertRequest(BaseModel):
    """Request for momentum trap alert"""
    ticker: str
    rating: float
    warning: str
    action: str
    channel: str = Field("whatsapp", description="Alert channel")
    recipient: str = Field(..., description="Phone number or email")


class BuyOpportunityAlertRequest(BaseModel):
    """Request for buy opportunity alert"""
    ticker: str
    rating: float
    signal: str
    category: str
    context: str
    channel: str = Field("whatsapp", description="Alert channel")
    recipient: str = Field(..., description="Phone number or email")


class AlertHistoryResponse(BaseModel):
    """Alert history entry"""
    channel: str
    recipient: str
    message: str
    severity: str
    success: bool
    timestamp: str


class AlertStatsResponse(BaseModel):
    """Alert statistics"""
    total_alerts: int
    successful_alerts: int
    failed_alerts: int
    alerts_by_channel: Dict[str, int]
    alerts_by_severity: Dict[str, int]
    recent_alerts: List[AlertHistoryResponse]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/send", response_model=SendAlertResponse)
async def send_alert(request: SendAlertRequest, background_tasks: BackgroundTasks):
    """
    Send alert via WhatsApp, SMS, or Email.
    
    **Channels:**
    - **whatsapp**: Via Twilio (requires WhatsApp Business API)
    - **sms**: Via Twilio
    - **email**: Via SMTP
    
    **Severity Levels:**
    - **info**: ℹ️ Information
    - **warning**: ⚠️ Warning
    - **critical**: 🚨 Critical
    - **emergency**: 🆘 Emergency
    
    **Example:**
    ```json
    {
      "channel": "whatsapp",
      "recipient": "+919876543210",
      "message": "NVDA rating upgraded to 85/100!",
      "severity": "info"
    }
    ```
    """
    try:
        if not ALERT_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Alert service not available"
            )
        
        logger.info(f"Sending {request.severity} alert via {request.channel} to {request.recipient}")
        
        # Get alert service
        alert_service = get_alert_service()
        
        # Map severity string to enum
        severity_map = {
            'info': AlertSeverity.INFO,
            'warning': AlertSeverity.WARNING,
            'critical': AlertSeverity.CRITICAL,
            'emergency': AlertSeverity.EMERGENCY
        }
        severity = severity_map[request.severity]
        
        # Send alert
        result = alert_service.send_alert(
            channel=request.channel,
            recipient=request.recipient,
            message=request.message,
            severity=severity,
            subject=request.subject
        )
        
        return SendAlertResponse(
            success=result.get('success', False),
            message_id=result.get('message_id'),
            channel=request.channel,
            recipient=request.recipient,
            status=result.get('status', 'unknown'),
            timestamp=result.get('timestamp', datetime.now().isoformat()),
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert send failed: {e}")
        raise HTTPException(status_code=500, detail=f"Alert send failed: {str(e)}")


@router.post("/simulate-crash", response_model=SimulateCrashResponse)
async def simulate_market_crash(request: SimulateCrashRequest):
    """
    Simulate market crash and optionally send alerts.
    
    **Crash Severity:**
    - **mild**: 15% drop (recovery: 7 days)
    - **moderate**: 30% drop (recovery: 30 days)
    - **severe**: 45% drop (recovery: 90 days)
    - **catastrophic**: 60% drop (recovery: 180 days)
    
    **Use Cases:**
    - Test alert system
    - Emergency preparedness
    - Downside scenario planning
    
    **Example:**
    ```json
    {
      "tickers": ["NVDA", "MSFT", "GOOGL"],
      "severity": "moderate",
      "notify": true,
      "notification_channel": "whatsapp",
      "recipient": "+919876543210"
    }
    ```
    """
    try:
        if not ALERT_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Alert service not available"
            )
        
        logger.info(f"Simulating {request.severity} crash for {len(request.tickers)} stocks")
        
        # Get current ratings (use mock data if ASRE service unavailable)
        original_ratings = {}
        
        if ASRE_SERVICE_AVAILABLE:
            asre_service = ASREService()
            for ticker in request.tickers:
                try:
                    rating_data = await asre_service.get_stock_rating(ticker)
                    original_ratings[ticker] = rating_data.get('rfinal', 70.0)
                except Exception as e:
                    logger.warning(f"Could not get rating for {ticker}: {e}")
                    original_ratings[ticker] = 70.0
        else:
            # Mock ratings
            for ticker in request.tickers:
                original_ratings[ticker] = 75.0
        
        # Map severity string to enum
        severity_map = {
            'mild': CrashSeverity.MILD,
            'moderate': CrashSeverity.MODERATE,
            'severe': CrashSeverity.SEVERE,
            'catastrophic': CrashSeverity.CATASTROPHIC
        }
        severity = severity_map[request.severity]
        
        # Get alert service
        alert_service = get_alert_service()
        
        # Simulate crash
        results = alert_service.simulate_market_crash(
            tickers=request.tickers,
            original_ratings=original_ratings,
            severity=severity,
            notify=request.notify,
            notification_channel=request.notification_channel,
            recipient=request.recipient
        )
        
        # Convert to response models
        simulations = [
            CrashSimulationResult(**result)
            for result in results
        ]
        
        notifications_sent = len(results) if request.notify and request.recipient else 0
        
        return SimulateCrashResponse(
            simulations=simulations,
            notifications_sent=notifications_sent,
            severity=request.severity,
            timestamp=datetime.now().isoformat(),
            message=f"Simulated {request.severity} crash for {len(request.tickers)} stocks. "
                   f"{'Alerts sent.' if notifications_sent > 0 else 'No alerts sent.'}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crash simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.post("/rating-change", response_model=SendAlertResponse)
async def send_rating_change_alert(request: RatingAlertRequest):
    """
    Send formatted rating change alert.
    
    **Auto-formatted message includes:**
    - Old → New rating
    - Change direction (📈/📉)
    - Signal
    - Context
    
    **Example:**
    ```json
    {
      "ticker": "NVDA",
      "old_rating": 75.0,
      "new_rating": 85.5,
      "signal": "BUY",
      "context": "Strong earnings beat",
      "channel": "whatsapp",
      "recipient": "+919876543210"
    }
    ```
    """
    try:
        if not ALERT_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Alert service not available")
        
        logger.info(f"Sending rating change alert for {request.ticker}")
        
        alert_service = get_alert_service()
        formatter = AlertFormatter()
        
        # Format message
        message = formatter.format_rating_change(
            ticker=request.ticker,
            old_rating=request.old_rating,
            new_rating=request.new_rating,
            signal=request.signal,
            context=request.context or ""
        )
        
        # Determine severity based on change magnitude
        change = abs(request.new_rating - request.old_rating)
        if change >= 20:
            severity = AlertSeverity.CRITICAL
        elif change >= 10:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        # Send alert
        result = alert_service.send_alert(
            channel=request.channel,
            recipient=request.recipient,
            message=message,
            severity=severity
        )
        
        return SendAlertResponse(
            success=result.get('success', False),
            message_id=result.get('message_id'),
            channel=request.channel,
            recipient=request.recipient,
            status=result.get('status', 'unknown'),
            timestamp=result.get('timestamp', datetime.now().isoformat()),
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rating change alert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Alert failed: {str(e)}")


@router.post("/momentum-trap", response_model=SendAlertResponse)
async def send_momentum_trap_alert(request: MomentumTrapAlertRequest):
    """
    Send formatted momentum trap warning alert.
    
    **Example:**
    ```json
    {
      "ticker": "COIN",
      "rating": 45.0,
      "warning": "Weak fundamentals despite strong price action",
      "action": "AVOID or REDUCE position",
      "channel": "whatsapp",
      "recipient": "+919876543210"
    }
    ```
    """
    try:
        if not ALERT_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Alert service not available")
        
        logger.info(f"Sending momentum trap alert for {request.ticker}")
        
        alert_service = get_alert_service()
        formatter = AlertFormatter()
        
        # Format message
        message = formatter.format_momentum_trap(
            ticker=request.ticker,
            rating=request.rating,
            warning=request.warning,
            action=request.action
        )
        
        # Send alert with critical severity
        result = alert_service.send_alert(
            channel=request.channel,
            recipient=request.recipient,
            message=message,
            severity=AlertSeverity.CRITICAL
        )
        
        return SendAlertResponse(
            success=result.get('success', False),
            message_id=result.get('message_id'),
            channel=request.channel,
            recipient=request.recipient,
            status=result.get('status', 'unknown'),
            timestamp=result.get('timestamp', datetime.now().isoformat()),
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Momentum trap alert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Alert failed: {str(e)}")


@router.post("/buy-opportunity", response_model=SendAlertResponse)
async def send_buy_opportunity_alert(request: BuyOpportunityAlertRequest):
    """
    Send formatted buy opportunity alert.
    
    **Example:**
    ```json
    {
      "ticker": "NVDA",
      "rating": 85.5,
      "signal": "STRONG_BUY",
      "category": "HIGH_QUALITY",
      "context": "Price dipped 8% below SMA-200",
      "channel": "whatsapp",
      "recipient": "+919876543210"
    }
    ```
    """
    try:
        if not ALERT_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Alert service not available")
        
        logger.info(f"Sending buy opportunity alert for {request.ticker}")
        
        alert_service = get_alert_service()
        formatter = AlertFormatter()
        
        # Format message
        message = formatter.format_buy_opportunity(
            ticker=request.ticker,
            rating=request.rating,
            signal=request.signal,
            category=request.category,
            context=request.context
        )
        
        # Send alert
        result = alert_service.send_alert(
            channel=request.channel,
            recipient=request.recipient,
            message=message,
            severity=AlertSeverity.WARNING  # Opportunities are warnings (action needed)
        )
        
        return SendAlertResponse(
            success=result.get('success', False),
            message_id=result.get('message_id'),
            channel=request.channel,
            recipient=request.recipient,
            status=result.get('status', 'unknown'),
            timestamp=result.get('timestamp', datetime.now().isoformat()),
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Buy opportunity alert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Alert failed: {str(e)}")


@router.get("/history", response_model=List[AlertHistoryResponse])
async def get_alert_history(
    recipient: Optional[str] = Query(None, description="Filter by recipient"),
    limit: int = Query(100, description="Maximum number of alerts to return")
):
    """
    Get alert history.
    
    **Example:** `/api/alerts/history?recipient=+919876543210&limit=50`
    """
    try:
        if not ALERT_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Alert service not available")
        
        alert_service = get_alert_service()
        
        history = alert_service.get_alert_history(
            recipient=recipient,
            limit=limit
        )
        
        return [AlertHistoryResponse(**entry) for entry in history]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/stats", response_model=AlertStatsResponse)
async def get_alert_stats():
    """
    Get alert statistics.
    
    **Returns:**
    - Total alerts sent
    - Success/failure counts
    - Breakdown by channel
    - Breakdown by severity
    - Recent alerts
    """
    try:
        if not ALERT_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Alert service not available")
        
        alert_service = get_alert_service()
        history = alert_service.get_alert_history(limit=1000)
        
        # Calculate stats
        total = len(history)
        successful = sum(1 for a in history if a['success'])
        failed = total - successful
        
        # By channel
        by_channel = {}
        for alert in history:
            channel = alert['channel']
            by_channel[channel] = by_channel.get(channel, 0) + 1
        
        # By severity
        by_severity = {}
        for alert in history:
            severity = alert['severity']
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Recent alerts (last 10)
        recent = [AlertHistoryResponse(**a) for a in history[-10:]]
        
        return AlertStatsResponse(
            total_alerts=total,
            successful_alerts=successful,
            failed_alerts=failed,
            alerts_by_channel=by_channel,
            alerts_by_severity=by_severity,
            recent_alerts=recent
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Check alert service health.
    
    **Returns:**
    - Service availability
    - Twilio status
    - Email status
    """
    try:
        status = {
            'alert_service_available': ALERT_SERVICE_AVAILABLE,
            'twilio_configured': False,
            'email_configured': False
        }
        
        if ALERT_SERVICE_AVAILABLE:
            alert_service = get_alert_service()
            status['twilio_configured'] = alert_service.whatsapp.client is not None
            status['email_configured'] = alert_service.email.smtp_enabled
        
        return {
            'status': 'healthy' if ALERT_SERVICE_AVAILABLE else 'degraded',
            'services': status,
            'message': 'Alert services operational' if ALERT_SERVICE_AVAILABLE else 'Alert services unavailable'
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'message': 'Health check failed'
        }


@router.post("/test-whatsapp")
async def test_whatsapp(
    recipient: str = Query(..., description="Phone number with country code (e.g., +919876543210)")
):
    """
    Send test WhatsApp message.
    
    **Example:** `/api/alerts/test-whatsapp?recipient=+919876543210`
    """
    try:
        if not ALERT_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Alert service not available")
        
        message = "🧪 Test Alert\n\nThis is a test message from ASRE Alert System.\n\nIf you received this, WhatsApp alerts are working!"
        
        result = send_whatsapp_alert(recipient, message, AlertSeverity.INFO)
        
        return {
            'success': result.get('success', False),
            'status': result.get('status'),
            'message_id': result.get('message_id'),
            'timestamp': result.get('timestamp')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")


@router.post("/test-sms")
async def test_sms(
    recipient: str = Query(..., description="Phone number with country code")
):
    """
    Send test SMS message.
    
    **Example:** `/api/alerts/test-sms?recipient=+919876543210`
    """
    try:
        if not ALERT_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Alert service not available")
        
        message = "Test Alert: ASRE SMS working!"
        
        result = send_sms_alert(recipient, message, AlertSeverity.INFO)
        
        return {
            'success': result.get('success', False),
            'status': result.get('status'),
            'message_id': result.get('message_id'),
            'timestamp': result.get('timestamp')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")
