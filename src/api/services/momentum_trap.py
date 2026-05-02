"""
Momentum Trap Detector
Identifies when stocks have strong momentum but weak fundamentals (pump risk).

A momentum trap occurs when:
- Technical score (T-Score) is high (overbought)
- Fundamental score (F-Score) is weak
- High divergence between momentum and fundamentals

This helps avoid buying overpriced stocks with weak underlying business.
"""

from typing import Dict, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class TrapSeverity(str, Enum):
    """Trap severity levels"""
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MarketRegion(str, Enum):
    """Market regions with different thresholds"""
    US = "US"
    INDIA = "INDIA"
    GLOBAL = "GLOBAL"


# Market-specific thresholds
THRESHOLDS = {
    MarketRegion.US: {
        "f_weak": 45,          # F-Score below this is weak
        "t_overbought": 70,    # T-Score above this is overbought
        "divergence_high": 30, # Difference threshold
        "m_excessive": 75,     # M-Score excessive momentum
    },
    MarketRegion.INDIA: {
        "f_weak": 40,          # More lenient for emerging markets
        "t_overbought": 75,    # Higher threshold
        "divergence_high": 35,
        "m_excessive": 80,
    },
    MarketRegion.GLOBAL: {
        "f_weak": 42,
        "t_overbought": 72,
        "divergence_high": 32,
        "m_excessive": 77,
    }
}


# ============================================================================
# MOMENTUM TRAP DETECTOR
# ============================================================================

class MomentumTrapDetector:
    """
    Advanced momentum trap detection with market-specific thresholds.
    
    Detects dangerous divergences between fundamentals and technicals
    that indicate overvaluation or pump-and-dump scenarios.
    """
    
    def __init__(self, market_region: MarketRegion = MarketRegion.US):
        """
        Initialize detector with market-specific thresholds.
        
        Args:
            market_region: Market region (US, INDIA, GLOBAL)
        """
        self.market_region = market_region
        self.thresholds = THRESHOLDS[market_region]
        logger.debug(f"Initialized MomentumTrapDetector for {market_region}")
    
    def detect_divergence(
        self,
        fscore: float,
        tscore: float,
        mscore: float
    ) -> bool:
        """
        Detect if momentum trap exists.
        
        Args:
            fscore: Fundamental score (0-100)
            tscore: Technical score (0-100)
            mscore: Momentum score (0-100)
            
        Returns:
            True if momentum trap detected
        """
        # Primary condition: Weak fundamentals + Strong technicals
        weak_fundamentals = fscore < self.thresholds["f_weak"]
        overbought = tscore > self.thresholds["t_overbought"]
        
        # Calculate divergence (T-Score higher than F-Score by significant margin)
        divergence = tscore - fscore
        high_divergence = divergence > self.thresholds["divergence_high"]
        
        # Secondary condition: Excessive momentum
        excessive_momentum = mscore > self.thresholds["m_excessive"]
        
        # Trap detected if:
        # 1. Weak fundamentals AND overbought technicals
        # 2. High divergence between T and F scores
        # 3. Optional: Excessive momentum adds to risk
        is_trap = (weak_fundamentals and overbought) or high_divergence
        
        if is_trap:
            logger.warning(
                f"Momentum trap detected: F={fscore:.1f}, T={tscore:.1f}, "
                f"M={mscore:.1f}, Divergence={divergence:.1f}"
            )
        
        return is_trap
    
    def calculate_trap_severity(
        self,
        fscore: float,
        tscore: float,
        mscore: Optional[float] = None
    ) -> Tuple[float, TrapSeverity]:
        """
        Calculate trap severity score (0-100).
        
        Args:
            fscore: Fundamental score (0-100)
            tscore: Technical score (0-100)
            mscore: Momentum score (0-100, optional)
            
        Returns:
            Tuple of (severity_score, severity_level)
        """
        # Base severity from divergence
        divergence = max(0, tscore - fscore)
        divergence_severity = min(100, (divergence / 50) * 100)  # Normalize to 0-100
        
        # Fundamental weakness factor
        f_weakness = max(0, self.thresholds["f_weak"] - fscore)
        f_weakness_severity = min(100, (f_weakness / self.thresholds["f_weak"]) * 100)
        
        # Technical overbought factor
        t_overbought = max(0, tscore - self.thresholds["t_overbought"])
        t_overbought_severity = min(100, (t_overbought / 30) * 100)
        
        # Momentum excess factor (if provided)
        m_severity = 0
        if mscore is not None:
            m_excess = max(0, mscore - self.thresholds["m_excessive"])
            m_severity = min(100, (m_excess / 25) * 100)
        
        # Weighted severity calculation
        severity_score = (
            divergence_severity * 0.40 +      # 40% weight on divergence
            f_weakness_severity * 0.30 +      # 30% weight on weak fundamentals
            t_overbought_severity * 0.20 +    # 20% weight on overbought
            m_severity * 0.10                 # 10% weight on momentum excess
        )
        
        # Determine severity level
        if severity_score >= 75:
            severity_level = TrapSeverity.CRITICAL
        elif severity_score >= 60:
            severity_level = TrapSeverity.HIGH
        elif severity_score >= 40:
            severity_level = TrapSeverity.MODERATE
        elif severity_score >= 20:
            severity_level = TrapSeverity.LOW
        else:
            severity_level = TrapSeverity.NONE
        
        logger.debug(
            f"Trap severity: {severity_score:.1f} ({severity_level}) - "
            f"F={fscore:.1f}, T={tscore:.1f}, M={mscore}"
        )
        
        return round(severity_score, 2), severity_level
    
    def get_trap_explanation(
        self,
        ticker: str,
        fscore: float,
        tscore: float,
        mscore: float,
        rfinal: Optional[float] = None
    ) -> Dict:
        """
        Generate human-readable trap warning with actionable insights.
        
        Args:
            ticker: Stock ticker symbol
            fscore: Fundamental score
            tscore: Technical score
            mscore: Momentum score
            rfinal: Final ASRE rating (optional)
            
        Returns:
            Dictionary with explanation, warnings, and recommendations
        """
        # Check if trap exists
        is_trap = self.detect_divergence(fscore, tscore, mscore)
        severity_score, severity_level = self.calculate_trap_severity(fscore, tscore, mscore)
        
        # Build explanation
        explanation = {
            "ticker": ticker,
            "is_trap": is_trap,
            "severity_score": severity_score,
            "severity_level": severity_level.value,
            "color_code": self._get_color_code(severity_level),
            "visual_indicator": self._get_visual_indicator(severity_level),
            "scores": {
                "fundamental": round(fscore, 1),
                "technical": round(tscore, 1),
                "momentum": round(mscore, 1),
                "divergence": round(tscore - fscore, 1)
            },
            "warning": self._generate_warning(ticker, severity_level, fscore, tscore),
            "explanation": self._generate_explanation(fscore, tscore, mscore, severity_level),
            "recommendations": self._generate_recommendations(severity_level, rfinal),
            "risk_factors": self._identify_risk_factors(fscore, tscore, mscore)
        }
        
        return explanation
    
    def _get_color_code(self, severity: TrapSeverity) -> str:
        """Get color code for severity level"""
        color_map = {
            TrapSeverity.NONE: "#22c55e",       # Green
            TrapSeverity.LOW: "#84cc16",        # Light green
            TrapSeverity.MODERATE: "#fbbf24",   # Yellow
            TrapSeverity.HIGH: "#f97316",       # Orange
            TrapSeverity.CRITICAL: "#ef4444",   # Red
        }
        return color_map[severity]
    
    def _get_visual_indicator(self, severity: TrapSeverity) -> str:
        """Get visual indicator emoji"""
        indicator_map = {
            TrapSeverity.NONE: "✅",
            TrapSeverity.LOW: "⚠️",
            TrapSeverity.MODERATE: "🟡",
            TrapSeverity.HIGH: "🟠",
            TrapSeverity.CRITICAL: "🚨",
        }
        return indicator_map[severity]
    
    def _generate_warning(
        self,
        ticker: str,
        severity: TrapSeverity,
        fscore: float,
        tscore: float
    ) -> str:
        """Generate appropriate warning message"""
        if severity == TrapSeverity.CRITICAL:
            return (
                f"🚨 CRITICAL WARNING: {ticker} shows severe momentum trap signals. "
                f"Stock price ({tscore:.0f}%) far exceeds fundamental value ({fscore:.0f}%). "
                f"HIGH RISK of sharp correction. Avoid buying or consider selling."
            )
        elif severity == TrapSeverity.HIGH:
            return (
                f"⚠️ HIGH RISK: {ticker} is significantly overbought relative to fundamentals. "
                f"Technical score ({tscore:.0f}%) much higher than fundamental score ({fscore:.0f}%). "
                f"Consider waiting for pullback."
            )
        elif severity == TrapSeverity.MODERATE:
            return (
                f"⚠️ CAUTION: {ticker} shows moderate divergence between price and fundamentals. "
                f"Exercise caution and consider smaller position sizes."
            )
        elif severity == TrapSeverity.LOW:
            return (
                f"ℹ️ WATCH: {ticker} shows minor divergence. Monitor for improvement in "
                f"fundamentals or technical pullback."
            )
        else:
            return f"✅ CLEAR: {ticker} shows healthy alignment between price and fundamentals."
    
    def _generate_explanation(
        self,
        fscore: float,
        tscore: float,
        mscore: float,
        severity: TrapSeverity
    ) -> str:
        """Generate detailed explanation"""
        divergence = tscore - fscore
        
        if severity in [TrapSeverity.CRITICAL, TrapSeverity.HIGH]:
            return (
                f"The stock is trading at technical levels ({tscore:.1f}) that are "
                f"{divergence:.1f} points higher than its fundamental value ({fscore:.1f}). "
                f"This large gap, combined with momentum score of {mscore:.1f}, suggests "
                f"the stock may be overvalued due to hype or speculation rather than "
                f"underlying business strength. This is a classic 'momentum trap' where "
                f"late buyers often get caught in a reversal."
            )
        elif severity == TrapSeverity.MODERATE:
            return (
                f"There's a {divergence:.1f}-point gap between technical ({tscore:.1f}) "
                f"and fundamental ({fscore:.1f}) scores. While not extreme, this divergence "
                f"suggests some overvaluation. The stock may still have upside, but risk "
                f"is elevated compared to fundamentally-driven rallies."
            )
        else:
            return (
                f"The stock shows healthy alignment between price ({tscore:.1f}) and "
                f"fundamentals ({fscore:.1f}). The {divergence:.1f}-point divergence "
                f"is within normal ranges, indicating the rally is supported by business fundamentals."
            )
    
    def _generate_recommendations(
        self,
        severity: TrapSeverity,
        rfinal: Optional[float] = None
    ) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        
        if severity == TrapSeverity.CRITICAL:
            recommendations.extend([
                "🛑 AVOID buying at current levels",
                "📉 If holding, consider taking profits or selling",
                "⏳ Wait for 20-30% pullback before reconsidering",
                "📊 Monitor for fundamental improvement (F-Score > 60)",
                "🔔 Set price alerts for technical breakdown"
            ])
        elif severity == TrapSeverity.HIGH:
            recommendations.extend([
                "⚠️ Avoid new positions or reduce position size by 50%",
                "📉 Consider selling if holding with profit",
                "⏳ Wait for pullback to stronger support levels",
                "📈 Look for fundamental catalysts before buying",
                "🎯 Set trailing stop-loss at -8% to -10%"
            ])
        elif severity == TrapSeverity.MODERATE:
            recommendations.extend([
                "⚠️ Reduce position size to 50-70% of normal allocation",
                "📊 Monitor for fundamental improvements",
                "🎯 Use tighter stop-loss orders (-5% to -7%)",
                "⏳ Consider waiting for better entry point",
                "💡 Look for confirmation from earnings reports"
            ])
        elif severity == TrapSeverity.LOW:
            recommendations.extend([
                "✅ Can consider buying with caution",
                "📊 Use standard position sizing",
                "🎯 Set stop-loss at -5%",
                "📈 Monitor for fundamental improvement",
                "💡 Ideal for short-term tactical positions"
            ])
        else:
            recommendations.extend([
                "✅ Healthy stock for investment",
                "📈 Suitable for normal position sizing",
                "💎 Good fundamentals support price action",
                "🎯 Standard risk management applies",
                "💡 Can hold for medium to long-term"
            ])
        
        # Add ASRE-specific recommendation if available
        if rfinal is not None:
            if rfinal >= 80:
                recommendations.append(f"🎯 ASRE rating ({rfinal:.1f}) confirms strong buy signal")
            elif rfinal >= 65:
                recommendations.append(f"📈 ASRE rating ({rfinal:.1f}) supports buy signal")
            elif rfinal < 50:
                recommendations.append(f"⚠️ ASRE rating ({rfinal:.1f}) suggests caution despite momentum")
        
        return recommendations
    
    def _identify_risk_factors(
        self,
        fscore: float,
        tscore: float,
        mscore: float
    ) -> list:
        """Identify specific risk factors"""
        risks = []
        
        divergence = tscore - fscore
        
        # Divergence risks
        if divergence > 40:
            risks.append("Extreme divergence between price and fundamentals (40+ points)")
        elif divergence > 30:
            risks.append("High divergence between price and fundamentals (30+ points)")
        
        # Fundamental risks
        if fscore < 30:
            risks.append(f"Very weak fundamentals (F-Score: {fscore:.0f}/100)")
        elif fscore < 45:
            risks.append(f"Below-average fundamentals (F-Score: {fscore:.0f}/100)")
        
        # Technical risks
        if tscore > 85:
            risks.append(f"Severely overbought (T-Score: {tscore:.0f}/100)")
        elif tscore > 70:
            risks.append(f"Overbought conditions (T-Score: {tscore:.0f}/100)")
        
        # Momentum risks
        if mscore > 85:
            risks.append(f"Excessive momentum (M-Score: {mscore:.0f}/100)")
        elif mscore > 75:
            risks.append(f"High momentum may not be sustainable (M-Score: {mscore:.0f}/100)")
        
        # Combined risks
        if fscore < 40 and tscore > 75:
            risks.append("Weak company with inflated stock price - high reversal risk")
        
        if not risks:
            risks.append("No significant risk factors identified")
        
        return risks


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def detect_momentum_trap(
    fscore: float,
    tscore: float,
    mscore: float,
    market: MarketRegion = MarketRegion.US
) -> bool:
    """
    Quick trap detection.
    
    Args:
        fscore: Fundamental score (0-100)
        tscore: Technical score (0-100)
        mscore: Momentum score (0-100)
        market: Market region
        
    Returns:
        True if momentum trap detected
    """
    detector = MomentumTrapDetector(market)
    return detector.detect_divergence(fscore, tscore, mscore)


def get_trap_severity(
    fscore: float,
    tscore: float,
    mscore: float,
    market: MarketRegion = MarketRegion.US
) -> Tuple[float, str]:
    """
    Quick severity calculation.
    
    Returns:
        Tuple of (severity_score, severity_level_name)
    """
    detector = MomentumTrapDetector(market)
    score, level = detector.calculate_trap_severity(fscore, tscore, mscore)
    return score, level.value


def analyze_trap_risk(
    ticker: str,
    fscore: float,
    tscore: float,
    mscore: float,
    rfinal: Optional[float] = None,
    market: MarketRegion = MarketRegion.US
) -> Dict:
    """
    Complete trap analysis with recommendations.
    
    Returns:
        Dictionary with full trap analysis
    """
    detector = MomentumTrapDetector(market)
    return detector.get_trap_explanation(ticker, fscore, tscore, mscore, rfinal)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MomentumTrapDetector',
    'TrapSeverity',
    'MarketRegion',
    'detect_momentum_trap',
    'get_trap_severity',
    'analyze_trap_risk',
]
