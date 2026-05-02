"""
Portfolio Analyzer Service
Analyzes portfolio health, risk distribution, and generates recommendations.

Features:
- Overall portfolio health scoring
- Risk categorization (Safe/Moderate/High-risk)
- High-risk stock identification
- Potential loss estimation
- Actionable recommendations
- Rebalancing suggestions
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
from datetime import datetime

from api.services.asre_service import ASREService

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class RiskCategory(str, Enum):
    """Portfolio risk categories"""
    SAFE = "safe"
    MODERATE = "moderate"
    HIGH_RISK = "high_risk"


class PortfolioHealth(str, Enum):
    """Overall portfolio health levels"""
    EXCELLENT = "EXCELLENT"      # 80-100
    GOOD = "GOOD"               # 65-79
    FAIR = "FAIR"               # 50-64
    POOR = "POOR"               # 35-49
    CRITICAL = "CRITICAL"       # 0-34


# Risk thresholds for categorization
RISK_THRESHOLDS = {
    RiskCategory.SAFE: (65, 100),        # ASRE rating 65-100
    RiskCategory.MODERATE: (45, 64),     # ASRE rating 45-64
    RiskCategory.HIGH_RISK: (0, 44),     # ASRE rating 0-44
}

# Loss estimation factors
LOSS_FACTORS = {
    "critical": 0.30,    # 30% potential loss for rating < 35
    "high": 0.20,        # 20% potential loss for rating 35-44
    "moderate": 0.10,    # 10% potential loss for rating 45-54
    "low": 0.05,         # 5% potential loss for rating 55-64
}


# ============================================================================
# PORTFOLIO ANALYZER
# ============================================================================

class PortfolioAnalyzer:
    """
    Comprehensive portfolio analysis with risk assessment and recommendations.
    
    Analyzes portfolio holdings to provide:
    - Weighted health score
    - Risk distribution
    - High-risk stock identification
    - Loss estimation
    - Rebalancing suggestions
    """
    
    def __init__(self):
        """Initialize portfolio analyzer"""
        self.asre_service = ASREService
        logger.debug("Initialized PortfolioAnalyzer")
    
    def analyze_portfolio(
        self,
        holdings: List[Dict],
        currency: str = "USD"
    ) -> Dict:
        """
        Comprehensive portfolio analysis.
        
        Args:
            holdings: List of holdings with ticker, shares, value
            currency: Portfolio currency (USD, INR, etc.)
            
        Returns:
            Complete portfolio analysis with scores and recommendations
        """
        logger.info(f"Analyzing portfolio with {len(holdings)} holdings")
        
        # Calculate total portfolio value
        total_value = sum(h.get('value', 0) for h in holdings)
        
        if total_value <= 0:
            raise ValueError("Total portfolio value must be positive")
        
        # Get ASRE ratings for all holdings
        enriched_holdings = self._enrich_holdings_with_ratings(holdings)
        
        # Calculate overall health score (weighted by value)
        overall_score = self._calculate_weighted_score(enriched_holdings, total_value)
        
        # Calculate risk breakdown
        risk_breakdown = self.calculate_risk_breakdown(enriched_holdings, total_value)
        
        # Identify high-risk stocks
        high_risk_stocks = self.identify_high_risk_stocks(enriched_holdings)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            enriched_holdings,
            overall_score,
            risk_breakdown
        )
        
        # Suggest rebalancing
        rebalancing = self.suggest_rebalancing(enriched_holdings, total_value)
        
        # Determine health level
        health_level = self._determine_health_level(overall_score)
        
        return {
            "overall_score": round(overall_score, 2),
            "health_level": health_level.value,
            "total_value": round(total_value, 2),
            "currency": currency,
            "holdings_count": len(holdings),
            "risk_breakdown": risk_breakdown,
            "high_risk_stocks": high_risk_stocks,
            "recommendations": recommendations,
            "rebalancing": rebalancing,
            "timestamp": datetime.now().isoformat(),
            "portfolio_composition": self._get_composition_summary(
                enriched_holdings,
                total_value
            )
        }
    
    def calculate_risk_breakdown(
        self,
        enriched_holdings: List[Dict],
        total_value: float
    ) -> Dict:
        """
        Calculate risk distribution across portfolio.
        
        Args:
            enriched_holdings: Holdings with ASRE ratings
            total_value: Total portfolio value
            
        Returns:
            Risk breakdown by category
        """
        breakdown = {
            RiskCategory.SAFE.value: {"count": 0, "value": 0, "percentage": 0},
            RiskCategory.MODERATE.value: {"count": 0, "value": 0, "percentage": 0},
            RiskCategory.HIGH_RISK.value: {"count": 0, "value": 0, "percentage": 0},
        }
        
        for holding in enriched_holdings:
            rating = holding.get('rating', {}).get('rfinal', 50)
            value = holding.get('value', 0)
            category = self._categorize_risk(rating)
            
            breakdown[category.value]["count"] += 1
            breakdown[category.value]["value"] += value
        
        # Calculate percentages
        for category in breakdown.values():
            category["percentage"] = round(
                (category["value"] / total_value * 100) if total_value > 0 else 0,
                2
            )
            category["value"] = round(category["value"], 2)
        
        logger.debug(f"Risk breakdown: {breakdown}")
        return breakdown
    
    def identify_high_risk_stocks(
        self,
        enriched_holdings: List[Dict]
    ) -> List[Dict]:
        """
        Identify stocks with rating < 45 (high risk).
        
        Args:
            enriched_holdings: Holdings with ASRE ratings
            
        Returns:
            List of high-risk stocks with details
        """
        high_risk = []
        
        for holding in enriched_holdings:
            rating_data = holding.get('rating', {})
            rating = rating_data.get('rfinal', 50)
            
            if rating < 45:  # High-risk threshold
                ticker = holding.get('ticker')
                value = holding.get('value', 0)
                shares = holding.get('shares', 0)
                
                # Calculate potential loss
                potential_loss = self.calculate_potential_loss(value, rating)
                
                # Determine action
                if rating < 35:
                    action = "SELL IMMEDIATELY"
                    urgency = "CRITICAL"
                elif rating < 40:
                    action = "SELL SOON"
                    urgency = "HIGH"
                else:
                    action = "CONSIDER SELLING"
                    urgency = "MODERATE"
                
                high_risk.append({
                    "ticker": ticker,
                    "rating": round(rating, 1),
                    "value": round(value, 2),
                    "shares": shares,
                    "potential_loss": round(potential_loss, 2),
                    "action": action,
                    "urgency": urgency,
                    "signal": rating_data.get('signal', 'UNKNOWN')
                })
        
        # Sort by rating (worst first)
        high_risk.sort(key=lambda x: x['rating'])
        
        logger.info(f"Identified {len(high_risk)} high-risk stocks")
        return high_risk
    
    def generate_recommendations(
        self,
        enriched_holdings: List[Dict],
        overall_score: float,
        risk_breakdown: Dict
    ) -> List[str]:
        """
        Generate actionable portfolio recommendations.
        
        Args:
            enriched_holdings: Holdings with ratings
            overall_score: Portfolio health score
            risk_breakdown: Risk distribution
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Overall health recommendations
        if overall_score >= 80:
            recommendations.append(
                "✅ Excellent portfolio health! Continue monitoring for any changes."
            )
        elif overall_score >= 65:
            recommendations.append(
                "📈 Good portfolio. Consider trimming high-risk positions to improve further."
            )
        elif overall_score >= 50:
            recommendations.append(
                "⚠️ Fair portfolio. Rebalancing recommended to reduce risk exposure."
            )
        else:
            recommendations.append(
                "🚨 Poor portfolio health. Immediate action required to minimize losses."
            )
        
        # High-risk exposure recommendations
        high_risk_pct = risk_breakdown[RiskCategory.HIGH_RISK.value]["percentage"]
        if high_risk_pct > 30:
            recommendations.append(
                f"🔴 {high_risk_pct:.1f}% in high-risk stocks. Reduce to <20% for better stability."
            )
        elif high_risk_pct > 20:
            recommendations.append(
                f"⚠️ {high_risk_pct:.1f}% in high-risk stocks. Consider reducing to <15%."
            )
        
        # Specific stock recommendations
        for holding in enriched_holdings:
            ticker = holding.get('ticker')
            rating_data = holding.get('rating', {})
            rating = rating_data.get('rfinal', 50)
            value = holding.get('value', 0)
            
            if rating < 35:
                potential_loss = self.calculate_potential_loss(value, rating)
                recommendations.append(
                    f"🔴 SELL {ticker} immediately (Rating: {rating:.1f}) - "
                    f"Could save ${potential_loss:,.0f}"
                )
            elif rating >= 80 and rating_data.get('context', '').find('DIP') >= 0:
                recommendations.append(
                    f"💎 {ticker} is a quality dip (Rating: {rating:.1f}) - "
                    f"Consider adding more"
                )
        
        # Diversification recommendations
        if len(enriched_holdings) < 5:
            recommendations.append(
                "📊 Consider adding 2-3 more stocks for better diversification."
            )
        elif len(enriched_holdings) > 15:
            recommendations.append(
                "📊 Portfolio may be over-diversified. Focus on top 10-12 positions."
            )
        
        # Safe allocation recommendations
        safe_pct = risk_breakdown[RiskCategory.SAFE.value]["percentage"]
        if safe_pct < 50:
            recommendations.append(
                f"💡 Only {safe_pct:.1f}% in safe stocks. Target 50-60% for stability."
            )
        
        logger.debug(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def calculate_potential_loss(
        self,
        current_value: float,
        rating: float
    ) -> float:
        """
        Estimate potential loss if high-risk stock not sold.
        
        Args:
            current_value: Current holding value
            rating: ASRE rating
            
        Returns:
            Estimated potential loss amount
        """
        if rating < 35:
            loss_factor = LOSS_FACTORS["critical"]
        elif rating < 45:
            loss_factor = LOSS_FACTORS["high"]
        elif rating < 55:
            loss_factor = LOSS_FACTORS["moderate"]
        elif rating < 65:
            loss_factor = LOSS_FACTORS["low"]
        else:
            loss_factor = 0  # No significant loss expected
        
        potential_loss = current_value * loss_factor
        
        logger.debug(
            f"Potential loss: ${potential_loss:.2f} "
            f"(Value: ${current_value:.2f}, Rating: {rating:.1f})"
        )
        
        return potential_loss
    
    def suggest_rebalancing(
        self,
        enriched_holdings: List[Dict],
        total_value: float
    ) -> Dict:
        """
        Suggest optimal portfolio rebalancing.
        
        Args:
            enriched_holdings: Holdings with ratings
            total_value: Total portfolio value
            
        Returns:
            Rebalancing suggestions
        """
        # Target allocation: 60% safe, 30% moderate, 10% high-risk
        target_allocation = {
            RiskCategory.SAFE.value: 0.60,
            RiskCategory.MODERATE.value: 0.30,
            RiskCategory.HIGH_RISK.value: 0.10,
        }
        
        # Calculate current allocation
        current_allocation = {}
        for category in RiskCategory:
            category_value = sum(
                h.get('value', 0)
                for h in enriched_holdings
                if self._categorize_risk(h.get('rating', {}).get('rfinal', 50)) == category
            )
            current_allocation[category.value] = category_value / total_value if total_value > 0 else 0
        
        # Calculate rebalancing needs
        rebalancing_actions = []
        
        for category, target_pct in target_allocation.items():
            current_pct = current_allocation.get(category, 0)
            diff_pct = target_pct - current_pct
            diff_value = diff_pct * total_value
            
            if abs(diff_pct) > 0.05:  # 5% threshold
                if diff_value > 0:
                    action = f"Increase {category} allocation by ${abs(diff_value):,.0f} ({abs(diff_pct)*100:.1f}%)"
                else:
                    action = f"Decrease {category} allocation by ${abs(diff_value):,.0f} ({abs(diff_pct)*100:.1f}%)"
                
                rebalancing_actions.append(action)
        
        # Identify best stocks to add
        best_to_add = []
        for holding in enriched_holdings:
            rating = holding.get('rating', {}).get('rfinal', 0)
            if rating >= 75:
                best_to_add.append({
                    "ticker": holding.get('ticker'),
                    "rating": round(rating, 1),
                    "current_value": round(holding.get('value', 0), 2)
                })
        
        best_to_add.sort(key=lambda x: x['rating'], reverse=True)
        
        # Identify stocks to reduce/remove
        to_reduce = []
        for holding in enriched_holdings:
            rating = holding.get('rating', {}).get('rfinal', 0)
            if rating < 50:
                to_reduce.append({
                    "ticker": holding.get('ticker'),
                    "rating": round(rating, 1),
                    "current_value": round(holding.get('value', 0), 2)
                })
        
        to_reduce.sort(key=lambda x: x['rating'])
        
        return {
            "current_allocation": {
                k: round(v * 100, 1) for k, v in current_allocation.items()
            },
            "target_allocation": {
                k: round(v * 100, 1) for k, v in target_allocation.items()
            },
            "actions": rebalancing_actions,
            "best_to_add": best_to_add[:5],  # Top 5
            "to_reduce": to_reduce[:5],  # Worst 5
            "rebalancing_needed": len(rebalancing_actions) > 0
        }
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _enrich_holdings_with_ratings(
        self,
        holdings: List[Dict]
    ) -> List[Dict]:
        """Fetch ASRE ratings for all holdings"""
        enriched = []
        
        for holding in holdings:
            ticker = holding.get('ticker')
            if not ticker:
                logger.warning("Holding without ticker, skipping")
                continue
            
            try:
                rating = self.asre_service.get_stock_rating(ticker)
                enriched.append({
                    **holding,
                    'rating': rating
                })
            except Exception as e:
                logger.error(f"Failed to get rating for {ticker}: {e}")
                # Add with default rating
                enriched.append({
                    **holding,
                    'rating': {'rfinal': 50, 'signal': 'UNKNOWN'}
                })
        
        return enriched
    
    def _calculate_weighted_score(
        self,
        enriched_holdings: List[Dict],
        total_value: float
    ) -> float:
        """Calculate portfolio-weighted ASRE score"""
        if total_value <= 0:
            return 0
        
        weighted_sum = sum(
            h.get('rating', {}).get('rfinal', 50) * h.get('value', 0)
            for h in enriched_holdings
        )
        
        return weighted_sum / total_value if total_value > 0 else 0
    
    def _categorize_risk(self, rating: float) -> RiskCategory:
        """Categorize risk based on ASRE rating"""
        for category, (min_rating, max_rating) in RISK_THRESHOLDS.items():
            if min_rating <= rating <= max_rating:
                return category
        return RiskCategory.MODERATE  # Default
    
    def _determine_health_level(self, score: float) -> PortfolioHealth:
        """Determine health level from score"""
        if score >= 80:
            return PortfolioHealth.EXCELLENT
        elif score >= 65:
            return PortfolioHealth.GOOD
        elif score >= 50:
            return PortfolioHealth.FAIR
        elif score >= 35:
            return PortfolioHealth.POOR
        else:
            return PortfolioHealth.CRITICAL
    
    def _get_composition_summary(
        self,
        enriched_holdings: List[Dict],
        total_value: float
    ) -> Dict:
        """Get portfolio composition summary"""
        # Top holdings
        holdings_with_pct = [
            {
                "ticker": h.get('ticker'),
                "value": round(h.get('value', 0), 2),
                "percentage": round(h.get('value', 0) / total_value * 100, 2),
                "rating": round(h.get('rating', {}).get('rfinal', 0), 1)
            }
            for h in enriched_holdings
        ]
        holdings_with_pct.sort(key=lambda x: x['value'], reverse=True)
        
        # Average rating
        avg_rating = sum(
            h.get('rating', {}).get('rfinal', 0) for h in enriched_holdings
        ) / len(enriched_holdings) if enriched_holdings else 0
        
        return {
            "top_holdings": holdings_with_pct[:5],  # Top 5 by value
            "average_rating": round(avg_rating, 2),
            "concentration_risk": holdings_with_pct[0]['percentage'] if holdings_with_pct else 0
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_portfolio(holdings: List[Dict], currency: str = "USD") -> Dict:
    """Quick portfolio analysis"""
    analyzer = PortfolioAnalyzer()
    return analyzer.analyze_portfolio(holdings, currency)


def get_high_risk_stocks(holdings: List[Dict]) -> List[Dict]:
    """Quick high-risk identification"""
    analyzer = PortfolioAnalyzer()
    enriched = analyzer._enrich_holdings_with_ratings(holdings)
    return analyzer.identify_high_risk_stocks(enriched)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PortfolioAnalyzer',
    'RiskCategory',
    'PortfolioHealth',
    'analyze_portfolio',
    'get_high_risk_stocks',
]
