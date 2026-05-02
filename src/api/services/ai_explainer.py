"""
AI Explainer Service
Uses Groq LLM to generate natural language explanations for ASRE ratings.

Provides:
- Stock rating explanations in simple language
- Momentum trap warnings
- Conversational Q&A about stocks
- Comparison explanations

Uses optimized prompts for retail investors with fallback responses.
"""

from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from api.config import settings
from api.services.momentum_trap import (
    analyze_trap_risk,
)

logger = logging.getLogger(__name__)


# ============================================================================
# AI CLIENT WRAPPER
# ============================================================================

class GroqAIClient:
    """Wrapper for Groq API with error handling and token optimization"""
    
    def __init__(self):
        """Initialize Groq client"""
        if not GROQ_AVAILABLE:
            logger.warning("Groq library not available. Install with: pip install groq")
            self.client = None
            return
        
        if not settings.GROQ_API_KEY:
            logger.warning("Groq API key not configured")
            self.client = None
            return
        
        try:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            self.model = settings.GROQ_MODEL
            logger.info(f"Groq AI client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate AI response.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum response length
            temperature: Creativity (0.0-1.0)
            system_prompt: System context
            
        Returns:
            Generated text or None if failed
        """
        if not self.client:
            logger.warning("Groq client not available")
            return None
        
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            result = response.choices[0].message.content
            
            # Log token usage
            if hasattr(response, 'usage'):
                logger.debug(
                    f"Groq API usage - Prompt: {response.usage.prompt_tokens}, "
                    f"Completion: {response.usage.completion_tokens}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return None


# ============================================================================
# AI EXPLAINER SERVICE
# ============================================================================

class AIExplainer:
    """
    AI-powered explanations for ASRE ratings and market insights.
    
    Uses optimized prompts for retail investors with:
    - Simple, jargon-free language
    - Actionable insights
    - Risk warnings
    - Comparison logic
    """
    
    # System prompt for retail investor context
    SYSTEM_PROMPT = """You are a friendly, knowledgeable stock market advisor helping retail investors understand ASRE (Advanced Stock Rating Engine) ratings. 

Your communication style:
- Use simple, clear language (avoid jargon)
- Be concise but informative (2-3 short paragraphs max)
- Focus on actionable insights
- Include specific numbers and percentages
- Always mention risks and limitations
- Use analogies when helpful
- Be honest about uncertainties

ASRE Rating System (0-100):
- 80-100: STRONG BUY (Exceptional opportunity)
- 65-79: BUY (Good opportunity)
- 50-64: HOLD (Neutral, wait and watch)
- 35-49: CAUTION (Warning signs, be careful)
- 0-34: STRONG SELL (High risk, avoid)

Components:
- F-Score (Fundamentals): Business health (revenue, profit, growth)
- T-Score (Technical): Price trends and patterns
- M-Score (Momentum): Market sentiment and buying pressure

Never give financial advice like "you should buy/sell". Instead say "the data suggests" or "investors might consider"."""
    
    def __init__(self):
        """Initialize AI explainer with Groq client"""
        self.ai_client = GroqAIClient()
    
    def explain_stock_rating(
        self,
        ticker: str,
        asre_data: Dict,
        include_trap_analysis: bool = True
    ) -> Dict:
        """
        Generate natural language explanation of ASRE rating.
        
        Args:
            ticker: Stock ticker symbol
            asre_data: ASRE rating data from ASREService
            include_trap_analysis: Include momentum trap analysis
            
        Returns:
            Dictionary with explanation and metadata
        """
        logger.info(f"Generating explanation for {ticker}")
        
        # Extract key data
        rfinal = asre_data.get('rfinal', 0)
        fscore = asre_data.get('fscore', 0)
        tscore = asre_data.get('tscore', 0)
        mscore = asre_data.get('mscore', 0)
        signal = asre_data.get('signal', 'UNKNOWN')
        category = asre_data.get('category', 'UNKNOWN')
        dip_quality = asre_data.get('dip_quality')
        dip_stage = asre_data.get('dip_stage')
        
        # Build prompt
        prompt = self._build_rating_prompt(
            ticker, rfinal, fscore, tscore, mscore,
            signal, category, dip_quality, dip_stage
        )
        
        # Generate AI explanation
        ai_explanation = self.ai_client.generate(
            prompt=prompt,
            max_tokens=400,
            temperature=0.7,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Fallback if AI fails
        if not ai_explanation:
            ai_explanation = self._generate_fallback_explanation(
                ticker, rfinal, fscore, tscore, mscore, signal, category
            )
        
        # Add momentum trap analysis if requested
        trap_analysis = None
        if include_trap_analysis:
            trap_analysis = analyze_trap_risk(
                ticker, fscore, tscore, mscore, rfinal
            )
        
        return {
            "ticker": ticker,
            "explanation": ai_explanation,
            "rating": rfinal,
            "signal": signal,
            "category": category,
            "scores": {
                "fundamental": fscore,
                "technical": tscore,
                "momentum": mscore
            },
            "trap_analysis": trap_analysis,
            "timestamp": datetime.now().isoformat(),
            "ai_generated": ai_explanation is not None
        }
    
    def explain_momentum_trap(
        self,
        ticker: str,
        fscore: float,
        tscore: float,
        mscore: float
    ) -> Dict:
        """
        Generate specific momentum trap warning.
        
        Args:
            ticker: Stock ticker
            fscore: Fundamental score
            tscore: Technical score
            mscore: Momentum score
            
        Returns:
            Dictionary with trap explanation and warnings
        """
        logger.info(f"Analyzing momentum trap for {ticker}")
        
        # Get trap analysis
        trap_data = analyze_trap_risk(ticker, fscore, tscore, mscore)
        
        # If no trap, return simple message
        if not trap_data['is_trap']:
            return {
                "ticker": ticker,
                "is_trap": False,
                "severity": "NONE",
                "warning": f"✅ {ticker} shows healthy alignment between fundamentals and price. No momentum trap detected.",
                "explanation": "The stock's technical performance is supported by its fundamental strength.",
                "recommendations": ["Normal investment considerations apply"]
            }
        
        # Build prompt for AI trap explanation
        prompt = f"""Explain this momentum trap situation for {ticker}:

Fundamental Score: {fscore:.1f}/100
Technical Score: {tscore:.1f}/100
Momentum Score: {mscore:.1f}/100
Severity: {trap_data['severity_level']}

The stock is technically overbought but has weak fundamentals. Explain WHY this is risky in 2-3 sentences for a beginner investor."""
        
        # Generate AI explanation
        ai_trap_explanation = self.ai_client.generate(
            prompt=prompt,
            max_tokens=250,
            temperature=0.6,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Use fallback if AI fails
        if not ai_trap_explanation:
            ai_trap_explanation = trap_data['explanation']
        
        return {
            "ticker": ticker,
            "is_trap": True,
            "severity": trap_data['severity_level'],
            "severity_score": trap_data['severity_score'],
            "warning": trap_data['warning'],
            "explanation": ai_trap_explanation,
            "recommendations": trap_data['recommendations'][:5],  # Top 5
            "risk_factors": trap_data['risk_factors'][:3],  # Top 3
            "visual_indicator": trap_data['visual_indicator'],
            "color_code": trap_data['color_code']
        }
    
    def chat_with_ai(
        self,
        message: str,
        context: Optional[Dict] = None,
        ticker: Optional[str] = None
    ) -> Dict:
        """
        Conversational Q&A about stocks.
        
        Args:
            message: User's question
            context: Previous conversation context
            ticker: Stock ticker for context
            
        Returns:
            Dictionary with AI response
        """
        logger.info(f"Processing chat message: {message[:50]}...")
        
        # Build context-aware prompt
        prompt_parts = []
        
        if ticker:
            prompt_parts.append(f"Question about {ticker}:")
        
        prompt_parts.append(message)
        
        if context:
            if 'previous_rating' in context:
                prompt_parts.append(
                    f"\nContext: ASRE rating is {context['previous_rating']:.1f}/100"
                )
            if 'previous_signal' in context:
                prompt_parts.append(f"Signal: {context['previous_signal']}")
        
        prompt = "\n".join(prompt_parts)
        
        # Generate response
        ai_response = self.ai_client.generate(
            prompt=prompt,
            max_tokens=350,
            temperature=0.7,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Fallback response
        if not ai_response:
            ai_response = self._generate_chat_fallback(message, ticker)
        
        return {
            "message": ai_response,
            "ticker": ticker,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "suggestions": self._generate_followup_suggestions(message, ticker)
        }
    
    def explain_comparison(
        self,
        stock_list: List[Dict],
        focus: str = "rating"
    ) -> Dict:
        """
        Explain why one stock ranks higher than another.
        
        Args:
            stock_list: List of stock ratings (sorted by rank)
            focus: What to focus on ('rating', 'fundamentals', 'risk')
            
        Returns:
            Dictionary with comparison explanation
        """
        logger.info(f"Generating comparison for {len(stock_list)} stocks")
        
        if len(stock_list) < 2:
            return {
                "explanation": "Need at least 2 stocks to compare.",
                "comparison": []
            }
        
        # Build comparison prompt
        top_stock = stock_list[0]
        comparison_data = []
        
        for i, stock in enumerate(stock_list[:5], 1):  # Top 5
            comparison_data.append(
                f"{i}. {stock['ticker']}: Rating {stock.get('rfinal', 0):.1f}/100 "
                f"(F:{stock.get('fscore', 0):.0f}, T:{stock.get('tscore', 0):.0f}, "
                f"M:{stock.get('mscore', 0):.0f})"
            )
        
        prompt = f"""Compare these stocks and explain why {top_stock['ticker']} ranks #1:

{chr(10).join(comparison_data)}

Focus on: {focus}

Explain in 2-3 sentences why the top stock is ranked higher. Use specific scores."""
        
        # Generate AI explanation
        ai_comparison = self.ai_client.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.6,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        # Fallback
        if not ai_comparison:
            ai_comparison = self._generate_comparison_fallback(stock_list)
        
        return {
            "explanation": ai_comparison,
            "top_pick": {
                "ticker": top_stock['ticker'],
                "rating": top_stock.get('rfinal', 0),
                "signal": top_stock.get('signal', 'UNKNOWN')
            },
            "comparison": comparison_data,
            "timestamp": datetime.now().isoformat()
        }
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _build_rating_prompt(
        self,
        ticker: str,
        rfinal: float,
        fscore: float,
        tscore: float,
        mscore: float,
        signal: str,
        category: str,
        dip_quality: Optional[float],
        dip_stage: Optional[str]
    ) -> str:
        """Build optimized prompt for rating explanation"""
        
        prompt = f"""Explain this ASRE rating for {ticker}:

Final Rating: {rfinal:.1f}/100 - {signal}
Category: {category}

Component Scores:
- Fundamental (F-Score): {fscore:.1f}/100
- Technical (T-Score): {tscore:.1f}/100
- Momentum (M-Score): {mscore:.1f}/100"""
        
        if dip_quality and dip_quality > 60:
            prompt += f"\n- Dip Quality: {dip_quality:.1f}/100 ({dip_stage} stage)"
        
        prompt += "\n\nExplain WHY this rating in 2-3 short paragraphs for a beginner investor. Include what the scores mean and whether it's a good buying opportunity."
        
        return prompt
    
    def _generate_fallback_explanation(
        self,
        ticker: str,
        rfinal: float,
        fscore: float,
        tscore: float,
        mscore: float,
        signal: str,
        category: str
    ) -> str:
        """Generate fallback explanation when AI fails"""
        
        # Determine strength
        if rfinal >= 80:
            strength = "exceptionally strong"
            outlook = "This is a high-confidence buying opportunity backed by solid fundamentals."
        elif rfinal >= 65:
            strength = "good"
            outlook = "This represents a solid investment opportunity with positive momentum."
        elif rfinal >= 50:
            strength = "neutral"
            outlook = "Consider waiting for stronger signals before investing."
        else:
            strength = "weak"
            outlook = "Caution advised - multiple risk factors present."
        
        # Build explanation
        explanation = f"""{ticker} receives an ASRE rating of {rfinal:.1f}/100, indicating a {strength} investment profile with a {signal} signal.

The rating is driven by a Fundamental Score of {fscore:.0f}/100, Technical Score of {tscore:.0f}/100, and Momentum Score of {mscore:.0f}/100. The stock falls into the {category} category based on its overall characteristics.

{outlook} As with any investment, consider your risk tolerance and investment timeline before making decisions."""
        
        return explanation
    
    def _generate_chat_fallback(
        self,
        message: str,
        ticker: Optional[str]
    ) -> str:
        """Generate fallback chat response"""
        
        message_lower = message.lower()
        
        # Pattern matching for common questions
        if any(word in message_lower for word in ['buy', 'should i', 'invest']):
            return (
                f"Based on ASRE data, {'the stock' if not ticker else ticker} shows "
                f"specific characteristics that investors should consider. I recommend "
                f"reviewing the F-Score (fundamentals), T-Score (technicals), and "
                f"M-Score (momentum) to make an informed decision. Remember, no single "
                f"rating should drive investment decisions - always do your own research."
            )
        
        elif any(word in message_lower for word in ['risk', 'safe', 'risky']):
            return (
                f"Risk assessment involves multiple factors including fundamental strength, "
                f"technical momentum, and market conditions. Check the momentum trap "
                f"analysis for specific risk warnings. Generally, higher F-Scores (>70) "
                f"indicate lower fundamental risk."
            )
        
        elif any(word in message_lower for word in ['rating', 'score', 'asre']):
            return (
                f"ASRE ratings combine three key metrics: Fundamentals (business health), "
                f"Technicals (price trends), and Momentum (buying pressure). Scores above "
                f"80 are exceptional, 65-80 are good buys, 50-65 are holds, and below 50 "
                f"warrant caution. Each component tells a different part of the story."
            )
        
        else:
            return (
                f"I can help explain ASRE ratings, analyze stocks, or answer questions "
                f"about fundamentals, technicals, and momentum. Try asking about specific "
                f"scores or what makes a stock a good investment."
            )
    
    def _generate_comparison_fallback(self, stock_list: List[Dict]) -> str:
        """Generate fallback comparison explanation"""
        
        if len(stock_list) < 2:
            return "Need at least 2 stocks to compare."
        
        top = stock_list[0]
        runner_up = stock_list[1] if len(stock_list) > 1 else None
        
        explanation = (
            f"{top['ticker']} ranks highest with a rating of {top.get('rfinal', 0):.1f}/100, "
            f"driven by strong fundamentals (F:{top.get('fscore', 0):.0f}) and positive momentum "
            f"(M:{top.get('mscore', 0):.0f}). "
        )
        
        if runner_up:
            explanation += (
                f"While {runner_up['ticker']} is also solid at {runner_up.get('rfinal', 0):.1f}/100, "
                f"{top['ticker']}'s superior fundamental score gives it the edge."
            )
        
        return explanation
    
    def _generate_followup_suggestions(
        self,
        message: str,
        ticker: Optional[str]
    ) -> List[str]:
        """Generate follow-up question suggestions"""
        
        suggestions = []
        
        if ticker:
            suggestions.extend([
                f"What are the risk factors for {ticker}?",
                f"Is {ticker} in a momentum trap?",
                f"Compare {ticker} with similar stocks"
            ])
        else:
            suggestions.extend([
                "What makes a good ASRE rating?",
                "How do I interpret F-Score and T-Score?",
                "What is a momentum trap?"
            ])
        
        return suggestions


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global instance (singleton pattern)
_explainer = None

def get_explainer() -> AIExplainer:
    """Get or create AIExplainer instance"""
    global _explainer
    if _explainer is None:
        _explainer = AIExplainer()
    return _explainer


def explain_rating(ticker: str, asre_data: Dict) -> Dict:
    """Quick rating explanation"""
    return get_explainer().explain_stock_rating(ticker, asre_data)


def explain_trap(ticker: str, fscore: float, tscore: float, mscore: float) -> Dict:
    """Quick trap explanation"""
    return get_explainer().explain_momentum_trap(ticker, fscore, tscore, mscore)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'AIExplainer',
    'GroqAIClient',
    'get_explainer',
    'explain_rating',
    'explain_trap',
]
