"""
API Configuration Module
Loads and validates environment variables, defines application settings.

This module is imported by all other modules and provides centralized configuration.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Determine project root (3 levels up from this file)
# D:\asre-project\src\api\config.py -> D:\asre-project
PROJECT_ROOT = Path(__file__).parent.parent.parent  # D:\asre-project
ENV_FILE = PROJECT_ROOT / ".env"
SOURCE_DIR = PROJECT_ROOT / "src"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
    logger.info(f"✅ Loaded environment from: {ENV_FILE}")
else:
    logger.warning(f"⚠️  .env file not found at: {ENV_FILE}")
    logger.warning("Using default/system environment variables")


class Settings:
    """
    Application Settings
    
    All configuration values are loaded from environment variables with sensible defaults.
    Critical settings (API keys) will raise warnings if not configured.
    """
    
    # ============================================================================
    # API SERVER CONFIGURATION
    # ============================================================================
    
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    """API host address. Use 0.0.0.0 to accept connections from any IP."""
    
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    """API port number."""
    
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    """Debug mode. Enables verbose logging and API docs."""
    
    ENV: str = os.getenv("ENV", "development")
    """Environment: development, staging, or production."""
    
    API_VERSION: str = "1.0.0"
    """API version for documentation."""
    
    API_TITLE: str = "ASRE API"
    """API title for documentation."""
    
    API_DESCRIPTION: str = "Advanced Stock Rating Engine - AI-Powered Investment Intelligence"
    """API description for documentation."""
    
    # ============================================================================
    # CORS CONFIGURATION
    # ============================================================================
    
    _cors_origins_str: str = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://localhost:5173, https://arkangel.co.in"
    )
    CORS_ORIGINS: List[str] = [origin.strip() for origin in _cors_origins_str.split(",")]
    """List of allowed CORS origins for React frontend."""
    
    CORS_ALLOW_CREDENTIALS: bool = True
    """Allow cookies and authentication headers in CORS requests."""
    
    CORS_ALLOW_METHODS: List[str] = ["*"]
    """Allowed HTTP methods for CORS."""
    
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    """Allowed HTTP headers for CORS."""
    
    # ============================================================================
    # GROQ AI CONFIGURATION
    # ============================================================================
    
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    """Groq API key for AI explanations. Get from: https://console.groq.com"""
    
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    """Groq model to use. Options: mixtral-8x7b-32768, llama2-70b-4096"""
    
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "500"))
    """Maximum tokens for Groq responses."""
    
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
    """Temperature for Groq responses (0.0-1.0). Higher = more creative."""
    
    GROQ_TIMEOUT: int = int(os.getenv("GROQ_TIMEOUT", "30"))
    """Timeout for Groq API calls (seconds)."""
    
    # ============================================================================
    # TWILIO CONFIGURATION (WhatsApp Alerts)
    # ============================================================================
    
    TWILIO_ACCOUNT_SID: Optional[str] = os.getenv("TWILIO_ACCOUNT_SID")
    """Twilio Account SID for WhatsApp alerts."""
    
    TWILIO_AUTH_TOKEN: Optional[str] = os.getenv("TWILIO_AUTH_TOKEN")
    """Twilio Auth Token for WhatsApp alerts."""
    
    TWILIO_WHATSAPP_NUMBER: str = os.getenv("TWILIO_WHATSAPP_NUMBER", "+14155238886")
    """Twilio WhatsApp sender number (sandbox default)."""
    
    TWILIO_ENABLED: bool = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)
    """Auto-detect if Twilio is configured."""
    
    # ============================================================================
    # FILE PATHS
    # ============================================================================
    
    PROJECT_ROOT: Path = PROJECT_ROOT
    """Project root directory."""
    
    SRC_DIR: Path = PROJECT_ROOT / "src"
    """Source code directory."""
    
    API_DIR: Path = SRC_DIR / "api"
    """API directory."""
    
    DATA_DIR: Path = PROJECT_ROOT / "data"
    """Data storage directory."""
    
    CACHE_DIR: Path = DATA_DIR / "cache"
    """Cache directory for fundamentals data."""
    
    FUNDAMENTALS_CACHE_DIR: Path = CACHE_DIR / "fundamentals"
    """Fundamentals data cache directory."""
    
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    """Logs directory."""
    
    TEMP_DIR: Path = PROJECT_ROOT / "temp"
    """Temporary files directory (PDFs, uploads)."""
    
    # ============================================================================
    # ASRE ALGORITHM CONFIGURATION
    # ============================================================================
    
    SUPPORTED_STOCKS: List[str] = [
        # IT / Technology
        "TCS", "INFY", "WIPRO", "HCLTECH",
        # Banking / Financial
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BAJFINANCE",
        # Conglomerates / Energy
        "RELIANCE", "ONGC",
        # FMCG / Consumer
        "HINDUNILVR", "ITC",
        # Auto
        "MARUTI", "TATAMOTORS",
        # Pharma
        "SUNPHARMA", "DRREDDY",
    ]
    """
    Indian NSE tickers (stored without .NS suffix for clean URL routing).
    The asre_service layer appends .NS before calling the ASRE pipeline.
    Chosen from INDIA_SECTOR_INDEX_MAP in data_loader_indian.py for full sector coverage.
    """
    
    CACHE_EXPIRY_HOURS: int = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))
    """Hours before cached stock data expires."""
    
    MAX_CONCURRENT_FETCHES: int = int(os.getenv("MAX_CONCURRENT_FETCHES", "5"))
    """Maximum concurrent stock data fetches."""
    
    # ============================================================================
    # API RATE LIMITING
    # ============================================================================
    
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "True").lower() in ("true", "1", "yes")
    """Enable rate limiting for API endpoints."""
    
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    """Maximum API requests per minute per IP."""
    
    GROQ_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("GROQ_RATE_LIMIT_PER_MINUTE", "10"))
    """Maximum Groq AI requests per minute (protect free tier quota)."""
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if DEBUG else "WARNING")
    """Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""
    
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """Log message format."""
    
    LOG_FILE_ENABLED: bool = os.getenv("LOG_FILE_ENABLED", "True").lower() in ("true", "1", "yes")
    """Enable logging to file."""
    
    LOG_FILE_PATH: Path = LOGS_DIR / f"api_{ENV}.log"
    """Log file path."""
    
    # ============================================================================
    # SECURITY
    # ============================================================================
    
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    """Secret key for signing tokens (future feature)."""
    
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = [".csv", ".xlsx"]
    """Allowed file extensions for backtest uploads."""
    
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "5"))
    """Maximum file upload size in MB."""
    
    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================
    
    ENABLE_AI_EXPLANATIONS: bool = bool(GROQ_API_KEY)
    """Auto-enable AI if Groq key is configured."""
    
    ENABLE_WHATSAPP_ALERTS: bool = TWILIO_ENABLED
    """Auto-enable WhatsApp if Twilio is configured."""
    
    ENABLE_BACKTEST_UPLOAD: bool = True
    """Enable backtest CSV upload feature."""
    
    ENABLE_PORTFOLIO_ANALYZER: bool = True
    """Enable portfolio analysis feature."""
    
    ENABLE_COMMUNITY_SIGNALS: bool = True
    """Enable community trading signals (mock data for demo)."""
    
    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    
    @classmethod
    def ensure_directories(cls):
        """Create required directories if they don't exist."""
        for directory in [
            cls.DATA_DIR,
            cls.CACHE_DIR,
            cls.FUNDAMENTALS_CACHE_DIR,
            cls.LOGS_DIR,
            cls.TEMP_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"✅ Directory ready: {directory}")
    
    @classmethod
    def validate(cls):
        """Validate configuration and log warnings for missing critical settings."""
        issues = []
        warnings = []
        
        # Critical: Groq API key
        if not cls.GROQ_API_KEY:
            warnings.append(
                "⚠️  GROQ_API_KEY not set. AI explanations will be disabled.\n"
                "   Get your free API key: https://console.groq.com"
            )
        
        # Optional: Twilio
        if not cls.TWILIO_ENABLED:
            warnings.append(
                "⚠️  Twilio credentials not set. WhatsApp alerts will be disabled.\n"
                "   This is optional for hackathon demo."
            )
        
        # Check if ASRE source directory exists
        if not cls.SRC_DIR.exists():
            issues.append(f"❌ CRITICAL: Source directory not found: {cls.SRC_DIR}")
        
        # Log all warnings
        if warnings:
            logger.warning("\n" + "\n".join(warnings))
        
        # Raise exception for critical issues
        if issues:
            error_msg = "\n".join(issues)
            logger.error(error_msg)
            raise RuntimeError(f"Configuration validation failed:\n{error_msg}")
        
        logger.info("✅ Configuration validated successfully")
    
    @classmethod
    def get_status(cls) -> dict:
        """Get configuration status for health checks."""
        return {
            "environment": cls.ENV,
            "debug": cls.DEBUG,
            "api_version": cls.API_VERSION,
            "features": {
                "ai_explanations": cls.ENABLE_AI_EXPLANATIONS,
                "whatsapp_alerts": cls.ENABLE_WHATSAPP_ALERTS,
                "backtest_upload": cls.ENABLE_BACKTEST_UPLOAD,
                "portfolio_analyzer": cls.ENABLE_PORTFOLIO_ANALYZER,
                "community_signals": cls.ENABLE_COMMUNITY_SIGNALS,
            },
            "paths": {
                "project_root": str(cls.PROJECT_ROOT),
                "data_dir": str(cls.DATA_DIR),
                "cache_dir": str(cls.CACHE_DIR),
            },
            "rate_limits": {
                "api_per_minute": cls.RATE_LIMIT_PER_MINUTE,
                "groq_per_minute": cls.GROQ_RATE_LIMIT_PER_MINUTE,
            },
            "integrations": {
                "groq_configured": bool(cls.GROQ_API_KEY),
                "twilio_configured": cls.TWILIO_ENABLED,
            },
        }
    
    @classmethod
    def print_config(cls):
        """Print configuration summary (for debugging)."""
        print("\n" + "="*80)
        print("🔧 ASRE API CONFIGURATION")
        print("="*80)
        print(f"Environment:        {cls.ENV}")
        print(f"Debug Mode:         {cls.DEBUG}")
        print(f"API Host:           {cls.API_HOST}:{cls.API_PORT}")
        print(f"Project Root:       {cls.PROJECT_ROOT}")
        print(f"Supported Stocks:   {len(cls.SUPPORTED_STOCKS)} tickers")
        print(f"\n🤖 AI Features:")
        print(f"  Groq API:         {'✅ Configured' if cls.GROQ_API_KEY else '❌ Not configured'}")
        print(f"  Groq Model:       {cls.GROQ_MODEL}")
        print(f"\n📱 Alert Features:")
        print(f"  Twilio:           {'✅ Configured' if cls.TWILIO_ENABLED else '❌ Not configured'}")
        print(f"\n🌐 CORS Origins:")
        for origin in cls.CORS_ORIGINS:
            print(f"  - {origin}")
        print("="*80 + "\n")


# Create singleton instance
settings = Settings()

# Initialize on import
try:
    settings.ensure_directories()
    settings.validate()
    
    if settings.DEBUG:
        settings.print_config()
    
except Exception as e:
    logger.error(f"❌ Configuration initialization failed: {e}")
    sys.exit(1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_log_level() -> int:
    """Convert string log level to logging constant."""
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(settings.LOG_LEVEL.upper(), logging.INFO)


def configure_logging():
    """Configure application-wide logging."""
    log_level = get_log_level()
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),  # Console output
        ]
    )
    
    # Add file handler if enabled
    if settings.LOG_FILE_ENABLED:
        try:
            # FIX: Use UTF-8 encoding for file handler to support emojis
            file_handler = logging.FileHandler(
                settings.LOG_FILE_PATH, 
                encoding='utf-8'  # ← THIS IS THE FIX
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
            logging.getLogger().addHandler(file_handler)
            logger.info(f"[OK] Logging to file: {settings.LOG_FILE_PATH}")  # Changed emoji to [OK]
        except Exception as e:
            logger.warning(f"[WARN] Failed to enable file logging: {e}")
    
    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    logger.info(f"[OK] Logging configured (level: {settings.LOG_LEVEL})")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "settings",
    "Settings",
    "configure_logging",
    "get_log_level",
]
