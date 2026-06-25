"""
ASRE API - Main Application Entry Point
========================================
FastAPI application with all routes, middleware, and error handling.

Features:
- 8 route modules (stocks, portfolio, dip_analysis, investing, backtest, ai_chat, alerts, community)
- CORS for frontend access
- Global exception handling
- Request timing middleware
- Rate limiting (100 req/min per IP)
- API documentation (Swagger + ReDoc)
- Health checks
- Comprehensive logging
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.config import settings


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce noise from verbose libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="ASRE API",
    description="""
    **Advanced Stock Rating Engine API**
    
    AI-powered stock analysis platform providing:
    - Real-time ASRE ratings (0-100 scale)
    - Portfolio health analysis
    - Momentum trap detection
    - Dip buying opportunities
    - Trade backtesting
    - WhatsApp/SMS alerts
    - AI-powered explanations (Groq)
    - Community discovery
    
    Supported Stocks: NVDA, MSFT, GOOGL, META, AAPL, JPM, MA, V, TSLA, BAC, AMZN, NFLX, ORCL
    """,
    version="1.0.0",
    contact={
        "name": "ASRE Team",
        "email": "support@asre.app"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS Middleware - Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

# GZip Compression - Reduce response size
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request ID and timing middleware
@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    """Add request ID and timing to all requests"""
    
    # Generate request ID
    request_id = f"{int(time.time())}-{id(request)}"
    
    # Log incoming request
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    # Time the request
    start_time = time.time()
    
    # Process request
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"[{request_id}] Request failed: {e}")
        raise
    
    # Calculate process time
    process_time = time.time() - start_time
    
    # Add headers
    response.headers["X-Process-Time"] = f"{process_time:.3f}s"
    response.headers["X-Request-ID"] = request_id
    
    # Log response
    logger.info(
        f"[{request_id}] {response.status_code} - {process_time:.3f}s"
    )
    
    return response


# Rate limiting middleware (simple in-memory)
request_counts: Dict[str, list] = {}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting (100 requests per minute per IP)"""
    
    # Skip rate limiting for health checks
    if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    client_ip = request.client.host
    current_time = time.time()
    
    # Initialize or clean old requests
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    # Remove requests older than 1 minute
    request_counts[client_ip] = [
        t for t in request_counts[client_ip] 
        if current_time - t < 60
    ]
    
    # Check rate limit (100 requests per minute)
    if len(request_counts[client_ip]) >= 100:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded. Maximum 100 requests per minute.",
                "retry_after": 60
            }
        )
    
    # Add current request
    request_counts[client_ip].append(current_time)
    
    return await call_next(request)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages"""
    logger.warning(f"Validation error: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Don't expose internal errors in production
    detail = str(exc) if settings.DEBUG else "Internal server error"
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


# ============================================================================
# ROUTE REGISTRATION (ALL 8 MODULES)
# ============================================================================

logger.info("Loading API routes...")

# 1. Stocks routes
try:
    from api.routes.stocks import router as stocks_router
    app.include_router(stocks_router)
    logger.info("✅ Loaded stocks routes")
except Exception as e:
    logger.error(f"❌ Failed to load stocks routes: {e}")

# 2. Portfolio routes
try:
    from api.routes.portfolio import router as portfolio_router
    app.include_router(portfolio_router)
    logger.info("✅ Loaded portfolio routes")
except Exception as e:
    logger.error(f"❌ Failed to load portfolio routes: {e}")

# 3. Dip Analysis routes
try:
    from api.routes.dip_analysis import router as dip_analysis_router
    app.include_router(dip_analysis_router)
    logger.info("✅ Loaded dip analysis routes")
except Exception as e:
    logger.error(f"❌ Failed to load dip analysis routes: {e}")

# 4. Investing routes
try:
    from api.routes.investing import router as investing_router
    app.include_router(investing_router)
    logger.info("✅ Loaded investing routes")
except Exception as e:
    logger.error(f"❌ Failed to load investing routes: {e}")

# 5. Backtest routes
try:
    from api.routes.backtest import router as backtest_router
    app.include_router(backtest_router)
    logger.info("✅ Loaded backtest routes")
except Exception as e:
    logger.error(f"❌ Failed to load backtest routes: {e}")

# 6. AI Chat routes
try:
    from api.routes.ai_chat import router as ai_router
    app.include_router(ai_router)
    logger.info("✅ Loaded AI chat routes")
except Exception as e:
    logger.error(f"❌ Failed to load AI routes: {e}")

# 7. Alert routes
try:
    from api.routes.alerts import router as alerts_router
    app.include_router(alerts_router)
    logger.info("✅ Loaded alert routes")
except Exception as e:
    logger.error(f"❌ Failed to load alert routes: {e}")

# 8. Community routes
try:
    from api.routes.community import router as community_router
    app.include_router(community_router)
    logger.info("✅ Loaded community routes")
except Exception as e:
    logger.error(f"❌ Failed to load community routes: {e}")


# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get(
    "/",
    response_class=HTMLResponse,
    summary="API Homepage",
    tags=["general"]
)
async def root():
    """API homepage with quick links"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ASRE API</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                max-width: 900px;
                margin: 50px auto;
                padding: 20px;
                line-height: 1.6;
            }}
            h1 {{ color: #2563eb; }}
            .status {{ color: #10b981; font-weight: bold; }}
            .links {{ margin-top: 30px; }}
            .link-card {{
                background: #f3f4f6;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #2563eb;
            }}
            .link-card a {{
                color: #2563eb;
                text-decoration: none;
                font-weight: 500;
            }}
            .link-card a:hover {{
                text-decoration: underline;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: #f9fafb;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2563eb;
            }}
        </style>
    </head>
    <body>
        <h1>🚀 ASRE API</h1>
        <p class="status">● Status: Operational</p>
        <p>Advanced Stock Rating Engine - AI-powered stock analysis platform</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">13</div>
                <div>Supported Stocks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len([r for r in app.routes])}</div>
                <div>API Endpoints</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">8</div>
                <div>Route Modules</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">v1.0</div>
                <div>Version</div>
            </div>
        </div>
        
        <div class="links">
            <h2>Quick Links</h2>
            
            <div class="link-card">
                <strong>📚 Interactive API Documentation</strong><br>
                <a href="/docs" target="_blank">Swagger UI →</a> | 
                <a href="/redoc" target="_blank">ReDoc →</a>
            </div>
            
            <div class="link-card">
                <strong>💊 Health Check</strong><br>
                <a href="/health">API Health Status →</a>
            </div>
            
            <div class="link-card">
                <strong>📊 Example Endpoints</strong><br>
                <a href="/api/stocks">List Supported Stocks →</a><br>
                <a href="/api/stocks/NVDA">Get NVDA Rating →</a>
            </div>
        </div>
        
        <hr style="margin: 40px 0; border: none; border-top: 1px solid #e5e7eb;">
        
        <h3>Features</h3>
        <ul>
            <li>✅ Real-time ASRE ratings (0-100 scale)</li>
            <li>✅ Portfolio health analysis</li>
            <li>✅ Momentum trap detection</li>
            <li>✅ Dip buying opportunities</li>
            <li>✅ Trade backtesting</li>
            <li>✅ WhatsApp/SMS alerts</li>
            <li>✅ AI-powered explanations (Groq)</li>
            <li>✅ Community discovery</li>
        </ul>
        
        <p style="margin-top: 40px; color: #6b7280; font-size: 14px;">
            Built with FastAPI • Python • ASRE Algorithm<br>
            ENV: {settings.ENV.upper()}
        </p>
    </body>
    </html>
    """
    return html_content


@app.get(
    "/health",
    tags=["general"],
    summary="Health Check"
)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        from api.services.asre_service import ASREService
        
        # Get ASRE service health
        asre_health = ASREService.health_check()
        
        # Check cache stats
        cache_stats = ASREService.get_cache_stats()
        
        return {
            "status": "healthy" if asre_health['asre_available'] else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "ENV": settings.ENV,
            "services": {
                "asre_engine": {
                    "available": asre_health['asre_available'],
                    "supported_stocks": asre_health['supported_stocks_count']
                },
                "cache": {
                    "enabled": True,
                    "size": cache_stats['size'],
                    "expiry_hours": asre_health['cache_expiry_hours']
                },
                "ai_explanations": {
                    "enabled": bool(settings.GROQ_API_KEY),
                    "provider": "Groq"
                },
                "alerts": {
                    "whatsapp": bool(settings.TWILIO_ACCOUNT_SID),
                    "sms": bool(settings.TWILIO_ACCOUNT_SID)
                }
            },
            "uptime": "operational"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get(
    "/stats",
    tags=["general"],
    summary="API Statistics"
)
async def api_stats():
    """Get API usage statistics"""
    try:
        from api.services.asre_service import ASREService
        
        cache_stats = ASREService.get_cache_stats()
        
        return {
            "endpoints": len([r for r in app.routes]),
            "route_modules": 8,
            "cache": cache_stats,
            "rate_limits": {
                "active_ips": len(request_counts),
                "requests_tracked": sum(len(reqs) for reqs in request_counts.values())
            },
            "supported_stocks": settings.SUPPORTED_STOCKS,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return {"error": str(e)}


# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

def _warmup_cache():
    """
    Pre-compute ASRE ratings for all supported tickers in the background so
    user-facing requests hit the warm 24h in-memory cache instead of paying
    the ~40s cold compute (which would otherwise risk a hosted-gateway
    timeout). Runs serially on a daemon thread; failures are non-fatal.

    Enable with ASRE_WARMUP=1 (off by default to keep dev startup fast).
    NOTE: the rating cache is per-process — under multiple uvicorn workers
    each worker warms independently; use a shared cache for true multi-worker.
    """
    try:
        from api.services.asre_service import ASREService
    except Exception as exc:  # pragma: no cover
        logger.warning("warmup: ASREService unavailable (%s)", exc)
        return
    tickers = list(settings.SUPPORTED_STOCKS)
    logger.info("warmup: pre-computing %d tickers in background...", len(tickers))
    for tk in tickers:
        try:
            ASREService.get_stock_rating(tk, force_refresh=False)
            logger.info("warmup: %s cached.", tk)
        except Exception as exc:
            logger.warning("warmup: %s failed (%s)", tk, exc)
    logger.info("warmup: complete.")


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 80)
    logger.info("🚀 ASRE API Starting...")
    logger.info(f"   ENV: {settings.ENV}")
    logger.info(f"   Debug Mode: {settings.DEBUG}")
    logger.info(f"   Host: {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"   Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logger.info(f"   Health: http://{settings.API_HOST}:{settings.API_PORT}/health")
    logger.info(f"   Supported Stocks: {len(settings.SUPPORTED_STOCKS)}")
    logger.info("=" * 80)

    if os.getenv("ASRE_WARMUP", "0") == "1":
        import threading
        threading.Thread(target=_warmup_cache, name="asre-warmup",
                         daemon=True).start()


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("=" * 80)
    logger.info("👋 ASRE API Shutting down...")
    logger.info("=" * 80)


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Print startup banner
    print("\n" + "=" * 80)
    print("🚀 Starting ASRE API Server")
    print("=" * 80)
    print(f"ENV: {settings.ENV}")
    print(f"Debug: {settings.DEBUG}")
    print(f"Host: {settings.API_HOST}:{settings.API_PORT}")
    print(f"Docs: http://localhost:{settings.API_PORT}/docs")
    print(f"Health: http://localhost:{settings.API_PORT}/health")
    print("=" * 80 + "\n")
    
    # Run server
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info",
        access_log=True
    )
