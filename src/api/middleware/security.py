"""
api/middleware/security.py — Production security hardening for the ASRE API.

Provides, as small composable helpers wired up in main.py:

  * API-key auth      — X-API-Key header, constant-time comparison, public
                        docs/health endpoints exempt. Disabled unless
                        settings.ASRE_API_KEY is set.
  * Rate limiting     — in-memory per-IP sliding window. A global limit plus a
                        tighter limit on compute-heavy routes. Returns 429 with
                        a Retry-After header.
  * Request size cap  — rejects Content-Length over settings.MAX_REQUEST_BYTES
                        with 413.
  * Security headers   — nosniff / frame-deny / referrer-policy / etc., added to
                        every response (Cache-Control: no-store on API routes).

All helpers are framework-light (operate on a Starlette/FastAPI Request) so the
ordering stays explicit and auditable in main.py.
"""

from __future__ import annotations

import hmac
import re
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from api.config import settings

# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------

# Never require auth (public docs, health, landing, stats).
AUTH_EXEMPT_PATHS = frozenset(
    {"/", "/health", "/stats", "/docs", "/openapi.json", "/redoc"}
)

# Never rate-limited (liveness probes / landing).
RATE_EXEMPT_PATHS = frozenset({"/", "/health"})

# Security headers applied to every response.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}

# Compute-heavy routes get a tighter rate limit. Tickers are UPPERCASE, so a
# case-sensitive match cleanly distinguishes /api/stocks/TCS (heavy) from static
# siblings like /api/stocks/health or /api/dip-analysis/quick-scan (not heavy).
_HEAVY_TICKER_RE = re.compile(
    r"^/api/(?:stocks|dip-analysis)/[A-Z][A-Z0-9]*(?:\.[A-Z]{1,3})?(?:/[a-z\-]+)?$"
)


def is_auth_exempt(path: str) -> bool:
    return path in AUTH_EXEMPT_PATHS


def is_rate_exempt(path: str) -> bool:
    return path in RATE_EXEMPT_PATHS


def is_api_path(path: str) -> bool:
    return path.startswith("/api/")


def is_heavy_route(path: str) -> bool:
    """True for compute-heavy routes that warrant a tighter per-IP limit."""
    if path.startswith("/api/backtest"):
        return True
    return bool(_HEAVY_TICKER_RE.match(path))


# ---------------------------------------------------------------------------
# API-key authentication
# ---------------------------------------------------------------------------

def validate_api_key(request: Request) -> Optional[JSONResponse]:
    """Return a 401 JSONResponse if the request lacks a valid API key.

    No-op when ASRE_API_KEY is unset (open API) or the path is public.
    Uses hmac.compare_digest for constant-time comparison (no timing oracle).
    """
    expected = settings.ASRE_API_KEY
    if not expected:
        return None
    if is_auth_exempt(request.url.path):
        return None

    provided = request.headers.get("X-API-Key", "")
    if not provided or not hmac.compare_digest(str(provided), str(expected)):
        return JSONResponse(
            status_code=401,
            content={
                "error": "Unauthorized",
                "message": "X-API-Key header required",
            },
        )
    return None


# ---------------------------------------------------------------------------
# Request size cap
# ---------------------------------------------------------------------------

def enforce_request_size(request: Request) -> Optional[JSONResponse]:
    """Return a 413 JSONResponse if Content-Length exceeds the configured cap."""
    cl = request.headers.get("content-length")
    if cl is None:
        return None
    try:
        size = int(cl)
    except (TypeError, ValueError):
        return None
    if size > settings.MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": "Payload too large",
                "message": (
                    f"Request body exceeds the "
                    f"{settings.MAX_REQUEST_BYTES} byte limit."
                ),
            },
        )
    return None


# ---------------------------------------------------------------------------
# Rate limiting (in-memory, per-IP sliding window)
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """Per-IP sliding-window limiter with separate global and heavy buckets."""

    def __init__(self, window_seconds: int = 60) -> None:
        self._window = window_seconds
        self._lock = threading.Lock()
        self._global: Dict[str, Deque[float]] = defaultdict(deque)
        self._heavy: Dict[str, Deque[float]] = defaultdict(deque)

    @staticmethod
    def _prune(dq: Deque[float], cutoff: float) -> None:
        while dq and dq[0] < cutoff:
            dq.popleft()

    def check(self, ip: str, limit: int, heavy: bool) -> Optional[int]:
        """Record a hit; return None if allowed, else Retry-After seconds."""
        now = time.time()
        cutoff = now - self._window
        bucket = self._heavy if heavy else self._global
        with self._lock:
            dq = bucket[ip]
            self._prune(dq, cutoff)
            if len(dq) >= limit:
                retry_after = max(1, int(self._window - (now - dq[0])))
                return retry_after
            dq.append(now)
            return None


# Module-level singleton (one limiter per process).
rate_limiter = SlidingWindowRateLimiter(window_seconds=60)


def enforce_rate_limit(request: Request) -> Optional[JSONResponse]:
    """Return a 429 JSONResponse if the client exceeded its rate budget.

    Applies the global per-minute limit on all non-exempt routes, plus a
    tighter limit on compute-heavy routes. No-op when rate limiting is disabled.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return None
    path = request.url.path
    if is_rate_exempt(path):
        return None

    client_ip = request.client.host if request.client else "unknown"

    # Tighter heavy-route budget first (checked before the global bucket so a
    # heavy-route hit doesn't get double-counted against the global allowance
    # only to then be rejected here).
    if is_heavy_route(path):
        retry = rate_limiter.check(
            client_ip, settings.HEAVY_RATE_LIMIT_PER_MINUTE, heavy=True
        )
        if retry is not None:
            return _rate_limited_response(
                retry, settings.HEAVY_RATE_LIMIT_PER_MINUTE,
                scope="compute-heavy",
            )

    retry = rate_limiter.check(
        client_ip, settings.RATE_LIMIT_PER_MINUTE, heavy=False
    )
    if retry is not None:
        return _rate_limited_response(
            retry, settings.RATE_LIMIT_PER_MINUTE, scope="global",
        )
    return None


def _rate_limited_response(retry_after: int, limit: int, scope: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": "Too many requests",
            "message": (
                f"Rate limit exceeded ({scope}: {limit}/min). "
                f"Retry after {retry_after}s."
            ),
            "retry_after": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

def apply_security_headers(response, path: str) -> None:
    """Attach standard security headers; no-store cache policy on API routes."""
    for k, v in SECURITY_HEADERS.items():
        response.headers[k] = v
    if is_api_path(path):
        response.headers["Cache-Control"] = "no-store"
