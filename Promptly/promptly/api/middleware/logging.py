"""
Logging Middleware
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
import json
from datetime import datetime

from ..config import get_settings

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("promptly.api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware"""

    async def dispatch(self, request: Request, call_next):
        """Log request and response"""
        request_id = request.headers.get("X-Request-ID", str(time.time()))
        start_time = time.time()

        # Log request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_host": request.client.host,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"Request started: {json.dumps(log_data)}")

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            log_data.update({
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2)
            })

            logger.info(f"Request completed: {json.dumps(log_data)}")

            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(duration)

            return response

        except Exception as e:
            duration = time.time() - start_time
            log_data.update({
                "status": "error",
                "error": str(e),
                "duration_ms": round(duration * 1000, 2)
            })
            logger.error(f"Request failed: {json.dumps(log_data)}")
            raise
