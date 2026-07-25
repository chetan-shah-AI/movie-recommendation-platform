import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.logger import app_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        start_time = time.time()

        app_logger.info(
            f"Incoming request: {request.method} {request.url.path}"
        )

        response = await call_next(request)

        duration = round(time.time() - start_time, 3)

        app_logger.info(
            f"Completed request: "
            f"{request.method} {request.url.path} "
            f"Status={response.status_code} "
            f"Duration={duration}s"
        )

        return response