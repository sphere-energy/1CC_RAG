import logging
import sys

from brotli_asgi import BrotliMiddleware
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.chat.router import router as chat_router
from src.core.auth import init_cognito_verifier
from src.core.circuit_breaker import init_circuit_breakers
from src.core.config import get_settings
from src.core.database import init_database
from src.core.exceptions import APIException
from src.core.middleware import (
    CorrelationIdMiddleware,
    RequestLoggingMiddleware,
    get_correlation_id,
)
from src.limiter import limiter

# Get settings
settings = get_settings()

# Configure JSON logging
logHandler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s",
    rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
)
logHandler.setFormatter(formatter)

# Configure root logger
logging.root.setLevel(settings.log_level)
logging.root.addHandler(logHandler)

logger = logging.getLogger(__name__)

infrastructure_initialized = False

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Production RAG API with AWS Cognito auth, PostgreSQL persistence, and circuit breakers",
    version=settings.app_version,
)

# Add custom middlewares (order matters!)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Add Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Brotli Compression
app.add_middleware(BrotliMiddleware)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=settings.cors_allow_methods_list,
    allow_headers=settings.cors_allow_headers_list,
)


# Custom Exception Handlers
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle custom API exceptions."""
    correlation_id = get_correlation_id()
    logger.error(
        "APIException",
        extra={
            "correlation_id": correlation_id,
            "error_type": exc.error_type,
            "message": exc.message,
            "detail": exc.detail,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_type": exc.error_type,
            "message": exc.message,
            "detail": exc.detail,
            "correlation_id": correlation_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    correlation_id = get_correlation_id()
    logger.error(
        "Validation error",
        extra={"correlation_id": correlation_id, "errors": exc.errors()},
    )
    return JSONResponse(
        status_code=422,
        content={
            "error_type": "validation_error",
            "message": "Request validation failed",
            "detail": exc.errors(),
            "correlation_id": correlation_id,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    correlation_id = get_correlation_id()
    logger.error(
        "Unhandled exception",
        extra={"correlation_id": correlation_id},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error_type": "internal_server_error",
            "message": "An internal server error occurred",
            "detail": {},
            "correlation_id": correlation_id,
        },
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Create database tables on startup (for development)."""
    global infrastructure_initialized
    logger.info("Application startup")
    if not infrastructure_initialized:
        logger.info("Initializing application infrastructure")
        init_database(settings)
        if settings.environment != "test":
            init_cognito_verifier(settings)
        init_circuit_breakers(settings)
        infrastructure_initialized = True
        logger.info("Infrastructure initialized successfully")

    # In production, use Alembic migrations instead
    # db.create_tables()


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Application shutdown")


# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    correlation_id = get_correlation_id()
    return {
        "status": "ok",
        "version": settings.app_version,
        "service": settings.app_name,
        "correlation_id": correlation_id,
    }
