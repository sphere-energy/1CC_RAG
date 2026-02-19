import logging
from pybreaker import CircuitBreaker, CircuitBreakerError, CircuitBreakerListener
from src.core.config import Settings
from src.core.exceptions import BedrockException, QdrantException

logger = logging.getLogger(__name__)


class LoggingCircuitBreakerListener(CircuitBreakerListener):
    """Custom circuit breaker listener for logging events."""

    def before_call(self, cb, func, *args, **kwargs):
        """Called before the circuit breaker calls the function."""
        pass

    def on_success(self, cb):
        """Called when a function call succeeds."""
        pass

    def on_failure(self, cb, exc):
        """Called when a function call fails."""
        logger.warning("Circuit breaker '%s' recorded failure: %s", cb.name, exc)

    def on_open(self, cb):
        """Called when the circuit breaker opens."""
        logger.error("Circuit breaker '%s' opened", cb.name)

    def on_close(self, cb):
        """Called when the circuit breaker closes."""
        logger.info("Circuit breaker '%s' closed", cb.name)


# Global circuit breaker instances
_bedrock_breaker: CircuitBreaker = None
_qdrant_breaker: CircuitBreaker = None


def init_circuit_breakers(settings: Settings):
    """
    Initialize circuit breakers.

    Args:
        settings (Settings): Application settings.
    """
    global _bedrock_breaker, _qdrant_breaker

    listener = LoggingCircuitBreakerListener()

    _bedrock_breaker = CircuitBreaker(
        fail_max=settings.circuit_breaker_fail_max,
        reset_timeout=settings.circuit_breaker_timeout,
        name="bedrock_breaker",
        listeners=[listener],
    )

    _qdrant_breaker = CircuitBreaker(
        fail_max=settings.circuit_breaker_fail_max,
        reset_timeout=settings.circuit_breaker_timeout,
        name="qdrant_breaker",
        listeners=[listener],
    )

    logger.info("Circuit breakers initialized")


def get_bedrock_breaker() -> CircuitBreaker:
    """
    Get Bedrock circuit breaker instance.

    Returns:
        CircuitBreaker: Bedrock circuit breaker.
    """
    if _bedrock_breaker is None:
        raise RuntimeError("Circuit breakers not initialized")
    return _bedrock_breaker


def get_qdrant_breaker() -> CircuitBreaker:
    """
    Get Qdrant circuit breaker instance.

    Returns:
        CircuitBreaker: Qdrant circuit breaker.
    """
    if _qdrant_breaker is None:
        raise RuntimeError("Circuit breakers not initialized")
    return _qdrant_breaker
