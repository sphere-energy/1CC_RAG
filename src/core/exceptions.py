from typing import Any, Dict, Optional


class APIException(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "api_error",
        detail: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize API exception.

        Args:
            message (str): Human-readable error message.
            status_code (int): HTTP status code.
            error_type (str): Machine-readable error type.
            detail (Optional[Dict[str, Any]]): Additional error details.
        """
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.detail = detail or {}
        super().__init__(self.message)


class BedrockException(APIException):
    """Exception for AWS Bedrock errors."""

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        """
        Initialize Bedrock exception.

        Args:
            message (str): Error message.
            detail (Optional[Dict[str, Any]]): Additional error details.
        """
        super().__init__(
            message=message,
            status_code=503,
            error_type="bedrock_error",
            detail=detail,
        )


class QdrantException(APIException):
    """Exception for Qdrant errors."""

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        """
        Initialize Qdrant exception.

        Args:
            message (str): Error message.
            detail (Optional[Dict[str, Any]]): Additional error details.
        """
        super().__init__(
            message=message,
            status_code=503,
            error_type="qdrant_error",
            detail=detail,
        )


class ValidationException(APIException):
    """Exception for validation errors."""

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        """
        Initialize Validation exception.

        Args:
            message (str): Error message.
            detail (Optional[Dict[str, Any]]): Additional error details.
        """
        super().__init__(
            message=message,
            status_code=400,
            error_type="validation_error",
            detail=detail,
        )


class ConfigurationException(APIException):
    """Exception for configuration errors."""

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        """
        Initialize Configuration exception.

        Args:
            message (str): Error message.
            detail (Optional[Dict[str, Any]]): Additional error details.
        """
        super().__init__(
            message=message,
            status_code=500,
            error_type="configuration_error",
            detail=detail,
        )
