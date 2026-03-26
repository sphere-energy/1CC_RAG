from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    """

    environment: str = "dev"

    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "1cc_legislation"
    qdrant_timeout_seconds: float = 10.0
    qdrant_retries: int = 3

    # AWS Configuration
    aws_region: str = "eu-central-1"
    bedrock_embedding_model_id: str = (
        "arn:aws:bedrock:eu-central-1:407179558514:"
        "inference-profile/eu.cohere.embed-v4:0"
    )
    bedrock_text_model_id: str = (
        "arn:aws:bedrock:eu-central-1:407179558514:"
        "inference-profile/eu.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )

    # AWS Cognito Configuration
    cognito_region: str = "eu-central-1"
    cognito_user_pool_id: str = Field(..., min_length=3)
    cognito_client_id: str | None = None  # Optional client validation

    @property
    def cognito_jwks_url(self) -> str:
        """Construct JWKS URL from pool ID and region."""
        return (
            f"https://cognito-idp.{self.cognito_region}.amazonaws.com/"
            f"{self.cognito_user_pool_id}/.well-known/jwks.json"
        )

    # Database Configuration
    database_url: str = Field(..., min_length=16)
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_echo: bool = False  # Set to True for SQL query logging

    # Circuit Breaker Configuration
    circuit_breaker_fail_max: int = 5
    circuit_breaker_timeout: int = 60  # seconds

    # CORS Configuration
    cors_origins: str = "http://localhost:3000,http://localhost:8000"
    cors_allow_methods: str = "GET,POST,OPTIONS"
    cors_allow_headers: str = "Authorization,Content-Type,X-Request-ID"

    # Rate Limiting
    rate_limit: str = "5/minute"

    # Temporary local testing switch
    allow_unauthenticated_requests: bool = False

    # External dependency resiliency
    external_timeout_seconds: float = 120.0
    external_retries: int = 3
    external_backoff_base_seconds: float = 0.25
    external_backoff_max_seconds: float = 3.0

    # Prompt/memory safeguards
    max_prompt_characters: int = 22000
    max_history_messages: int = 24
    summary_trigger_messages: int = 12

    # Logging
    log_level: str = "INFO"

    # Application
    app_name: str = "1CC RAG API"
    app_version: str = "1.0.0-production"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars from other services (e.g., KMS)
    )

    @staticmethod
    def parse_csv(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return self.parse_csv(self.cors_origins)

    @property
    def cors_allow_methods_list(self) -> list[str]:
        return self.parse_csv(self.cors_allow_methods)

    @property
    def cors_allow_headers_list(self) -> list[str]:
        return self.parse_csv(self.cors_allow_headers)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in {"dev", "test", "prod"}:
            raise ValueError("environment must be one of: dev, test, prod")
        return normalized


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings.
    """
    return Settings()
