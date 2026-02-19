import json
import logging
import random
import time
from collections.abc import Iterator

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pybreaker import CircuitBreakerError

from src.core.circuit_breaker import get_bedrock_breaker
from src.core.config import Settings
from src.core.exceptions import BedrockException

logger = logging.getLogger(__name__)


class BedrockClient:
    """
    Client for interacting with AWS Bedrock services.
    Handles text generation, streaming, and embedding generation with circuit breaker protection.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the Bedrock client.

        Args:
            settings (Settings): Application settings.

        Raises:
            BedrockException: If client initialization fails.
        """
        try:
            self.settings = settings
            config = Config(
                connect_timeout=settings.external_timeout_seconds,
                read_timeout=settings.external_timeout_seconds,
                retries={"max_attempts": 1, "mode": "standard"},
            )
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=settings.aws_region,
                config=config,
            )
            self.embedding_model_id = settings.bedrock_embedding_model_id
            self.text_model_id = settings.bedrock_text_model_id
            self.breaker = get_bedrock_breaker()
        except Exception as e:
            logger.error("Failed to initialize Bedrock client: %s", e)
            raise BedrockException(
                message="Failed to initialize Bedrock client",
                detail={"error": str(e)},
            )

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate text embedding using the Cohere Embed model with circuit breaker protection.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The generated embedding vector.

        Raises:
            BedrockException: If the Bedrock call fails or circuit is open.
        """
        try:
            return self.breaker.call(self._generate_embedding_impl, text)
        except CircuitBreakerError as e:
            logger.error("Circuit breaker open for Bedrock: %s", e)
            raise BedrockException(
                message="Bedrock service temporarily unavailable. Please try again later.",
                detail={"circuit_breaker": "open"},
            )

    def _generate_embedding_impl(self, text: str) -> list[float]:
        """Internal implementation of embedding generation."""
        logger.info("Generating embedding for text: %s...", text[:50])

        body = json.dumps(
            {
                "texts": [text],
                "input_type": "search_document",
                "embedding_types": ["float"],
            },
        )

        try:
            response = self._with_retries(
                self.bedrock_runtime.invoke_model,
                body=body,
                modelId=self.embedding_model_id,
                accept="*/*",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            embedding = response_body.get("embeddings")["float"][0]
            logger.info("Successfully generated embedding")
            return embedding
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            logger.error("Bedrock ClientError: %s - %s", error_code, error_message)

            if error_code == "ThrottlingException":
                raise BedrockException(
                    message="Rate limit exceeded. Please try again later.",
                    detail={"error_code": error_code, "error_message": error_message},
                )
            raise BedrockException(
                message="Failed to generate embedding",
                detail={"error_code": error_code, "error_message": error_message},
            )
        except (BotoCoreError, Exception) as e:
            logger.error("Failed to generate embedding: %s", e)
            raise BedrockException(
                message="Failed to generate embedding",
                detail={"error": str(e)},
            )

    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Anthropic Claude model with circuit breaker protection.

        Args:
            prompt (str): The input prompt for the model.

        Returns:
            str: The generated text response.

        Raises:
            BedrockException: If the Bedrock call fails or circuit is open.
        """
        try:
            return self.breaker.call(self._generate_text_impl, prompt)
        except CircuitBreakerError as e:
            logger.error("Circuit breaker open for Bedrock: %s", e)
            raise BedrockException(
                message="Bedrock service temporarily unavailable. Please try again later.",
                detail={"circuit_breaker": "open"},
            )

    def _generate_text_impl(self, prompt: str) -> str:
        """Internal implementation of text generation."""
        logger.info("Generating text for prompt: %s...", prompt[:50])

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10000,
                "temperature": 0.7,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    },
                ],
            },
        )

        try:
            response = self._with_retries(
                self.bedrock_runtime.invoke_model,
                body=body,
                modelId=self.text_model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())

            if "error" in response_body:
                raise BedrockException(
                    message="Text generation error",
                    detail={"error": response_body["error"]},
                )

            content = response_body.get("content", [])
            if content and content[0]["type"] == "text":
                logger.info("Successfully generated text")
                return content[0]["text"]
            return ""

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            logger.error("Bedrock ClientError: %s - %s", error_code, error_message)

            if error_code == "ThrottlingException":
                raise BedrockException(
                    message="Rate limit exceeded. Please try again later.",
                    detail={"error_code": error_code, "error_message": error_message},
                )
            raise BedrockException(
                message="Failed to generate text",
                detail={"error_code": error_code, "error_message": error_message},
            )
        except (BotoCoreError, Exception) as e:
            logger.error("Failed to generate text: %s", e)
            raise BedrockException(
                message="Failed to generate text",
                detail={"error": str(e)},
            )

    def generate_text_stream(self, prompt: str) -> Iterator[str]:
        """
        Generate text using streaming with circuit breaker protection.

        Args:
            prompt (str): The input prompt for the model.

        Yields:
            str: Text chunks as they are generated.

        Raises:
            BedrockException: If the Bedrock call fails or circuit is open.
        """
        try:
            # Note: Circuit breakers don't work well with generators
            # We check the state before starting
            if self.breaker.current_state == "open":
                raise BedrockException(
                    message="Bedrock service temporarily unavailable. Please try again later.",
                    detail={"circuit_breaker": "open"},
                )

            yield from self._generate_text_stream_impl(prompt)

        except CircuitBreakerError as e:
            logger.error("Circuit breaker open for Bedrock: %s", e)
            raise BedrockException(
                message="Bedrock service temporarily unavailable. Please try again later.",
                detail={"circuit_breaker": "open"},
            )

    def _generate_text_stream_impl(self, prompt: str) -> Iterator[str]:
        """Internal implementation of streaming text generation."""
        logger.info("Generating text stream for prompt: %s...", prompt[:50])

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10000,
                "temperature": 0.7,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    },
                ],
            },
        )

        try:
            response = self._with_retries(
                self.bedrock_runtime.invoke_model_with_response_stream,
                body=body,
                modelId=self.text_model_id,
                accept="application/json",
                contentType="application/json",
            )

            stream = response.get("body")
            if stream:
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_obj = json.loads(chunk.get("bytes").decode())

                        if chunk_obj.get("type") == "content_block_delta":
                            delta = chunk_obj.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield text
                        elif chunk_obj.get("type") == "message_stop":
                            logger.info("Streaming completed")
                            break

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            logger.error("Bedrock ClientError: %s - %s", error_code, error_message)

            if error_code == "ThrottlingException":
                raise BedrockException(
                    message="Rate limit exceeded. Please try again later.",
                    detail={"error_code": error_code, "error_message": error_message},
                )
            else:
                raise BedrockException(
                    message="Failed to generate text stream",
                    detail={"error_code": error_code, "error_message": error_message},
                )
        except (BotoCoreError, Exception) as e:
            logger.error("Failed to generate text stream: %s", e)
            raise BedrockException(
                message="Failed to generate text stream",
                detail={"error": str(e)},
            )

    def _with_retries(self, func, **kwargs):
        attempts = self.settings.external_retries
        last_error = None

        for attempt in range(1, attempts + 1):
            try:
                return func(**kwargs)
            except ClientError as exc:
                last_error = exc
                code = exc.response.get("Error", {}).get("Code", "")
                retryable = code in {
                    "ThrottlingException",
                    "InternalServerException",
                    "ModelTimeoutException",
                    "ServiceUnavailableException",
                }
                if not retryable or attempt == attempts:
                    break

                backoff = min(
                    self.settings.external_backoff_max_seconds,
                    self.settings.external_backoff_base_seconds * (2 ** (attempt - 1)),
                )
                sleep_for = backoff + random.uniform(0, 0.2)
                logger.warning(
                    "Bedrock transient error (%s) attempt %d/%d, retrying in %.2fs",
                    code,
                    attempt,
                    attempts,
                    sleep_for,
                )
                time.sleep(sleep_for)
            except BotoCoreError as exc:
                last_error = exc
                if attempt == attempts:
                    break
                backoff = min(
                    self.settings.external_backoff_max_seconds,
                    self.settings.external_backoff_base_seconds * (2 ** (attempt - 1)),
                )
                sleep_for = backoff + random.uniform(0, 0.2)
                logger.warning(
                    "Bedrock core transient error attempt %d/%d, retrying in %.2fs",
                    attempt,
                    attempts,
                    sleep_for,
                )
                time.sleep(sleep_for)

        raise last_error
