import logging

import requests
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.core.config import Settings, get_settings
from src.core.exceptions import APIException

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


class CognitoTokenVerifier:
    """Verify AWS Cognito JWT tokens."""

    def __init__(self, settings: Settings):
        """
        Initialize token verifier.

        Args:
            settings (Settings): Application settings.
        """
        self.settings = settings
        self.jwks = None
        self._fetch_jwks()

    def _fetch_jwks(self):
        """Fetch JSON Web Key Set from Cognito."""
        try:
            response = requests.get(self.settings.cognito_jwks_url, timeout=10)
            response.raise_for_status()
            self.jwks = response.json()
            logger.info("JWKS fetched successfully from Cognito")
        except Exception as e:
            logger.error("Failed to fetch JWKS: %s", e)
            raise APIException(
                message="Failed to initialize authentication",
                status_code=500,
                error_type="auth_initialization_error",
                detail={"error": str(e)},
            )

    def verify_token(self, token: str) -> dict:
        """
        Verify JWT token and return claims.

        Args:
            token (str): JWT token to verify.

        Returns:
            dict: Token claims including sub, email, username.

        Raises:
            HTTPException: If token is invalid.
        """
        try:
            # Get the key ID from the token header
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            # Find the matching key in JWKS
            key = None
            for jwk in self.jwks.get("keys", []):
                if jwk.get("kid") == kid:
                    key = jwk
                    break

            if not key:
                logger.warning("Unable to find matching key in JWKS")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token: unable to verify signature",
                )

            # Verify the token
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self.settings.cognito_client_id,
                issuer=f"https://cognito-idp.{self.settings.cognito_region}.amazonaws.com/{self.settings.cognito_user_pool_id}",
                options={"verify_aud": self.settings.cognito_client_id is not None},
            )

            logger.info("Token verified successfully for user: %s", claims.get("sub"))
            return claims

        except JWTError as e:
            logger.warning("JWT verification failed: %s", e)
            raise HTTPException(status_code=401, detail=f"Invalid token: {e!s}")
        except Exception as e:
            logger.error("Unexpected error during token verification: %s", e)
            raise HTTPException(status_code=401, detail="Invalid token")


# Global verifier instance
_verifier: CognitoTokenVerifier | None = None


def init_cognito_verifier(settings: Settings) -> CognitoTokenVerifier:
    """
    Initialize the global Cognito verifier.

    Args:
        settings (Settings): Application settings.

    Returns:
        CognitoTokenVerifier: Verifier instance.
    """
    global _verifier
    if _verifier is None:
        _verifier = CognitoTokenVerifier(settings)
    return _verifier


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> dict:
    """
    FastAPI dependency to get current authenticated user.

    Args:
        credentials (HTTPAuthorizationCredentials): Authorization header.

    Returns:
        dict: User claims from JWT token.

    Raises:
        HTTPException: If authentication fails.
    """
    settings = get_settings()
    if credentials is None or not credentials.credentials:
        if settings.allow_unauthenticated_requests:
            logger.warning("ALLOW_UNAUTHENTICATED_REQUESTS is enabled")
            return {
                "sub": "guest-local-user",
                "email": "guest@local.invalid",
                "cognito:username": "guest-local",
                "auth_mode": "unauthenticated",
            }
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    if _verifier is None:
        logger.error("Cognito verifier not initialized")
        raise HTTPException(
            status_code=500,
            detail="Authentication system not initialized",
        )

    token = credentials.credentials
    return _verifier.verify_token(token)
