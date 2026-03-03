import os
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("COGNITO_USER_POOL_ID", "eu-central-1_test")

from src.chat.router import get_chat_service
from src.chat.service import ChatService
from src.core.auth import get_current_user
from src.core.exceptions import APIException
from src.main import app


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides = {}
    yield
    app.dependency_overrides = {}


@pytest.fixture
def mock_chat_service():
    mock_service = MagicMock(spec=ChatService)
    mock_service.user = MagicMock()
    mock_service.user.email = "test@example.com"
    mock_service.generate_response.return_value = (
        "This is a mock response from the RAG API.",
        uuid4(),
        uuid4(),
        {"intent": "legal_lookup", "degraded_mode": False, "sources": []},
    )
    mock_service.generate_response_stream.return_value = (iter([]), uuid4())
    app.dependency_overrides[get_chat_service] = lambda: mock_service
    return mock_service


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_chat_requires_authentication(client, mock_chat_service):
    response = client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
    )
    assert response.status_code == 401


def test_chat_endpoint_valid_request(client, mock_chat_service):

    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }

    response = client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert response.json()["response"] == "This is a mock response from the RAG API."
    assert "metadata" in response.json()


def test_chat_endpoint_streaming(client, mock_chat_service):
    conversation_id = uuid4()

    def stream():
        yield {"event": "progress", "data": "retrieval_complete"}
        yield {"event": "data", "data": "Hello"}

    mock_chat_service.generate_response_stream.return_value = (
        stream(),
        conversation_id,
    )

    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }

    response = client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 200
    assert "conversation_id" in response.text


def test_chat_endpoint_invalid_request_missing_field(client, mock_chat_service):
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }
    response = client.post(
        "/api/v1/chat",
        json={"stream": False},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 422


def test_chat_endpoint_invalid_request_empty_messages(client, mock_chat_service):
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }
    response = client.post(
        "/api/v1/chat",
        json={"messages": [], "stream": False},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "No messages provided"


def test_chat_endpoint_api_exception_maps_status_code(client, mock_chat_service):
    mock_chat_service.generate_response.side_effect = APIException(
        message="Forbidden",
        status_code=403,
        error_type="authorization_error",
    )

    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }
    response = client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 403
