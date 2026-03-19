import os

# Force test environment (must be before imports)
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["COGNITO_USER_POOL_ID"] = "eu-central-1_test"
os.environ["ALLOW_UNAUTHENTICATED_REQUESTS"] = "false"

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.core.config import get_settings

# Clear cached settings before they're used
get_settings.cache_clear()

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


# --- Conversation Rename Tests ---


def test_rename_conversation_requires_authentication(client, mock_chat_service):
    conversation_id = str(uuid4())
    response = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "New Title"},
    )
    assert response.status_code == 401


def test_rename_conversation_success(client, mock_chat_service):
    conversation_id = uuid4()
    mock_chat_service.rename_conversation.return_value = {
        "id": conversation_id,
        "title": "New Title",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }

    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }

    response = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "New Title"},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "New Title"
    mock_chat_service.rename_conversation.assert_called_once_with(
        conversation_id, "New Title"
    )


def test_rename_conversation_not_found(client, mock_chat_service):
    conversation_id = uuid4()
    mock_chat_service.rename_conversation.side_effect = APIException(
        message="Conversation not found",
        status_code=404,
        error_type="not_found",
    )

    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }

    response = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "New Title"},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"


def test_rename_conversation_unauthorized(client, mock_chat_service):
    conversation_id = uuid4()
    mock_chat_service.rename_conversation.side_effect = APIException(
        message="You are not authorized to rename this conversation",
        status_code=403,
        error_type="authorization_error",
    )

    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }

    response = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "New Title"},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"]


def test_rename_conversation_invalid_uuid(client, mock_chat_service):
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }

    response = client.patch(
        "/api/v1/conversations/not-a-uuid",
        json={"title": "New Title"},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid conversation ID"


def test_rename_conversation_empty_title(client, mock_chat_service):
    conversation_id = uuid4()

    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }

    response = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": ""},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 422  # Pydantic validation error


def test_rename_conversation_title_too_long(client, mock_chat_service):
    conversation_id = uuid4()

    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-user-id",
        "email": "test@example.com",
    }

    response = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "x" * 121},  # 121 chars, max is 120
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 422  # Pydantic validation error
