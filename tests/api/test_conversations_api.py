from uuid import UUID

from fastapi.testclient import TestClient

from api.main import app


def test_conversation_lifecycle_without_startup_resources():
    client = TestClient(app)
    create_response = client.post(
        "/api/v1/conversations",
        json={"user_id": "test-user"},
    )

    assert create_response.status_code == 201
    conversation_id = create_response.json()["id"]
    UUID(conversation_id)

    get_response = client.get(
        f"/api/v1/conversations/{conversation_id}"
    )
    assert get_response.status_code == 200
    assert get_response.json()["id"] == conversation_id

    messages_response = client.get(
        f"/api/v1/conversations/{conversation_id}/messages"
    )
    assert messages_response.status_code == 200
    assert messages_response.json() == []

    delete_response = client.delete(
        f"/api/v1/conversations/{conversation_id}"
    )
    assert delete_response.status_code == 204

    missing_response = client.get(
        f"/api/v1/conversations/{conversation_id}"
    )
    assert missing_response.status_code == 404
    assert missing_response.json()["error"]["code"] == (
        "CONVERSATION_NOT_FOUND"
    )


def test_message_rejects_unknown_model_before_calling_openai():
    client = TestClient(app)
    create_response = client.post(
        "/api/v1/conversations",
        json={"user_id": "test-user"},
    )
    conversation_id = create_response.json()["id"]

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "content": "What is RAG?",
            "model": "string",
            "temperature": 0,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
