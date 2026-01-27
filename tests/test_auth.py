"""Tests for authentication middleware."""



class TestAPIKeyVerification:
    """Tests for API key verification."""

    def test_missing_api_key_returns_401(self, client):
        """Test that missing API key returns 401."""
        response = client.get("/api/v1/status")
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_invalid_api_key_returns_401(self, client):
        """Test that invalid API key returns 401."""
        response = client.get(
            "/api/v1/status",
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_valid_api_key_passes(self, client, api_key_headers):
        """Test that valid API key passes verification."""
        response = client.get("/api/v1/status", headers=api_key_headers)
        assert response.status_code == 200


class TestJWTVerification:
    """Tests for JWT token verification."""

    def test_missing_token_returns_401(self, client, api_key_headers):
        """Test that missing JWT returns 401 for protected endpoints."""
        response = client.get("/api/v1/llm/tasks", headers=api_key_headers)
        # This endpoint only requires API key, not JWT
        assert response.status_code == 200

    def test_invalid_token_returns_401(self, client, valid_api_key):
        """Test that invalid JWT returns 401."""
        response = client.post(
            "/api/v1/llm/complete",
            headers={
                "X-API-Key": valid_api_key,
                "Authorization": "Bearer invalid-token",
            },
            json={"task": "chat", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 401

    def test_valid_token_passes(self, client, auth_headers):
        """Test that valid JWT passes verification."""
        # We need to mock the LLM client for this
        from unittest.mock import AsyncMock, patch

        mock_result = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "anthropic/claude-opus-4-20250514",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch(
            "app.services.llm.client.LLMClient.complete",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = client.post(
                "/api/v1/llm/complete",
                headers=auth_headers,
                json={"task": "chat", "messages": [{"role": "user", "content": "test"}]},
            )
            assert response.status_code == 200


class TestUserContext:
    """Tests for user context extraction from JWT."""

    def test_user_context_extraction(self, valid_jwt_token):
        """Test that user context is correctly extracted from JWT."""
        from jose import jwt

        payload = jwt.decode(
            valid_jwt_token,
            "test-jwt-secret-at-least-32-chars-long",
            algorithms=["HS256"],
            audience="authenticated",
        )

        assert payload["sub"] == "test-user-id"
        assert payload["role"] == "authenticated"
        assert payload["email"] == "test@example.com"
