"""Tests for health check endpoints."""



class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Test that /health returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_no_auth_required(self, client):
        """Test that /health does not require authentication."""
        response = client.get("/health")
        assert response.status_code == 200


class TestLivenessEndpoint:
    """Tests for /health/live endpoint."""

    def test_liveness_returns_200(self, client):
        """Test that /health/live returns 200."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_liveness_no_auth_required(self, client):
        """Test that /health/live does not require authentication."""
        response = client.get("/health/live")
        assert response.status_code == 200


class TestReadinessEndpoint:
    """Tests for /health/ready endpoint."""

    def test_readiness_returns_200_when_healthy(self, client):
        """Test that /health/ready returns 200 when all dependencies are healthy."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.json()["checks"]["database"] == "ok"
        assert response.json()["checks"]["redis"] == "ok"

    def test_readiness_no_auth_required(self, client):
        """Test that /health/ready does not require authentication."""
        response = client.get("/health/ready")
        assert response.status_code == 200

    def test_readiness_returns_503_when_db_unhealthy(self):
        """Test that /health/ready returns 503 when database is unhealthy."""
        from unittest.mock import AsyncMock, patch

        from fastapi.testclient import TestClient

        from app.main import create_app

        # Patch at the import location (where it's used in health.py)
        with patch(
            "app.api.health.check_db_health",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with patch(
                "app.api.health.check_redis_health",
                new_callable=AsyncMock,
                return_value=True,
            ):
                app = create_app()
                with TestClient(app) as test_client:
                    response = test_client.get("/health/ready")
                    assert response.status_code == 503
                    assert response.json()["status"] == "not ready"
                    assert response.json()["checks"]["database"] == "failed"

    def test_readiness_returns_503_when_redis_unhealthy(self):
        """Test that /health/ready returns 503 when Redis is unhealthy."""
        from unittest.mock import AsyncMock, patch

        from fastapi.testclient import TestClient

        from app.main import create_app

        # Patch at the import location (where it's used in health.py)
        with patch(
            "app.api.health.check_db_health",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch(
                "app.api.health.check_redis_health",
                new_callable=AsyncMock,
                return_value=False,
            ):
                app = create_app()
                with TestClient(app) as test_client:
                    response = test_client.get("/health/ready")
                    assert response.status_code == 503
                    assert response.json()["status"] == "not ready"
                    assert response.json()["checks"]["redis"] == "failed"
