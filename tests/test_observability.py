from fastapi.testclient import TestClient

from prior_auth.api import create_app


def test_metrics_endpoint_returns_prometheus_format() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "prior_auth_http_requests_total" in response.text


def test_http_request_metric_records_health_request() -> None:
    app = create_app()
    client = TestClient(app)

    health_response = client.get("/health")
    metrics_response = client.get("/metrics")

    assert health_response.status_code == 200
    assert metrics_response.status_code == 200
    assert 'path="/health"' in metrics_response.text
    assert 'status_code="200"' in metrics_response.text


def test_http_request_metric_records_not_found() -> None:
    app = create_app()
    client = TestClient(app)

    missing_response = client.get("/missing-endpoint")
    metrics_response = client.get("/metrics")

    assert missing_response.status_code == 404
    assert metrics_response.status_code == 200
    assert 'path="/missing-endpoint"' in metrics_response.text
    assert 'status_code="404"' in metrics_response.text