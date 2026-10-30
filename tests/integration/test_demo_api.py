"""Exercise the real HTTP boundary with synthetic data; no model downloads."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from coach_emotion import server


@pytest.fixture
def client():
    with TestClient(server.create_app()) as session:
        yield session


def test_demo_assets_and_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/health").json()["device"] == "cpu"
    assert client.get("/static/../../pyproject.toml").status_code == 404


def test_example_is_explicitly_synthetic(client):
    data = client.get("/api/example").json()
    assert data["synthetic"] is True
    assert len(data["rows"]) == 4


@pytest.mark.parametrize(
    "query,headers,status",
    [
        ("", {"content-type": "audio/wav"}, 400),
        ("?consent=true", {"content-type": "text/plain"}, 415),
        ("?consent=true", {"content-type": "audio/wav", "content-length": "8388609"}, 413),
        ("?consent=true", {"content-type": "audio/wav", "content-length": "bad"}, 400),
    ],
)
def test_rejected_requests_do_not_run_models(client, monkeypatch, query, headers, status):
    def unexpected(*args):
        pytest.fail("Rejected input reached model execution")

    monkeypatch.setattr(server, "analyze_upload", unexpected)
    response = client.post("/api/analyze" + query, content=b"bad", headers=headers)
    assert response.status_code == status
    assert client.app.state.pending == 0


def test_stream_size_limit_without_content_length(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_BYTES", 8)
    response = client.post(
        "/api/analyze?consent=true",
        content=iter([b"12345", b"67890"]),
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 413
    assert client.app.state.pending == 0


def test_malformed_wav_is_rejected(client):
    response = client.post(
        "/api/analyze?consent=true", content=b"broken wav", headers={"content-type": "audio/wav"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("fail", [False, True])
def test_temporary_upload_removed_on_success_or_failure(client, monkeypatch, fail):
    paths = []

    def infer(filename, consent):
        path = Path(filename)
        paths.append(path)
        assert path.read_bytes() == b"test payload"
        assert consent is True
        if fail:
            raise RuntimeError("private transcript and machine path")
        return [[0, 4, 0, "neutro"]], "Completed"

    monkeypatch.setattr(server, "analyze_upload", infer)
    response = client.post(
        "/api/analyze?consent=true", content=b"test payload", headers={"content-type": "audio/wav"}
    )
    assert response.status_code == (503 if fail else 200)
    assert "private transcript" not in response.text
    assert not paths[0].exists()
    assert not paths[0].parent.exists()
    assert client.app.state.pending == 0
    if not fail:
        assert response.json()["synthetic"] is False


def test_queue_overflow(client):
    client.app.state.pending = 4
    response = client.post(
        "/api/analyze?consent=true", content=b"", headers={"content-type": "audio/wav"}
    )
    assert response.status_code == 429
    assert client.app.state.pending == 4
