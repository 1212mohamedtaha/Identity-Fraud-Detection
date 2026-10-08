import pytest
from fastapi.testclient import TestClient

from verity.api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app(llm=None))


def test_index_and_packs(client):
    page = client.get("/")
    assert page.status_code == 200 and "Verity" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    packs = {p["name"]: p for p in client.get("/api/packs").json()}
    assert set(packs) == {"identity", "cv"}
    assert "legacy" in packs["identity"]["policies"]


def test_identity_session_over_http_hides_scores(client):
    view = client.post("/api/sessions", json={"pack": "identity", "inputs": {}}).json()
    assert view["question"]["choices"] and not view["question"]["free_text"]
    while not view["finished"]:
        view = client.post(f"/api/sessions/{view['id']}/answer", json={"answer": "D"}).json()
    assert view["verdict"]["status"] == "refuted"
    assert all(t["score"] is None for t in view["history"])      # no correctness leak
    again = client.post(f"/api/sessions/{view['id']}/answer", json={"answer": "A"})
    assert again.status_code == 409


def test_cv_session_over_http_shows_feedback(client):
    view = client.post("/api/sessions", json={"pack": "cv", "inputs": {"cv": "Python developer"}}).json()
    assert view["question"]["free_text"]
    view = client.post(f"/api/sessions/{view['id']}/answer", json={"answer": "no idea"}).json()
    assert view["history"][0]["feedback"]
    assert client.get(f"/api/sessions/{view['id']}").json()["asked"] == 1


@pytest.mark.parametrize("body,status", [
    ({"pack": "nope"}, 404),
    ({"pack": "cv", "inputs": {"cv": ""}}, 400),
    ({"pack": "identity", "policy": "nope"}, 400),
])
def test_start_errors(client, body, status):
    assert client.post("/api/sessions", json=body).status_code == status


def test_bad_answers(client):
    view = client.post("/api/sessions", json={"pack": "identity"}).json()
    assert client.post(f"/api/sessions/{view['id']}/answer", json={"answer": "Z"}).status_code == 400
    assert client.post("/api/sessions/missing/answer", json={"answer": "A"}).status_code == 404


def test_openapi_schema_is_published(client):
    schema = client.get("/openapi.json").json()
    assert "/api/sessions/{session_id}/answer" in schema["paths"]
