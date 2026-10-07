import pytest

from fraud_detection.web import create_app


@pytest.fixture
def client(engine):
    app = create_app(engine=engine)
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_serves_the_chat_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Identity verification" in response.data
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]


def test_no_dialogue_before_start(client):
    assert client.get("/api/dialogue").status_code == 204
    assert client.post("/api/dialogue/answer", json={"choice": "A"}).status_code == 404


def test_full_dialogue_over_http(client):
    started = client.post("/api/dialogue")
    assert started.status_code == 201
    turn = started.get_json()["turn"]
    assert turn["status"] == "question" and len(turn["choices"]) >= 2

    # Reloading the page resumes the same question.
    assert client.get("/api/dialogue").get_json()["turn"]["question"] == turn["question"]

    for _ in range(100):
        data = client.post("/api/dialogue/answer", json={"choice": "D"}).get_json()
        if data["turn"]["status"] != "question":
            break
    assert data["turn"]["status"] == "fraud"
    assert len(data["history"]) == data["turn"]["answered"]

    assert client.post("/api/dialogue/answer", json={"choice": "A"}).status_code == 409


@pytest.mark.parametrize("body", [{}, {"choice": 3}, {"choice": "Z"}])
def test_bad_answers_are_rejected(client, body):
    client.post("/api/dialogue")
    assert client.post("/api/dialogue/answer", json=body).status_code == 400


def test_restart_replaces_the_dialogue(client):
    client.post("/api/dialogue")
    client.post("/api/dialogue/answer", json={"choice": "D"})
    fresh = client.post("/api/dialogue").get_json()
    assert fresh["history"] == [] and fresh["turn"]["answered"] == 0


def test_session_store_expires_and_caps_entries():
    from fraud_detection.web.store import SessionStore

    store = SessionStore(ttl_seconds=-1)
    sid = store.create(object())
    assert store.get(sid) is None

    store = SessionStore(max_sessions=2)
    sids = [store.create(object()) for _ in range(3)]
    assert store.get(sids[0]) is None and store.get(sids[2]) is not None
