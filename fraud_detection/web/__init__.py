"""Flask application: a JSON API plus a single-page chat interface."""
import os
import secrets

from flask import Flask, jsonify, render_template, request, session as browser_session

from ..session import Engine
from .store import SessionStore

SESSION_KEY = "dialogue_id"


def create_app(engine=None, store=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_HTTPONLY=True,
    )
    app.extensions["engine"] = engine  # loaded lazily so importing the app stays cheap
    store = store or SessionStore()

    def get_engine():
        if app.extensions["engine"] is None:
            app.extensions["engine"] = Engine()
        return app.extensions["engine"]

    def current_entry():
        sid = browser_session.get(SESSION_KEY)
        return store.get(sid) if sid else None

    def snapshot(dialogue):
        return {"turn": dialogue.current.to_dict(), "history": dialogue.history}

    def error(message, status):
        return jsonify({"error": message}), status

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/dialogue")
    def start_dialogue():
        """Start a fresh dialogue, replacing any previous one of this browser."""
        old_sid = browser_session.get(SESSION_KEY)
        if old_sid:
            store.delete(old_sid)
        dialogue = get_engine().new_session()
        browser_session[SESSION_KEY] = store.create(dialogue)
        return jsonify(snapshot(dialogue)), 201

    @app.get("/api/dialogue")
    def get_dialogue():
        """Current dialogue state, so a page reload can resume the chat (204 if there is none)."""
        entry = current_entry()
        if entry is None:
            return "", 204
        with entry.lock:
            return jsonify(snapshot(entry.session))

    @app.post("/api/dialogue/answer")
    def answer():
        entry = current_entry()
        if entry is None:
            return error("No active dialogue; it may have expired", 404)
        payload = request.get_json(silent=True) or {}
        choice = payload.get("choice")
        if not isinstance(choice, str):
            return error("Body must be JSON like {\"choice\": \"A\"}", 400)
        with entry.lock:
            dialogue = entry.session
            if dialogue.finished:
                return error("This dialogue is already finished", 409)
            try:
                dialogue.answer(choice)
            except ValueError as exc:
                return error(str(exc), 400)
            return jsonify(snapshot(dialogue))

    return app
