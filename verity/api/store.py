"""Thread-safe in-memory store of live sessions, with expiry."""
import secrets
import threading
import time


class Entry:
    def __init__(self, session):
        self.session = session
        self.lock = threading.Lock()       # one request at a time per session
        self.last_seen = time.monotonic()


class SessionStore:
    def __init__(self, ttl_seconds=3600, max_sessions=1000):
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._entries = {}
        self._lock = threading.Lock()

    def add(self, session):
        session_id = secrets.token_urlsafe(16)
        with self._lock:
            self._evict()
            self._entries[session_id] = Entry(session)
        return session_id

    def get(self, session_id):
        with self._lock:
            self._evict()
            entry = self._entries.get(session_id)
            if entry:
                entry.last_seen = time.monotonic()
            return entry

    def _evict(self):
        now = time.monotonic()
        for key in [k for k, e in self._entries.items() if now - e.last_seen > self.ttl]:
            del self._entries[key]
        # Over capacity: drop the least recently used sessions.
        overflow = len(self._entries) - self.max_sessions + 1
        if overflow > 0:
            oldest = sorted(self._entries.items(), key=lambda item: item[1].last_seen)[:overflow]
            for key, _ in oldest:
                del self._entries[key]
