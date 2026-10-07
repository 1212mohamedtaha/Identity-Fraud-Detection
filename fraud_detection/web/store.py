"""Thread-safe in-memory store of live dialogue sessions with expiry."""
import secrets
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Entry:
    session: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_seen: float = field(default_factory=time.monotonic)


class SessionStore:
    def __init__(self, ttl_seconds=3600, max_sessions=1000):
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._entries = {}
        self._lock = threading.Lock()

    def create(self, session):
        sid = secrets.token_urlsafe(24)
        with self._lock:
            self._evict()
            self._entries[sid] = Entry(session)
        return sid

    def get(self, sid):
        """Return the live entry for ``sid`` (refreshing its expiry), or None."""
        with self._lock:
            self._evict()
            entry = self._entries.get(sid)
            if entry:
                entry.last_seen = time.monotonic()
            return entry

    def delete(self, sid):
        with self._lock:
            self._entries.pop(sid, None)

    def _evict(self):
        now = time.monotonic()
        for sid in [s for s, e in self._entries.items() if now - e.last_seen > self.ttl]:
            del self._entries[sid]
        # Drop the least recently used sessions if still over capacity.
        overflow = len(self._entries) - self.max_sessions + 1
        if overflow > 0:
            for sid, _ in sorted(self._entries.items(), key=lambda kv: kv[1].last_seen)[:overflow]:
                del self._entries[sid]
