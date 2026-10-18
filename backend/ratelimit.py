import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Allows each client at most `limit` calls in any `window_seconds` period.

    Kept in memory, so limits reset when the server restarts and are not shared between
    server processes. That is enough to stop one visitor using up a public demo's LLM quota.
    """

    def __init__(self, limit: int, window_seconds: float = 3600, clock=time.monotonic):
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def acquire(self, client: str) -> float | None:
        """Records a call. Returns None if allowed, or the seconds to wait if not."""
        if self._limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            calls = self._calls[client]
            while calls and calls[0] <= now - self._window:
                calls.popleft()
            if len(calls) >= self._limit:
                return calls[0] + self._window - now
            calls.append(now)
            return None
