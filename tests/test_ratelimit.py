from backend.ratelimit import RateLimiter


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_allows_up_to_the_limit_then_reports_the_wait():
    clock = Clock()
    limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)

    assert limiter.acquire("a") is None
    clock.now += 10
    assert limiter.acquire("a") is None
    clock.now += 5
    assert limiter.acquire("a") == 45.0  # the first call leaves the window in 45s


def test_calls_expire_after_the_window():
    clock = Clock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)

    limiter.acquire("a")
    clock.now += 60

    assert limiter.acquire("a") is None


def test_clients_are_counted_separately():
    limiter = RateLimiter(limit=1, window_seconds=60, clock=Clock())

    assert limiter.acquire("a") is None
    assert limiter.acquire("b") is None
    assert limiter.acquire("a") is not None


def test_a_limit_of_zero_means_unlimited():
    limiter = RateLimiter(limit=0, clock=Clock())

    assert all(limiter.acquire("a") is None for _ in range(1000))
