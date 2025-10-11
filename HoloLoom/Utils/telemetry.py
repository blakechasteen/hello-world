import time
from contextlib import contextmanager

_counters = {}
_timings = {}

def inc(name: str, by: int = 1):
    _counters[name] = _counters.get(name, 0) + by

@contextmanager
def span(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        bucket = _timings.setdefault(name, [])
        bucket.append(dt)

def snapshot():
    return {
        "counters": dict(_counters),
        "timings": {k: list(v) for k, v in _timings.items()},
    }
