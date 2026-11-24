"""Read-only liveness checks before a HIL actor may send an action."""
import math
import socket
import time


class ObservationDeadline:
    """Reject isolated stale predictions; latch a stop on repeated misses."""

    def __init__(self, maximum_age_ms, maximum_consecutive_misses=3):
        if not math.isfinite(maximum_age_ms) or maximum_age_ms <= 0:
            raise ValueError("observation deadline must be finite and positive")
        if maximum_consecutive_misses < 1:
            raise ValueError("maximum consecutive deadline misses must be positive")
        self.maximum_age_ms = maximum_age_ms
        self.maximum_consecutive_misses = maximum_consecutive_misses
        self.misses = 0

    def retry(self, age_ms, other_checks_healthy):
        if (age_ms is None or not math.isfinite(age_ms)
                or not other_checks_healthy or age_ms <= self.maximum_age_ms):
            self.misses = 0
            return False
        self.misses += 1
        return self.misses < self.maximum_consecutive_misses


def require_fresh_feedback(message, role, maximum_age):
    stamp = float(getattr(message, "timestamp", 0)) if message is not None else 0.0
    age = time.time() - stamp
    if not math.isfinite(age) or stamp <= 0 or age < -0.1 or age > maximum_age:
        raise RuntimeError(f"HIL {role} feedback is missing/stale ({age:.3f}s); motion refused")


class CANIdentityGuard:
    """A reused interface name must not hide a USB-CAN disconnect/re-enumeration."""
    def __init__(self, channels):
        self.identities = {name: socket.if_nametoindex(name) for name in channels}

    def check(self):
        for name, expected in self.identities.items():
            try:
                actual = socket.if_nametoindex(name)
            except OSError:
                raise RuntimeError(f"HIL CAN interface disappeared: {name}") from None
            if actual != expected:
                raise RuntimeError(f"HIL USB-CAN reconnected: {name}; restart the actor before sending commands")
