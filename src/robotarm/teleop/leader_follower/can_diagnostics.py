"""Bounded receive-only diagnostics for native homing failures."""
from contextlib import contextmanager
import threading
import time


@contextmanager
def home_receive_diagnostics(robot, channel):
    # Observe the existing SDK receive path: no socket, request or mode change.
    callbacks = getattr(getattr(robot, '_ctx', None), '_parser_packet_fun_list', None)
    available = isinstance(callbacks, list)
    lock = threading.Lock()
    frames = {}

    def observe(frame):
        try:
            identifier = int(frame.arbitration_id)
            payload = bytes(frame.data).hex()
            now = time.monotonic()
            with lock:
                previous = frames.get(identifier)
                if previous is None:
                    if len(frames) < 128:
                        frames[identifier] = (1, 0, payload, payload, now)
                else:
                    count, changes, first, last, _ = previous
                    frames[identifier] = (
                        count + 1, changes + int(last != payload), first, payload, now,
                    )
        except (AttributeError, TypeError, ValueError):
            # Diagnostics must never interrupt the SDK receive thread.
            return

    if available:
        callbacks.append(observe)
    try:
        yield
    except BaseException:
        now = time.monotonic()
        with lock:
            snapshot = dict(frames)
        if not available:
            print(f'[home-rx] {channel}: SDK receive observer unavailable.')
        elif not snapshot:
            print(f'[home-rx] {channel}: no CAN frames received during native homing.')
        else:
            print(f'[home-rx] {channel}: received CAN frames; raw bytes are not decoded joint angles:')
            for identifier, (count, changes, first, last, stamp) in sorted(snapshot.items()):
                print(
                    f'[home-rx] 0x{identifier:03X} count={count} changes={changes} '
                    f'age_ms={max(0, now - stamp) * 1000:.0f} first={first} last={last}'
                )
        raise
    finally:
        if available:
            callbacks.remove(observe)
