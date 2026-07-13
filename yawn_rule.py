from collections import deque
class YawnCounter:
    def __init__(self, n_required, window_seconds):
        self.n, self.window = n_required, window_seconds
        self._events, self._prev = deque(), False
    def update(self, yawning, now):
        if yawning and not self._prev:          # rising edge = one yawn
            self._events.append(now)
        self._prev = yawning
        while self._events and now - self._events[0] > self.window:
            self._events.popleft()
        return len(self._events) >= self.n, list(self._events)
