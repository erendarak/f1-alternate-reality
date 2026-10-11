"""Replay clock for the recorded race: play, pause, speed and seek over ``HistoricalRace``.

Like ``WorldRunner`` pacing, wall time only decides *where* in the recording we are; the
recording itself never changes. ``now`` is injectable so tests are exact and instant.
"""

import threading
import time
from collections.abc import Callable
from typing import Literal

ReplayStatus = Literal["READY", "PLAYING", "PAUSED", "FINISHED"]
REPLAY_SPEEDS: tuple[int, ...] = (1, 2, 5, 10, 25, 50)


class ReplayError(ValueError):
    """A replay command that can't be applied (bad speed, seek outside the race, ...)."""


class ReplayClock:
    def __init__(
        self,
        duration_s: float,
        *,
        speed: int = 10,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if speed not in REPLAY_SPEEDS:
            raise ReplayError(f"speed must be one of {REPLAY_SPEEDS}")
        self.duration_s = duration_s
        self._clock = clock
        self._lock = threading.Lock()
        self._speed = speed
        self._status: ReplayStatus = "READY"
        self._anchor_pos = 0.0
        self._anchor_wall = clock()

    def _position(self, now: float) -> float:
        if self._status != "PLAYING":
            return self._anchor_pos
        return min(self._anchor_pos + (now - self._anchor_wall) * self._speed, self.duration_s)

    def snapshot(self) -> tuple[float, ReplayStatus, int]:
        """``(replay_seconds, status, speed)``, consistent with each other."""
        with self._lock:
            now = self._clock()
            pos = self._position(now)
            if self._status == "PLAYING" and pos >= self.duration_s:
                self._status, self._anchor_pos = "FINISHED", self.duration_s
            return pos, self._status, self._speed

    def _reanchor(self, now: float) -> None:
        self._anchor_pos = self._position(now)
        self._anchor_wall = now

    def play(self) -> None:
        with self._lock:
            if self._status == "FINISHED":
                raise ReplayError("replay finished; seek back to watch again")
            now = self._clock()
            self._reanchor(now)
            self._status = "PLAYING"

    def pause(self) -> None:
        with self._lock:
            if self._status != "PLAYING":
                raise ReplayError(f"cannot pause while {self._status}")
            self._reanchor(self._clock())
            self._status = "PAUSED"

    def set_speed(self, speed: int) -> None:
        if speed not in REPLAY_SPEEDS:
            raise ReplayError(f"speed must be one of {REPLAY_SPEEDS}, got {speed}")
        with self._lock:
            self._reanchor(self._clock())
            self._speed = speed

    def seek(self, replay_s: float) -> None:
        if not 0.0 <= replay_s <= self.duration_s:
            raise ReplayError(f"seek target {replay_s:.1f}s outside 0..{self.duration_s:.1f}s")
        with self._lock:
            self._anchor_pos = replay_s
            self._anchor_wall = self._clock()
            if self._status in ("FINISHED", "READY"):
                self._status = "PAUSED"

    def step(self, seconds: float) -> None:
        """Move forward while paused (manual stepping)."""
        with self._lock:
            if self._status not in ("PAUSED", "READY"):
                raise ReplayError("pause the replay before stepping")
            self._anchor_pos = min(self._anchor_pos + seconds, self.duration_s)
            self._status = "PAUSED"
