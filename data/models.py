from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)
