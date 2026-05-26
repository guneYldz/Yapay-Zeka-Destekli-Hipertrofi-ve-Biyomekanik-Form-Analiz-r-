from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalculatedAngles:
    """Angle outputs produced by the future calculator implementation."""

    knee_angle_degrees: float | None = None
    torso_angle_degrees: float | None = None


class AngleCalculator:
    """Placeholder for the landmark -> angles pipeline (to be implemented later)."""

    def calculate(self, *args, **kwargs) -> CalculatedAngles:  # noqa: ANN002, ANN003
        raise NotImplementedError
