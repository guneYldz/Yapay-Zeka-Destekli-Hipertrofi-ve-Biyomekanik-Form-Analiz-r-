from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, sqrt


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0


def _vector(first: Point3D, second: Point3D) -> tuple[float, float, float]:
    return (first.x - second.x, first.y - second.y, first.z - second.z)


def calculate_angle(first: Point3D, joint: Point3D, third: Point3D) -> float:
    """Return angle in degrees for any 3-point coordinate set."""
    first_vector = _vector(first, joint)
    second_vector = _vector(third, joint)

    first_length = sqrt(sum(component * component for component in first_vector))
    second_length = sqrt(sum(component * component for component in second_vector))
    if first_length == 0 or second_length == 0:
        raise ValueError("Angle cannot be computed with a zero-length vector.")

    dot_product = sum(a * b for a, b in zip(first_vector, second_vector))
    cosine = dot_product / (first_length * second_length)
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))
