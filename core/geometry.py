from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float = 0.0

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        raise NotImplementedError

    def magnitude(self) -> float:
        raise NotImplementedError


def vector_angle_degrees(first: Vector3D, second: Vector3D) -> float:
    # TODO: pure geometry calculations will be restored in the implementation branch.
    raise NotImplementedError


def joint_angle_degrees(first: Vector3D, joint: Vector3D, third: Vector3D) -> float:
    raise NotImplementedError


def velocity_magnitude(distance: float, delta_time_seconds: float) -> float:
    raise NotImplementedError


def acceleration_magnitude(initial_velocity: float, final_velocity: float, delta_time_seconds: float) -> float:
    raise NotImplementedError
