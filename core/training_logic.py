from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from core.geometry import Vector3D


@dataclass(frozen=True)
class MovementTiming:
    eccentric_seconds: float
    concentric_seconds: float


@dataclass(frozen=True)
class AnalysisFinding:
    code: str
    message: str
    severity: str = "warning"


@dataclass(frozen=True)
class MovementAnalysis:
    exercise: str
    valid: bool
    findings: tuple[AnalysisFinding, ...] = field(default_factory=tuple)
    knee_angle_degrees: float | None = None
    torso_angle_degrees: float | None = None


class TrainingLogic:
    """Pure form-analysis engine used by the feature/form-analysis-logic branch slice."""

    squat_min_depth_angle = 110.0
    deadlift_min_torso_angle = 140.0
    min_eccentric_seconds = 1.5
    min_concentric_seconds = 1.0

    def analyze_squat(
        self,
        landmarks: Mapping[str, Vector3D],
        timing: MovementTiming,
    ) -> MovementAnalysis:
        # TODO: squat depth, knee tracking, and tempo checks will be added in the branch implementation.
        raise NotImplementedError

    def analyze_deadlift(
        self,
        landmarks: Mapping[str, Vector3D],
        timing: MovementTiming,
    ) -> MovementAnalysis:
        # TODO: deadlift torso-angle and tempo checks will be added in the branch implementation.
        raise NotImplementedError

    def _knee_angle(self, landmarks: Mapping[str, Vector3D], side: str) -> float:
        raise NotImplementedError

    def _tempo_findings(self, timing: MovementTiming) -> list[AnalysisFinding]:
        raise NotImplementedError
