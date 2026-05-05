from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class AnalysisWarning:
    code: str
    message: str


@dataclass(frozen=True)
class FormAnalysis:
    exercise: str
    is_valid: bool
    warnings: tuple[AnalysisWarning, ...] = field(default_factory=tuple)


class FormAnalyzer:
    """Threshold-based form checker that consumes angle values from calculator.py."""

    squat_min_depth_angle = 110.0
    deadlift_max_back_angle = 145.0
    deadlift_min_knee_angle = 150.0

    def analyze_squat(self, angles: Mapping[str, float]) -> FormAnalysis:
        warnings: list[AnalysisWarning] = []

        knee_angle = angles.get("knee_angle")
        if knee_angle is None:
            warnings.append(
                AnalysisWarning(
                    code="missing_knee_angle",
                    message="Knee angle is required for squat analysis.",
                )
            )
        elif knee_angle > self.squat_min_depth_angle:
            warnings.append(
                AnalysisWarning(
                    code="shallow_squat",
                    message="Squat derinliği yeterli değil.",
                )
            )

        return FormAnalysis(exercise="squat", is_valid=not warnings, warnings=tuple(warnings))

    def analyze_deadlift(self, angles: Mapping[str, float]) -> FormAnalysis:
        warnings: list[AnalysisWarning] = []

        back_angle = angles.get("back_angle")
        if back_angle is None:
            warnings.append(
                AnalysisWarning(
                    code="missing_back_angle",
                    message="Back angle is required for deadlift analysis.",
                )
            )
        elif back_angle < self.deadlift_max_back_angle:
            warnings.append(
                AnalysisWarning(
                    code="spine_risk",
                    message="Sırt açısı sakatlık riski taşıyor.",
                )
            )

        knee_angle = angles.get("knee_angle")
        if knee_angle is None:
            warnings.append(
                AnalysisWarning(
                    code="missing_knee_angle",
                    message="Knee angle is required for deadlift analysis.",
                )
            )
        elif knee_angle < self.deadlift_min_knee_angle:
            warnings.append(
                AnalysisWarning(
                    code="knee_risk",
                    message="Diz açısı deadlift için fazla kapalı.",
                )
            )

        return FormAnalysis(exercise="deadlift", is_valid=not warnings, warnings=tuple(warnings))
