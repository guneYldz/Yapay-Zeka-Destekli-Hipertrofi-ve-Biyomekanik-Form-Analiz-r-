from __future__ import annotations

from dataclasses import dataclass, field

from core.calculator import CalculatedAngles


@dataclass(frozen=True)
class Thresholds:
    """Thresholds used by the analyzer to flag form issues."""

    squat_max_knee_angle_degrees: float = 110.0
    deadlift_min_torso_angle_degrees: float = 140.0


@dataclass(frozen=True)
class Warning:
    code: str
    message: str
    severity: str = "warning"


@dataclass(frozen=True)
class AnalysisResult:
    exercise: str
    ok: bool
    warnings: tuple[Warning, ...] = field(default_factory=tuple)
    angles: CalculatedAngles | None = None


class FormAnalyzer:
    """Threshold-based form checks using precomputed angles."""

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.thresholds = thresholds or Thresholds()

    def analyze_squat(self, angles: CalculatedAngles) -> AnalysisResult:
        warnings: list[Warning] = []

        if angles.knee_angle_degrees is None:
            warnings.append(
                Warning(
                    code="missing_knee_angle",
                    message="Diz açısı bulunamadı.",
                    severity="error",
                )
            )
        elif angles.knee_angle_degrees > self.thresholds.squat_max_knee_angle_degrees:
            warnings.append(
                Warning(
                    code="squat_depth_insufficient",
                    message="Squat derinliği yetersiz (diz açısı eşiğin üzerinde).",
                    severity="warning",
                )
            )

        return AnalysisResult(
            exercise="squat",
            ok=len(warnings) == 0,
            warnings=tuple(warnings),
            angles=angles,
        )

    def analyze_deadlift(self, angles: CalculatedAngles) -> AnalysisResult:
        warnings: list[Warning] = []

        if angles.torso_angle_degrees is None:
            warnings.append(
                Warning(
                    code="missing_torso_angle",
                    message="Torso/sırt açısı bulunamadı.",
                    severity="error",
                )
            )
        elif angles.torso_angle_degrees < self.thresholds.deadlift_min_torso_angle_degrees:
            warnings.append(
                Warning(
                    code="back_angle_injury_risk",
                    message="Sırt açısı sakatlık riski taşıyor (torso açısı eşiğin altında).",
                    severity="warning",
                )
            )

        return AnalysisResult(
            exercise="deadlift",
            ok=len(warnings) == 0,
            warnings=tuple(warnings),
            angles=angles,
        )
