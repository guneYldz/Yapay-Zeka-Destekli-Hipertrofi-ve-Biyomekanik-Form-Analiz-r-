from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from data.models import Landmark

@dataclass(frozen=True)
class NormalizationConfig:
    reference_scale: float | None = None
    center_on_centroid: bool = True

class PoseProvider:
    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self.config = config or NormalizationConfig()

    def from_mediapipe_like(self, landmarks: Iterable[object]) -> list[Landmark]:
        return [
            Landmark(
                x=getattr(lm, "x", 0.0),
                y=getattr(lm, "y", 0.0),
                z=getattr(lm, "z", 0.0),
                visibility=getattr(lm, "visibility", 1.0)
            )
            for lm in landmarks
        ]

    def normalize(self, landmarks: Sequence[Landmark]) -> list[Landmark]:
        if not landmarks:
            return []

        cx = cy = cz = 0.0
        if self.config.center_on_centroid:
            valid = [lm for lm in landmarks if lm.visibility > 0.5]
            if not valid:
                valid = landmarks
            cx = sum(lm.x for lm in valid) / len(valid)
            cy = sum(lm.y for lm in valid) / len(valid)
            cz = sum(lm.z for lm in valid) / len(valid)

        scale = 1.0
        if self.config.reference_scale is not None:
            current_scale = self._calculate_scale(landmarks)
            scale = self.config.reference_scale / current_scale

        return [
            Landmark(
                x=(lm.x - cx) * scale,
                y=(lm.y - cy) * scale,
                z=(lm.z - cz) * scale,
                visibility=lm.visibility
            )
            for lm in landmarks
        ]

    def _calculate_scale(self, landmarks: Sequence[Landmark]) -> float:
        if len(landmarks) <= 24:
            return 1.0

        ls = landmarks[11]
        rs = landmarks[12]
        lh = landmarks[23]
        rh = landmarks[24]

        mx = (ls.x + rs.x) / 2.0 - (lh.x + rh.x) / 2.0
        my = (ls.y + rs.y) / 2.0 - (lh.y + rh.y) / 2.0
        mz = (ls.z + rs.z) / 2.0 - (lh.z + rh.z) / 2.0

        dist = math.sqrt(mx * mx + my * my + mz * mz)
        return dist if dist > 0.001 else 1.0
