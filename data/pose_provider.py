from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from data.models import Landmark


@dataclass(frozen=True)
class NormalizationConfig:
    reference_scale: float | None = None
    center_on_centroid: bool = True


class PoseProvider:
    """Wrapper around pose coordinates so core logic stays MediaPipe-agnostic."""

    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self.config = config or NormalizationConfig()

    def from_mediapipe_like(self, landmarks: Iterable[object]) -> list[Landmark]:
        # TODO: MediaPipe landmark conversion will be added in the feature branch.
        raise NotImplementedError

    def normalize(self, landmarks: Sequence[Landmark]) -> list[Landmark]:
        # TODO: normalization rules will live in the data branch slice.
        raise NotImplementedError

    def _calculate_scale(self, landmarks: Sequence[Landmark]) -> float:
        # TODO: scale heuristics will be calibrated later.
        raise NotImplementedError
