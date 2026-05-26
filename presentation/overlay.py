from __future__ import annotations

from dataclasses import dataclass

from domain.analyzer import AnalysisResult


@dataclass(frozen=True)
class OverlayItem:
    text: str
    color: tuple[int, int, int]
    priority: int = 0


class OverlayRenderer:
    """Skeleton for drawing analysis output without binding tests to OpenCV."""

    green = (0, 255, 0)
    red = (0, 0, 255)

    def render_analysis(self, analysis: AnalysisResult) -> list[OverlayItem]:
        # TODO: overlay composition will be implemented in the UI branch.
        raise NotImplementedError
