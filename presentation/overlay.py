from __future__ import annotations

from dataclasses import dataclass
from random import random
from typing import Dict, Iterable, Tuple

import cv2


@dataclass(frozen=True)
class OverlayConfig:
    line_thickness: int = 2
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.6
    font_thickness: int = 2


@dataclass(frozen=True)
class OverlayData:
    joint_points: Dict[str, Tuple[int, int]]
    joint_lines: Iterable[Tuple[str, str]]
    angles: Dict[str, float]
    is_risky: bool = False


class OverlayRenderer:
    """Video kareleri uzerine iskelet ve aci bilgisi cizer."""

    color_ok = (0, 200, 0)
    color_risk = (0, 0, 255)
    color_text = (245, 245, 245)

    def __init__(self, config: OverlayConfig | None = None) -> None:
        self.config = config or OverlayConfig()

    def render_frame(self, frame, data: OverlayData | None = None):
        if data is None:
            data = self._mock_data(frame)

        color = self.color_risk if data.is_risky else self.color_ok

        for joint_a, joint_b in data.joint_lines:
            if joint_a in data.joint_points and joint_b in data.joint_points:
                p1 = data.joint_points[joint_a]
                p2 = data.joint_points[joint_b]
                cv2.line(frame, p1, p2, color, self.config.line_thickness, cv2.LINE_AA)

        for _, point in data.joint_points.items():
            cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)

        self._draw_metrics(frame, data.angles, color)
        return frame

    def _draw_metrics(self, frame, angles: Dict[str, float], color) -> None:
        x, y = 12, 28
        for name, value in angles.items():
            text = f"{name}: {value:.1f}°"
            cv2.putText(
                frame,
                text,
                (x, y),
                self.config.font,
                self.config.font_scale,
                self.color_text,
                self.config.font_thickness,
                cv2.LINE_AA,
            )
            y += 24

    def _mock_data(self, frame) -> OverlayData:
        h, w = frame.shape[:2]

        joint_points = {
            "right_shoulder": (int(w * 0.55), int(h * 0.35)),
            "right_hip": (int(w * 0.55), int(h * 0.55)),
            "right_knee": (int(w * 0.53), int(h * 0.75)),
            "right_ankle": (int(w * 0.53), int(h * 0.90)),
        }

        joint_lines = [
            ("right_shoulder", "right_hip"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]

        angles = {
            "Hip": 90.0 + random() * 30.0,
            "Knee": 90.0 + random() * 40.0,
            "Back": 20.0 + random() * 20.0,
        }

        is_risky = angles["Knee"] > 120.0 or angles["Back"] > 35.0
        return OverlayData(
            joint_points=joint_points,
            joint_lines=joint_lines,
            angles=angles,
            is_risky=is_risky,
        )
