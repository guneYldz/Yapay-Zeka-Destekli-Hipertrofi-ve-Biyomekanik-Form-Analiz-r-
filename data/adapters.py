from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from data.models import Landmark


# mediapipe 33 landmark indeksleri
class LandmarkIndex:
    NOSE            = 0
    LEFT_EYE_INNER  = 1
    LEFT_EYE        = 2
    LEFT_EYE_OUTER  = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE       = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR        = 7
    RIGHT_EAR       = 8
    MOUTH_LEFT      = 9
    MOUTH_RIGHT     = 10
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13
    RIGHT_ELBOW     = 14
    LEFT_WRIST      = 15
    RIGHT_WRIST     = 16
    LEFT_PINKY      = 17
    RIGHT_PINKY     = 18
    LEFT_INDEX      = 19
    RIGHT_INDEX     = 20
    LEFT_THUMB      = 21
    RIGHT_THUMB     = 22
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
    LEFT_KNEE       = 25
    RIGHT_KNEE      = 26
    LEFT_ANKLE      = 27
    RIGHT_ANKLE     = 28
    LEFT_HEEL       = 29
    RIGHT_HEEL      = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    TOTAL = 33


@dataclass(frozen=True)
class PoseFrame:
    """Tek bir kareden gelen poz verisi. 33 landmark içerir."""

    landmarks: tuple[Landmark, ...]
    frame_index: int = 0
    timestamp_ms: float = 0.0

    def get(self, index: int) -> Landmark:
        if index < 0 or index >= len(self.landmarks):
            raise IndexError(f"Geçersiz indeks: {index} (toplam: {len(self.landmarks)})")
        return self.landmarks[index]

    def left_shoulder(self) -> Landmark:
        return self.get(LandmarkIndex.LEFT_SHOULDER)

    def right_shoulder(self) -> Landmark:
        return self.get(LandmarkIndex.RIGHT_SHOULDER)

    def left_hip(self) -> Landmark:
        return self.get(LandmarkIndex.LEFT_HIP)

    def right_hip(self) -> Landmark:
        return self.get(LandmarkIndex.RIGHT_HIP)

    def left_knee(self) -> Landmark:
        return self.get(LandmarkIndex.LEFT_KNEE)

    def right_knee(self) -> Landmark:
        return self.get(LandmarkIndex.RIGHT_KNEE)

    def left_ankle(self) -> Landmark:
        return self.get(LandmarkIndex.LEFT_ANKLE)

    def right_ankle(self) -> Landmark:
        return self.get(LandmarkIndex.RIGHT_ANKLE)

    def is_visible(self, index: int, threshold: float = 0.5) -> bool:
        return self.get(index).visibility >= threshold


class PoseAdapter(Protocol):
    def convert(self, raw_landmarks: object, frame_index: int = 0,
                timestamp_ms: float = 0.0) -> PoseFrame: ...


class MediaPipeAdapter:
    """
    MediaPipe çıktısını PoseFrame'e çevirir.
    frame_width/height verirsen piksel koordinatına dönüştürür,
    vermezsen 0..1 normalize kalır.
    """

    def __init__(self, visibility_threshold: float = 0.0,
                 frame_width: int = 1, frame_height: int = 1) -> None:
        self._vis_threshold = visibility_threshold
        self._fw = frame_width
        self._fh = frame_height

    def convert(self, raw_landmarks: object, frame_index: int = 0,
                timestamp_ms: float = 0.0) -> PoseFrame:
        landmarks_list = list(raw_landmarks)

        if len(landmarks_list) != LandmarkIndex.TOTAL:
            raise ValueError(
                f"Beklenen landmark sayısı {LandmarkIndex.TOTAL}, gelen: {len(landmarks_list)}"
            )

        converted: list[Landmark] = []
        for lm in landmarks_list:
            vis = float(getattr(lm, "visibility", 1.0))
            if vis < self._vis_threshold:
                vis = 0.0
            converted.append(Landmark(
                x=float(lm.x) * self._fw,
                y=float(lm.y) * self._fh,
                z=float(getattr(lm, "z", 0.0)),
                visibility=vis,
            ))

        return PoseFrame(
            landmarks=tuple(converted),
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def convert_batch(self, raw_landmark_list: list[object],
                      start_frame_index: int = 0, fps: float = 30.0) -> list[PoseFrame]:
        frames: list[PoseFrame] = []
        ms_per_frame = 1000.0 / fps if fps > 0 else 0.0
        for offset, raw in enumerate(raw_landmark_list):
            idx = start_frame_index + offset
            frames.append(self.convert(raw, frame_index=idx, timestamp_ms=idx * ms_per_frame))
        return frames


class MockAdapter:
    """
    Kamera olmadan test için sahte poz üretir.
    pose: 'squat', 'stand' veya 'random'
    """

    _SQUAT_TEMPLATE: dict[int, tuple[float, float, float]] = {
        LandmarkIndex.LEFT_SHOULDER:  (0.45, 0.30, -0.10),
        LandmarkIndex.RIGHT_SHOULDER: (0.55, 0.30, -0.10),
        LandmarkIndex.LEFT_HIP:       (0.44, 0.55,  0.00),
        LandmarkIndex.RIGHT_HIP:      (0.56, 0.55,  0.00),
        LandmarkIndex.LEFT_KNEE:      (0.42, 0.72,  0.05),
        LandmarkIndex.RIGHT_KNEE:     (0.58, 0.72,  0.05),
        LandmarkIndex.LEFT_ANKLE:     (0.43, 0.88,  0.10),
        LandmarkIndex.RIGHT_ANKLE:    (0.57, 0.88,  0.10),
    }

    _STAND_TEMPLATE: dict[int, tuple[float, float, float]] = {
        LandmarkIndex.LEFT_SHOULDER:  (0.46, 0.22, -0.05),
        LandmarkIndex.RIGHT_SHOULDER: (0.54, 0.22, -0.05),
        LandmarkIndex.LEFT_HIP:       (0.46, 0.50,  0.00),
        LandmarkIndex.RIGHT_HIP:      (0.54, 0.50,  0.00),
        LandmarkIndex.LEFT_KNEE:      (0.46, 0.70,  0.02),
        LandmarkIndex.RIGHT_KNEE:     (0.54, 0.70,  0.02),
        LandmarkIndex.LEFT_ANKLE:     (0.46, 0.90,  0.05),
        LandmarkIndex.RIGHT_ANKLE:    (0.54, 0.90,  0.05),
    }

    def __init__(self, pose: str = "squat", noise: float = 0.005,
                 seed: int | None = 42) -> None:
        if pose not in ("squat", "stand", "random"):
            raise ValueError(f"Geçersiz pose: '{pose}'. squat, stand veya random olmalı.")
        self._pose  = pose
        self._noise = noise
        self._rng   = random.Random(seed)

    def convert(self, raw_landmarks: object = None,  # type: ignore[assignment]
                frame_index: int = 0, timestamp_ms: float = 0.0) -> PoseFrame:
        return PoseFrame(
            landmarks=tuple(self._generate_landmarks()),
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def generate_sequence(self, num_frames: int, fps: float = 30.0) -> list[PoseFrame]:
        ms_per_frame = 1000.0 / fps if fps > 0 else 0.0
        return [
            self.convert(frame_index=i, timestamp_ms=i * ms_per_frame)
            for i in range(num_frames)
        ]

    def _generate_landmarks(self) -> list[Landmark]:
        if self._pose == "random":
            return [
                Landmark(
                    x=self._rng.random(),
                    y=self._rng.random(),
                    z=self._rng.uniform(-0.5, 0.5),
                    visibility=self._rng.uniform(0.5, 1.0),
                )
                for _ in range(LandmarkIndex.TOTAL)
            ]

        template = self._SQUAT_TEMPLATE if self._pose == "squat" else self._STAND_TEMPLATE
        landmarks: list[Landmark] = []

        for idx in range(LandmarkIndex.TOTAL):
            if idx in template:
                bx, by, bz = template[idx]
                x   = bx + self._rng.gauss(0, self._noise)
                y   = by + self._rng.gauss(0, self._noise)
                z   = bz + self._rng.gauss(0, self._noise * 0.5)
                vis = self._rng.uniform(0.85, 1.0)
            else:
                x   = 0.5 + self._rng.gauss(0, 0.05)
                y   = 0.5 + self._rng.gauss(0, 0.1)
                z   = self._rng.gauss(0, 0.02)
                vis = self._rng.uniform(0.3, 0.6)
            landmarks.append(Landmark(x=x, y=y, z=z, visibility=vis))

        return landmarks


def create_adapter(source: str = "mediapipe", **kwargs: object) -> MediaPipeAdapter | MockAdapter:
    """'mediapipe' veya 'mock' seçeneğine göre adaptör döndürür."""
    if source == "mediapipe":
        return MediaPipeAdapter(**kwargs)  # type: ignore[arg-type]
    if source == "mock":
        return MockAdapter(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Geçersiz kaynak: '{source}'. mediapipe veya mock olmalı.")
