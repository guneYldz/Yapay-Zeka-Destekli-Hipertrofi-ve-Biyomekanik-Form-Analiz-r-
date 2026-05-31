from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class Joint(Enum):
    NOSE = auto()
    LEFT_EYE = auto()
    RIGHT_EYE = auto()
    LEFT_SHOULDER = auto()
    RIGHT_SHOULDER = auto()
    LEFT_ELBOW = auto()
    RIGHT_ELBOW = auto()
    LEFT_WRIST = auto()
    RIGHT_WRIST = auto()
    LEFT_HIP = auto()
    RIGHT_HIP = auto()
    LEFT_KNEE = auto()
    RIGHT_KNEE = auto()
    LEFT_ANKLE = auto()
    RIGHT_ANKLE = auto()


class ExerciseType(Enum):
    SQUAT = auto()
    BENCH_PRESS = auto()


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass(frozen=True)
class PoseFrame:
    landmarks: Dict[Joint, Point3D] = field(default_factory=dict)

    def get_landmark(self, joint: Joint) -> Optional[Point3D]:
        return self.landmarks.get(joint)


@dataclass(frozen=True)
class FormIssue:
    description: str
    risk_score: int  # 0 to 100
    affected_joints: List[Joint] = field(default_factory=list)
