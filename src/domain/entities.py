"""
Domain Entities for Form Analyzer.
MANDATORY PURE: No external dependencies allowed.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class Joint(Enum):
    """Anatomical joints mapped for pose estimation."""
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
    """Supported exercise types."""
    SQUAT = auto()
    BENCH_PRESS = auto()


class RiskLevel(Enum):
    """Calculated risk level based on form issues."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class Point3D:
    """3D coordinate representation."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass(frozen=True)
class PoseFrame:
    """A single frame of pose estimation representing joint locations."""
    landmarks: Dict[Joint, Point3D] = field(default_factory=dict)
    
    def get_landmark(self, joint: Joint) -> Optional[Point3D]:
        """Safely retrieve a landmark."""
        return self.landmarks.get(joint)


@dataclass(frozen=True)
class FormIssue:
    """Represents a specific form issue detected in a frame."""
    description: str
    risk_score: int  # 0 to 100
    affected_joints: List[Joint] = field(default_factory=list)
