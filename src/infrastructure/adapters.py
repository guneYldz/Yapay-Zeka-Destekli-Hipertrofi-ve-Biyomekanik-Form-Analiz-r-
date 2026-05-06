"""
Infrastructure adapters to convert external formats to Domain Entities.
"""

from typing import Any, Dict

from src.domain.entities import Joint, Point3D, PoseFrame

# Map MediaPipe indices to our Domain Joint Enum
# Following the standard mediapipe pose landmark indices
MEDIAPIPE_JOINT_MAP = {
    0: Joint.NOSE,
    2: Joint.LEFT_EYE,
    5: Joint.RIGHT_EYE,
    11: Joint.LEFT_SHOULDER,
    12: Joint.RIGHT_SHOULDER,
    13: Joint.LEFT_ELBOW,
    14: Joint.RIGHT_ELBOW,
    15: Joint.LEFT_WRIST,
    16: Joint.RIGHT_WRIST,
    23: Joint.LEFT_HIP,
    24: Joint.RIGHT_HIP,
    25: Joint.LEFT_KNEE,
    26: Joint.RIGHT_KNEE,
    27: Joint.LEFT_ANKLE,
    28: Joint.RIGHT_ANKLE,
}


class MediaPipePoseAdapter:
    """Adapts a MediaPipe NormalizedLandmarkList to a Domain PoseFrame."""
    
    @staticmethod
    def to_pose_frame(mp_landmarks: Any) -> PoseFrame:
        """
        Converts MediaPipe landmark list to PoseFrame.
        mp_landmarks is assumed to be an object with a `landmark` attribute,
        which is an iterable of objects with x, y, z, visibility.
        """
        landmarks_dict = {}
        
        try:
            for idx, mp_lm in enumerate(mp_landmarks.landmark):
                if idx in MEDIAPIPE_JOINT_MAP:
                    joint = MEDIAPIPE_JOINT_MAP[idx]
                    landmarks_dict[joint] = Point3D(
                        x=mp_lm.x,
                        y=mp_lm.y,
                        z=mp_lm.z,
                        visibility=getattr(mp_lm, 'visibility', 1.0)
                    )
        except AttributeError:
            # Handle cases where mp_landmarks is just a list or similar mock
            pass
            
        return PoseFrame(landmarks=landmarks_dict)


class MockPoseAdapter:
    """Mock adapter for testing or reading JSON directly."""
    
    @staticmethod
    def from_dict(data: Dict[str, Dict[str, float]]) -> PoseFrame:
        """
        Converts a simple dictionary to PoseFrame.
        Expects format: {"LEFT_HIP": {"x": 0.5, "y": 0.5, "z": 0.0}, ...}
        """
        landmarks_dict = {}
        for joint_name, coords in data.items():
            try:
                joint = Joint[joint_name]
                landmarks_dict[joint] = Point3D(
                    x=coords.get("x", 0.0),
                    y=coords.get("y", 0.0),
                    z=coords.get("z", 0.0),
                    visibility=coords.get("visibility", 1.0)
                )
            except KeyError:
                pass  # Ignore invalid joint names
                
        return PoseFrame(landmarks=landmarks_dict)
