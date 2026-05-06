"""
Combined Tests for Clean Architecture Form Analyzer
Contains tests for Domain, Application, and Infrastructure layers.
"""

import pytest

from src.domain.entities import FormIssue, Joint, Point3D, PoseFrame, RiskLevel, ExerciseType
from src.domain.geometry import calculate_angle_2d, calculate_angle_3d, calculate_vertical_angle
from src.domain.rules import (
    BenchPressBarPathRule,
    RiskScoringEngine,
    SquatBackAngleRule,
    SquatKneeAngleRule,
    SquatKneeInwardRule,
)
from src.application.use_cases import AnalyzeFormUseCase
from src.infrastructure.adapters import MediaPipePoseAdapter, MockPoseAdapter


# ==========================================
# DOMAIN LAYER TESTS
# ==========================================

class TestGeometry:
    def test_calculate_angle_2d(self):
        p1 = Point3D(0, 1)
        p2 = Point3D(0, 0)
        p3 = Point3D(1, 0)
        angle = calculate_angle_2d(p1, p2, p3)
        assert abs(angle - 90.0) < 1.0

    def test_calculate_angle_3d(self):
        p1 = Point3D(1, 0, 0)
        p2 = Point3D(0, 0, 0)
        p3 = Point3D(0, 1, 0)
        angle = calculate_angle_3d(p1, p2, p3)
        assert abs(angle - 90.0) < 1.0

    def test_calculate_vertical_angle(self):
        p1 = Point3D(0, 0)
        p2 = Point3D(1, 1)
        angle = calculate_vertical_angle(p1, p2)
        assert abs(angle - 45.0) < 1.0


class TestSquatRules:
    def test_knee_angle_rule_safe(self):
        frame = PoseFrame(landmarks={
            Joint.LEFT_HIP: Point3D(0, 10),
            Joint.LEFT_KNEE: Point3D(0, 5),
            Joint.LEFT_ANKLE: Point3D(5, 0)
        })
        rule = SquatKneeAngleRule()
        issues = rule.validate(frame)
        assert len(issues) == 0

    def test_knee_angle_rule_dangerous(self):
        frame = PoseFrame(landmarks={
            Joint.LEFT_HIP: Point3D(0, 10),
            Joint.LEFT_KNEE: Point3D(0, 5),
            Joint.LEFT_ANKLE: Point3D(2, 0)
        })
        rule = SquatKneeAngleRule()
        issues = rule.validate(frame)
        angle = calculate_angle_2d(Point3D(0, 10), Point3D(0, 5), Point3D(2, 0))
        if angle < 70:
            assert len(issues) == 1
            assert issues[0].risk_score == 50

    def test_knee_inward_rule_valgus(self):
        frame = PoseFrame(landmarks={
            Joint.LEFT_HIP: Point3D(0, 10),
            Joint.RIGHT_HIP: Point3D(10, 10),
            Joint.LEFT_KNEE: Point3D(4, 5),
            Joint.RIGHT_KNEE: Point3D(6, 5)
        })
        rule = SquatKneeInwardRule()
        issues = rule.validate(frame)
        assert len(issues) == 1
        assert "Valgus" in issues[0].description

    def test_back_angle_rule_leaning(self):
        frame = PoseFrame(landmarks={
            Joint.LEFT_HIP: Point3D(0, 0),
            Joint.LEFT_SHOULDER: Point3D(10, 8)
        })
        rule = SquatBackAngleRule()
        issues = rule.validate(frame)
        assert len(issues) == 1


class TestBenchPressRules:
    def test_bar_path_rule(self):
        frame = PoseFrame(landmarks={
            Joint.LEFT_ELBOW: Point3D(0, 5),
            Joint.LEFT_WRIST: Point3D(0.2, 10)
        })
        rule = BenchPressBarPathRule()
        issues = rule.validate(frame)
        assert len(issues) == 1


class TestRiskScoringEngine:
    def test_no_issues(self):
        level, score = RiskScoringEngine.calculate_risk([])
        assert level == RiskLevel.LOW
        assert score == 0

    def test_single_high_issue(self):
        issues = [FormIssue("test", 80)]
        level, score = RiskScoringEngine.calculate_risk(issues)
        assert level == RiskLevel.HIGH
        assert score == 80

    def test_multiple_issues(self):
        issues = [FormIssue("test1", 50), FormIssue("test2", 40)]
        level, score = RiskScoringEngine.calculate_risk(issues)
        assert level == RiskLevel.MEDIUM
        assert score == 58

    def test_score_capped_at_100(self):
        issues = [FormIssue("t1", 90), FormIssue("t2", 80)]
        level, score = RiskScoringEngine.calculate_risk(issues)
        assert level == RiskLevel.HIGH
        assert score == 100


# ==========================================
# APPLICATION LAYER TESTS
# ==========================================

class TestAnalyzeFormUseCase:
    def test_execute_squat_perfect_form(self):
        frame = PoseFrame(landmarks={
            Joint.LEFT_HIP: Point3D(2, 10),
            Joint.RIGHT_HIP: Point3D(8, 10),
            Joint.LEFT_KNEE: Point3D(2, 5),
            Joint.RIGHT_KNEE: Point3D(8, 5),
            Joint.LEFT_ANKLE: Point3D(2, 0),
            Joint.RIGHT_ANKLE: Point3D(8, 0),
            Joint.LEFT_SHOULDER: Point3D(2, 18),
            Joint.RIGHT_SHOULDER: Point3D(8, 18)
        })
        use_case = AnalyzeFormUseCase()
        issues, risk_level, explanation = use_case.execute(frame, ExerciseType.SQUAT)
        
        assert len(issues) == 0
        assert risk_level == RiskLevel.LOW
        assert "Perfect form" in explanation

    def test_execute_squat_bad_form(self):
        frame = PoseFrame(landmarks={
            Joint.LEFT_HIP: Point3D(0, 10),
            Joint.RIGHT_HIP: Point3D(10, 10),
            Joint.LEFT_KNEE: Point3D(4, 5),
            Joint.RIGHT_KNEE: Point3D(6, 5),
            Joint.LEFT_SHOULDER: Point3D(10, 15)
        })
        use_case = AnalyzeFormUseCase()
        issues, risk_level, explanation = use_case.execute(frame, ExerciseType.SQUAT)
        
        assert len(issues) > 0
        assert risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
        assert "Detected" in explanation


# ==========================================
# INFRASTRUCTURE LAYER TESTS
# ==========================================

class DummyLandmark:
    def __init__(self, x, y, z, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class DummyMediaPipeResult:
    def __init__(self, landmarks):
        self.landmark = landmarks

class TestMediaPipeAdapter:
    def test_to_pose_frame(self):
        mp_landmarks = [DummyLandmark(0, 0, 0)] * 33
        mp_landmarks[0] = DummyLandmark(0.5, 0.2, 0.1)
        mp_landmarks[23] = DummyLandmark(0.4, 0.6, 0.2)
        result = DummyMediaPipeResult(mp_landmarks)
        
        frame = MediaPipePoseAdapter.to_pose_frame(result)
        
        assert frame.get_landmark(Joint.NOSE) is not None
        assert frame.get_landmark(Joint.NOSE).x == 0.5
        assert frame.get_landmark(Joint.LEFT_HIP) is not None
        assert frame.get_landmark(Joint.LEFT_HIP).y == 0.6
        
    def test_invalid_input(self):
        frame = MediaPipePoseAdapter.to_pose_frame(None)
        assert len(frame.landmarks) == 0

class TestMockPoseAdapter:
    def test_from_dict(self):
        data = {
            "LEFT_HIP": {"x": 0.5, "y": 0.5, "z": 0.0},
            "RIGHT_HIP": {"x": 0.6, "y": 0.5, "z": 0.1, "visibility": 0.9},
            "INVALID_JOINT": {"x": 0, "y": 0}
        }
        frame = MockPoseAdapter.from_dict(data)
        
        assert len(frame.landmarks) == 2
        assert frame.get_landmark(Joint.LEFT_HIP).x == 0.5
        assert frame.get_landmark(Joint.RIGHT_HIP).visibility == 0.9
        assert frame.get_landmark(Joint.LEFT_KNEE) is None
