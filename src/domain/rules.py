"""
Domain rules for exercise validation and risk scoring.
MANDATORY PURE: No external dependencies allowed.
"""

from abc import ABC, abstractmethod
from typing import List

from src.domain.entities import FormIssue, Joint, PoseFrame, RiskLevel, ExerciseType
from src.domain.geometry import calculate_angle_2d, calculate_vertical_angle


class ValidationRule(ABC):
    """Abstract base class for all form validation rules."""
    
    @abstractmethod
    def validate(self, frame: PoseFrame) -> List[FormIssue]:
        """Validate the pose frame and return a list of issues found."""
        pass


class SquatKneeAngleRule(ValidationRule):
    """Squat rule: Checks if the knee angle is dangerously deep (< 70)."""
    
    def validate(self, frame: PoseFrame) -> List[FormIssue]:
        issues = []
        
        # Check both legs
        for hip_j, knee_j, ankle_j, side in [
            (Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE, "Left"),
            (Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE, "Right")
        ]:
            hip = frame.get_landmark(hip_j)
            knee = frame.get_landmark(knee_j)
            ankle = frame.get_landmark(ankle_j)
            
            if hip and knee and ankle:
                angle = calculate_angle_2d(hip, knee, ankle)
                if angle < 70.0:
                    issues.append(FormIssue(
                        description=f"{side} knee angle is dangerously deep ({angle:.1f}° < 70°).",
                        risk_score=50,
                        affected_joints=[knee_j]
                    ))
                    
        return issues


class SquatKneeInwardRule(ValidationRule):
    """Squat rule: Checks for knee valgus (knees caving inward)."""
    
    def validate(self, frame: PoseFrame) -> List[FormIssue]:
        issues = []
        
        left_hip = frame.get_landmark(Joint.LEFT_HIP)
        right_hip = frame.get_landmark(Joint.RIGHT_HIP)
        left_knee = frame.get_landmark(Joint.LEFT_KNEE)
        right_knee = frame.get_landmark(Joint.RIGHT_KNEE)
        
        if left_hip and right_hip and left_knee and right_knee:
            # Assuming camera is front-facing. Left x > Right x in mediapipe typically,
            # but we just check distance ratio between hips and knees
            hip_dist = abs(left_hip.x - right_hip.x)
            knee_dist = abs(left_knee.x - right_knee.x)
            
            if hip_dist > 0 and knee_dist < hip_dist * 0.6:
                issues.append(FormIssue(
                    description="Knees are caving inward (Knee Valgus). High risk of injury.",
                    risk_score=80,
                    affected_joints=[Joint.LEFT_KNEE, Joint.RIGHT_KNEE]
                ))
                
        return issues


class SquatBackAngleRule(ValidationRule):
    """Squat rule: Checks if the back leans too far forward."""
    
    def validate(self, frame: PoseFrame) -> List[FormIssue]:
        issues = []
        
        hip = frame.get_landmark(Joint.LEFT_HIP) or frame.get_landmark(Joint.RIGHT_HIP)
        shoulder = frame.get_landmark(Joint.LEFT_SHOULDER) or frame.get_landmark(Joint.RIGHT_SHOULDER)
        
        if hip and shoulder:
            angle = calculate_vertical_angle(hip, shoulder)
            if angle > 45.0:
                issues.append(FormIssue(
                    description=f"Back leans too far forward ({angle:.1f}°). Keep your chest up.",
                    risk_score=40,
                    affected_joints=[Joint.LEFT_HIP, Joint.LEFT_SHOULDER]
                ))
                
        return issues


class BenchPressBarPathRule(ValidationRule):
    """Bench Press rule: Checks basic arm extension mechanics."""
    
    def validate(self, frame: PoseFrame) -> List[FormIssue]:
        issues = []
        
        # Simple heuristic: Check if wrists are too far horizontally from elbows
        for wrist_j, elbow_j, side in [
            (Joint.LEFT_WRIST, Joint.LEFT_ELBOW, "Left"),
            (Joint.RIGHT_WRIST, Joint.RIGHT_ELBOW, "Right")
        ]:
            wrist = frame.get_landmark(wrist_j)
            elbow = frame.get_landmark(elbow_j)
            
            if wrist and elbow:
                # Basic check for flaring elbows vs wrist alignment
                horiz_dist = abs(wrist.x - elbow.x)
                if horiz_dist > 0.15:  # Arbitrary threshold for demonstration
                    issues.append(FormIssue(
                        description=f"{side} wrist is not stacked over elbow during press.",
                        risk_score=35,
                        affected_joints=[wrist_j, elbow_j]
                    ))
                    
        return issues


class RiskScoringEngine:
    """Calculates an aggregate risk score and categorizes the risk level."""
    
    @staticmethod
    def calculate_risk(issues: List[FormIssue]) -> tuple[RiskLevel, int]:
        """
        Combine multiple issues using a weighted scoring system.
        Returns RiskLevel and total score out of 100.
        """
        if not issues:
            return RiskLevel.LOW, 0
            
        # Example weighting: take the maximum single risk and add 20% of the sum of others
        scores = sorted([issue.risk_score for issue in issues], reverse=True)
        max_score = scores[0]
        additional_score = sum(scores[1:]) * 0.2
        
        total_score = min(100, int(max_score + additional_score))
        
        if total_score < 30:
            level = RiskLevel.LOW
        elif total_score < 70:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.HIGH
            
        return level, total_score


class RuleRegistry:
    """Registry to fetch the correct rules for an exercise."""
    
    @staticmethod
    def get_rules_for(exercise: ExerciseType) -> List[ValidationRule]:
        if exercise == ExerciseType.SQUAT:
            return [
                SquatKneeAngleRule(),
                SquatKneeInwardRule(),
                SquatBackAngleRule()
            ]
        elif exercise == ExerciseType.BENCH_PRESS:
            return [
                BenchPressBarPathRule()
            ]
        return []
