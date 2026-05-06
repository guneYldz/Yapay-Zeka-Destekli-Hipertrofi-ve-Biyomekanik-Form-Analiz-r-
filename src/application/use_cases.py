"""
Application use cases.
Orchestrates domain entities and rules.
"""

from typing import List, Tuple

from src.domain.entities import ExerciseType, FormIssue, PoseFrame, RiskLevel
from src.domain.rules import RiskScoringEngine, RuleRegistry


class AnalyzeFormUseCase:
    """Use case to analyze exercise form from a pose frame."""
    
    def execute(self, frame: PoseFrame, exercise: ExerciseType) -> Tuple[List[FormIssue], RiskLevel, str]:
        """
        Analyzes the frame and returns:
        1. List of FormIssues
        2. RiskLevel
        3. Explanation string
        """
        issues: List[FormIssue] = []
        
        # Get rules for the specific exercise
        rules = RuleRegistry.get_rules_for(exercise)
        
        # Validate using each rule
        for rule in rules:
            rule_issues = rule.validate(frame)
            issues.extend(rule_issues)
            
        # Calculate overall risk
        risk_level, score = RiskScoringEngine.calculate_risk(issues)
        
        # Generate explanation
        if not issues:
            explanation = "Perfect form! Keep it up."
        else:
            explanation = f"Detected {len(issues)} issue(s). Overall risk score: {score}/100. "
            if risk_level == RiskLevel.HIGH:
                explanation += "DANGER: Please stop and correct your form immediately!"
            elif risk_level == RiskLevel.MEDIUM:
                explanation += "WARNING: Be careful, you are at moderate risk."
            else:
                explanation += "Note: Minor form deviations found."
                
        return issues, risk_level, explanation
