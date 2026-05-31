from typing import List, Tuple

from domain.entities import ExerciseType, FormIssue, PoseFrame, RiskLevel
from domain.rules import RiskScoringEngine, RuleRegistry


class AnalyzeFormUseCase:
    def execute(self, frame: PoseFrame, exercise: ExerciseType) -> Tuple[List[FormIssue], RiskLevel, str]:
        issues: List[FormIssue] = []

        rules = RuleRegistry.get_rules_for(exercise)

        for rule in rules:
            rule_issues = rule.validate(frame)
            issues.extend(rule_issues)

        risk_level, score = RiskScoringEngine.calculate_risk(issues)

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
