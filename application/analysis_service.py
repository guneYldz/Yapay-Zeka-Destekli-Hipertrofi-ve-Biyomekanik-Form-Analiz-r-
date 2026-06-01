from domain.rules import RuleRegistry
from domain.entities import ExerciseType, PoseFrame, Landmark
from typing import List

class AnalysisService:
    @staticmethod
    def analyze_squat(landmarks_list) -> dict:
        """
        OpenCV'den gelen ham landmark verilerini alır, 
        Domain katmanındaki kurallara gönderir ve sonucu döner.
        """
        # Ham veriyi Domain varlığına (Entity) çevir
        pose_frame = PoseFrame(
            landmarks=[Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks_list]
        )
        
        # Domain kurallarını çalıştır
        rules = RuleRegistry.get_rules_for(ExerciseType.SQUAT)
        all_issues = []
        for rule in rules:
            issues = rule.validate(pose_frame)
            all_issues.extend(issues)
            
        # Sonuçları UI'nın anlayacağı basitleştirilmiş bir formata çevir
        # (Bu kısım form_analyzer.py'deki draw fonksiyonlarını besleyecek)
        return {
            "issues": [issue.message for issue in all_issues],
            "severity": "danger" if any(i.level == "high" for i in all_issues) else "ok"
        }
