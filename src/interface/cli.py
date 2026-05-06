"""
Simple CLI Interface for the Form Analyzer.
"""

import json
import sys
from typing import Dict

from src.application.use_cases import AnalyzeFormUseCase
from src.domain.entities import ExerciseType
from src.infrastructure.adapters import MockPoseAdapter


def main() -> None:
    """
    Main CLI entry point.
    In a real app, this might read from a file or stdin.
    Here we provide an example usage with hardcoded/mock JSON.
    """
    print("=== AI Form Analyzer CLI ===")
    
    # Mock JSON input representing a bad squat (knees caving in, knee angle sharp)
    mock_json_input = """
    {
        "exercise": "SQUAT",
        "pose": {
            "LEFT_HIP": {"x": 0.5, "y": 0.5},
            "RIGHT_HIP": {"x": 0.6, "y": 0.5},
            "LEFT_KNEE": {"x": 0.55, "y": 0.7},
            "RIGHT_KNEE": {"x": 0.55, "y": 0.7},
            "LEFT_ANKLE": {"x": 0.5, "y": 0.9},
            "RIGHT_ANKLE": {"x": 0.6, "y": 0.9},
            "LEFT_SHOULDER": {"x": 0.5, "y": 0.2},
            "RIGHT_SHOULDER": {"x": 0.6, "y": 0.2}
        }
    }
    """
    
    data: Dict = json.loads(mock_json_input)
    
    # 1. Parse Exercise Type
    try:
        exercise_str = data.get("exercise", "SQUAT").upper()
        exercise = ExerciseType[exercise_str]
    except KeyError:
        print(f"Error: Unsupported exercise type '{exercise_str}'")
        sys.exit(1)
        
    # 2. Adapt Input to Domain Entity
    pose_data = data.get("pose", {})
    frame = MockPoseAdapter.from_dict(pose_data)
    
    # 3. Execute Use Case
    use_case = AnalyzeFormUseCase()
    issues, risk_level, explanation = use_case.execute(frame, exercise)
    
    # 4. Present Output
    print(f"\\nAnalysis for {exercise.name}:")
    print("-" * 30)
    print(f"Risk Level: {risk_level.value}")
    print(f"Explanation: {explanation}\\n")
    
    if issues:
        print("Detailed Issues:")
        for idx, issue in enumerate(issues, 1):
            affected = ", ".join([j.name for j in issue.affected_joints])
            print(f"  {idx}. {issue.description} (Score: {issue.risk_score}, Joints: {affected})")
    else:
        print("No issues detected.")


if __name__ == "__main__":
    main()
