from __future__ import annotations

import unittest

from core.analyzer import FormAnalyzer, Thresholds
from core.calculator import CalculatedAngles


class TestFormAnalyzer(unittest.TestCase):
    def test_squat_warns_when_depth_insufficient(self) -> None:
        analyzer = FormAnalyzer(Thresholds(squat_max_knee_angle_degrees=110.0))
        angles = CalculatedAngles(knee_angle_degrees=125.0)

        result = analyzer.analyze_squat(angles)

        self.assertFalse(result.ok)
        self.assertEqual(result.exercise, "squat")
        self.assertIn("squat_depth_insufficient", {warning.code for warning in result.warnings})

    def test_squat_ok_when_depth_sufficient(self) -> None:
        analyzer = FormAnalyzer(Thresholds(squat_max_knee_angle_degrees=110.0))
        angles = CalculatedAngles(knee_angle_degrees=95.0)

        result = analyzer.analyze_squat(angles)

        self.assertTrue(result.ok)
        self.assertEqual(result.warnings, ())

    def test_squat_errors_when_knee_angle_missing(self) -> None:
        analyzer = FormAnalyzer()
        angles = CalculatedAngles(knee_angle_degrees=None)

        result = analyzer.analyze_squat(angles)

        self.assertFalse(result.ok)
        self.assertIn("missing_knee_angle", {warning.code for warning in result.warnings})

    def test_deadlift_warns_when_back_angle_risky(self) -> None:
        analyzer = FormAnalyzer(Thresholds(deadlift_min_torso_angle_degrees=140.0))
        angles = CalculatedAngles(torso_angle_degrees=125.0)

        result = analyzer.analyze_deadlift(angles)

        self.assertFalse(result.ok)
        self.assertEqual(result.exercise, "deadlift")
        self.assertIn("back_angle_injury_risk", {warning.code for warning in result.warnings})

    def test_deadlift_ok_when_back_angle_safe(self) -> None:
        analyzer = FormAnalyzer(Thresholds(deadlift_min_torso_angle_degrees=140.0))
        angles = CalculatedAngles(torso_angle_degrees=150.0)

        result = analyzer.analyze_deadlift(angles)

        self.assertTrue(result.ok)
        self.assertEqual(result.warnings, ())

    def test_deadlift_errors_when_torso_angle_missing(self) -> None:
        analyzer = FormAnalyzer()
        angles = CalculatedAngles(torso_angle_degrees=None)

        result = analyzer.analyze_deadlift(angles)

        self.assertFalse(result.ok)
        self.assertIn("missing_torso_angle", {warning.code for warning in result.warnings})


if __name__ == "__main__":
    unittest.main()
