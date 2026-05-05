from __future__ import annotations

import unittest

from core.calculator import Point3D, calculate_angle
from core.analyzer import FormAnalyzer


class FormAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = FormAnalyzer()

    def test_calculate_angle_is_generic_for_three_points(self) -> None:
        angle = calculate_angle(
            Point3D(0.0, 1.0, 0.0),
            Point3D(0.0, 0.0, 0.0),
            Point3D(1.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(angle, 90.0, places=6)

    def test_deadlift_reports_risky_back_angle(self) -> None:
        back_angle = calculate_angle(
            Point3D(0.0, 2.0, 0.0),
            Point3D(0.0, 1.0, 0.0),
            Point3D(1.0, 1.0, 0.0),
        )
        knee_angle = calculate_angle(
            Point3D(0.0, 1.0, 0.0),
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.0, -1.0, 0.0),
        )
        analysis = self.analyzer.analyze_deadlift({"back_angle": back_angle, "knee_angle": knee_angle})

        self.assertFalse(analysis.is_valid)
        self.assertIn("spine_risk", {warning.code for warning in analysis.warnings})
        self.assertEqual(analysis.exercise, "deadlift")

    def test_deadlift_accepts_safe_back_angle(self) -> None:
        back_angle = calculate_angle(
            Point3D(0.0, 2.0, 0.0),
            Point3D(0.0, 1.0, 0.0),
            Point3D(0.0, 0.0, 0.0),
        )
        knee_angle = calculate_angle(
            Point3D(0.0, 1.0, 0.0),
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.0, -1.0, 0.0),
        )
        analysis = self.analyzer.analyze_deadlift({"back_angle": back_angle, "knee_angle": knee_angle})

        self.assertTrue(analysis.is_valid)
        self.assertEqual(analysis.warnings, ())

    def test_deadlift_reports_closed_knee_angle(self) -> None:
        back_angle = calculate_angle(
            Point3D(0.0, 2.0, 0.0),
            Point3D(0.0, 1.0, 0.0),
            Point3D(0.0, 0.0, 0.0),
        )
        knee_angle = calculate_angle(
            Point3D(1.0, 1.0, 0.0),
            Point3D(0.0, 0.0, 0.0),
            Point3D(-1.0, 0.0, 0.0),
        )
        analysis = self.analyzer.analyze_deadlift({"back_angle": back_angle, "knee_angle": knee_angle})

        self.assertFalse(analysis.is_valid)
        self.assertIn("knee_risk", {warning.code for warning in analysis.warnings})
        self.assertEqual(analysis.exercise, "deadlift")


if __name__ == "__main__":
    unittest.main()
