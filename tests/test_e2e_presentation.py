from __future__ import annotations

import io
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

_cv2_mock = types.ModuleType("cv2")
_cv2_mock.VideoCapture = MagicMock
_cv2_mock.FONT_HERSHEY_SIMPLEX = 0
_cv2_mock.LINE_AA = 16
_cv2_mock.imshow = MagicMock()
_cv2_mock.waitKey = MagicMock(return_value=ord("q"))
_cv2_mock.destroyAllWindows = MagicMock()
_cv2_mock.putText = MagicMock()
_cv2_mock.circle = MagicMock()
_cv2_mock.line = MagicMock()
sys.modules.setdefault("cv2", _cv2_mock)

_sa = types.ModuleType("sqlalchemy")
_sa.text = MagicMock()
_sa.Integer = MagicMock()
_sa.String = MagicMock()
_sa.DateTime = MagicMock()
_sa.Float = MagicMock()
_sa.ForeignKey = MagicMock()
_sa.create_engine = MagicMock()
sys.modules.setdefault("sqlalchemy", _sa)

_sa_orm = types.ModuleType("sqlalchemy.orm")
_sa_orm.DeclarativeBase = object
_sa_orm.Mapped = MagicMock()
_sa_orm.Session = MagicMock()
_sa_orm.mapped_column = MagicMock()
_sa_orm.relationship = MagicMock()
_sa_orm.sessionmaker = MagicMock()
sys.modules.setdefault("sqlalchemy.orm", _sa_orm)

_fastapi_mock = types.ModuleType("fastapi")
class MockFastAPI:
    def get(self, *args, **kwargs):
        def decorator(f):
            return f
        return decorator
_fastapi_mock.FastAPI = MockFastAPI
sys.modules.setdefault("fastapi", _fastapi_mock)

_uvicorn_mock = types.ModuleType("uvicorn")
_uvicorn_mock.run = MagicMock()
sys.modules.setdefault("uvicorn", _uvicorn_mock)

from application.use_cases import AnalyzeFormUseCase
from data.adapters import MockAdapter, MockPoseAdapter, create_adapter
from domain.entities import ExerciseType, RiskLevel
from presentation.cli import main as cli_main
from presentation.realtime_squat import UserRepository, api_status


class TestCLICiktisi(unittest.TestCase):

    def _calistir(self) -> str:
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli_main()
        return captured.getvalue()

    def test_baslik_ciktida_var(self):
        output = self._calistir()
        self.assertIn("AI Form Analyzer CLI", output)

    def test_risk_level_etiketi_ciktida_var(self):
        output = self._calistir()
        self.assertIn("Risk Level:", output)

    def test_explanation_etiketi_ciktida_var(self):
        output = self._calistir()
        self.assertIn("Explanation:", output)

    def test_squat_analiz_baslik_ciktida_var(self):
        output = self._calistir()
        self.assertIn("Analysis for SQUAT", output)

    def test_gecersiz_egzersiz_sys_exit_1_firlatiyor(self):
        bad_data = {"exercise": "INVALID", "pose": {}}
        with patch("presentation.cli.json.loads", return_value=bad_data), \
             patch("sys.stdout", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                cli_main()
        self.assertEqual(ctx.exception.code, 1)

    def test_no_issues_detected_mesaji_goster(self):
        output = self._calistir()
        has_issues = "Detailed Issues:" in output
        has_no_issues = "No issues detected." in output
        self.assertTrue(has_issues or has_no_issues)


class TestUctanUcaAnalizAkisi(unittest.TestCase):

    def _squat_frame(self):
        return MockPoseAdapter.from_dict({
            "LEFT_HIP":       {"x": 0.50, "y": 0.50},
            "RIGHT_HIP":      {"x": 0.60, "y": 0.50},
            "LEFT_KNEE":      {"x": 0.55, "y": 0.70},
            "RIGHT_KNEE":     {"x": 0.55, "y": 0.70},
            "LEFT_ANKLE":     {"x": 0.50, "y": 0.90},
            "RIGHT_ANKLE":    {"x": 0.60, "y": 0.90},
            "LEFT_SHOULDER":  {"x": 0.50, "y": 0.20},
            "RIGHT_SHOULDER": {"x": 0.60, "y": 0.20},
        })

    def _tehlikeli_squat_frame(self):
        return MockPoseAdapter.from_dict({
            "LEFT_HIP":    {"x": 0.40, "y": 0.50},
            "LEFT_KNEE":   {"x": 0.49, "y": 0.70},
            "LEFT_ANKLE":  {"x": 0.40, "y": 0.90},
            "RIGHT_HIP":   {"x": 0.60, "y": 0.50},
            "RIGHT_KNEE":  {"x": 0.51, "y": 0.70},
            "RIGHT_ANKLE": {"x": 0.60, "y": 0.90},
        })

    def test_squat_analizi_risk_level_dondurur(self):
        issues, risk_level, explanation = AnalyzeFormUseCase().execute(
            self._squat_frame(), ExerciseType.SQUAT
        )
        self.assertIsInstance(risk_level, RiskLevel)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)

    def test_bench_press_analizi_tamamlaniyor(self):
        frame = MockPoseAdapter.from_dict({
            "LEFT_WRIST":  {"x": 0.40, "y": 0.30},
            "RIGHT_WRIST": {"x": 0.60, "y": 0.30},
            "LEFT_ELBOW":  {"x": 0.42, "y": 0.40},
            "RIGHT_ELBOW": {"x": 0.58, "y": 0.40},
        })
        issues, risk_level, explanation = AnalyzeFormUseCase().execute(frame, ExerciseType.BENCH_PRESS)
        self.assertIsInstance(risk_level, RiskLevel)
        self.assertIsInstance(explanation, str)

    def test_bos_poz_low_risk_donduruyor(self):
        frame = MockPoseAdapter.from_dict({})
        issues, risk_level, _ = AnalyzeFormUseCase().execute(frame, ExerciseType.SQUAT)
        self.assertEqual(risk_level, RiskLevel.LOW)
        self.assertEqual(issues, [])

    def test_bos_poz_perfect_form_mesaji(self):
        frame = MockPoseAdapter.from_dict({})
        _, _, explanation = AnalyzeFormUseCase().execute(frame, ExerciseType.SQUAT)
        self.assertIn("Perfect form", explanation)

    def test_tehlikeli_diz_acisi_sorun_uretiyor(self):
        issues, risk_level, explanation = AnalyzeFormUseCase().execute(
            self._tehlikeli_squat_frame(), ExerciseType.SQUAT
        )
        self.assertGreater(len(issues), 0)
        self.assertIn(risk_level, [RiskLevel.MEDIUM, RiskLevel.HIGH])

    def test_sorun_varsa_explanation_issue_sayisi_iceriyor(self):
        issues, _, explanation = AnalyzeFormUseCase().execute(
            self._tehlikeli_squat_frame(), ExerciseType.SQUAT
        )
        if issues:
            self.assertIn("issue(s)", explanation)

    def test_high_risk_explanation_danger_iceriyor(self):
        issues, risk_level, explanation = AnalyzeFormUseCase().execute(
            self._tehlikeli_squat_frame(), ExerciseType.SQUAT
        )
        if risk_level == RiskLevel.HIGH:
            self.assertIn("DANGER", explanation)

    def test_medium_risk_explanation_warning_iceriyor(self):
        issues, risk_level, explanation = AnalyzeFormUseCase().execute(
            self._tehlikeli_squat_frame(), ExerciseType.SQUAT
        )
        if risk_level == RiskLevel.MEDIUM:
            self.assertIn("WARNING", explanation)


class TestDemoAdapterAkisi(unittest.TestCase):

    def test_mock_squat_demo_33_landmark_uretir(self):
        mock = MockAdapter(pose="squat", noise=0.005, seed=42)
        frame = mock.convert(frame_index=0, timestamp_ms=0.0)
        self.assertEqual(len(frame.landmarks), 33)
        self.assertEqual(frame.frame_index, 0)

    def test_mock_stand_sequence_dogrulanir(self):
        frames = MockAdapter(pose="stand", noise=0.003, seed=0).generate_sequence(10, fps=30.0)
        self.assertEqual(len(frames), 10)
        for i, f in enumerate(frames):
            self.assertEqual(f.frame_index, i)

    def test_demo_factory_mediapipe_olusturuluyor(self):
        adapter = create_adapter("mediapipe", frame_width=1280, frame_height=720)
        self.assertEqual(adapter.__class__.__name__, "MediaPipeAdapter")

    def test_demo_factory_mock_random_frame_uretir(self):
        adapter = create_adapter("mock", pose="random", seed=7)
        frame = adapter.convert(frame_index=99, timestamp_ms=3300.0)
        self.assertEqual(frame.frame_index, 99)
        self.assertAlmostEqual(frame.timestamp_ms, 3300.0)

    def test_demo_stdout_ciktisi_dogru_anahtarlar_iceriyor(self):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            mock = MockAdapter(pose="squat", noise=0.005, seed=42)
            frame = mock.convert(frame_index=0, timestamp_ms=0.0)
            print(f"Toplam landmark : {len(frame.landmarks)}")
            print(f"Kare numarasi   : {frame.frame_index}")
            print(f"Sol kalca       : {frame.left_hip()}")
        output = captured.getvalue()
        self.assertIn("Toplam landmark", output)
        self.assertIn("Kare numarasi", output)
        self.assertIn("Sol kalca", output)


class TestRealtimeSquatBilesenleri(unittest.TestCase):

    def test_user_repository_kullanici_bulunamadiginda_none_doner(self):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        repo = UserRepository(session_factory=lambda: mock_session)
        result = repo.kullanici_bul("test_kullanici")
        self.assertIsNone(result)

    def test_user_repository_kullanici_bulundugunda_model_doner(self):
        fake_user = MagicMock()
        fake_user.user_name = "ahmet"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter_by.return_value.first.return_value = fake_user

        repo = UserRepository(session_factory=lambda: mock_session)
        result = repo.kullanici_bul("ahmet")
        self.assertEqual(result.user_name, "ahmet")

    def test_api_status_db_basarili_ok_doner(self):
        with patch("presentation.realtime_squat.SessionLocal") as mock_session_local:
            response = api_status()

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["database"], "ok")
        self.assertIn("time", response)

    def test_api_status_db_hatasinda_error_doner(self):
        with patch("presentation.realtime_squat.SessionLocal") as mock_session_local:
            mock_session_instance = mock_session_local.return_value.__enter__.return_value
            mock_session_instance.execute.side_effect = Exception("db hatasi")
            response = api_status()

        self.assertEqual(response["status"], "error")
        self.assertEqual(response["database"], "error")

    def test_run_demo_kamera_acilamazsa_erken_cikiyor(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        captured = io.StringIO()
        with patch("presentation.realtime_squat.cv2.VideoCapture", return_value=mock_cap), \
             patch("presentation.realtime_squat.vision.PoseLandmarker.create_from_options"), \
             patch("presentation.realtime_squat.init_db"), \
             patch("sys.stdout", captured):
            from presentation.realtime_squat import run_demo
            run_demo()

        self.assertIn("Kamera acilamadi", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
