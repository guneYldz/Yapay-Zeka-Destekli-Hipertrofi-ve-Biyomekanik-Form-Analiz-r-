from __future__ import annotations

import unittest
from types import SimpleNamespace

from data.adapters import LandmarkIndex, MediaPipeAdapter, MockAdapter, create_adapter


def _fake_landmarks(n=33, x=0.5, y=0.5, z=0.0, vis=1.0):
    """Sahte landmark nesnesi listesi üretir (MediaPipe arayüzünü taklit eder)."""
    return [SimpleNamespace(x=x, y=y, z=z, visibility=vis) for _ in range(n)]


class TestMediaPipeAdapter(unittest.TestCase):

    def setUp(self):
        self.adapter = MediaPipeAdapter()

    def test_donus_landmark_sayisi_33_olmali(self):
        raw = _fake_landmarks(33)
        frame = self.adapter.convert(raw)
        self.assertEqual(len(frame.landmarks), 33)

    def test_koordinatlar_dogru_aktariliyor(self):
        raw = _fake_landmarks(33, x=0.3, y=0.7, z=0.1, vis=0.9)
        frame = self.adapter.convert(raw)
        lm = frame.get(0)
        self.assertAlmostEqual(lm.x, 0.3)
        self.assertAlmostEqual(lm.y, 0.7)
        self.assertAlmostEqual(lm.z, 0.1)

    def test_piksel_donusumu_frame_boyutlariyla(self):
        adapter = MediaPipeAdapter(frame_width=1280, frame_height=720)
        raw = _fake_landmarks(33, x=0.5, y=0.5)
        frame = adapter.convert(raw)
        lm = frame.get(0)
        self.assertAlmostEqual(lm.x, 640.0)
        self.assertAlmostEqual(lm.y, 360.0)

    def test_frame_index_ve_timestamp_aktariliyor(self):
        raw = _fake_landmarks(33)
        frame = self.adapter.convert(raw, frame_index=5, timestamp_ms=166.6)
        self.assertEqual(frame.frame_index, 5)
        self.assertAlmostEqual(frame.timestamp_ms, 166.6)

    def test_visibility_threshold_altindakiler_sifirlanir(self):
        adapter = MediaPipeAdapter(visibility_threshold=0.8)
        raw = _fake_landmarks(33, vis=0.5)
        frame = adapter.convert(raw)
        self.assertEqual(frame.get(0).visibility, 0.0)

    def test_yanlis_landmark_sayisi_hata_firlatiyor(self):
        raw = _fake_landmarks(20)
        with self.assertRaises(ValueError):
            self.adapter.convert(raw)

    def test_convert_batch_dogru_kare_sayisi_dondurur(self):
        raws = [_fake_landmarks(33) for _ in range(5)]
        frames = self.adapter.convert_batch(raws)
        self.assertEqual(len(frames), 5)

    def test_convert_batch_frame_indeksleri_sirali(self):
        raws = [_fake_landmarks(33) for _ in range(3)]
        frames = self.adapter.convert_batch(raws, start_frame_index=10)
        indeksler = [f.frame_index for f in frames]
        self.assertEqual(indeksler, [10, 11, 12])

    def test_convert_batch_timestamp_fps_ile_hesaplaniyor(self):
        raws = [_fake_landmarks(33) for _ in range(2)]
        frames = self.adapter.convert_batch(raws, fps=10.0)
        self.assertAlmostEqual(frames[0].timestamp_ms, 0.0)
        self.assertAlmostEqual(frames[1].timestamp_ms, 100.0)


class TestMockAdapter(unittest.TestCase):

    def test_squat_pozunda_33_landmark_uretiliyor(self):
        mock = MockAdapter(pose="squat", seed=0)
        frame = mock.convert()
        self.assertEqual(len(frame.landmarks), 33)

    def test_stand_pozunda_33_landmark_uretiliyor(self):
        mock = MockAdapter(pose="stand", seed=0)
        frame = mock.convert()
        self.assertEqual(len(frame.landmarks), 33)

    def test_random_pozunda_33_landmark_uretiliyor(self):
        mock = MockAdapter(pose="random", seed=0)
        frame = mock.convert()
        self.assertEqual(len(frame.landmarks), 33)

    def test_ayni_seed_ayni_sonucu_veriyor(self):
        m1 = MockAdapter(pose="squat", seed=42)
        m2 = MockAdapter(pose="squat", seed=42)
        f1 = m1.convert()
        f2 = m2.convert()
        self.assertEqual(f1.landmarks[LandmarkIndex.LEFT_HIP].x,
                         f2.landmarks[LandmarkIndex.LEFT_HIP].x)

    def test_frame_index_ve_timestamp_iletiliyor(self):
        mock = MockAdapter()
        frame = mock.convert(frame_index=7, timestamp_ms=233.3)
        self.assertEqual(frame.frame_index, 7)
        self.assertAlmostEqual(frame.timestamp_ms, 233.3)

    def test_generate_sequence_dogru_uzunlukta(self):
        mock = MockAdapter(pose="stand")
        frames = mock.generate_sequence(num_frames=10, fps=30.0)
        self.assertEqual(len(frames), 10)

    def test_generate_sequence_timestamp_artiyor(self):
        mock = MockAdapter()
        frames = mock.generate_sequence(num_frames=3, fps=30.0)
        ts_listesi = [f.timestamp_ms for f in frames]
        self.assertEqual(ts_listesi, sorted(ts_listesi))

    def test_gecersiz_pose_hata_firlatiyor(self):
        with self.assertRaises(ValueError):
            MockAdapter(pose="deadlift")

    def test_squat_kalca_y_diz_y_den_yukarda(self):
        # squat pozunda kalça diz'den daha yukarıda olmalı (y ekseni aşağı doğru artar)
        mock = MockAdapter(pose="squat", noise=0.0, seed=1)
        frame = mock.convert()
        kalca_y = frame.left_hip().y
        diz_y   = frame.left_knee().y
        self.assertLess(kalca_y, diz_y)


class TestCreateAdapter(unittest.TestCase):

    def test_mediapipe_adapter_olusturuluyor(self):
        adapter = create_adapter("mediapipe")
        self.assertIsInstance(adapter, MediaPipeAdapter)

    def test_mock_adapter_olusturuluyor(self):
        adapter = create_adapter("mock", pose="stand")
        self.assertIsInstance(adapter, MockAdapter)

    def test_gecersiz_kaynak_hata_firlatiyor(self):
        with self.assertRaises(ValueError):
            create_adapter("kamera")


if __name__ == "__main__":
    unittest.main()
