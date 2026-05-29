"""
data/adapter_demo.py
====================
MockAdapter ve MediaPipeAdapter'in hizli test edilmesi icin demo scripti.

Kullanim:
    python -m data.adapter_demo
"""

from __future__ import annotations

from data.adapters import LandmarkIndex, MockAdapter, create_adapter


def demo_mock_squat() -> None:
    """Squat pozundaki sahte verileri gosterir."""
    print("\n-- MockAdapter (squat) --")
    mock = MockAdapter(pose="squat", noise=0.005, seed=42)
    frame = mock.convert(frame_index=0, timestamp_ms=0.0)

    print(f"Toplam landmark : {len(frame.landmarks)}")
    print(f"Kare numarasi   : {frame.frame_index}")
    print(f"Zaman damgasi   : {frame.timestamp_ms} ms")
    print(f"Sol kalca       : {frame.left_hip()}")
    print(f"Sag kalca       : {frame.right_hip()}")
    print(f"Sol diz         : {frame.left_knee()}")
    print(f"Sag diz         : {frame.right_knee()}")
    print(f"Sol omuz gorunur: {frame.is_visible(LandmarkIndex.LEFT_SHOULDER)}")


def demo_mock_sequence() -> None:
    """10 ardisik sahte kare uretir ve timestamp'leri gosterir."""
    print("\n-- MockAdapter sequence (10 kare) --")
    mock = MockAdapter(pose="stand", noise=0.003)
    frames = mock.generate_sequence(num_frames=10, fps=30.0)
    for f in frames:
        hip = f.right_hip()
        print(f"  Kare {f.frame_index:02d} | ts={f.timestamp_ms:7.1f} ms | "
              f"sag_kalca=({hip.x:.3f}, {hip.y:.3f})")


def demo_factory() -> None:
    """create_adapter fabrika fonksiyonunu test eder."""
    print("\n-- create_adapter() fabrika --")
    adapter = create_adapter("mock", pose="random", seed=7)
    frame = adapter.convert(frame_index=99, timestamp_ms=3300.0)
    print(f"Random frame  : {frame.frame_index} | landmark[0]: {frame.get(0)}")

    mp_adapter = create_adapter("mediapipe", frame_width=1280, frame_height=720)
    print(f"MediaPipeAdapter olusturuldu: {mp_adapter.__class__.__name__}")


if __name__ == "__main__":
    demo_mock_squat()
    demo_mock_sequence()
    demo_factory()
    print("\n[OK] Tum demo adimlari basariyla tamamlandi.\n")
