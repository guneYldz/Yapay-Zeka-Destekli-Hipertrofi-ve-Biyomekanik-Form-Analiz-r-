import os
import threading
from datetime import datetime

import cv2
from fastapi import FastAPI
from sqlalchemy import text

from data.database import SessionLocal, UserModel, init_db
from data.adapters import MediaPipePoseAdapter
from application.use_cases import AnalyzeFormUseCase
from domain.entities import ExerciseType, RiskLevel
from presentation.overlay import OverlayRenderer, OverlayData
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class UserRepository:
    def __init__(self, session_factory=SessionLocal) -> None:
        self._session_factory = session_factory

    def kullanici_bul(self, user_name: str) -> UserModel | None:
        with self._session_factory() as session:
            return session.query(UserModel).filter_by(user_name=user_name).first()


app = FastAPI()


@app.get("/api/v1/status")
def api_status():
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok", "database": "ok", "time": datetime.utcnow().isoformat()}
    except Exception:
        return {"status": "error", "database": "error", "time": datetime.utcnow().isoformat()}


def _run_api() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


def run_demo():
    init_db()

    renderer = OverlayRenderer()
    cap = cv2.VideoCapture(0)

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)

    use_case = AnalyzeFormUseCase()

    if not cap.isOpened():
        print("[HATA] Kamera acilamadi.")
        return

    print("[BASLADI] Kamera akisi acildi. Cikmak icin q.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            
            class DummyLandmarks:
                def __init__(self, landmark_list):
                    self.landmark = landmark_list
            
            pose_frame = MediaPipePoseAdapter.to_pose_frame(DummyLandmarks(landmarks))
            issues, risk_level, explanation = use_case.execute(pose_frame, ExerciseType.SQUAT)

            h, w = frame.shape[:2]
            joint_points = {}
            for joint, pt in pose_frame.landmarks.items():
                joint_points[joint.name.lower()] = (int(pt.x * w), int(pt.y * h))

            joint_lines = [
                ("left_shoulder", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                ("right_shoulder", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
                ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")
            ]

            data = OverlayData(
                joint_points=joint_points,
                joint_lines=joint_lines,
                angles={},
                is_risky=risk_level == RiskLevel.HIGH
            )
            frame = renderer.render_frame(frame, data=data)

            color = (0, 0, 255) if risk_level == RiskLevel.HIGH else (0, 200, 0)
            cv2.putText(frame, explanation, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        else:
            empty_data = OverlayData(joint_points={}, joint_lines=[], angles={}, is_risky=False)
            frame = renderer.render_frame(frame, data=empty_data)

        cv2.imshow("Real-Time Squat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if os.getenv("START_API") == "1":
        threading.Thread(target=_run_api, daemon=True).start()
    run_demo()
