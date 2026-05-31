"""
Gercek zamanli squat demo akisi.
"""

import os
import threading
from datetime import datetime

import cv2
from fastapi import FastAPI
from sqlalchemy import text

from data.database import SessionLocal, UserModel, init_db
from presentation.overlay import OverlayRenderer


class UserRepository:
    """Repository deseni: kullanici sorgularini yonetir."""

    def __init__(self, session_factory=SessionLocal) -> None:
        self._session_factory = session_factory

    def kullanici_bul(self, user_name: str) -> UserModel | None:
        # Kullanici kaydini arar.
        with self._session_factory() as session:
            return session.query(UserModel).filter_by(user_name=user_name).first()


app = FastAPI()


@app.get("/api/v1/status")
def api_status():
    # Sistem ve veritabani durumunu kontrol eder.
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok", "database": "ok", "time": datetime.utcnow().isoformat()}
    except Exception:
        return {"status": "error", "database": "error", "time": datetime.utcnow().isoformat()}


def _run_api() -> None:
    # API servisini istege bagli calistirir.
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


def run_demo():
    init_db()

    renderer = OverlayRenderer()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[HATA] Kamera acilamadi.")
        return

    print("[BASLADI] Kamera akisi acildi. Cikmak icin q.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = renderer.render_frame(frame, data=None)
        cv2.imshow("Real-Time Squat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if os.getenv("START_API") == "1":
        threading.Thread(target=_run_api, daemon=True).start()
    run_demo()
