from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

from data.database import AnalysisResultModel, SessionLocal


app = FastAPI()


class PoseDataRequest(BaseModel):
    user_id: int
    session_id: int
    rep_number: int
    knee_angle: float
    hip_angle: float


@app.get("/api/v1/status")
def status():
    # Sistemin ayakta oldugunu dogrular.
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/api/v1/analyze")
def analyze_pose(payload: PoseDataRequest):
    # Kural tabanli analiz mantigi
    if payload.knee_angle > 100.0:
        error_type = "Yetersiz Derinlik (Yarim Squat)"
        risk_level = "Orta"
    elif payload.knee_angle < 60.0:
        error_type = "Asiri Derinlik (Sakatlik Riski)"
        risk_level = "Yuksek"
    else:
        error_type = None
        risk_level = "Dusuk"

    with SessionLocal() as session:
        model = AnalysisResultModel(
            session_id=payload.session_id,
            rep_number=payload.rep_number,
            max_knee_angle=payload.knee_angle,
            error_type=error_type or "",
            risk_level=risk_level,
            timestamp=datetime.utcnow(),
        )
        session.add(model)
        session.commit()
        session.refresh(model)

    return {
        "status": "ok",
        "analysis_id": model.id,
        "detected_error": error_type,
        "risk_level": risk_level,
    }
