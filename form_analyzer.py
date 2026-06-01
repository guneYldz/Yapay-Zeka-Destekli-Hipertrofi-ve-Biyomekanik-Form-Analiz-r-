import cv2
import numpy as np
import math
import time
import sys
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode
)

MODEL_FILE = "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

if not os.path.exists(MODEL_FILE):
    print(f"[İNDİRİLİYOR] {MODEL_FILE} — lütfen bekleyin...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("[TAMAM] Model indirildi!\n")
    except Exception as e:
        print(f"[HATA] Model indirilemedi: {e}")
        sys.exit(1)

# --- MODERN RENK PALETİ (BGR) ---
C_PRIMARY = (129, 185, 16)   # #10B981 (Emerald Green)
C_PRIMARY_L = (153, 211, 52) # #34D399
C_BG = (42, 23, 15)          # #0F172A (Slate Dark)
C_ACCENT = (11, 158, 245)    # #F59E0B (Amber)
C_DANGER = (50, 50, 220)     # Red
C_TEXT = (245, 250, 248)     # Off-white
C_SKEL = (180, 220, 80)      # Cyan-Green glow

# Açı eşikleri
HIP_DEEP     = 80
HIP_PARALLEL = 100
HIP_SHALLOW  = 130
KNEE_DANGER  = 165
BACK_THRESH  = 35

IDX = {
    'r_shoulder': 12, 'l_shoulder': 11,
    'r_hip': 24,      'l_hip': 23,
    'r_knee': 26,     'l_knee': 25,
    'r_ankle': 28,    'l_ankle': 27,
}

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28)
]

def angle3(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    r = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(np.degrees(r))
    return 360 - ang if ang > 180 else ang

def vert_angle(p1, p2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

def draw_skeleton(frame, lms, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    
    # Eklemleri birbirine bağlayan parlak çizgiler
    for a, b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], C_SKEL, 3, cv2.LINE_AA)
            # Hafif bir glow efekti için üstüne daha ince beyaz çizgi
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1, cv2.LINE_AA)
            
    # Temel eklem noktaları
    major_joints = [11, 12, 23, 24, 25, 26, 27, 28]
    for idx in major_joints:
        if idx < len(pts):
            cv2.circle(frame, pts[idx], 6, C_PRIMARY, -1, cv2.LINE_AA)
            cv2.circle(frame, pts[idx], 8, (255, 255, 255), 1, cv2.LINE_AA)
    
    return pts

def draw_hud_panel(frame, metrics, reps, status, risk_level):
    h, w = frame.shape[:2]
    
    # --- Üst Header Paneli ---
    header_h = 70
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, header_h), C_BG, -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    
    # Başlık
    cv2.putText(frame, "HYPERTROPHY AI", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, C_PRIMARY_L, 1, cv2.LINE_AA)
    cv2.putText(frame, "BIOMECHANICAL ANALYZER", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1, cv2.LINE_AA)
    
    # Tekrar Sayacı (Büyük)
    cv2.putText(frame, f"REPS", (w//2 - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{reps:02d}", (w//2 - 65, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, C_ACCENT, 2, cv2.LINE_AA)
    
    # Durum Etiketi
    color = C_PRIMARY if risk_level == "ok" else (C_ACCENT if risk_level == "warn" else C_DANGER)
    ts = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    cv2.putText(frame, status, (w - ts[0] - 20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)

    # --- Yan Metrik Panelleri ---
    for i, m in enumerate(metrics):
        py = 100 + i * 85
        px = 15
        width = 200
        height = 70
        
        # Panel Kutusu
        ov = frame.copy()
        cv2.rectangle(ov, (px, py), (px + width, py + height), C_BG, -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        
        # Sol kenar renkli çubuk
        cv2.rectangle(frame, (px, py), (px + 4, py + height), m['c'], -1)
        
        # Etiket ve Değer
        cv2.putText(frame, m['label'], (px + 12, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        val_str = f"{int(m['v'])}" if isinstance(m['v'], (int, float)) else str(m['v'])
        cv2.putText(frame, val_str, (px + 12, py + 55), cv2.FONT_HERSHEY_DUPLEX, 0.9, m['c'], 2, cv2.LINE_AA)
        if isinstance(m['v'], (int, float)):
             cv2.putText(frame, "deg", (px + 12 + len(val_str)*20, py + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, m['c'], 1, cv2.LINE_AA)

def draw_vignette_warning(frame):
    h, w = frame.shape[:2]
    pulse = (math.sin(time.time() * 8) + 1) / 2
    ov = frame.copy()
    # Kenarlara kırmızı vignette
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 150), 15)
    cv2.addWeighted(ov, pulse * 0.3, frame, 1 - pulse * 0.3, 0, frame)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera bulunamadı!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    reps = 0
    stage = None
    
    with open(MODEL_FILE, "rb") as f:
        model_data = f.read()

    opts = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model_data),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(opts) as detector:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            status = "SISTEM HAZIR"
            risk_level = "ok"
            metrics = []

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                pts = draw_skeleton(frame, lms, h, w)

                def xy(i): return pts[i]

                # Landmarks
                sh = xy(IDX['r_shoulder'])
                hp = xy(IDX['r_hip'])
                kn = xy(IDX['r_knee'])
                an = xy(IDX['r_ankle'])

                # Aci Hesaplamalari
                hip_a = angle3(sh, hp, kn)
                knee_a = angle3(hp, kn, an)
                back_a = vert_angle(hp, sh)

                # Renk Mantigi
                hip_c = C_PRIMARY if HIP_DEEP < hip_a < HIP_SHALLOW else (C_ACCENT if hip_a > HIP_SHALLOW else C_DANGER)
                knee_c = C_PRIMARY if knee_a < KNEE_DANGER else C_DANGER
                back_c = C_PRIMARY if back_a < BACK_THRESH else C_DANGER

                # Squat Mantigi
                if hip_a < HIP_PARALLEL:
                    stage = "down"
                    status = "DERINE IN" if hip_a > HIP_DEEP else "MUKEMMEL"
                if hip_a > 150 and stage == "down":
                    stage = "up"
                    reps += 1
                
                if stage == "down": status = "YUKSEL" if hip_a < HIP_PARALLEL else "DERINE IN"

                # Risk Analizi
                if back_a > BACK_THRESH or hip_a < HIP_DEEP:
                    risk_level = "danger"
                    status = "FORMU DUZELT!"
                    draw_vignette_warning(frame)
                elif hip_a > HIP_SHALLOW and stage == "down":
                    risk_level = "warn"

                metrics = [
                    {'label': 'KALCA ACISI', 'v': hip_a, 'c': hip_c},
                    {'label': 'DIZ ACISI', 'v': knee_a, 'c': knee_c},
                    {'label': 'SIRT EGIMI', 'v': back_a, 'c': back_c},
                    {'label': 'SQUAT FAZI', 'v': stage.upper() if stage else 'BEKLE', 'c': C_ACCENT}
                ]
            else:
                status = "KULLANICI BEKLENIYOR"
                risk_level = "warn"

            draw_hud_panel(frame, metrics, reps, status, risk_level)

            cv2.imshow("Hypertrophy AI Coaching", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): reps, stage = 0, None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if "--no-gui" in sys.argv:
        main()
    else:
        from presentation.gui import run_gui
        run_gui()
