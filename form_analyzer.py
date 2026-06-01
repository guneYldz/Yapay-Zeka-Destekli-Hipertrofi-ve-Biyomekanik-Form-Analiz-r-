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
        print(f"Manuel indirin → {MODEL_URL}")
        sys.exit(1)

# BGR renk değerleri
C_OK     = (0,   215, 100)
C_WARN   = (0,   165, 255)
C_DANGER = (30,  30,  220)
C_PANEL  = (18,  18,  35)
C_TEXT   = (225, 225, 240)
C_GOLD   = (50,  195, 255)
C_BLUE   = (240, 170,  60)
C_SKEL   = (80,  220, 180)

# açı eşikleri
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
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
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
    pts = [(lm.x * w, lm.y * h) for lm in lms]
    for a, b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            p1 = (int(pts[a][0]), int(pts[a][1]))
            p2 = (int(pts[b][0]), int(pts[b][1]))
            cv2.line(frame, p1, p2, C_SKEL, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return pts


def draw_arc(frame, center, angle_deg, color, r=45):
    cx, cy = int(center[0]), int(center[1])
    cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, int(angle_deg)-90, color, 3, cv2.LINE_AA)
    mid_r = math.radians(-90 + angle_deg / 2)
    tx = int(cx + (r + 18) * math.cos(mid_r))
    ty = int(cy + (r + 18) * math.sin(mid_r))
    cv2.putText(frame, f"{int(angle_deg)}", (tx - 12, ty + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_panel(frame, metrics):
    for i, m in enumerate(metrics):
        x1, y1, x2, y2 = 8, 80 + i * 76, 230, 140 + i * 76
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), C_PANEL, -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x1+4, y2), m['c'], -1)
        cv2.putText(frame, m['label'], (x1+12, y1+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_TEXT, 1, cv2.LINE_AA)
        val = f"{m['v']:.1f}°" if isinstance(m['v'], float) else str(m['v'])
        cv2.putText(frame, val, (x1+12, y1+54),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, m['c'], 2, cv2.LINE_AA)


def draw_header(frame, reps, status, color):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 70), (8, 8, 18), -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, "BIYOMEKANIK FORM ANALIZORU",
                (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, C_GOLD, 1, cv2.LINE_AA)
    cv2.putText(frame, f"TEKRAR: {reps}",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLUE, 1, cv2.LINE_AA)
    ts = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
    cv2.putText(frame, status, (w - ts[0] - 15, 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_danger_banner(frame, msgs):
    h, w = frame.shape[:2]
    pulse = 0.6 + 0.4 * abs(math.sin(time.time() * 6))
    ov = frame.copy()
    cv2.rectangle(ov, (0, h-95), (w, h), (0, 0, 140), -1)
    cv2.addWeighted(ov, pulse * 0.75, frame, 1 - pulse * 0.75, 0, frame)
    y = h - 55
    for msg in msgs[:2]:
        ts = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2)[0]
        tx = (w - ts[0]) // 2
        cv2.putText(frame, msg, (tx+2, y+2), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, msg, (tx, y), cv2.FONT_HERSHEY_DUPLEX, 0.72, (70, 70, 255), 2, cv2.LINE_AA)
        y += 34


def draw_hints(frame):
    h, w = frame.shape[:2]
    for i, t in enumerate(["Q: Cikis", "R: Sifirla", "S: Goruntu"]):
        cv2.putText(frame, t, (w-160, h-70+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 130), 1, cv2.LINE_AA)


# ─── Açı analiz motoru (saf fonksiyon – birim test dostu) ────────────────────
def analyze_angles(hip_a: float, knee_a: float, back_a: float) -> dict:
    """
    Verilen kalça, diz ve sırt açılarını biyomekanik eşiklerle karşılaştırır.

    Parametreler
    ------------
    hip_a  : Kalça eklem açısı (derece)
    knee_a : Diz eklem açısı (derece)
    back_a : Sırın düşeyden sapma açısı (derece)

    Döndürür
    --------
    dict:
        warnings : list[str]  – Uyarı mesajları listesi
        status   : str        – Genel durum etiketi
        risk     : str        – "ok" | "warn" | "danger"
    """
    warnings: list = []
    status = "FORM UYGUN"
    risk = "ok"

    # ── Kalça kontrolü ────────────────────────────────────────────────────────
    if hip_a < HIP_DEEP:
        warnings.append("!! DIKKAT: SAKATLIK RISKI YUKSEK - COK DERIN SQUAT !!")
        status = "SAKATLIK RISKI!"
        risk = "danger"
    elif hip_a > HIP_SHALLOW:
        warnings.append("Yetersiz derinlik - daha derin inin!")
        if risk != "danger":
            status = "DERINLESTIRIN"
            risk = "warn"

    # ── Sırt kontrolü ─────────────────────────────────────────────────────────
    if back_a > BACK_THRESH:
        warnings.append(f"!! SIRT EGILMESI {back_a:.0f} - Omurgayi Dik Tut !!")
        status = "SIRT FORMUNU DUZELTIN!"
        risk = "danger"

    # ── Diz kontrolü ──────────────────────────────────────────────────────────
    if knee_a > KNEE_DANGER:
        warnings.append("Diz acisi kritik! Kontrolu kaybediyorsunuz!")
        if risk != "danger":
            risk = "danger"

    return {"warnings": warnings, "status": status, "risk": risk}


# ─── Ana döngü ────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera bulunamadı!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    reps   = 0
    stage  = None
    ss_idx = 0

    print("=" * 58)
    print("  Biyomekanik Form Analizörü BAŞLADI")
    print("  Tam vücudunuz kameraya göründüğünden emin olun.")
    print("  Q: Çıkış  |  R: Tekrar sıfırla  |  S: Ekran görüntüsü")
    print("=" * 58)

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
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            warnings = []
            status   = "POZA HAZIR"
            status_c = C_OK
            hip_a = knee_a = back_a = 0.0

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                draw_skeleton(frame, lms, h, w)

                def xy(i): return (lms[i].x * w, lms[i].y * h)

                shoulder = xy(IDX['r_shoulder'])
                hip_p    = xy(IDX['r_hip'])
                knee_p   = xy(IDX['r_knee'])
                ankle    = xy(IDX['r_ankle'])

                hip_a  = angle3(shoulder, hip_p, knee_p)
                knee_a = angle3(hip_p, knee_p, ankle)
                back_a = vert_angle(hip_p, shoulder)

                hip_c  = C_OK if HIP_DEEP < hip_a < HIP_SHALLOW else C_DANGER
                knee_c = C_OK if knee_a < KNEE_DANGER else C_WARN
                back_c = C_OK if back_a < BACK_THRESH else C_DANGER

                draw_arc(frame, hip_p,  hip_a,  hip_c,  r=48)
                draw_arc(frame, knee_p, knee_a, knee_c, r=36)

                if hip_a < HIP_PARALLEL:
                    stage = "down"
                if hip_a > 150 and stage == "down":
                    stage = "up"
                    reps += 1

                if hip_a < HIP_DEEP:
                    warnings.append("!! DIKKAT: SAKATLIK RISKI YUKSEK - COK DERIN SQUAT !!")
                    status, status_c = "SAKATLIK RISKI!", C_DANGER
                elif hip_a > HIP_SHALLOW:
                    warnings.append("Yetersiz derinlik - daha derin inin!")
                    if status_c != C_DANGER:
                        status, status_c = "DERINLESTIRIN", C_WARN

                if back_a > BACK_THRESH:
                    warnings.append(f"!! SIRT EGILMESI {back_a:.0f} - Omurgayi Dik Tut !!")
                    status, status_c = "SIRT FORMUNU DUZELTIN!", C_DANGER

                if knee_a > KNEE_DANGER:
                    warnings.append("Diz acisi kritik! Kontrolu kaybediyorsunuz!")
                    status_c = C_DANGER

                draw_panel(frame, [
                    {'label': 'KALCA ACISI', 'v': hip_a,  'c': hip_c},
                    {'label': 'DIZ ACISI',   'v': knee_a, 'c': knee_c},
                    {'label': 'SIRT EGIMI',  'v': back_a, 'c': back_c},
                    {'label': 'SQUAT FAZI',  'v': (stage.upper() if stage else 'BEKLE'), 'c': C_GOLD},
                ])

                if warnings:
                    draw_danger_banner(frame, warnings)
            else:
                cv2.putText(
                    frame, "Poz algilanamadi - tam vucudunuz gorunsun!",
                    (60, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WARN, 2, cv2.LINE_AA
                )
                status, status_c = "POZ ALGILANAMADI", C_WARN

            draw_header(frame, reps, status, status_c)
            draw_hints(frame)

            cv2.imshow("Biyomekanik Form Analizoru  [Q=Cikis]", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nToplam tekrar: {reps}")
                break
            elif key == ord('r'):
                reps, stage = 0, None
                print("Sayaç sıfırlandı.")
            elif key == ord('s'):
                ss_idx += 1
                fn = f"form_goruntu_{ss_idx}.png"
                cv2.imwrite(fn, frame)
                print(f"Kaydedildi: {fn}")

    cap.release()
    cv2.destroyAllWindows()
    print("Program kapatıldı.")


if __name__ == "__main__":
    if "--no-gui" in sys.argv:
        main()
    else:
        from presentation.gui import run_gui
        run_gui()
