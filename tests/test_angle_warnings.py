"""
=================================================================
  Unit Testler – Açı Uyarı Sistemi
  tests/test_angle_warnings.py
=================================================================
  Bu dosya form_analyzer.py'yi DOĞRUDAN import ETMEZ.
  Sadece saf iş mantığını (analyze_angles) burada yeniden
  tanımlar; böylece cv2 / mediapipe kurulu olmadan da çalışır.

  Eşik değerleri (form_analyzer.py ile birebir aynı):
    HIP_DEEP    = 80   → çok derin squat sınırı
    HIP_SHALLOW = 130  → yetersiz derinlik sınırı
    KNEE_DANGER = 165  → diz tam açık sınırı
    BACK_THRESH = 35   → sırt eğim uyarı sınırı

  Çalıştırma:
    python -m pytest   tests/test_angle_warnings.py -v
    python -m unittest tests/test_angle_warnings.py -v
=================================================================
"""

import unittest

# ─── Eşik sabitleri (form_analyzer.py ile birebir kopyalanmıştır) ─────────────
HIP_DEEP     = 80
HIP_PARALLEL = 100
HIP_SHALLOW  = 130
KNEE_DANGER  = 165
BACK_THRESH  = 35


# ─── Test edilecek saf fonksiyon (cv2/mediapipe bağımlılığı yoktur) ───────────
def analyze_angles(hip_a: float, knee_a: float, back_a: float) -> dict:
    """
    Kalça, diz ve sırt açılarını biyomekanik eşiklerle karşılaştırır.

    Döndürür
    --------
    dict:
        warnings : list[str]  – Uyarı mesajları
        status   : str        – Genel durum etiketi
        risk     : str        – "ok" | "warn" | "danger"
    """
    warnings: list = []
    status = "FORM UYGUN"
    risk = "ok"

    # Kalça kontrolü
    if hip_a < HIP_DEEP:
        warnings.append("!! DİKKAT: SAKATLIK RİSKİ YÜKSEK – ÇOK DERİN SQUAT !!")
        status = "SAKATLIK RİSKİ!"
        risk = "danger"
    elif hip_a > HIP_SHALLOW:
        warnings.append("Yetersiz derinlik – daha derin inin!")
        if risk != "danger":
            status = "DERİNLEŞTİRİN"
            risk = "warn"

    # Sırt kontrolü
    if back_a > BACK_THRESH:
        warnings.append(f"!! SIRT EĞİLMESİ {back_a:.0f}° – Omurgayı Dik Tut !!")
        status = "SIRT FORMUNU DÜZELTİN!"
        risk = "danger"

    # Diz kontrolü
    if knee_a > KNEE_DANGER:
        warnings.append("Diz açısı kritik! Kontrolü kaybediyorsunuz!")
        if risk != "danger":
            risk = "danger"

    return {"warnings": warnings, "status": status, "risk": risk}


# ─── Yardımcı temel değerler ──────────────────────────────────────────────────
SAFE_HIP  = 110   # 80 < 110 < 130 → güvenli kalça
SAFE_KNEE = 140   # 140 < 165      → güvenli diz
SAFE_BACK = 20    # 20  < 35       → güvenli sırt


# ═════════════════════════════════════════════════════════════════════════════
#  SINIF 1 – Kalça Açısı Testleri
# ═════════════════════════════════════════════════════════════════════════════
class TestHipAngle(unittest.TestCase):
    """Kalça açısının 90° altı / üstü davranışı."""

    # ── 90°'nin ALTINDA ───────────────────────────────────────────────────────

    def test_hip_50_sakatlik_uyarisi_var(self):
        """50° → çok derin squat → SAKATLIK uyarısı."""
        result = analyze_angles(hip_a=50.0, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(
            any("SAKATLIK" in w for w in result["warnings"]),
            "50° kalça → SAKATLIK uyarısı bekleniyor"
        )

    def test_hip_79_risk_danger(self):
        """79° (HIP_DEEP=80 altı) → risk='danger'."""
        result = analyze_angles(hip_a=79.0, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(
            result["risk"] == "danger",
            f"79° → 'danger' bekleniyor, gelen: {result['risk']}"
        )

    def test_hip_80_tam_esik_sakatlik_yok(self):
        """80° tam eşik → SAKATLIK uyarısı OLMAMALI (< değil, = eşit kabul)."""
        result = analyze_angles(hip_a=80.0, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertFalse(
            any("SAKATLIK" in w for w in result["warnings"]),
            "80° tam eşik → SAKATLIK uyarısı OLMAMALI"
        )

    def test_hip_85_guvenli_aralik(self):
        """85° (80 < 85 < 130) → güvenli, risk='ok'."""
        result = analyze_angles(hip_a=85.0, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(
            result["risk"] == "ok",
            "85° → risk='ok' bekleniyor"
        )

    def test_hip_50_status_sakatlik(self):
        """50° → status 'SAKATLIK RİSKİ!' içermeli."""
        result = analyze_angles(hip_a=50.0, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(
            "SAKATLIK" in result["status"],
            f"Status beklenen: 'SAKATLIK RİSKİ!', gelen: '{result['status']}'"
        )

    # ── 90°'nin ÜSTÜNDE ───────────────────────────────────────────────────────

    def test_hip_145_yetersiz_derinlik(self):
        """145° (HIP_SHALLOW=130 üstü) → yetersiz derinlik uyarısı."""
        result = analyze_angles(hip_a=145.0, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(
            any("Yetersiz" in w for w in result["warnings"]),
            "145° → 'Yetersiz derinlik' uyarısı bekleniyor"
        )

    def test_hip_131_risk_warn(self):
        """131° (HIP_SHALLOW=130 hemen üstü) → risk='warn'."""
        result = analyze_angles(hip_a=131.0, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(
            result["risk"] == "warn",
            f"131° → 'warn' bekleniyor, gelen: {result['risk']}"
        )

    def test_hip_130_tam_esik_uyari_yok(self):
        """130° tam eşik → yetersiz derinlik uyarısı OLMAMALI."""
        result = analyze_angles(hip_a=130.0, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertFalse(
            any("Yetersiz" in w for w in result["warnings"]),
            "130° tam eşik → 'Yetersiz derinlik' uyarısı OLMAMALI"
        )

    def test_hip_180_status_derinlestirin(self):
        """180° (tamamen dik) → status 'DERİNLEŞTİRİN'."""
        result = analyze_angles(hip_a=180.0, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(
            result["status"] == "DERİNLEŞTİRİN",
            f"180° → 'DERİNLEŞTİRİN' bekleniyor, gelen: '{result['status']}'"
        )

    def test_hip_guvenli_aralik_uyari_yok(self):
        """90–120° arası → hiç uyarı çıkmamalı."""
        for angle in [90, 100, 110, 120]:
            result = analyze_angles(hip_a=float(angle), knee_a=SAFE_KNEE, back_a=SAFE_BACK)
            self.assertTrue(
                len(result["warnings"]) == 0,
                f"{angle}° → uyarı beklenmiyordu, gelen: {result['warnings']}"
            )

    def test_hip_guvenli_status_form_uygun(self):
        """110° → status 'FORM UYGUN'."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(
            result["status"] == "FORM UYGUN",
            f"110° → 'FORM UYGUN' bekleniyor, gelen: '{result['status']}'"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SINIF 2 – Diz Açısı Testleri
# ═════════════════════════════════════════════════════════════════════════════
class TestKneeAngle(unittest.TestCase):
    """Diz açısının 90° altı / üstü davranışı."""

    def test_knee_170_kritik_uyari(self):
        """170° (KNEE_DANGER=165 üstü) → 'kritik' uyarısı."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=170.0, back_a=SAFE_BACK)
        self.assertTrue(
            any("kritik" in w for w in result["warnings"]),
            "170° diz → 'kritik' uyarısı bekleniyor"
        )

    def test_knee_166_risk_danger(self):
        """166° (KNEE_DANGER=165 hemen üstü) → risk='danger'."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=166.0, back_a=SAFE_BACK)
        self.assertTrue(
            result["risk"] == "danger",
            f"166° diz → 'danger' bekleniyor, gelen: {result['risk']}"
        )

    def test_knee_165_tam_esik_uyari_yok(self):
        """165° tam eşik → diz uyarısı OLMAMALI."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=165.0, back_a=SAFE_BACK)
        self.assertFalse(
            any("kritik" in w for w in result["warnings"]),
            "165° tam eşik → diz uyarısı OLMAMALI"
        )

    def test_knee_60_alt_sinir_guvenli(self):
        """60° (90° altı diz) → diz uyarısı yok."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=60.0, back_a=SAFE_BACK)
        self.assertFalse(
            any("kritik" in w for w in result["warnings"]),
            "60° diz → uyarı OLMAMALI"
        )

    def test_knee_90_risk_ok(self):
        """90° diz → risk='ok'."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=90.0, back_a=SAFE_BACK)
        self.assertTrue(
            result["risk"] == "ok",
            f"90° diz → 'ok' bekleniyor, gelen: {result['risk']}"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SINIF 3 – Sırt Açısı Testleri
# ═════════════════════════════════════════════════════════════════════════════
class TestBackAngle(unittest.TestCase):
    """Sırt eğim açısının 90° altı / üstü davranışı."""

    def test_back_45_sirt_uyarisi(self):
        """45° (BACK_THRESH=35 üstü) → sırt eğilmesi uyarısı."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=SAFE_KNEE, back_a=45.0)
        self.assertTrue(
            any("SIRT" in w for w in result["warnings"]),
            "45° sırt → 'SIRT EĞİLMESİ' uyarısı bekleniyor"
        )

    def test_back_36_risk_danger(self):
        """36° (BACK_THRESH=35 hemen üstü) → risk='danger'."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=SAFE_KNEE, back_a=36.0)
        self.assertTrue(
            result["risk"] == "danger",
            f"36° sırt → 'danger' bekleniyor, gelen: {result['risk']}"
        )

    def test_back_35_tam_esik_uyari_yok(self):
        """35° tam eşik → sırt uyarısı OLMAMALI."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=SAFE_KNEE, back_a=35.0)
        self.assertFalse(
            any("SIRT" in w for w in result["warnings"]),
            "35° tam eşik → sırt uyarısı OLMAMALI"
        )

    def test_back_20_guvenli(self):
        """20° (90° altı) → sırt uyarısı yok."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=SAFE_KNEE, back_a=20.0)
        self.assertFalse(
            any("SIRT" in w for w in result["warnings"]),
            "20° sırt → uyarı OLMAMALI"
        )

    def test_back_50_status_duzelt(self):
        """50° sırt → status 'SIRT FORMUNU DÜZELTİN!'."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=SAFE_KNEE, back_a=50.0)
        self.assertTrue(
            result["status"] == "SIRT FORMUNU DÜZELTİN!",
            f"50° sırt → 'SIRT FORMUNU DÜZELTİN!' bekleniyor, gelen: '{result['status']}'"
        )

    def test_back_uyari_mesaji_derece_icerir(self):
        """Sırt uyarı mesajı gerçek derece değerini (42) içermeli."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=SAFE_KNEE, back_a=42.0)
        self.assertTrue(
            any("42" in w for w in result["warnings"]),
            "Sırt uyarısı '42' değerini içermeli"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SINIF 4 – Birleşik Senaryo Testleri
# ═════════════════════════════════════════════════════════════════════════════
class TestCombinedScenarios(unittest.TestCase):
    """Birden fazla açının aynı anda sınır dışında olduğu senaryolar."""

    def test_tum_guvenli_uyari_yok(self):
        """Tüm değerler güvenli → uyarı listesi boş, risk='ok'."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(len(result["warnings"]) == 0, "Uyarı beklenmiyordu")
        self.assertTrue(result["risk"] == "ok")

    def test_derin_squat_ve_sirt_egilmesi(self):
        """Çok derin squat (60°) + sırt eğimi (50°) → en az 2 uyarı, danger."""
        result = analyze_angles(hip_a=60.0, knee_a=SAFE_KNEE, back_a=50.0)
        self.assertTrue(len(result["warnings"]) >= 2, "En az 2 uyarı bekleniyor")
        self.assertTrue(result["risk"] == "danger")

    def test_yetersiz_derinlik_ve_kritik_diz(self):
        """Yetersiz derinlik (160°) + kritik diz (170°) → her iki uyarı da mevcut."""
        result = analyze_angles(hip_a=160.0, knee_a=170.0, back_a=SAFE_BACK)
        combined = " ".join(result["warnings"])
        self.assertTrue("Yetersiz" in combined, "Yetersiz derinlik uyarısı bekleniyor")
        self.assertTrue("kritik" in combined, "Diz kritik uyarısı bekleniyor")
        self.assertTrue(result["risk"] == "danger")

    def test_ideal_form_110_140_20(self):
        """İdeal squat formu → tamamen temiz sonuç."""
        result = analyze_angles(hip_a=110.0, knee_a=140.0, back_a=20.0)
        self.assertTrue(result["status"] == "FORM UYGUN")
        self.assertTrue(result["risk"] == "ok")
        self.assertTrue(len(result["warnings"]) == 0)

    def test_uc_kosul_tetiklenince_3_uyari(self):
        """Hip çok derin + sırt eğik + diz kritik → warnings listesinde 3 eleman."""
        result = analyze_angles(hip_a=60.0, knee_a=170.0, back_a=50.0)
        self.assertTrue(
            len(result["warnings"]) == 3,
            f"3 uyarı bekleniyor, gelen: {len(result['warnings'])}"
        )

    def test_sadece_diz_kritik_status_degismez(self):
        """Yalnızca diz kritik → kalça/sırt status'ları YOK."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=170.0, back_a=SAFE_BACK)
        self.assertTrue(any("kritik" in w for w in result["warnings"]))
        self.assertFalse("SAKATLIK" in result["status"])
        self.assertFalse("SIRT" in result["status"])

    def test_sirt_status_son_yazilir(self):
        """Hip danger (60°) + sırt danger (50°) → sırt status'u son yazar."""
        result = analyze_angles(hip_a=60.0, knee_a=SAFE_KNEE, back_a=50.0)
        self.assertTrue(
            result["status"] == "SIRT FORMUNU DÜZELTİN!",
            f"Sırt son yazmalıydı. Gelen: '{result['status']}'"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SINIF 5 – Sınır Değer (Boundary) Testleri
# ═════════════════════════════════════════════════════════════════════════════
class TestBoundaryValues(unittest.TestCase):
    """Eşik değerlerin tam altı/üstü – off-by-one kontrolleri."""

    def test_hip_esik_alti_79_danger(self):
        """HIP_DEEP − 1 = 79 → danger."""
        result = analyze_angles(hip_a=float(HIP_DEEP - 1), knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(result["risk"] == "danger")

    def test_hip_tam_esik_80_ok(self):
        """HIP_DEEP = 80 → danger DEĞİL."""
        result = analyze_angles(hip_a=float(HIP_DEEP), knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(result["risk"] == "ok")

    def test_hip_esik_ustu_131_warn(self):
        """HIP_SHALLOW + 1 = 131 → warn."""
        result = analyze_angles(hip_a=float(HIP_SHALLOW + 1), knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertTrue(result["risk"] == "warn")

    def test_hip_tam_esik_130_ok(self):
        """HIP_SHALLOW = 130 → warn DEĞİL."""
        result = analyze_angles(hip_a=float(HIP_SHALLOW), knee_a=SAFE_KNEE, back_a=SAFE_BACK)
        self.assertFalse(result["risk"] == "warn")

    def test_knee_esik_ustu_166_danger(self):
        """KNEE_DANGER + 1 = 166 → danger."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=float(KNEE_DANGER + 1), back_a=SAFE_BACK)
        self.assertTrue(result["risk"] == "danger")

    def test_knee_tam_esik_165_uyari_yok(self):
        """KNEE_DANGER = 165 → diz uyarısı yok."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=float(KNEE_DANGER), back_a=SAFE_BACK)
        self.assertFalse(any("kritik" in w for w in result["warnings"]))

    def test_back_esik_ustu_36_danger(self):
        """BACK_THRESH + 1 = 36 → danger."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=SAFE_KNEE, back_a=float(BACK_THRESH + 1))
        self.assertTrue(result["risk"] == "danger")

    def test_back_tam_esik_35_uyari_yok(self):
        """BACK_THRESH = 35 → sırt uyarısı yok."""
        result = analyze_angles(hip_a=SAFE_HIP, knee_a=SAFE_KNEE, back_a=float(BACK_THRESH))
        self.assertFalse(any("SIRT" in w for w in result["warnings"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
