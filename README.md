# Yapay Zeka Destekli Hipertrofi ve Biyomekanik Form Analizörü

Bu repo, katmanlı mimari (Layered Architecture) yaklaşımıyla, poz verisi (MediaPipe vb.) üzerinden form analizi üretmeyi hedefler.

## Katmanlar

- `domain/`: Saf matematik/biomekanik, kurallar ve analiz (UI ve CV bağımlılığı yok).
- `application/`: Use-case ve orkestrasyon katmanı.
- `data/`: Pose (landmark) edinimi ve adapter katmanı (MediaPipe entegrasyonu burada).
- `presentation/`: CLI ve görsel arayuzler.
- `tests/`: Unit testler (CV bağımlılığı olmadan, mock/simülasyon verilerle).

## Form Analizi

- `domain/calculator.py`: İleride landmark -> açı hesaplayacak modülün iskeleti ve `CalculatedAngles` tipi.
- `domain/analyzer.py`: `CalculatedAngles` girdisini alıp threshold (eşik) mantığıyla uyarılar üreten `FormAnalyzer`.

## Testleri Çalıştırma

```bash
python -m unittest discover -s tests -v
```
