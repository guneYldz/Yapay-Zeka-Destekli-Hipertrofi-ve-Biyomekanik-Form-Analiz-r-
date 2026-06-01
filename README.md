# Yapay Zeka Destekli Biyomekanik Form Analiz Sistemi

Bu çalışma, bilgisayarlı görü (Computer Vision) ve yapay öğrenme teknikleri kullanılarak sporcuların egzersiz biyomekaniğini gerçek zamanlı olarak takip eden ve analiz eden bir yazılım projesidir. Sistem, MediaPipe Pose Estimation teknolojisi üzerine inşa edilmiş olup, kullanıcıya anlık teknik geri bildirim sağlayarak sakatlık riskini azaltmayı ve antrenman verimliliğini artırmayı hedefler.

## 📌 Proje Özellikleri

*   **Gerçek Zamanlı Poz Tahmini:** MediaPipe Pose Landmarker (Heavy Model) kullanılarak vücuttaki 33 ana eklem noktası üzerinden milisaniyelik analiz.
*   **Biyomekanik Kural Motoru:** Squat ve benzeri bileşik hareketlerde kalça, diz ve omurga açılarını (Heuristic-based) analiz eden algoritma altyapısı.
*   **Görsel Veri Paneli (HUD):** OpenCV pencereleri üzerinden kullanıcıya sunulan ham veri göstergeleri ve teknik uyarı sistemleri.
*   **Kullanıcı Yönetimi:** SQLite tabanlı veritabanı persistancı ile kullanıcı kayıt, giriş ve fiziksel veri (Boy, Kilo, VKİ) takibi.
*   **Modern Arayüz:** Flet (Flutter based) framework kullanılarak geliştirilen, katmanlı mimari (Separation of Concerns) prensiplerine uygun kullanıcı paneli.

## 🛠 Kullanılan Teknolojiler

*   **Dil:** Python 3.x
*   **Poz Algılama:** MediaPipe Vision Tasks
*   **Görüntü İşleme:** OpenCV (Open Source Computer Vision Library)
*   **Kullanıcı Arayüzü:** Flet (Pythonic Flutter UI)
*   **Veri Yönetimi:** SQLite3 (Local RDBMS)
*   **Matematiksel Hesaplamalar:** NumPy ve Trigonometrik Fonksiyonlar

## 📁 Proje Yapılandırması

*   `presentation/`: Flet tabanlı kullanıcı arayüzü katmanı. Doğrudan veri erişimi yapmaz.
*   `application/`: Uygulama servisleri (Use Cases). Arayüz ile iş mantığını birbirine bağlayan orkestrasyon katmanı (`user_service.py`, `analysis_service.py`).
*   `domain/`: Sistemin kalbi olan biyomekanik kurallar, geometrik hesaplamalar ve temel varlıklar (Entities).
*   `data/`: SQLite veritabanı erişimi ve ham veri yönetimi.
*   `form_analyzer.py`: Sistemin ana giriş noktası ve gerçek zamanlı görüntü işleme döngüsü.

## 🚀 Kurulum ve Çalıştırma

1.  **Gerekli Kütüphanelerin Yüklenmesi:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Sistemin Başlatılması:**
    ```bash
    python form_analyzer.py
    ```

## 📐 Analiz Metodolojisi

Sistem, kamera görüntüsü üzerinden aldığı koordinatları kullanarak şu biyomekanik eşikleri denetler:
*   **Kalça Açısı:** Squat derinliğinin yeterli olup olmadığını paralel (`100°`) ve derin (`80°`) eşikleriyle ölçer.
*   **Sırt Eğimi:** Omurganın düşey eksenden sapmasını (`35°` eşiği) takip ederek sakatlık riski uyarısı verir.
*   **Diz Açısı:** Eklemin aşırı bükülme durumlarını denetler.

---
*Bu çalışma, Bilgisayarlı Görü ve İnsan Bilgisayar Etkileşimi dersleri kapsamında bir proje ödevi olarak geliştirilmiştir.*
