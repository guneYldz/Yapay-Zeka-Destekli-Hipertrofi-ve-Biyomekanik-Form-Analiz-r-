"""
Layered Architecture note:
- data: MediaPipe-like landmark ingestion and normalization only.
- core: pure biomechanics and training decisions with no UI/runtime dependency.
- ui: rendering and visual alert projection only.
- tests: isolated verification with mock data, keeping computer vision dependencies out of unit tests.
"""
