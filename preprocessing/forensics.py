# preprocessing/forensics.py
"""
Forensics Score v2 (FORNS++)
Enhanced signal-based tamper/rephoto detector.
Detects halftone/moiré (FFT), JPEG double compression, and local ELA anomalies.
Returns a single score (0–100, higher = more suspicious).
"""

import cv2
import numpy as np

class ForensicsDetector:
    def __init__(self, ela_quality: int = 85):
        self.ela_quality = ela_quality

    def score(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        # --- 1) FFT Periodicity (adaptive ring) ---
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.log1p(np.abs(fshift))

        cy, cx = h // 2, w // 2
        R1 = max(5, min(h, w) // 40)
        R2 = max(R1 + 5, min(h, w) // 8)

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        inner = (dist < R1)
        outer = (dist < R2) & (~inner)

        ring_diff = mag[outer].mean() - mag[inner].mean()
        fft_score = float(np.clip(ring_diff * 3.5, 0, 100))

        # --- 2) JPEG Blockiness (normalized) ---
        H = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        V = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.abs(H) + np.abs(V)

        boundary = (edges[:, 7::8].mean() + edges[7::8, :].mean()) / 2.0
        global_mean = edges.mean() + 1e-5
        ratio = boundary / global_mean
        block_score = float(np.clip((ratio - 1.0) * 60.0, 0, 100))

        # --- 3) Local ELA anomaly (localized compression inconsistency) ---
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.ela_quality]
        _, buf = cv2.imencode('.jpg', img_bgr, encode_param)
        rec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(img_bgr, rec)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        local_var = cv2.absdiff(diff_gray, blur)
        ela_score = float(np.clip(local_var.mean() * 2.0, 0, 100))

        # --- Weighted Combination ---
        forns = 0.40 * fft_score + 0.35 * block_score + 0.25 * ela_score
        return float(np.clip(forns, 0, 100))

