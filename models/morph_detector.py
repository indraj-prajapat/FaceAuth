import cv2
import numpy as np

class MorphDetector:
    def __init__(self, discrepancy_flag=25.0, penalty=15.0):
        self.discrepancy_flag = discrepancy_flag
        self.penalty = penalty
        self.model = None  # Placeholder for real model if any

    def detect_prob_single(self, img_bgr):
        if self.model:
            prob = float(self.model.predict(img_bgr))
            return np.clip(prob, 0, 100)
        
        # Heuristic fallback with enhanced signals
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Frequency analysis
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.log1p(np.abs(fshift))
        h, w = mag.shape
        center = mag[h//2-12:h//2+12, w//2-12:w//2+12].mean()
        band = mag[h//2-36:h//2+36, w//2-36:w//2+36].mean()
        freq_score = max(0.0, (band - center) * 6.0)
        
        # 2. Symmetry analysis
        edges = cv2.Canny(gray, 60, 160)
        left, right = edges[:, :w//2].mean(), edges[:, w//2:].mean()
        sym_score = abs(left - right) * 5.0
        
        # 3. Gamma variance
        vals = []
        for g in (0.8, 1.0, 1.2, 1.4):
            aug = np.clip((gray / 255.0)**g, 0, 1)
            vals.append(float(aug.mean()))
        var_score = np.std(vals) * 600.0
        
        # ✅ NEW: 4. Texture inconsistency (Local Binary Patterns variance)
        h_regions, w_regions = gray.shape[0] // 4, gray.shape[1] // 4
        region_stds = []
        for i in range(4):
            for j in range(4):
                region = gray[i*h_regions:(i+1)*h_regions, j*w_regions:(j+1)*w_regions]
                region_stds.append(np.std(region))
        texture_score = np.std(region_stds) * 2.0  # Inconsistent texture patches
        
        # ✅ NEW: 5. Gradient magnitude variance (blending artifacts)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_score = np.std(grad_mag) * 0.5
        
        # ✅ NEW: 6. Color channel correlation (morphs show channel misalignment)
        if len(img_bgr.shape) == 3:
            b, g, r = cv2.split(img_bgr)
            corr_bg = np.corrcoef(b.flatten(), g.flatten())[0, 1]
            corr_br = np.corrcoef(b.flatten(), r.flatten())[0, 1]
            corr_gr = np.corrcoef(g.flatten(), r.flatten())[0, 1]
            avg_corr = (corr_bg + corr_br + corr_gr) / 3.0
            color_score = (1.0 - avg_corr) * 30.0  # Lower correlation = higher morph
        else:
            color_score = 0.0
        
        # Weighted combination (adjusted weights)
        prob = (0.30 * freq_score + 
                0.25 * sym_score + 
                0.15 * var_score + 
                0.12 * texture_score + 
                0.10 * grad_score + 
                0.08 * color_score)
        
        return np.clip(prob, 0, 100)

    def detect_both(self, orig_bgr, enh_bgr=None):
        p_orig = self.detect_prob_single(orig_bgr)
        if enh_bgr is None:
            return p_orig, False
        p_enh = self.detect_prob_single(enh_bgr)
        delta = abs(p_orig - p_enh)
        discrepancy = delta > self.discrepancy_flag
        if discrepancy:
            combined = np.clip(max(p_orig, p_enh) + self.penalty, 0, 100)
        else:
            combined = (p_orig + p_enh) / 2.0
        return combined, discrepancy

    def _heuristic_prob(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1) Mid/high-frequency energy ratio (morphs often smooth boundaries)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        mag = np.log1p(np.abs(fshift))
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        r1, r2 = min(cy, cx) // 6, min(cy, cx) // 3
        center = mag[cy - r1:cy + r1, cx - r1:cx + r1].mean()
        mid = mag[cy - r2:cy + r2, cx - r2:cx + r2].mean()
        freq_score = float(np.clip((mid - center) * 4, 0, 100))

        # 2) Edge symmetry discrepancy (left vs right)
        edges = cv2.Canny(img, 50, 150)
        left = edges[:, :w//2].mean()
        right = edges[:, w//2:].mean()
        sym_gap = abs(left - right) * 2.5
        sym_score = float(np.clip(sym_gap, 0, 100))

        # 3) Augmentation variance proxy (small random gamma jitter)
        sims = []
        for g in [0.8, 1.0, 1.2, 1.4]:
            aug = np.clip((img / 255.0) ** g, 0, 1)
            sims.append(aug.mean())
        var = np.std(sims)
        var_score = float(np.clip(var * 500, 0, 100))

        # Weighted sum (tuned light; conservative toward higher prob)
        prob = 0.45 * freq_score + 0.35 * sym_score + 0.20 * var_score
        return float(np.clip(prob, 0, 100))
