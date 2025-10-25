"""
Quality Assessment Module
Evaluates image quality before processing
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import QUALITY_THRESHOLDS

class QualityAssessor:
    def __init__(self):
        self.thresholds = QUALITY_THRESHOLDS
    def _safe_crop(image: np.ndarray, bbox: list[int]) -> np.ndarray | None:
        print(f"[DEBUG] _safe_crop called with bbox: {bbox}")
        H, W = image.shape[:2]
        print(f"[DEBUG] Image dimensions: {W}x{H}")
        x, y, w, h = map(int, bbox)
        print(f"[DEBUG] Original bbox values: x={x}, y={y}, w={w}, h={h}")
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        print(f"[DEBUG] Cropping coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        # Validate proper ordering and non-zero size
        if x1 >= x2 or y1 >= y2:
            return None
        print(f"[DEBUG] Cropped region size: {(x2 - x1)}x{(y2 - y1)}")
        return image[y1:y2, x1:x2]
    def assess_quality(self, image, face_bbox=None, landmarks=None):
        """
        Comprehensive quality assessment

        Args:
            image: Input image (numpy array)
            face_bbox: Face bounding box [x, y, w, h]
            landmarks: Facial landmarks (5 points)

        Returns:
            dict: Quality assessment results
        """
        results = {
            "status": "accept",
            "quality_score": 100,
            "reason": None,
            "checks": {}
        }

        # Check if face detection succeeded
        if face_bbox is None:
            results["status"] = "reject"
            results["reason"] = "no_face_detected"
            results["quality_score"] = 0
            return results
        print(f"[DEBUG] Face bbox for quality assessment: {face_bbox}")
      
        # Extract face region
        face = image
        print(f"[DEBUG] Extracted face shape: {face.shape}")
        # Face size check
        if face is None or face.size == 0:
            results["status"] = "reject"
            results["reason"] = "invalid_bbox_crop"
            results["quality_score"] = 0
            return results
        
        # Blur detection
        blur_score = self._detect_blur(face)
        results["checks"]["blur_score"] = blur_score
        if blur_score < self.thresholds["min_blur_score"]:
            results["status"] = "reject"
            results["reason"] = "too_blurry"
            results["quality_score"] = min(results["quality_score"], 30)
        print(f"[DEBUG] Blur score: {blur_score}")
        # Brightness check
        brightness = self._check_brightness(face)
        results["checks"]["brightness"] = brightness
        if (brightness < self.thresholds["min_brightness"] or 
            brightness > self.thresholds["max_brightness"]):
            if results["status"] == "accept":
                results["status"] = "warning"
            results["reason"] = "brightness_issue"
            results["quality_score"] = min(results["quality_score"], 60)
        print(f"[DEBUG] Brightness: {brightness}")
        # Pose estimation (if landmarks available)
        if landmarks is not None:
            yaw, pitch, roll = self._estimate_pose(landmarks)
            results["checks"]["pose"] = {"yaw": yaw, "pitch": pitch, "roll": roll}

            if (abs(yaw) > self.thresholds["max_yaw"] or 
                abs(pitch) > self.thresholds["max_pitch"]):
                results["status"] = "reject"
                results["reason"] = "extreme_pose"
                results["quality_score"] = min(results["quality_score"], 40)
        print(f"[DEBUG] Pose - Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")
        # Calculate final quality score
        if results["status"] == "accept":
            results["quality_score"] = self._calculate_quality_score(results["checks"])
        print(f"[DEBUG] Final quality score: {results['quality_score']}")
        return results

    def _detect_blur(self, face):
        """Detect blur using Laplacian variance"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def _check_brightness(self, face):
        """Check average brightness"""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        return np.mean(gray)

    def _estimate_pose(self, landmarks):
        """
        Estimate head pose from 5-point landmarks
        Simplified estimation
        """
        # landmarks: [[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x5,y5]]
        # 0,1: eyes, 2: nose, 3,4: mouth corners

        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        # Yaw (horizontal rotation)
        eye_center = (left_eye + right_eye) / 2
        eye_to_nose = nose[0] - eye_center[0]
        eye_dist = np.linalg.norm(right_eye - left_eye)
        yaw = np.degrees(np.arctan2(eye_to_nose, eye_dist / 2)) if eye_dist > 0 else 0

        # Pitch (vertical rotation)
        eye_nose_dist_y = abs(nose[1] - eye_center[1])
        expected_ratio = 0.4  # Typical ratio for frontal face
        actual_ratio = eye_nose_dist_y / eye_dist if eye_dist > 0 else 0
        pitch = (actual_ratio - expected_ratio) * 100

        # Roll (tilt)
        roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], 
                                       right_eye[0] - left_eye[0]))

        return yaw, pitch, roll

    def _calculate_quality_score(self, checks):
        """Calculate overall quality score (0-100)"""
        score = 100

        # Blur penalty
        if "blur_score" in checks:
            if checks["blur_score"] < 200:
                score -= 20
            elif checks["blur_score"] < 300:
                score -= 10

        # Brightness penalty
        if "brightness" in checks:
            brightness = checks["brightness"]
            if brightness < 80 or brightness > 180:
                score -= 15
            elif brightness < 100 or brightness > 160:
                score -= 5

        # Pose penalty
        if "pose" in checks:
            pose = checks["pose"]
            if abs(pose["yaw"]) > 30 or abs(pose["pitch"]) > 30:
                score -= 20
            elif abs(pose["yaw"]) > 15 or abs(pose["pitch"]) > 15:
                score -= 10

        return max(score, 0)
