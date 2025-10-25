"""
Face Detection and Alignment Module
"""
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_and_align(self, image):
        """
        Detect face and return aligned face + metadata.
        If no face is detected, use the whole image as a face.
        """
        padded = cv2.copyMakeBorder(
            image,
            20, 20, 20, 20,
            cv2.BORDER_REFLECT
        )
        faces = self.app.get(padded)

        if len(faces) == 0:
            # Treat entire image as the face
            h, w = image.shape[:2]
            bbox = [0, 0, w, h]
            landmarks = np.array([
                [w * 0.3, h * 0.3],  # approximate left eye
                [w * 0.7, h * 0.3],  # approximate right eye
                [w * 0.5, h * 0.5],  # approximate nose tip
                [w * 0.35, h * 0.7], # approximate left mouth corner
                [w * 0.65, h * 0.7], # approximate right mouth corner
            ], dtype=np.float32)
            # Return the whole image as aligned face, no transform needed
            return {
                'aligned_face': cv2.resize(image, (112, 112)),
                'bbox': bbox,
                'landmarks': landmarks,
                'confidence': 1.0  # Max confidence since we're defaulting
            }
        print(f"Detected {len(faces)} faces")
        # Take the largest/most confident face
        face = faces[0]

        # Extract bbox
        bbox = face.bbox.astype(int)
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y

        # Align face to 112x112
        aligned_face = self._align_face(image, face.kps)

        return {
            'aligned_face': aligned_face,
            'bbox': [x, y, w, h],
            'landmarks': face.kps,
            'confidence': face.det_score
        }

    def _align_face(self, image, landmarks):
        """Align face to canonical 112x112 size"""
        # Reference points for alignment
        ref_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        # Compute similarity transform
        tform = self._similarity_transform(landmarks, ref_pts)

        # Warp image
        aligned = cv2.warpAffine(image, tform, (112, 112), 
                                  flags=cv2.INTER_LINEAR)

        return aligned

    def _similarity_transform(self, src_pts, dst_pts):
        """Compute similarity transformation matrix"""
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        # Compute transformation using OpenCV
        tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        return tform
