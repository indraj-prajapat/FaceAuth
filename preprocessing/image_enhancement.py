# preprocessing/image_enhancement.py
"""
Face Enhancement Module
Implements super-resolution/restoration + brightness normalization + denoising
Processes original and (if needed) enhanced branch as specified
"""

import cv2
import tempfile
import numpy as np
import os
import subprocess
import shutil


class ImageEnhancer:
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image using CodeFormer CLI (calls the inference script as a subprocess).

        Steps:
        1. Save input image to a temporary directory.
        2. Call CodeFormer via subprocess with appropriate arguments.
        3. Retrieve enhanced output (if generated).
        4. Clean up temporary files.
        """

        try:
            # ✅ Create temporary workspace
            tmp_dir = tempfile.mkdtemp()
            input_path = os.path.join(tmp_dir, "input.png")
            output_dir = os.path.join(tmp_dir, "restored")

            print(f"[DEBUG] Temp input: {input_path}")
            print(f"[DEBUG] Temp output dir: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            # ✅ Save input image to temp file
            success = cv2.imwrite(input_path, image)
            if not success:
                print("❌ Failed to write temporary input image.")
                return image

            # ✅ Build subprocess command for CodeFormer
            cmd = [
                "python",
                "A:\\AIT\\FaceNew\\New\\CodeFormer\\inference_codeformer.py",
                "-w", "0.5",
                "--input_path", input_path,
                "--output_path", output_dir,
                "--bg_upsampler", "none"
            ]
            print(f"[DEBUG] Running command: {' '.join(cmd)}")

            # ✅ Run the subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)

            # ✅ Check subprocess exit status and logs
            print(f"[DEBUG] Return code: {result.returncode}")
            if result.stdout:
                print("[DEBUG] STDOUT:")
                print(result.stdout[:500])  # truncate long outputs
            if result.stderr:
                print("[DEBUG] STDERR:")
                print(result.stderr[:500])

            # ✅ Locate CodeFormer output
            enhanced_path = os.path.join(output_dir, "final_results", "input.png")
            print(f"[DEBUG] Expected enhanced path: {enhanced_path}")

            if os.path.exists(enhanced_path):
                print("[✅] CodeFormer enhancement successful.")
                enhanced = cv2.imread(enhanced_path)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return enhanced
            else:
                print("⚠️ CodeFormer output not found — returning original image.")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return image

        except Exception as e:
            print(f"❌ CodeFormer CLI enhancement failed: {e}")
            if 'tmp_dir' in locals() and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return image

    def enhance_if_needed(self, face_bgr: np.ndarray, quality_score: float):
        """
        Decide whether to apply enhancement based on quality score.

        Args:
            face_bgr (np.ndarray): Face image in BGR format.
            quality_score (float): Quality score from detector/forensics.

        Returns:
            dict: {
                'original': original image,
                'enhanced': enhanced version (if applicable),
                'used': 'original' | 'both',
                'ops': list of enhancement ops applied
            }
        """
        print(f"[DEBUG] Input quality score: {quality_score}")

        # ✅ Skip enhancement for already good-quality images
        if quality_score >= 70:
            print("[INFO] Quality >= 70 — skipping enhancement.")
            return {'original': face_bgr, 'enhanced': None, 'used': 'original', 'ops': []}

        print("[INFO] Quality < 70 — running CodeFormer enhancement.")
        enhanced = self.enhance_image(face_bgr)

        ops = ['CodeFormer']

        # ✅ Log operation summary
        print("[DEBUG] Enhancement complete. Returning results.")
        return {'original': face_bgr, 'enhanced': enhanced, 'used': 'both', 'ops': ops}
