import cv2
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
sys.path.insert(0, ROOT_DIR)

from insightface.app import FaceAnalysis
from preprocessing.face_detection import FaceDetector

# Initialize detector
detector = FaceDetector()

# Initialize FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load image
image = cv2.imread('face.png')

print(f"Original image shape: {image.shape}")

# Detect and align face
det = detector.detect_and_align(image)
aligned_face = det['aligned_face']

print(f"Aligned face shape: {aligned_face.shape}")
print(f"Aligned face dtype: {aligned_face.dtype}")
print(f"Min/Max values: {aligned_face.min()}/{aligned_face.max()}")

# ========================================================================
# CORRECT APPROACH: Use a DIFFERENT method to extract embedding
# ========================================================================
# Instead of app.get(), use the model's internal embedding extractor directly

# Method 1: Use get() with padding (ADD CONTEXT)
print("\n[Method 1] Adding padding for context...")
h, w = aligned_face.shape[:2]
padded = cv2.copyMakeBorder(
    aligned_face,
    20, 20, 20, 20,  # Top, bottom, left, right padding
    cv2.BORDER_REFLECT
)

print(f"Padded image shape: {padded.shape}")

faces = app.get(padded)
print(f"Faces detected in padded image: {len(faces)}")

if len(faces) > 0:
    embedding = faces[0].embedding
    print(f"Raw embedding: {embedding}")
    embedding = embedding / np.linalg.norm(embedding)
    
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding norm: {np.linalg.norm(embedding):.6f}")
    print(f"✓ Embedding min/max: {embedding.min():.4f}/{embedding.max():.4f}")
    print(f"✓ Embedding all zeros? {np.all(embedding == 0)}")
else:
    print("❌ Still no faces detected!")

print("\n")

# Method 2: Direct model inference (RECOMMENDED)
print("[Method 2] Direct model inference...")
import onnxruntime as ort

try:
    # Get the embedding model directly
    embedding_model = app.model_zoo.get_model('recognition')
    
    if embedding_model is not None:
        # Prepare input
        input_blob = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        input_blob = np.transpose(input_blob, (2, 0, 1))  # HWC -> CHW
        input_blob = np.expand_dims(input_blob, 0)  # Add batch dimension
        input_blob = (input_blob.astype(np.float32) - 127.5) / 128.0
        
        # Get embedding
        embedding = embedding_model.forward(input_blob)
        embedding = embedding[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        print(f"✓ Direct embedding shape: {embedding.shape}")
        print(f"✓ Direct embedding norm: {np.linalg.norm(embedding):.6f}")
        print(f"✓ Direct embedding min/max: {embedding.min():.4f}/{embedding.max():.4f}")
    else:
        print("Embedding model not found")
        
except Exception as e:
    print(f"Error: {e}")
