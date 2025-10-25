"""
Configuration file for Face Authentication System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"
DB_DIR = BASE_DIR / "database"

# Model configurations
MODEL_CONFIG = {
    "arcface": {
        "name": "buffalo_l",  # InsightFace model
        "embedding_dim": 512
    },
    "adaface": {
        "weights_path": WEIGHTS_DIR / "adaface_ir50_webface4m.ckpt",
        "embedding_dim": 512
    },
    "elasticface": {
        "weights_path": WEIGHTS_DIR / "elasticface_ir50.pth",
        "embedding_dim": 512
    }
}

# Face detection configuration
DETECTION_CONFIG = {
    "min_face_size": 80,
    "confidence_threshold": 0.95,
    "image_size": 112
}

# Quality assessment thresholds
QUALITY_THRESHOLDS = {
    "min_blur_score": 30,      # Reduced from 100 to allow more blur
    "max_yaw": 60,             # Increased from 45 for more tolerance in face rotation
    "max_pitch": 60,           # Increased from 45 for more tolerance in face tilt
    "min_brightness": 30,      # Reduced from 50 to allow darker images
    "max_brightness": 220,     # Increased from 200 to allow brighter images
    "acceptable_quality_score": 50  # Reduced from 70 to accept images with lower quality score
}


# Risk scoring weights
RISK_WEIGHTS = {
    "sim": 0.35,
    "agree": 0.18,
    "margin": 0.15,
    "morph": 0.25,
    "cohort": 0.05,
    "uncertainty": 0.02
}

# Decision thresholds
DECISION_THRESHOLDS = {
    "low_risk": 25,
    "high_risk": 55,
    "high_similarity": 90,
    "strong_agreement": 75,
    "clear_margin": 15,
    "high_morph": 75,
    "high_cohort": 80
}

# FAISS configuration
FAISS_CONFIG = {
    "use_gpu": False,
    "index_type": "FlatIP",  # Inner Product (cosine similarity)
    "n_probe": 10,
    "top_k": 10
}

# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "face_auth"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "superuser")
}

# Redis configuration (for cohortness tracking)
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": 0,
    "cohort_ttl": 30 * 24 * 60 * 60  # 30 days in seconds
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True
}


