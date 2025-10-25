"""
Quick Start Script for Face Authentication System
Runs the full pipeline step-by-step:
Quality → Enhancement → Detection → Embedding → Search → 7-Signal Fusion → Risk → Decision
"""

import cv2
import numpy as np
from preprocessing.face_detection import FaceDetector
from preprocessing.quality_assessment import QualityAssessor
from preprocessing.image_enhancement import ImageEnhancer
from preprocessing.forensics import ForensicsDetector
from core.feature_extraction import FeatureExtractor
from core.faiss_search import FAISSSearcher
from core.signal_computation import SignalComputer
from core.risk_scoring import RiskScorer
from models.morph_detector import MorphDetector


def main(image_path: str, applicant_id: str = "TEST_ID"):
    print("=" * 80)
    print("FACE AUTHENTICATION SYSTEM - QUICK START TEST")
    print("=" * 80)

    # Initialize all core components
    print("\n[1/7] Initializing components...")
    detector = FaceDetector()
    quality = QualityAssessor()
    enhancer = ImageEnhancer()
    forensic = ForensicsDetector()
    extractor = FeatureExtractor()
    searcher = FAISSSearcher()
    signal_comp = SignalComputer()
    scorer = RiskScorer()
    morph_detector = MorphDetector()
    print("✓ All components initialized successfully.\n")

    # Load test image
    print("[2/7] Reading input image...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ Could not load image: {image_path}")
        return
    print(f"✓ Image loaded: {image_path}")

    # Face detection
    print("\n[3/7] Detecting face...")
    detection = detector.detect_and_align(img)
    if detection is None:
        print("✗ No face detected. Please use a clear frontal face image.")
        return
    print(f"✓ Face detected. Confidence: {detection['confidence']:.2f}")

    # Quality assessment
    print("\n[4/7] Assessing image quality...")
    q = quality.assess_quality(
        img,
        detection["bbox"],
        detection["landmarks"]
    )
    print(f"Quality status: {q['status']}, score: {q['quality_score']:.2f}")
    if q["status"] == "reject":
        print(f"✗ Image rejected: {q['reason']}")
        return

    # Enhancement
    print("\n[5/7] Running face enhancement (if needed)...")
    enhanced = enhancer.enhance_if_needed(detection["aligned_face"], q["quality_score"])
    print(f"Enhancement operations: {enhanced['ops']}")
    enhanced_face = enhanced["enhanced"]
    use_enhanced = enhanced_face is not None

    # Forensics analysis
    forns_score = forensic.score(detection["aligned_face"])
    print(f"Forensics Score (FORNS): {forns_score:.2f}")

    # Morph detection
    morph_prob, morph_discrepancy = morph_detector.detect_both(
        detection["aligned_face"],
        enhanced_face if use_enhanced else None
    )
    print(f"Morph Probability: {morph_prob:.2f}")
    print(f"Morph Discrepancy: {morph_discrepancy}")

    # Feature extraction
    print("\n[6/7] Extracting embeddings...")
    emb_original = extractor.extract_embeddings(detection["aligned_face"])
    emb_enhanced = extractor.extract_embeddings(enhanced_face) if use_enhanced else emb_original
    print("✓ Embeddings extracted from both branches.")

    # Search (build DB if empty)
    print("\n[7/7] Running similarity search and risk scoring...")
    if searcher.get_db_size() == 0:
        searcher.add_to_database(emb_original, applicant_id)
        print("✓ First entry added. Risk = 0.0, Decision = unique")
        return

    res_orig = searcher.search(emb_original, k=5)
    res_enh = searcher.search(emb_enhanced, k=5) if use_enhanced else None

    # Compute signals for both branches
    s_orig = signal_comp.compute_all(res_orig, morph_prob, cohort_count=0, forns=forns_score)
    s_enh = signal_comp.compute_all(res_enh, morph_prob, cohort_count=0, forns=forns_score) if res_enh else None

    chosen_signals = s_orig
    chosen_branch = "original"
    if s_enh and s_enh["uncertainty"] < s_orig["uncertainty"]:
        chosen_signals = s_enh
        chosen_branch = "enhanced"

    risk_score = scorer.compute(chosen_signals)
    decision = scorer.decide(
        risk_score,
        chosen_signals,
        quality_pass=q["status"] != "reject",
        morph_discrepancy=morph_discrepancy
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Branch used: {chosen_branch}")
    print(f"Risk score: {risk_score:.2f}")
    print(f"Decision: {decision['status']} -> {decision['action']}")
    print(f"Risk level: {decision['risk_level']}")
    print(f"Reason: {decision.get('reason', '-')}")
    print("-")

    print("Signals:")
    for k, v in chosen_signals.items():
        if isinstance(v, (float, int)):
            print(f"  {k.upper()}: {v:.2f}")
    print("=" * 80)
    print("✓ Quick start test completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main('face.png')
