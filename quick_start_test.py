"""
Quick Start Test Script
Tests the Face Authentication System with sample images
"""
import cv2
import numpy as np
from preprocessing.face_detection import FaceDetector
from preprocessing.quality_assessment import QualityAssessor
from core.feature_extraction import FeatureExtractor
from core.faiss_search import FAISSSearcher
from core.signal_computation import SignalComputer
from core.risk_scoring import RiskScorer

def test_system():
    print("="*80)
    print("FACE AUTHENTICATION SYSTEM - QUICK START TEST")
    print("="*80)

    # Initialize components
    print("\n[1/6] Initializing components...")
    try:
        detector = FaceDetector()
        print("  ✓ Face detector loaded")
    except Exception as e:
        print(f"  ✗ Face detector failed: {e}")
        return

    try:
        quality_assessor = QualityAssessor()
        print("  ✓ Quality assessor loaded")
    except Exception as e:
        print(f"  ✗ Quality assessor failed: {e}")
        return

    try:
        extractor = FeatureExtractor()
        print("  ✓ Feature extractor loaded")
    except Exception as e:
        print(f"  ✗ Feature extractor failed: {e}")
        return

    try:
        searcher = FAISSSearcher()
        signal_computer = SignalComputer()
        risk_scorer = RiskScorer()
        print("  ✓ Search and scoring modules loaded")
    except Exception as e:
        print(f"  ✗ Module loading failed: {e}")
        return

    print("\n[2/6] Creating test images...")
    # Create synthetic test images (since we don't have real images)
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    cv2.imwrite('test_image.jpg', test_image)
    print("  ✓ Test image created: test_image.jpg")

    print("\n[3/6] Testing face detection...")
    image = cv2.imread('test_image.jpg')
    detection = detector.detect_and_align(image)

    if detection is None:
        print("  ⚠ No face detected in test image (expected for random image)")
        print("  → Please use a real face image for actual testing")
        print("\nTo test with a real image:")
        print("  1. Place a face photo as 'my_face.jpg' in this directory")
        print("  2. Run: python quick_start.py")
        return
    else:
        print(f"  ✓ Face detected with confidence: {detection['confidence']:.2f}")

    print("\n[4/6] Testing quality assessment...")
    quality = quality_assessor.assess_quality(
        image, 
        detection['bbox'],
        detection['landmarks']
    )
    print(f"  Status: {quality['status']}")
    print(f"  Quality score: {quality['quality_score']}")

    if quality['status'] == 'reject':
        print(f"  ⚠ Image rejected: {quality['reason']}")
        return

    print("\n[5/6] Testing feature extraction...")
    embeddings = extractor.extract_embeddings(detection['aligned_face'])
    print(f"  ✓ Extracted embeddings:")
    print(f"    - ArcFace: {embeddings['arcface'].shape}")
    print(f"    - AdaFace: {embeddings['adaface'].shape}")
    print(f"    - ElasticFace: {embeddings['elastic'].shape}")

    print("\n[6/6] Testing database operations...")
    # Add to database
    searcher.add_to_database(embeddings, "TEST_ID_001")
    print(f"  ✓ Added to database")
    print(f"  Database size: {searcher.get_db_size()}")

    # Search
    search_results = searcher.search(embeddings, k=5)
    print(f"  ✓ Search completed")
    print(f"    Top similarity: {search_results['arcface'][0][0]:.4f}")

    # Compute signals
    signals = signal_computer.compute_all_signals(
        search_results, 
        morph_prob=10, 
        cohort_count=0
    )

    # Risk scoring
    risk_score = risk_scorer.compute_risk_score(signals)
    decision = risk_scorer.make_decision(risk_score, signals)

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Decision: {decision['status']}")
    print(f"Risk Score: {risk_score:.2f}")
    print(f"Risk Level: {decision['risk_level']}")
    print(f"Action: {decision['action']}")
    print("\nSignals:")
    print(f"  - Similarity: {signals['sim']:.2f}")
    print(f"  - Agreement: {signals['agree']:.2f}")
    print(f"  - Margin: {signals['margin']:.2f}")
    print(f"  - Morph: {signals['morph']:.2f}")
    print(f"  - Cohort: {signals['cohort']:.2f}")
    print(f"  - Uncertainty: {signals['uncertainty']:.2f}")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print("\nNext steps:")
    print("1. Start the API server: python api/main.py")
    print("2. Visit http://localhost:8000/docs for API documentation")
    print("3. Test with real face images using the /api/verify endpoint")

if __name__ == "__main__":
    try:
        test_system()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        print("\nPlease check:")
        print("1. All dependencies are installed (pip install -r requirements.txt)")
        print("2. You have activated the virtual environment")
        print("3. InsightFace models are downloaded")
