# Face Authentication System
## IndiaAI Face Authentication Challenge Implementation

A secure, scalable AI-based face verification and de-duplication system for examination and government applications.

## Features

- **Multi-Model Ensemble**: Uses ArcFace, AdaFace, and ElasticFace for robust face recognition
- **Quality Assessment**: Automatic image quality checks (blur, brightness, pose)
- **Morph Detection**: Detects morphed/blended faces
- **Risk Scoring**: Multi-signal risk assessment with configurable thresholds
- **Scalable Search**: FAISS-based similarity search for fast database queries
- **REST API**: FastAPI-based API for easy integration

## System Architecture

```
Input Photo → Quality Check → Face Detection → Feature Extraction
                                                      ↓
                                            (3 Models in Parallel)
                                                      ↓
                                              FAISS Search
                                                      ↓
                                          Signal Computation
                                                      ↓
                                            Risk Scoring
                                                      ↓
                                  Decision: Low/Medium/High Risk
```

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL (optional, for production)
- Redis (optional, for cohortness tracking)

### Step 1: Clone/Extract the project

```bash
cd face-authentication-system
```

### Step 2: Create virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## For image enhancement clone Codeformer from git

```bash
# git clone this repository
git clone https://github.com/sczhou/CodeFormer
cd CodeFormer

# create new anaconda env
conda create -n codeformer python=3.8 -y
conda activate codeformer

# install python dependencies
pip3 install -r requirements.txt
python basicsr/setup.py develop
conda install -c conda-forge dlib (only for face detection or cropping with dlib)
```
### Step 4: Download model weights

The system uses InsightFace models which will be auto-downloaded on first run.
For custom models (AdaFace, ElasticFace), download from:

- AdaFace: https://github.com/mk-minchul/AdaFace
- ElasticFace: https://github.com/fdbtrs/ElasticFace

Place weights in the `weights/` directory.

### Step 5: Set up database (Optional)

For production use with PostgreSQL:

```bash
# Create database
createdb face_auth

# Run schema
psql face_auth < database/schema.sql
```

For development, the system works without a database (in-memory only).

## Running the System

### Option 1: Run API Server

```bash
python api/main.py
```

The API will be available at: http://localhost:8000

API Documentation: http://localhost:8000/docs

### Option 2: Use as Python Library

```python
from preprocessing.face_detection import FaceDetector
from preprocessing.quality_assessment import QualityAssessor
from core.feature_extraction import FeatureExtractor
from core.faiss_search import FAISSSearcher
from core.signal_computation import SignalComputer
from core.risk_scoring import RiskScorer
import cv2

# Initialize components
detector = FaceDetector()
quality_assessor = QualityAssessor()
feature_extractor = FeatureExtractor()
faiss_searcher = FAISSSearcher()
signal_computer = SignalComputer()
risk_scorer = RiskScorer()

# Process an image
image = cv2.imread('face.jpg')
detection = detector.detect_and_align(image)

if detection:
    quality = quality_assessor.assess_quality(
        image, detection['bbox'], detection['landmarks']
    )

    if quality['status'] == 'accept':
        embeddings = feature_extractor.extract_embeddings(
            detection['aligned_face']
        )

        # Search database
        results = faiss_searcher.search(embeddings, k=10)

        # Compute signals and risk
        signals = signal_computer.compute_all_signals(results, 10, 0)
        risk = risk_scorer.compute_risk_score(signals)
        decision = risk_scorer.make_decision(risk, signals)

        print(f"Decision: {decision['status']}")
        print(f"Risk Score: {risk:.2f}")
```

## API Usage

### Verify Face

```bash
curl -X POST "http://localhost:8000/api/verify" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "applicant_id=APP12345"
```

Response:
```json
{
  "status": "success",
  "branch": "ArcFace",
  "applicant_id": "APP12345",
  "probe": {
    "probe_id": "PROBE123",
    "filename_orig": "face_orig.jpg",
    "filename_enh": "face_enh.jpg",
    "image_url_orig": "http://localhost:8000/static/probes/face_orig.jpg",
    "image_url_enh": "http://localhost:8000/static/probes/face_enh.jpg"
  },
  "decision": "unique",
  "action": "Auto-cleared as unique",
  "reason": "No duplicate found",
  "risk_score": 15.23,
  "risk_level": "LOW",
  "signals": {
    "similarity": 45.2,
    "agreement": 33.3,
    "margin": 5.8,
    "morph": 10.0,
    "forns": 0.0,
    "cohort": 0.0,
    "uncertainty": 8.5
  },
  "best_match": {
    "match_id": "DB123",
    "similarity": 78.4,
    "image_url": "http://localhost:8000/static/db/DB123.jpg"
  },
  "matches": [
    {"match_id": "DB123", "similarity": 78.4},
    {"match_id": "DB124", "similarity": 75.6}
  ],
  "quality_score": 85.0
}

```

### Get Statistics

```bash
curl http://localhost:8000/api/stats
```

## Configuration

Edit `config.py` to customize:

- **Risk Weights**: Adjust importance of different signals
- **Decision Thresholds**: Set cutoffs for low/medium/high risk
- **Quality Thresholds**: Configure quality assessment criteria
- **FAISS Settings**: Change index type for larger databases

## Project Structure

```
face-authentication-system/
├── api/                    # FastAPI application
│   └── main.py
├── core/                   # Core processing modules
│   ├── feature_extraction.py
│   ├── faiss_search.py
│   ├── signal_computation.py
│   └── risk_scoring.py
├── preprocessing/          # Image preprocessing
│   ├── face_detection.py
│   ├── quality_assessment.py
│   └── image_enhancement.py
├── models/                 # Model implementations
├── database/               # Database schema
│   └── schema.sql
├── weights/                # Pre-trained model weights
├── config.py               # Configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Run Frontend (bio-scan-pro)

The frontend is a React web dashboard for interacting with the API.



```bash
cd bio-scan-pro
npm install
npm run dev

```

The frontend will start on http://localhost:8080
It connects automatically to the backend running at http://localhost:8000





## Signal Explanation

1. **Similarity (SIM)**: Average cosine similarity across 3 models (0-100)
   - >90: Very strong match (likely duplicate)
   - 80-90: Strong match
   - 70-80: Weak match
   - <70: No match

2. **Agreement (AGREE)**: % of models agreeing on top candidate (0-100)
   - ≥75%: Strong consensus
   - 50-75%: Moderate
   - <50%: Low consensus

3. **Margin (MARGIN)**: Gap between top-1 and top-2 matches (0-100)
   - >15: Clear winner
   - 5-15: Ambiguous
   - <5: Near-tie (suspicious)

4. **Morph (MORPH)**: Morphed face probability (0-100)
   - >70: High risk
   - 40-70: Suspicious
   - <40: Low risk

5. **Cohort (COHORT)**: Times a DB image has been matched (normalized 0-100)
   - >80: 10+ matches (highly suspicious)
   - 40-80: 5-10 matches
   - <40: <5 matches

6. **Uncertainty (UNC)**: Model disagreement level (0-100)
   - >40: High uncertainty
   - 20-40: Medium
   - <20: Low

## Risk Levels

- **LOW (≤25)**: Auto-process (accept as unique or confirm duplicate)
- **MEDIUM (26-55)**: Manual review required
- **HIGH (>55)**: Quarantine + forensic investigation

## Performance

- **Query Latency**: <1 second for 5K-100K database
- **Throughput**: 100+ requests/minute
- **Storage**: ~2KB per face (embeddings only)
- **Target FAR**: <0.1% (False Accept Rate)
- **Target FRR**: <2% (False Reject Rate)

## Troubleshooting

### Issue: "No module named 'insightface'"

```bash
pip install insightface
```

### Issue: ONNX runtime error

```bash
pip install onnxruntime
```

### Issue: FAISS installation fails

```bash
# For CPU-only
pip install faiss-cpu

# For GPU
pip install faiss-gpu
```

### Issue: Face detection fails

Check image quality:
- Ensure face is clearly visible
- Good lighting
- Minimal blur
- Frontal pose (±45° acceptable)

## License

This project is for the IndiaAI Face Authentication Challenge.

## Contact

For issues and questions, please refer to the IndiaAI challenge forum.
