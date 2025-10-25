from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi import Request

import os, cv2, numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime

# Your modules
from preprocessing.face_detection import FaceDetector
from preprocessing.quality_assessment import QualityAssessor
from preprocessing.image_enhancement import ImageEnhancer
from preprocessing.forensics import ForensicsDetector
from core.feature_extraction import MultiEmbedder
from core.faiss_search import FAISSSearcher
from core.signal_computation import SignalComputer
from core.risk_scoring import RiskScorer
from models.morph_detector import MorphDetector

# SQL / DB
from sqlalchemy import Column
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlmodel import SQLModel, Field, Session, create_engine, select

class Probe(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    probe_id: str = Field(index=True)
    applicant_id: Optional[str] = Field(default=None, index=True)
    filename_orig: Optional[str] = None
    filename_enh: Optional[str] = None
    image_url_orig: Optional[str] = None
    image_url_enh: Optional[str] = None
    quality_score: Optional[float] = 0.0
    branch_used: Optional[str] = "original"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DbEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    db_id: str = Field(index=True, unique=True)
    db_image_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ReviewItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    probe_id: str = Field(index=True)
    risk_level: str = Field(default="MEDIUM", index=True)
    why_review: Optional[str] = None

    summary: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(SQLITE_JSON))
    probe_images: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(SQLITE_JSON))
    matches: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(SQLITE_JSON))

    branch: Optional[str] = "original"
    status: Optional[str] = "pending"
    reviewer_notes: Optional[str] = None
    final_decision: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

# -----------------------------------------------------------------------------
# App and DB
# -----------------------------------------------------------------------------
app = FastAPI(title="Face Authentication System (SQL)")

origins = ["http://localhost:3000","http://localhost:8080", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.getcwd(), "static")
PROBES_DIR = os.path.join(STATIC_DIR, "probes")
DB_DIR = os.path.join(STATIC_DIR, "db")
os.makedirs(PROBES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

sqlite_url = "sqlite:///./faceauth.db"
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# -----------------------------------------------------------------------------
# Core components (in-memory ops)
# -----------------------------------------------------------------------------
detector = FaceDetector()
quality = QualityAssessor()
enhancer = ImageEnhancer()
forensics = ForensicsDetector()
extractor = MultiEmbedder()
faiss_searcher = FAISSSearcher()
signals = SignalComputer()
scorer = RiskScorer()
morph = MorphDetector()
cohort_counts: Dict[int, int] = {}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _numpy_from_upload(contents: bytes):
    arr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _now_id(prefix="probe"):
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

def _probe_paths(probe_id: str):
    orig_fname = f"{probe_id}_orig.jpg"
    enh_fname = f"{probe_id}_enh.jpg"
    orig_path = os.path.join(PROBES_DIR, orig_fname)
    enh_path = os.path.join(PROBES_DIR, enh_fname)
    orig_url = f"/static/probes/{orig_fname}"
    enh_url = f"/static/probes/{enh_fname}"
    return orig_path, enh_path, orig_url, enh_url, orig_fname, enh_fname

def _db_paths(db_id: str):
    db_fname = f"{db_id}.jpg"
    db_path = os.path.join(DB_DIR, db_fname)
    db_url = f"/static/db/{db_fname}"
    return db_path, db_url

def _cohort_count_for(db_index: int):
    return int(cohort_counts.get(db_index, 0))

def _get_db_id_by_index(db_index: int):
    try:
        return faiss_searcher.get_db_id_by_index(int(db_index))
    except Exception:
        return None

def _build_matches(faiss_searcher, search_res, k=5):
    try:
        if not isinstance(search_res, dict):
            return []
        cand_indices = set()
        for m in ['arcface', 'adaface', 'elastic']:
            if m in search_res:
                try:
                    idxs = list(search_res[m][1])
                    valid_idxs = [int(i) for i in idxs if i != -1]
                    cand_indices.update(valid_idxs)
                except Exception:
                    pass

        rows = []
        for db_idx in cand_indices:
            try:
                sims = {}
                for m in ['arcface', 'adaface', 'elastic']:
                    if m in search_res:
                        sim_arr, idx_arr = search_res[m]
                        sim_val = 0.0
                        for s, i in zip(sim_arr, idx_arr):
                            if int(i) == db_idx:
                                try:
                                    val = float(s) * 100.0
                                    if not np.isfinite(val):
                                        val = 0.0
                                    sim_val = val
                                except Exception:
                                    sim_val = 0.0
                                break
                        sims[m] = sim_val
                    else:
                        sims[m] = 0.0

                avg_sim = (sims.get('arcface', 0.0) + sims.get('adaface', 0.0) + sims.get('elastic', 0.0)) / 3.0
                db_id = _get_db_id_by_index(db_idx)
                db_path, db_url = (None, None)
                if db_id:
                    db_path, db_url = _db_paths(str(db_id))

                row = {
                    "db_index": int(db_idx),
                    "db_id": str(db_id) if db_id is not None else None,
                    "avg_similarity": round(float(avg_sim), 2),
                    "db_image_url": db_url,
                    "arcface_similarity": round(float(sims.get('arcface', 0.0)), 2),
                    "adaface_similarity": round(float(sims.get('adaface', 0.0)), 2),
                    "elastic_similarity": round(float(sims.get('elastic', 0.0)), 2),
                }
                rows.append(row)
            except Exception:
                continue

        rows.sort(key=lambda r: r['avg_similarity'], reverse=True)
        return rows[:k]
    except Exception:
        return []

def to_native(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, set):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def _abs_url(request: Request, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    base = str(request.base_url).rstrip("/")
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if path.startswith("/"):
        return f"{base}{path}"
    return f"{base}/{path}"

def _best_match_block(search_res):
    try:
        if 'arcface' in search_res and len(search_res['arcface'][1]) > 0:
            top_idx = int(search_res['arcface'][1][0])
            top_score = float(search_res['arcface'][0][0]) * 100.0
            db_id = _get_db_id_by_index(top_idx)
            _, db_url = _db_paths(str(db_id)) if db_id else (None, None)
            return {
                "db_index": top_idx,
                "db_id": db_id,
                "similarity": round(top_score, 2),
                "db_image_url": db_url
            }
    except Exception:
        pass
    return None

# -----------------------------------------------------------------------------
# API: Verify single (persists Probe, optional DbEntry, optional ReviewItem)
# -----------------------------------------------------------------------------
@app.post("/api/verify")
async def verify_face(
    request: Request,
    file: UploadFile = File(...),
    applicant_id: Optional[str] = None,
    session: Session = Depends(get_session)
):
    try:
        contents = await file.read()
        img = _numpy_from_upload(contents)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        det = detector.detect_and_align(img)
        if det is None or det.get('confidence', 0.0) < 0.25:
            return JSONResponse({"status": "reject", "reason": "not confident about face in image", "decision": "reject"})

        face = det['aligned_face']

        q = quality.assess_quality(face, det['bbox'], det['landmarks'])
        if q['status'] == 'reject':
            return JSONResponse({"status": "reject", "reason": q['reason'], "quality_score": float(q['quality_score']), "decision": "reject"})
        quality_pass = (q['status'] in ['accept', 'warning'])

        enh_out = enhancer.enhance_if_needed(face, q['quality_score'])
        orig_face = enh_out['original']
        enh_face = enh_out['enhanced']

        probe_id = applicant_id or _now_id("probe")
        orig_path, enh_path, orig_url, enh_url, orig_fname, enh_fname = _probe_paths(probe_id)
        cv2.imwrite(orig_path, orig_face)
        if enh_face is not None:
            cv2.imwrite(enh_path, enh_face)

        # Store Probe
        probe_row = Probe(
            probe_id=probe_id,
            applicant_id=applicant_id,
            filename_orig=orig_fname,
            filename_enh=(enh_fname if enh_face is not None else None),
            image_url_orig=orig_url,
            image_url_enh=(enh_url if enh_face is not None else None),
            quality_score=float(q['quality_score']),
            branch_used="original"
        )
        session.add(probe_row)
        session.commit()
        session.refresh(probe_row)

        forns = forensics.score(orig_face)
        emb_o = extractor.extract_embeddings(orig_face)
        emb_e = extractor.extract_embeddings(enh_face) if enh_face is not None else None

        # Initial DB bootstrap
        if faiss_searcher.get_db_size() == 0:
            db_id = applicant_id or probe_id
            db_path, db_url = _db_paths(db_id)
            cv2.imwrite(db_path, orig_face)
            faiss_searcher.add_to_database(emb_o, db_id)
            session.add(DbEntry(db_id=db_id, db_image_url=db_url))
            session.commit()
            return JSONResponse(to_native({
                "status": "success",
                "decision": "unique",
                "action": "First entry added to database",
                "risk_score": 0.0,
                "risk_level": "LOW",
                "probe": {
                    "probe_id": probe_id,
                    "filename_orig": orig_fname,
                    "filename_enh": (enh_fname if enh_face is not None else None),
                    "image_url_orig": _abs_url(request, orig_url),
                    "image_url_enh": _abs_url(request, enh_url if enh_face is not None else None),
                },
                "best_match": None,
                "matches": [],
                "db": {"db_id": db_id, "db_image_url": _abs_url(request, db_url)}
            }))

        res_o = faiss_searcher.search(emb_o, k=10)
        res_e = faiss_searcher.search(emb_e, k=10) if enh_face is not None else None

        p_morph_o = morph.detect_prob_single(orig_face)
        p_morph_e = morph.detect_prob_single(enh_face) if enh_face is not None else p_morph_o
        morph_discrepancy = (abs(p_morph_o - p_morph_e) > 25.0)
        morph_prob = morph.detect_both(orig_face, enh_face)

        top_db_idx_o = int(res_o['arcface'][1][0]) if len(res_o['arcface'][1]) > 0 else -1
        cohort_o = _cohort_count_for(top_db_idx_o) if top_db_idx_o >= 0 else 0
        sig_o = signals.compute_all_signals(res_o, morph_prob, cohort_o, forns)

        sig_e = None
        if res_e is not None:
            top_db_idx_e = int(res_e['arcface'][1][0]) if len(res_e['arcface'][1]) > 0 else -1
            cohort_e = _cohort_count_for(top_db_idx_e) if top_db_idx_e >= 0 else 0
            sig_e = signals.compute_all_signals(res_e, morph_prob, cohort_e, forns)

        chosen = sig_o
        chosen_branch = "original"
        chosen_res = res_o
        if sig_e is not None and sig_e['uncertainty'] < sig_o['uncertainty']:
            chosen = sig_e
            chosen_branch = "enhanced"
            chosen_res = res_e

        risk, _, _ = scorer.compute_risk_score(chosen)
        decision = scorer.make_decision(risk, chosen, quality_pass=quality_pass, morph_discrepancy=morph_discrepancy)

        db_id = applicant_id or probe_id
        if decision['status'] == 'unique':
            db_path, db_url = _db_paths(db_id)
            cv2.imwrite(db_path, orig_face)
            faiss_searcher.add_to_database(emb_o, db_id)
            exists = session.exec(select(DbEntry).where(DbEntry.db_id == db_id)).first()
            if not exists:
                session.add(DbEntry(db_id=db_id, db_image_url=db_url))
                session.commit()

        if len(chosen_res['arcface'][0]) > 0 and chosen_res['arcface'][0][0] * 100.0 > 70.0:
            top_idx = int(chosen_res['arcface'][1][0])
            cohort_counts[top_idx] = cohort_counts.get(top_idx, 0) + 1

        top_matches = _build_matches(faiss_searcher, chosen_res, k=5)
        # attach absolute URLs for match images
        for m in top_matches:
            m["db_image_url"] = _abs_url(request, m.get("db_image_url"))

        if decision['status'] == 'review':
            morph_val = chosen.get('morph')
            morph_prob_num = float(morph_val[0] if isinstance(morph_val, (tuple, list)) and morph_val else morph_val or 0.0)
            review_row = ReviewItem(
                probe_id=probe_id,
                risk_level="MEDIUM",
                why_review=decision.get('reason', ''),
                summary={
                    'SIM': round(float(chosen.get('sim', 0.0)), 2),
                    'AGREE': round(float(chosen.get('agree', 0.0)), 2),
                    'MARGIN': round(float(chosen.get('margin', 0.0)), 2),
                    'MORPH': round(float(morph_prob_num), 2),
                    'FORNS': round(float(chosen.get('forns', 0.0)), 2),
                    'COHORT': round(float(chosen.get('cohort', 0.0)), 2),
                    'UNC': round(float(chosen.get('uncertainty', 0.0)), 2),
                },
                probe_images={
                    'aligned': f"/static/probes/{orig_fname}",
                    'enhanced': (f"/static/probes/{enh_fname}" if enh_face is not None else None)
                },
                matches=to_native(top_matches),
                branch=chosen_branch,
                status="pending"
            )
            session.add(review_row)
            session.commit()

        morph_val_resp = chosen.get('morph')
        morph_prob_resp = float(morph_val_resp[0] if isinstance(morph_val_resp, (tuple, list)) and morph_val_resp else morph_val_resp or 0.0)

        best_match = _best_match_block(chosen_res)
        if best_match:
            best_match["db_image_url"] = _abs_url(request, best_match.get("db_image_url"))

        # Near the end of your endpoint, replace the resp block:
        resp = {
            "status": "success",
            "branch": chosen_branch,
            "applicant_id": applicant_id,
            "probe": {
                "probe_id": probe_id,
                "filename_orig": orig_fname,
                "filename_enh": (enh_fname if enh_face is not None else None),
                "image_url_orig": _abs_url(request, f"/static/probes/{orig_fname}"),
                "image_url_enh": _abs_url(request, f"/static/probes/{enh_fname}") if enh_face is not None else None,
            },
            "decision": decision['status'],
            "action": decision.get('action'),
            "reason": decision.get('reason'),
            "risk_score": abs(float(risk)),  # ✅ abs() added
            "risk_level": decision['risk_level'],
            "signals": {
                "similarity": abs(float(chosen['sim'])),      # ✅ abs() added
                "agreement": abs(float(chosen['agree'])),     # ✅ abs() added
                "margin": abs(float(chosen['margin'])),       # ✅ abs() added
                "morph": abs(float(morph_prob_resp)),         # ✅ abs() added
                "forns": abs(float(chosen['forns'])),         # ✅ abs() added
                "cohort": abs(float(chosen['cohort'])),       # ✅ abs() added
                "uncertainty": abs(float(chosen['uncertainty'])) # ✅ abs() added
            },
            "best_match": best_match,
            "matches": top_matches,
            "quality_score": abs(float(q['quality_score']))  # ✅ abs() added
        }

        return JSONResponse(to_native(resp))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# API: Stats
# -----------------------------------------------------------------------------
@app.get("/api/stats")
async def get_stats(session: Session = Depends(get_session)):
    rq_count = session.exec(select(ReviewItem)).all()
    return {"database_size": faiss_searcher.get_db_size(), "review_queue": len(rq_count), "status": "operational"}

# -----------------------------------------------------------------------------
# API: Review queue (SQL)
# -----------------------------------------------------------------------------
@app.get("/api/review/queue")
async def get_review_queue(
    request: Request,
    risk_level: Optional[str] = Query(None),
    limit: int = Query(50),
    session: Session = Depends(get_session)
):
    stmt = select(ReviewItem).order_by(ReviewItem.created_at.desc()).limit(limit)
    if risk_level:
        stmt = select(ReviewItem).where(ReviewItem.risk_level == risk_level.upper()).order_by(ReviewItem.created_at.desc()).limit(limit)
    items = session.exec(stmt).all()
    payload = []
    for item in items:
        d = item.dict()
        # convert relative probe_images URLs to absolute for frontend display
        if d.get("probe_images"):
            for k, v in list(d["probe_images"].items()):
                d["probe_images"][k] = _abs_url(request, v) if v else None
        # convert matches db_image_url to absolute
        if d.get("matches"):
            for m in d["matches"]:
                if isinstance(m, dict) and "db_image_url" in m:
                    m["db_image_url"] = _abs_url(request, m["db_image_url"])
        payload.append(to_native(d))
    return JSONResponse(content=jsonable_encoder(payload))
APP_ROOT = os.getcwd()  # or better: Path(__file__).resolve().parent.parent

def url_to_fs_path(url_path: str) -> str:
    """
    Convert a /static/... URL path into an absolute filesystem path.
    Handles leading slash and normalizes separators.
    """
    # url_path like "/static/probes/file.jpg"
    rel = url_path.lstrip("/")  # "static/probes/file.jpg"
    abs_path = os.path.join(APP_ROOT, rel)
    return os.path.normpath(abs_path)
@app.post("/api/review/decision")
async def review_decision(
    request: Request,
    probe_id: str = Body(...),
    decision: str = Body(..., embed=True),
    reviewer_notes: Optional[str] = Body(None),
    session: Session = Depends(get_session)
):
    row = session.exec(
        select(ReviewItem).where(ReviewItem.probe_id == probe_id).order_by(ReviewItem.created_at.desc())
    ).first()
    if not row:
        return {"status": "not_found", "message": "Probe not in queue"}
    print(row)
    print(decision, reviewer_notes)
    # Update review item fields
    row.reviewer_notes = reviewer_notes or ""
    final = decision.lower()
    row.final_decision = final
    row.status = 'reviewed' if final in ('approve', 'reject', 'escalate') else 'pending'

    # Act on approve / reject
    probe = session.exec(select(Probe).where(Probe.probe_id == probe_id)).first()

    if final == 'approve':
        aligned_rel = None
        if row and isinstance(row.probe_images, dict):
            aligned_rel = row.probe_images.get('aligned')  # "/static/probes/....jpg"

        aligned_path = url_to_fs_path(aligned_rel) if aligned_rel else None

        # Fallback to Probe only if review payload missing
        if (not aligned_path) or (not os.path.exists(aligned_path)):
            if probe and probe.image_url_orig:
                aligned_path = url_to_fs_path(probe.image_url_orig)

        if (not aligned_path) or (not os.path.exists(aligned_path)):
            return {"status": "error", "message": "Aligned probe image path not found for approval"}

        # Normalize for OpenCV
        aligned_path = os.path.normpath(aligned_path)
        print("Aligned path for approval:", aligned_path)

        img = cv2.imread(aligned_path)
        if img is None:
            return {"status": "error", "message": "Failed to read aligned image from disk"}

        db_id = (probe.applicant_id if (probe and probe.applicant_id) else (probe.probe_id if probe else row.probe_id))
        db_path, db_url = _db_paths(db_id)

        # Ensure DB image exists
        if not os.path.exists(db_path):
            cv2.imwrite(db_path, img)

        emb = extractor.extract_embeddings(img)
        if emb is None:
            return {"status": "error", "message": "Failed to compute embedding for approve"}
        faiss_searcher.add_to_database(emb, db_id)

        exists = session.exec(select(DbEntry).where(DbEntry.db_id == db_id)).first()
        if not exists:
            session.add(DbEntry(db_id=db_id, db_image_url=db_url))

        session.delete(row)
        session.commit()
        return {"status": "ok", "message": "Approved, added to FAISS and DB, removed from review queue"}

    if final == 'reject':
        # Remove probe files and row, and remove from review queue
        if probe:
            if probe.image_url_orig:
                p = os.path.join(STATIC_DIR, probe.image_url_orig.lstrip("/"))
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            if probe.image_url_enh:
                p = os.path.join(STATIC_DIR, probe.image_url_enh.lstrip("/"))
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            try:
                session.delete(probe)
            except Exception:
                pass
        session.delete(row)
        session.commit()
        return {
            "status": "ok",
            "message": "Rejected, probe removed and item deleted from review queue",
        }

    session.add(row)
    session.commit()
    return {"status": "ok", "message": "Decision recorded"}

# -----------------------------------------------------------------------------
# API: Batch deduplicate (persists Probe, optional DbEntry, optional ReviewItem)
# -----------------------------------------------------------------------------
@app.post("/api/batch/deduplicate")
async def batch_deduplicate(
    request: Request,
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session)
):
    results = {}
    processed_count = 0

    for file in files:
        fn = file.filename
        try:
            contents = await file.read()
            img = _numpy_from_upload(contents)
            if img is None:
                results[fn] = {"status": "failed", "reason": "Could not read image", "decision": "reject"}
                processed_count += 1
                continue

            det = detector.detect_and_align(img)
            if det is None or det.get('confidence', 0.0) < 0.25:
                results[fn] = {"status": "reject", "reason": "No confident face detected", "decision": "reject"}
                processed_count += 1
                continue

            face = det['aligned_face']
            q = quality.assess_quality(face, det['bbox'], det['landmarks'])
            if q['status'] == 'reject':
                results[fn] = {"status": "reject", "reason": q['reason'], "quality_score": float(q['quality_score']), "decision": "reject"}
                processed_count += 1
                continue

            quality_pass = (q['status'] in ['accept', 'warning'])
            enh_out = enhancer.enhance_if_needed(face, q['quality_score'])
            orig_face = enh_out['original']
            enh_face = enh_out['enhanced']

            base_id = os.path.splitext(os.path.basename(fn))[0]
            probe_id = f"{base_id}_{processed_count}"
            orig_path, enh_path, orig_url, enh_url, orig_fname, enh_fname = _probe_paths(probe_id)
            cv2.imwrite(orig_path, orig_face)
            if enh_face is not None:
                cv2.imwrite(enh_path, enh_face)

            probe_row = Probe(
                probe_id=probe_id,
                applicant_id=None,
                filename_orig=orig_fname,
                filename_enh=(enh_fname if enh_face is not None else None),
                image_url_orig=orig_url,
                image_url_enh=(enh_url if enh_face is not None else None),
                quality_score=float(q['quality_score']),
                branch_used="original"
            )
            session.add(probe_row)
            session.commit()

            forns = forensics.score(orig_face)
            emb_o = extractor.extract_embeddings(orig_face)
            emb_e = extractor.extract_embeddings(enh_face) if enh_face is not None else None

            if faiss_searcher.get_db_size() == 0:
                db_id = f"{base_id}_id_{processed_count}"
                db_path, db_url = _db_paths(db_id)
                cv2.imwrite(db_path, orig_face)
                faiss_searcher.add_to_database(emb_o, db_id)
                session.add(DbEntry(db_id=db_id, db_image_url=db_url))
                session.commit()
                results[fn] = {
                    "status": "success",
                    "decision": "unique",
                    "action": "First entry added to database",
                    "risk_score": 0.0,
                    "risk_level": "LOW",
                    "probe": {
                        "probe_id": probe_id,
                        "filename_orig": orig_fname,
                        "filename_enh": (enh_fname if enh_face is not None else None),
                        "image_url_orig": _abs_url(request, orig_url),
                        "image_url_enh": _abs_url(request, enh_url if enh_face is not None else None),
                    },
                    "best_match": None,
                    "matches": [],
                    "db": {"db_id": db_id, "db_image_url": _abs_url(request, db_url)}
                }
                processed_count += 1
                continue

            res_o = faiss_searcher.search(emb_o, k=10)
            res_e = faiss_searcher.search(emb_e, k=10) if enh_face is not None else None

            p_morph_o = morph.detect_prob_single(orig_face)
            p_morph_e = morph.detect_prob_single(enh_face) if enh_face is not None else p_morph_o
            morph_discrepancy = (abs(p_morph_o - p_morph_e) > 25.0)
            morph_prob = morph.detect_both(orig_face, enh_face)

            top_db_idx_o = int(res_o['arcface'][1][0]) if len(res_o['arcface'][1]) > 0 else -1
            cohort_o = _cohort_count_for(top_db_idx_o) if top_db_idx_o >= 0 else 0
            sig_o = signals.compute_all_signals(res_o, morph_prob, cohort_o, forns)

            sig_e = None
            if res_e is not None:
                top_db_idx_e = int(res_e['arcface'][1][0]) if len(res_e['arcface'][1]) > 0 else -1
                cohort_e = _cohort_count_for(top_db_idx_e) if top_db_idx_e >= 0 else 0
                sig_e = signals.compute_all_signals(res_e, morph_prob, cohort_e, forns)

            chosen = sig_o
            chosen_branch = "original"
            chosen_res = res_o
            if sig_e is not None and sig_e['uncertainty'] < sig_o['uncertainty']:
                chosen = sig_e
                chosen_branch = "enhanced"
                chosen_res = res_e

            risk, components, audit_notes = scorer.compute_risk_score(chosen)
            decision = scorer.make_decision(risk, chosen, quality_pass=quality_pass, morph_discrepancy=morph_discrepancy)

            if decision['status'] == 'unique':
                db_id = f"{base_id}_id_{processed_count}"
                db_path, db_url = _db_paths(db_id)
                cv2.imwrite(db_path, orig_face)
                faiss_searcher.add_to_database(emb_o, db_id)
                exists = session.exec(select(DbEntry).where(DbEntry.db_id == db_id)).first()
                if not exists:
                    session.add(DbEntry(db_id=db_id, db_image_url=db_url))
                    session.commit()
                    if decision['status'] == 'review':
                        morph_val = chosen.get('morph')
                        morph_prob_num = float(morph_val[0] if isinstance(morph_val, (tuple, list)) and morph_val else morph_val or 0.0)
                        review_row = ReviewItem(
                            probe_id=probe_id,
                            risk_level="MEDIUM",
                            why_review=decision.get('reason', ''),
                            summary={
                                'SIM': round(float(chosen.get('sim', 0.0)), 2),
                                'AGREE': round(float(chosen.get('agree', 0.0)), 2),
                                'MARGIN': round(float(chosen.get('margin', 0.0)), 2),
                                'MORPH': round(float(morph_prob_num), 2),
                                'FORNS': round(float(chosen.get('forns', 0.0)), 2),
                                'COHORT': round(float(chosen.get('cohort', 0.0)), 2),
                                'UNC': round(float(chosen.get('uncertainty', 0.0)), 2),
                            },
                            probe_images={
                                'aligned': f"/static/probes/{orig_fname}",
                                'enhanced': (f"/static/probes/{enh_fname}" if enh_face is not None else None)
                            },
                            matches=to_native(top_matches),
                            branch=chosen_branch,
                            status="pending"
                        )
                        session.add(review_row)
                        session.commit()

            if len(chosen_res['arcface'][0]) > 0 and chosen_res['arcface'][0][0] * 100.0 > 70.0:
                top_idx = int(chosen_res['arcface'][1][0])
                cohort_counts[top_idx] = cohort_counts.get(top_idx, 0) + 1

            top_matches = _build_matches(faiss_searcher, chosen_res, k=5)
            for m in top_matches:
                m["db_image_url"] = _abs_url(request, m.get("db_image_url"))

            morph_val_resp = chosen.get('morph')
            morph_prob_resp = float(morph_val_resp[0] if isinstance(morph_val_resp, (tuple, list)) and morph_val_resp else morph_val_resp or 0.0)

            best_match = _best_match_block(chosen_res)
            if best_match:
                best_match["db_image_url"] = _abs_url(request, best_match.get("db_image_url"))

            results[fn] = {
                "status": "success",
                "branch": chosen_branch,
                "decision": decision['status'],
                "action": decision.get('action'),
                "reason": decision.get('reason'),
                "risk_score": float(risk),
                "risk_level": decision['risk_level'],
                "confidence": decision.get('confidence', 0.0),
                "probe": {
                    "probe_id": probe_id,
                    "filename_orig": orig_fname,
                    "filename_enh": (enh_fname if enh_face is not None else None),
                    "image_url_orig": _abs_url(request, orig_url),
                    "image_url_enh": _abs_url(request, enh_url if enh_face is not None else None),
                },
                "signals": {
                    "similarity": float(chosen['sim']),
                    "agreement": float(chosen['agree']),
                    "margin": float(chosen['margin']),
                    "morph": float(morph_prob_resp),
                    "forns": float(chosen['forns']),
                    "cohort": float(chosen['cohort']),
                    "uncertainty": float(chosen['uncertainty'])
                },
                "best_match": best_match,
                "matches": top_matches,
                "quality_score": float(q['quality_score']),
                "audit_notes": audit_notes
            }

            processed_count += 1
        except Exception as e:
            results[fn] = {"status": "error", "reason": str(e), "decision": "error"}
            processed_count += 1

    return JSONResponse(to_native({"status": "ok", "total_processed": processed_count, "results": results}))

# -----------------------------------------------------------------------------
# API: Get one probe record with image URLs (helper for UI)
# -----------------------------------------------------------------------------
@app.get("/api/probe/{probe_id}")
async def get_probe(request: Request, probe_id: str, session: Session = Depends(get_session)):
    probe = session.exec(select(Probe).where(Probe.probe_id == probe_id)).first()
    if not probe:
        raise HTTPException(status_code=404, detail="Probe not found")
    return {
        "probe_id": probe.probe_id,
        "image_url_orig": _abs_url(request, probe.image_url_orig),
        "image_url_enh": _abs_url(request, probe.image_url_enh) if probe.image_url_enh else None,
        "quality_score": probe.quality_score,
        "branch_used": probe.branch_used,
        "created_at": probe.created_at.isoformat()
    }

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
