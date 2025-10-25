-- Face Authentication Database Schema

-- Applications table
CREATE TABLE IF NOT EXISTS applications (
    id SERIAL PRIMARY KEY,
    applicant_id VARCHAR(50) UNIQUE NOT NULL,
    photo_path VARCHAR(255),
    embedding_arcface FLOAT[512],
    embedding_adaface FLOAT[512],
    embedding_elastic FLOAT[512],
    quality_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Match history (for cohortness tracking)
CREATE TABLE IF NOT EXISTS match_history (
    id SERIAL PRIMARY KEY,
    probe_id VARCHAR(50) NOT NULL,
    db_id INTEGER REFERENCES applications(id),
    similarity FLOAT NOT NULL,
    risk_score FLOAT NOT NULL,
    decision VARCHAR(20) NOT NULL,
    signals JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_match_db_timestamp ON match_history(db_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_match_probe ON match_history(probe_id);

-- Review queue
CREATE TABLE IF NOT EXISTS review_queue (
    id SERIAL PRIMARY KEY,
    probe_id VARCHAR(50) NOT NULL,
    matched_db_id INTEGER REFERENCES applications(id),
    risk_level VARCHAR(20) NOT NULL,
    signals JSONB,
    status VARCHAR(20) DEFAULT 'pending', -- pending/reviewed/escalated
    reviewer_id VARCHAR(50),
    reviewer_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status, risk_level);

-- Audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    action VARCHAR(50) NOT NULL,
    applicant_id VARCHAR(50),
    user_id VARCHAR(50),
    details JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
