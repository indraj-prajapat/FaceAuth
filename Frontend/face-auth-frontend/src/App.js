import React, { useEffect, useMemo, useState, useCallback } from "react";

const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

// ---------------- API HOOK ----------------
function useApi() {
  const verify = async (file, applicantId) => {
    const fd = new FormData();
    fd.append("file", file);
    if (applicantId) fd.append("applicant_id", applicantId);
    const res = await fetch(`${API_BASE}/api/verify`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Verify failed: ${res.status}`);
    return res.json();
  };

  const batchDeduplicate = async (files) => {
    const fd = new FormData();
    // key must match FastAPI parameter name: files: List[UploadFile] = File(...)
    // append each file under the same field name "files"
    for (const f of files) fd.append("files", f);
    const res = await fetch(`${API_BASE}/api/batch/deduplicate`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) throw new Error(`Batch failed: ${res.status}`);
    return res.json();
  };

  const getQueue = async (risk) => {
    const url = new URL(`${API_BASE}/api/review/queue`);
    if (risk) url.searchParams.set("risk_level", risk);
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(`Queue fetch failed: ${res.status}`);
    return res.json();
  };

  const postDecision = async (probe_id, decision, reviewer_notes = "") => {
    const res = await fetch(`${API_BASE}/api/review/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ probe_id, decision, reviewer_notes }),
    });
    if (!res.ok) throw new Error(`Decision failed: ${res.status}`);
    return res.json();
  };

  const getStats = async () => {
    const res = await fetch(`${API_BASE}/api/stats`);
    if (!res.ok) throw new Error(`Stats failed: ${res.status}`);
    return res.json();
  };

  return { verify, batchDeduplicate, getQueue, postDecision, getStats };
}

// ---------------- UI ATOMS ----------------
function StatBadge({ label, value }) {
  return (
    <div style={{ padding: "10px 14px", borderRadius: 10, background: "#0b1220", border: "1px solid #1f2937", color: "white", minWidth: 110 }}>
      <div style={{ fontSize: 12, color: "#9ca3af" }}>{label}</div>
      <div style={{ fontWeight: 700, fontSize: 16 }}>{value}</div>
    </div>
  );
}

function Card({ children, header, footer, style }) {
  return (
    <div style={{ background: "white", border: "1px solid #e5e7eb", borderRadius: 14, boxShadow: "0 1px 2px rgba(0,0,0,0.04)", ...style }}>
      {header && <div style={{ padding: 14, borderBottom: "1px solid #f1f5f9", fontWeight: 600 }}>{header}</div>}
      <div style={{ padding: 14 }}>{children}</div>
      {footer && <div style={{ padding: 12, borderTop: "1px solid #f1f5f9" }}>{footer}</div>}
    </div>
  );
}

function Pill({ children, color = "#eef2ff", text = "#3730a3" }) {
  return (
    <span style={{ padding: "4px 8px", borderRadius: 999, background: color, color: text, fontSize: 12, fontWeight: 600 }}>
      {children}
    </span>
  );
}

// ---------------- DOMAIN CARDS ----------------
function MatchCard({ match }) {
  return (
    <div style={{ width: 160, border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden", background: "white" }}>
      {match?.db_image_url && (
        <img src={match.db_image_url} alt="db" width={160} height={120} style={{ objectFit: "cover", display: "block" }} />
      )}
      <div style={{ padding: 8, fontSize: 12 }}>
        <div style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          db_id: {match?.db_id || "NA"}
        </div>
        <div>avg: {match?.avg_similarity ?? 0}%</div>
        <div>a:{match?.arcface_similarity ?? 0} e:{match?.elastic_similarity ?? 0}</div>
      </div>
    </div>
  );
}

function ProbePanel({ probe }) {
  return (
    <div>
      <div style={{ display: "flex", gap: 12 }}>
        {probe?.image_url_orig && (
          <img src={probe.image_url_orig} alt="probe" width={240} style={{ objectFit: "cover", borderRadius: 12, border: "1px solid #e5e7eb" }} />
        )}
        {probe?.image_url_enh && (
          <img src={probe.image_url_enh} alt="probe-enh" width={240} style={{ objectFit: "cover", borderRadius: 12, border: "1px solid #e5e7eb" }} />
        )}
      </div>
      <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
        {probe?.probe_id && <Pill>{probe.probe_id}</Pill>}
      </div>
    </div>
  );
}

function BestMatchPanel({ best }) {
  return (
    <div>
      {best?.db_image_url ? (
        <div>
          <img src={best.db_image_url} alt="best-match" width={240} style={{ objectFit: "cover", borderRadius: 12, border: "1px solid #e5e7eb" }} />
          <div style={{ marginTop: 8, fontSize: 13 }}>
            db_id: <Pill>{best.db_id || "NA"}</Pill> similarity: <Pill>{best.similarity ?? 0}%</Pill>
          </div>
        </div>
      ) : (
        <div style={{ color: "#6b7280" }}>No best match</div>
      )}
    </div>
  );
}

function MatchesPanel({ matches }) {
  const list = Array.isArray(matches) ? matches : [];
  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        {list.map((m, i) => (
          <MatchCard key={i} match={m} />
        ))}
      </div>
    </div>
  );
}

// ---------------- SECTIONS ----------------
function VerifySection() {
  const { verify } = useApi();
  const [file, setFile] = useState(null);
  const [applicant, setApplicant] = useState("");
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const onVerify = async () => {
    if (!file) return;
    setLoading(true);
    setErr("");
    setResp(null);
    try {
      const data = await verify(file, applicant || undefined);
      setResp(data);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card header="Verify">
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12 }}>
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <input placeholder="applicant_id (optional)" value={applicant} onChange={(e) => setApplicant(e.target.value)} style={{ padding: 8, border: "1px solid #ddd", borderRadius: 8 }} />
        <button disabled={!file || loading} onClick={onVerify} style={{ padding: "8px 12px", borderRadius: 8, background: "#0ea5e9", color: "white", border: 0 }}>
          {loading ? "Processing..." : "Submit"}
        </button>
      </div>
      {err && <div style={{ color: "crimson", marginBottom: 8 }}>{err}</div>}
      {resp && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
          <div>
            <h4 style={{ margin: "8px 0" }}>Probe</h4>
            <ProbePanel probe={resp.probe} />
            <div style={{ marginTop: 10, fontSize: 13 }}>
              <div>Decision: <Pill color="#ecfeff" text="#0e7490">{resp.decision}</Pill></div>
              <div>Risk: <Pill color="#fef3c7" text="#92400e">{resp.risk_level}</Pill> ({(resp.risk_score ?? 0).toFixed(2)})</div>
              <div>Similarity: {(resp.signals?.similarity ?? 0).toFixed(2)}</div>
              <div>Agreement: {(resp.signals?.agreement ?? 0).toFixed(2)}</div>
              <div>Margin: {(resp.signals?.margin ?? 0).toFixed(2)}</div>
              <div>Morph: {(resp.signals?.morph ?? 0).toFixed(2)}</div>
              <div>Forensics: {(resp.signals?.forns ?? 0).toFixed(2)}</div>
              <div>Cohort: {(resp.signals?.cohort ?? 0).toFixed(2)}</div>
              <div>Uncertainty: {(resp.signals?.uncertainty ?? 0).toFixed(2)}</div>
              <div>Quality: {(resp.quality_score ?? 0).toFixed(2)}</div>
            </div>
          </div>
          <div>
            <h4 style={{ margin: "8px 0" }}>Best Match</h4>
            <BestMatchPanel best={resp.best_match} />
          </div>
          <div>
            <h4 style={{ margin: "8px 0" }}>Top Matches</h4>
            <MatchesPanel matches={resp.matches} />
          </div>
        </div>
      )}
    </Card>
  );
}

function DropZone({ onFiles }) {
  const [drag, setDrag] = useState(false);
  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDrag(false);
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length) onFiles(files);
  }, [onFiles]);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      onClick={() => document.getElementById("batch-input")?.click()}
      style={{
        border: "2px dashed #93c5fd",
        background: drag ? "#eff6ff" : "#f8fafc",
        borderRadius: 12,
        padding: 20,
        textAlign: "center",
        color: "#1f2937",
        cursor: "pointer"
      }}
    >
      <div style={{ fontWeight: 600 }}>Drag & drop images here, or click to browse</div>
      <div style={{ fontSize: 12, color: "#6b7280" }}>Multiple files supported</div>
      <input id="batch-input" type="file" accept="image/*" multiple hidden onChange={(e) => onFiles(Array.from(e.target.files || []))} />
    </div>
  );
}

function BatchSection() {
  const { batchDeduplicate } = useApi();
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [busy, setBusy] = useState(false);
  const [resp, setResp] = useState(null);
  const [err, setErr] = useState("");

  const onFiles = (picked) => {
    setFiles(picked);
    // local previews
    const readers = picked.map((f) => new Promise((resolve) => {
      const r = new FileReader();
      r.onload = () => resolve({ name: f.name, src: r.result });
      r.readAsDataURL(f);
    }));
    Promise.all(readers).then(setPreviews);
  };

  const onSubmit = async () => {
    if (!files.length) return;
    setBusy(true);
    setErr("");
    setResp(null);
    try {
      const data = await batchDeduplicate(files);
      setResp(data);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  // Flatten results for grid
  const resultCards = useMemo(() => {
    if (!resp?.results) return [];
    return Object.entries(resp.results).map(([filename, r]) => ({ filename, ...r }));
  }, [resp]);

  return (
    <Card header="Batch Deduplicate">
      <DropZone onFiles={onFiles} />
      {previews.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div style={{ marginBottom: 6, color: "#6b7280", fontSize: 13 }}>
            Selected: {previews.length} file(s)
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {previews.map((p, i) => (
              <div key={i} style={{ width: 120, textAlign: "center" }}>
                <img src={p.src} alt={p.name} width={120} height={90} style={{ objectFit: "cover", borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <div style={{ fontSize: 11, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.name}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      <div style={{ marginTop: 12 }}>
        <button onClick={onSubmit} disabled={!files.length || busy} style={{ padding: "8px 12px", borderRadius: 8, background: "#16a34a", color: "white", border: 0 }}>
          {busy ? "Processing..." : "Run Deduplicate"}
        </button>
      </div>

      {err && <div style={{ color: "crimson", marginTop: 10 }}>{err}</div>}

      {resultCards.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h4 style={{ margin: "6px 0" }}>Results</h4>
          <div style={{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fill,minmax(280px,1fr))" }}>
            {resultCards.map((r, idx) => (
              <Card key={idx} header={r.filename} style={{ margin: 0 }}>
                <div style={{ display: "grid", gap: 10 }}>
                  <div style={{ display: "flex", gap: 10 }}>
                    {r?.probe?.image_url_orig && (
                      <img src={r.probe.image_url_orig} alt="probe" width={120} height={100} style={{ objectFit: "cover", borderRadius: 10, border: "1px solid #e5e7eb" }} />
                    )}
                    {r?.probe?.image_url_enh && (
                      <img src={r.probe.image_url_enh} alt="enh" width={120} height={100} style={{ objectFit: "cover", borderRadius: 10, border: "1px solid #e5e7eb" }} />
                    )}
                  </div>
                  <div style={{ fontSize: 13 }}>
                    <div>Decision: <Pill color="#ecfeff" text="#0e7490">{r.decision}</Pill></div>
                    <div>Risk: <Pill color="#fef3c7" text="#92400e">{r.risk_level}</Pill> ({(r.risk_score ?? 0).toFixed(2)})</div>
                    <div>Quality: {(r.quality_score ?? 0).toFixed(2)}</div>
                  </div>
                  {Array.isArray(r.matches) && r.matches.length > 0 && (
                    <div>
                      <div style={{ marginBottom: 6, fontWeight: 600 }}>Matches</div>
                      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                        {r.matches.map((m, i) => <MatchCard key={i} match={m} />)}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

function ReviewCard({ item, onDecision }) {
  const hasMatches = Array.isArray(item?.matches) && item.matches.length > 0;
  return (
    <Card>
      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 280px", gap: 16 }}>
        <div>
          <h4 style={{ margin: "6px 0" }}>Probe</h4>
          {item?.probe_images?.aligned && (
            <img src={item.probe_images.aligned} alt="probe-aligned" width={240} style={{ objectFit: "cover", borderRadius: 12, border: "1px solid #e5e7eb" }} />
          )}
          {item?.probe_images?.enhanced && (
            <img src={item.probe_images.enhanced} alt="probe-enhanced" width={240} style={{ objectFit: "cover", borderRadius: 12, border: "1px solid #e5e7eb", marginTop: 8 }} />
          )}
          <div style={{ fontSize: 12, color: "#6b7280", marginTop: 6 }}>
            probe_id: <Pill>{item.probe_id}</Pill>
          </div>
        </div>

        <div>
          <h4 style={{ margin: "6px 0" }}>Matches</h4>
          {hasMatches ? (
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              {item.matches.map((m, idx) => <MatchCard key={idx} match={m} />)}
            </div>
          ) : (
            <div style={{ color: "#9ca3af" }}>No candidates</div>
          )}
        </div>

        <div>
          <h4 style={{ margin: "6px 0" }}>Summary</h4>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, fontSize: 12 }}>
            <div>risk: {item.risk_level}</div>
            <div>status: {item.status}</div>
            <div>why: {item.why_review || "-"}</div>
            {item.summary && (
              <>
                <div>SIM: {item.summary.SIM}</div>
                <div>AGREE: {item.summary.AGREE}</div>
                <div>MARGIN: {item.summary.MARGIN}</div>
                <div>MORPH: {item.summary.MORPH}</div>
                <div>FORNS: {item.summary.FORNS}</div>
                <div>COHORT: {item.summary.COHORT}</div>
                <div>UNC: {item.summary.UNC}</div>
              </>
            )}
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            <button onClick={() => onDecision(item.probe_id, "approve")} style={{ background: "#22c55e", color: "white", border: 0, padding: "8px 12px", borderRadius: 8 }}>Approve</button>
            <button onClick={() => onDecision(item.probe_id, "reject")} style={{ background: "#ef4444", color: "white", border: 0, padding: "8px 12px", borderRadius: 8 }}>Reject</button>
          </div>
        </div>
      </div>
    </Card>
  );
}

function ReviewSection() {
  const { getQueue, postDecision, getStats } = useApi();
  const [queue, setQueue] = useState([]);
  const [stats, setStats] = useState({ database_size: 0, review_queue: 0, status: "operational" });
  const [riskFilter, setRiskFilter] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const refresh = async () => {
    setErr("");
    try {
      const [q, s] = await Promise.all([getQueue(riskFilter || undefined), getStats()]);
      setQueue(q);
      setStats(s);
    } catch (e) {
      setErr(String(e));
    }
  };

  useEffect(() => { refresh(); /* eslint-disable-next-line */ }, [riskFilter]);

  const onDecision = async (probe_id, decision) => {
    setBusy(true);
    setErr("");
    try {
      await postDecision(probe_id, decision, "");
      await refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card header={
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span>Manual Review</span>
        <div style={{ display: "flex", gap: 8 }}>
          <StatBadge label="DB size" value={stats.database_size} />
          <StatBadge label="Review items" value={stats.review_queue} />
          <StatBadge label="Status" value={stats.status} />
        </div>
      </div>
    }>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)} style={{ padding: 8, borderRadius: 8, border: "1px solid #ddd" }}>
          <option value="">All risks</option>
          <option value="LOW">LOW</option>
          <option value="MEDIUM">MEDIUM</option>
          <option value="HIGH">HIGH</option>
        </select>
        <button onClick={refresh} disabled={busy}>Refresh</button>
      </div>
      {err && <div style={{ color: "crimson", marginBottom: 8 }}>{err}</div>}
      {queue.length === 0 ? (
        <div style={{ color: "#6b7280" }}>No items in review.</div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          {queue.map((item) => <ReviewCard key={item.id} item={item} onDecision={onDecision} />)}
        </div>
      )}
    </Card>
  );
}

// ---------------- APP ----------------
export default function App() {
  return (
    <div style={{ padding: 18, fontFamily: "Inter, system-ui, Segoe UI, Arial", background: "#f8fafc", minHeight: "100vh" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>Face Authentication Console</h2>
        <div style={{ display: "flex", gap: 8 }}>
          <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" style={{ color: "#0ea5e9", textDecoration: "none" }}>API Docs</a>
        </div>
      </div>
      <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr" }}>
        <VerifySection />
        <BatchSection />
        <ReviewSection />
      </div>
      <div style={{ marginTop: 18, fontSize: 12, color: "#6b7280" }}>
        API base: {API_BASE}
      </div>
    </div>
  );
}
