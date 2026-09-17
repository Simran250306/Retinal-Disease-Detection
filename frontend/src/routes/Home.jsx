import { useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [apiBase, setApiBase] = useState(API_BASE);

  const detected = useMemo(() => results?.detected || [], [results]);

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    setResults(null);
    setError(null);
    if (f) {
      const url = URL.createObjectURL(f);
      setPreview(url);
    } else {
      setPreview(null);
    }
  };

  const onPredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${apiBase}/predict`, {
        method: "POST",
        body: form,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.error || "Prediction failed");
      }
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid">
      <section className="card">
        <h3>Image Input</h3>
        <div className="upload">
          <div className="input-row">
            <label className="mono">API base URL</label>
            <input
              className="mono input"
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
            />
          </div>

          <div className="input-row">
            <label className="mono">Upload image (JPG/PNG)</label>
            <input type="file" accept="image/*" onChange={onFileChange} />
          </div>

          <button className="button" onClick={onPredict} disabled={!file || loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </div>

        {preview && (
          <div className="preview">
            <img src={preview} alt="Preview" />
          </div>
        )}
      </section>

      <section className="card">
        <h3>Results</h3>
        {!results && !error && <div className="mono">Run a prediction to see results here.</div>}
        {error && <div className="error">{error}</div>}

        {results && (
          <div className="result">
            {results.top && (
              <div className="badge">
                Top prediction: {results.top.label} ({results.top.prob.toFixed(2)})
              </div>
            )}

            <div className="meta">
              <div className="mono">Model: {results.model?.path} ({results.model?.type})</div>
              <div className="mono">Threshold: {results.threshold?.toFixed(2)}</div>
            </div>

            <div>
              <div className="section-title">Detected (>= threshold)</div>
              {detected.length === 0 && <div className="mono">No labels passed the threshold.</div>}
              {detected.length > 0 && (
                <div className="chips">
                  {detected.map((d) => (
                    <span className="chip" key={d.label}>{d.label} ({d.prob.toFixed(2)})</span>
                  ))}
                </div>
              )}
            </div>

            <div>
              <div className="section-title">All Probabilities</div>
              <div className="bars">
                {results.results.map((r) => (
                  <div className="bar" key={r.label}>
                    <div>
                      <div style={{ fontWeight: 600 }}>{r.label}</div>
                      <div className="bar-line">
                        <div
                          className="bar-fill"
                          style={{ width: `${Math.min(100, r.prob * 100)}%` }}
                        ></div>
                      </div>
                    </div>
                    <div className="mono">{r.prob.toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}