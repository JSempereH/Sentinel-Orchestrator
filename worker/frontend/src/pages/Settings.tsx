import { useState } from "react";
import { workerApi, type UsageSnapshot } from "../api/client";

type TestState = "idle" | "testing" | "ok" | "error";

export function SettingsPage() {
  const [workerUrl, setWorkerUrl] = useState(() => localStorage.getItem("worker_url") ?? "");
  const [token, setToken] = useState(() => localStorage.getItem("worker_token") ?? "");
  const [saved, setSaved] = useState(false);

  const [testState, setTestState] = useState<TestState>("idle");
  const [testError, setTestError] = useState<string | null>(null);
  const [usage, setUsage] = useState<UsageSnapshot | null>(null);

  function handleSave() {
    if (workerUrl.trim()) localStorage.setItem("worker_url", workerUrl.trim());
    else localStorage.removeItem("worker_url");
    if (token.trim()) localStorage.setItem("worker_token", token.trim());
    else localStorage.removeItem("worker_token");
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  async function handleTest() {
    setTestState("testing");
    setTestError(null);
    setUsage(null);
    try {
      await workerApi.health();
      const snapshot = await workerApi.usage();
      setUsage(snapshot);
      setTestState("ok");
    } catch (err) {
      setTestError(err instanceof Error ? err.message : "Could not reach the worker");
      setTestState("error");
    }
  }

  return (
    <div className="page" style={{ maxWidth: 620 }}>
      <h2 style={{ marginBottom: 4 }}>Settings</h2>
      <p style={{ margin: "0 0 28px", color: "var(--text-muted)", fontSize: 13 }}>
        This page only configures how the browser talks to the sentinel-worker. Provider
        credentials (CDSE, CDS/ADS, OpenAQ, Earthdata) live in the worker's own <code>.env</code>{" "}
        file on the machine running it - they never pass through this UI.
      </p>

      <div className="card" style={{ padding: 24, marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 14, marginBottom: 20 }}>
          <div
            style={{
              width: 40, height: 40, borderRadius: 10, flexShrink: 0,
              background: "#e0f2fe",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20,
            }}
          >
            🔌
          </div>
          <div>
            <h3 style={{ marginBottom: 4 }}>Worker connection</h3>
            <p style={{ margin: 0, fontSize: 13, color: "var(--text-muted)" }}>
              Leave the URL blank to talk to the worker serving this page (the default). Set it
              only if this build is hosted separately from the worker it should control.
            </p>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="field">
            <label>Worker URL (optional)</label>
            <input
              className="input"
              type="text"
              value={workerUrl}
              onChange={(e) => { setWorkerUrl(e.target.value); setSaved(false); }}
              placeholder="http://localhost:8100"
            />
          </div>
          <div className="field">
            <label>Bearer token</label>
            <input
              className="input"
              type="password"
              value={token}
              onChange={(e) => { setToken(e.target.value); setSaved(false); }}
              autoComplete="off"
              placeholder="WORKER_API_TOKEN"
            />
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <button className="btn btn-primary" onClick={handleSave}>
              Save
            </button>
            <button className="btn btn-secondary" onClick={handleTest} disabled={testState === "testing"}>
              {testState === "testing" ? "Testing…" : "Test connection"}
            </button>
            {saved && <span style={{ color: "var(--success)", fontSize: 13, fontWeight: 500 }}>✓ Saved</span>}
          </div>

          {testState === "error" && (
            <div style={{ fontSize: 13, color: "var(--danger)" }}>{testError}</div>
          )}
          {testState === "ok" && usage && (
            <div>
              <div style={{ fontSize: 13, color: "var(--success)", fontWeight: 500, marginBottom: 8 }}>
                ✓ Reached the worker
              </div>
              <table className="table">
                <thead>
                  <tr>
                    <th>Provider</th>
                    <th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(usage).map(([provider, info]) => (
                    <tr key={provider}>
                      <td style={{ fontWeight: 500, verticalAlign: "top" }}>{provider}</td>
                      <td>
                        <pre style={{ margin: 0, fontSize: 11, whiteSpace: "pre-wrap", color: "var(--text-muted)" }}>
                          {JSON.stringify(info, null, 2)}
                        </pre>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
