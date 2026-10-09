import { useState } from "react";
import { workerApi, type Readiness, type ReadinessCheck } from "../api/client";

const STATUS_COLOR: Record<ReadinessCheck["status"], string> = {
  ok: "var(--success)",
  warning: "var(--warning)",
  error: "var(--danger)",
  not_configured: "var(--text-faint)",
};

type TestState = "idle" | "testing" | "ok" | "error";

export function SettingsPage() {
  const [workerUrl, setWorkerUrl] = useState(() => localStorage.getItem("worker_url") ?? "");
  const [token, setToken] = useState(() => localStorage.getItem("worker_token") ?? "");
  const [saved, setSaved] = useState(false);

  const [testState, setTestState] = useState<TestState>("idle");
  const [testError, setTestError] = useState<string | null>(null);
  const [ready, setReady] = useState<Readiness | null>(null);
  const [build, setBuild] = useState<string | null>(null);

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
    setReady(null);
    try {
      const health = await workerApi.health();
      setBuild(`${health.version}${health.git_commit ? ` (${health.git_commit.slice(0, 8)})` : ""}`);
      // Authenticates against every data provider: takes a few seconds.
      setReady(await workerApi.ready(true));
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
          {testState === "ok" && ready && (
            <div>
              <div style={{ fontSize: 13, color: STATUS_COLOR[ready.status], fontWeight: 500, marginBottom: 8 }}>
                {ready.status === "ok" ? "✓ Worker ready" : ready.status === "warning" ? "Worker ready, with warnings" : "Worker not ready"}
                {build && <span style={{ color: "var(--text-muted)", fontWeight: 400 }}> · version {build}</span>}
              </div>
              <table className="table">
                <thead>
                  <tr>
                    <th>Check</th>
                    <th>Status</th>
                    <th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(ready.checks).map(([name, check]) => (
                    <tr key={name}>
                      <td style={{ fontWeight: 500 }}>{name.replace("credentials.", "")}</td>
                      <td style={{ color: STATUS_COLOR[check.status], fontWeight: 500 }}>{check.status.replace("_", " ")}</td>
                      <td style={{ fontSize: 12, color: "var(--text-muted)" }}>{check.detail}</td>
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
