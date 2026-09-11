import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { SubmitPage } from "./pages/Submit";
import { JobsPage } from "./pages/Jobs";
import { SettingsPage } from "./pages/Settings";

const queryClient = new QueryClient();

type Page = "submit" | "jobs" | "settings";

const NAV: { id: Page; icon: string; label: string }[] = [
  { id: "submit", icon: "🛰", label: "New run" },
  { id: "jobs", icon: "📋", label: "Jobs" },
  { id: "settings", icon: "⚙", label: "Settings" },
];

function App() {
  const [page, setPage] = useState<Page>("submit");
  const [focusJobId, setFocusJobId] = useState<string | null>(null);

  return (
    <QueryClientProvider client={queryClient}>
      <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
        <nav
          style={{
            width: 220,
            background: "var(--sidebar-bg)",
            display: "flex",
            flexDirection: "column",
            flexShrink: 0,
            borderRight: "1px solid var(--sidebar-border)",
          }}
        >
          <div
            style={{
              padding: "20px 16px 18px",
              borderBottom: "1px solid var(--sidebar-border)",
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}
          >
            <div
              style={{
                width: 32, height: 32, borderRadius: 8,
                background: "var(--primary)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 16, flexShrink: 0,
              }}
            >
              🛰
            </div>
            <div>
              <div style={{ color: "#f1f5f9", fontWeight: 700, fontSize: 13, lineHeight: 1.2 }}>
                Sentinel Worker
              </div>
              <div style={{ color: "#475569", fontSize: 11 }}>v0.1.0</div>
            </div>
          </div>

          <div style={{ flex: 1, padding: "8px 8px 0" }}>
            {NAV.map(({ id, icon, label }) => {
              const active = page === id;
              return (
                <button
                  key={id}
                  onClick={() => { setPage(id); if (id !== "jobs") setFocusJobId(null); }}
                  style={{
                    width: "100%",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "9px 12px",
                    marginBottom: 2,
                    background: active ? "var(--sidebar-active)" : "transparent",
                    border: "none",
                    borderRadius: 6,
                    color: active ? "#60a5fa" : "var(--sidebar-text)",
                    fontSize: 13,
                    fontWeight: active ? 600 : 400,
                    fontFamily: "var(--font)",
                    cursor: "pointer",
                    textAlign: "left",
                  }}
                  onMouseEnter={(e) => {
                    if (!active) (e.currentTarget as HTMLElement).style.background = "var(--sidebar-hover)";
                  }}
                  onMouseLeave={(e) => {
                    if (!active) (e.currentTarget as HTMLElement).style.background = "transparent";
                  }}
                >
                  <span style={{ fontSize: 15, opacity: 0.85 }}>{icon}</span>
                  {label}
                </button>
              );
            })}
          </div>

          <div style={{ padding: "12px 16px", borderTop: "1px solid var(--sidebar-border)" }}>
            <div style={{ fontSize: 11, color: "#334155" }}>sentinel_analysis</div>
          </div>
        </nav>

        <main style={{ flex: 1, overflowY: "auto", background: "var(--bg)" }}>
          {page === "submit" && (
            <SubmitPage
              onSubmitted={(jobId) => {
                setFocusJobId(jobId);
                setPage("jobs");
              }}
            />
          )}
          {page === "jobs" && <JobsPage focusJobId={focusJobId} />}
          {page === "settings" && <SettingsPage />}
        </main>
      </div>
    </QueryClientProvider>
  );
}

export default App;
