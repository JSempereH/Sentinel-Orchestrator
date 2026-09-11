import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { jobsApi, type JobStatus, type JobSummary } from "../api/client";

const RUNNING_POLL_MS = 5_000;
const IDLE_POLL_MS = false as const;

function badgeClass(status: JobStatus): string {
  return `badge badge-${status.toLowerCase()}`;
}

export function JobsPage({ focusJobId }: { focusJobId?: string | null }) {
  const [expanded, setExpanded] = useState<string | null>(focusJobId ?? null);

  const { data: jobs, isLoading } = useQuery({
    queryKey: ["jobs"],
    queryFn: jobsApi.list,
    refetchInterval: (query) => {
      const rows = query.state.data ?? [];
      const anyActive = rows.some((j) => j.status === "PENDING" || j.status === "RUNNING");
      return anyActive ? RUNNING_POLL_MS : IDLE_POLL_MS;
    },
  });

  return (
    <div className="page">
      <h2 style={{ marginBottom: 4 }}>Jobs</h2>
      <p style={{ margin: "0 0 20px", color: "var(--text-muted)", fontSize: 13 }}>
        History of runs submitted to this worker. Updates automatically while any job is pending or running.
      </p>

      {isLoading && <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Loading…</div>}

      {jobs && jobs.length === 0 && (
        <div className="card" style={{ padding: 20, color: "var(--text-muted)", fontSize: 13 }}>
          No jobs yet - submit one from the "New run" page.
        </div>
      )}

      {jobs && jobs.length > 0 && (
        <div className="card">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Created</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <JobRow
                  key={job.job_id}
                  job={job}
                  expanded={expanded === job.job_id}
                  onToggle={() => setExpanded(expanded === job.job_id ? null : job.job_id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function JobRow({ job, expanded, onToggle }: { job: JobSummary; expanded: boolean; onToggle: () => void }) {
  const detailQuery = useQuery({
    queryKey: ["job", job.job_id],
    queryFn: () => jobsApi.get(job.job_id),
    enabled: expanded,
    refetchInterval: expanded && (job.status === "PENDING" || job.status === "RUNNING") ? RUNNING_POLL_MS : IDLE_POLL_MS,
  });

  return (
    <>
      <tr style={{ cursor: "pointer" }} onClick={onToggle}>
        <td style={{ fontWeight: 500 }}>{job.name ?? <span style={{ color: "var(--text-faint)" }}>(unnamed)</span>}</td>
        <td>
          <span className={badgeClass(job.status)}>{job.status}</span>
        </td>
        <td style={{ color: "var(--text-muted)" }}>
          {job.progress ? `${job.progress.sensor}: ${job.progress.done}/${job.progress.total}` : "—"}
        </td>
        <td style={{ color: "var(--text-muted)" }}>{job.created_at ? new Date(job.created_at).toLocaleString() : "—"}</td>
        <td style={{ color: "var(--text-faint)" }}>{expanded ? "▲" : "▼"}</td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={5} style={{ background: "#fafafa" }}>
            {detailQuery.isLoading && <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Loading detail…</div>}
            {detailQuery.data && (
              <div style={{ display: "flex", flexDirection: "column", gap: 10, padding: "4px 0" }}>
                {detailQuery.data.error_message && (
                  <div style={{ fontSize: 12, color: "var(--danger)" }}>{detailQuery.data.error_message}</div>
                )}
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", marginBottom: 4 }}>
                    REQUEST
                  </div>
                  <pre style={{ margin: 0, fontSize: 11, whiteSpace: "pre-wrap" }}>
                    {JSON.stringify(detailQuery.data.request_params, null, 2)}
                  </pre>
                </div>
                {job.status === "SUCCEEDED" && (
                  <button
                    className="btn btn-secondary btn-sm"
                    style={{ width: "fit-content" }}
                    onClick={(e) => { e.stopPropagation(); jobsApi.download(job.job_id); }}
                  >
                    ⬇ Download result
                  </button>
                )}
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
