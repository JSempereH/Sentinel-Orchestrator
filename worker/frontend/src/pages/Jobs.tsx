import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { ACTIVE_STATUSES, jobsApi, type JobMetrics, type JobStatus, type JobSummary } from "../api/client";

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
      const anyActive = rows.some((j) => ACTIVE_STATUSES.includes(j.status));
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

function formatMetrics(metrics: JobMetrics): string | null {
  const parts: string[] = [];
  if (metrics.duration_s !== undefined) parts.push(`${Math.round(metrics.duration_s / 60)} min`);
  if (metrics.peak_rss_mb !== undefined) parts.push(`peak RAM ${(metrics.peak_rss_mb / 1024).toFixed(1)} GB`);
  if (metrics.failed_products) parts.push(`${metrics.failed_products} product(s) skipped after failing`);
  return parts.length ? parts.join(" · ") : null;
}

function errorText(error: unknown): string {
  return String((error as AxiosError<{ detail?: string }>)?.response?.data?.detail ?? error);
}

function JobRow({ job, expanded, onToggle }: { job: JobSummary; expanded: boolean; onToggle: () => void }) {
  const queryClient = useQueryClient();
  const [showLog, setShowLog] = useState(false);
  const active = ACTIVE_STATUSES.includes(job.status);
  const detailQuery = useQuery({
    queryKey: ["job", job.job_id],
    queryFn: () => jobsApi.get(job.job_id),
    enabled: expanded,
    refetchInterval: expanded && active ? RUNNING_POLL_MS : IDLE_POLL_MS,
  });
  const logQuery = useQuery({
    queryKey: ["job-log", job.job_id],
    queryFn: () => jobsApi.log(job.job_id),
    enabled: expanded && showLog,
    refetchInterval: expanded && showLog && active ? RUNNING_POLL_MS : IDLE_POLL_MS,
  });
  const refresh = () => queryClient.invalidateQueries({ queryKey: ["jobs"] });
  const cancelMut = useMutation({ mutationFn: () => jobsApi.cancel(job.job_id), onSuccess: refresh });
  const deleteMut = useMutation({ mutationFn: () => jobsApi.remove(job.job_id), onSuccess: refresh });
  const metrics = formatMetrics(job.metrics ?? {});

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
                {metrics && <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{metrics}</div>}
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", marginBottom: 4 }}>
                    REQUEST
                  </div>
                  <pre style={{ margin: 0, fontSize: 11, whiteSpace: "pre-wrap" }}>
                    {JSON.stringify(detailQuery.data.request_params, null, 2)}
                  </pre>
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {job.status === "SUCCEEDED" && (
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={(e) => { e.stopPropagation(); jobsApi.download(job.job_id); }}
                    >
                      ⬇ Download result
                    </button>
                  )}
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={(e) => { e.stopPropagation(); setShowLog(!showLog); }}
                  >
                    {showLog ? "Hide log" : "Show log"}
                  </button>
                  {active && (
                    <button
                      className="btn btn-secondary btn-sm"
                      disabled={cancelMut.isPending}
                      onClick={(e) => { e.stopPropagation(); if (confirm("Cancel this job?")) cancelMut.mutate(); }}
                    >
                      Cancel job
                    </button>
                  )}
                  {!active && (
                    <button
                      className="btn btn-secondary btn-sm"
                      style={{ color: "var(--danger)" }}
                      disabled={deleteMut.isPending}
                      onClick={(e) => { e.stopPropagation(); if (confirm("Delete this job and all its files?")) deleteMut.mutate(); }}
                    >
                      Delete
                    </button>
                  )}
                </div>
                {(cancelMut.isError || deleteMut.isError) && (
                  <div style={{ fontSize: 12, color: "var(--danger)" }}>{errorText(cancelMut.error ?? deleteMut.error)}</div>
                )}
                {showLog && (
                  <pre style={{ margin: 0, fontSize: 11, maxHeight: 320, overflow: "auto", whiteSpace: "pre-wrap", background: "#fff", border: "1px solid var(--border)", borderRadius: 6, padding: 8 }}>
                    {logQuery.isLoading ? "Loading…" : logQuery.data || "(no log yet)"}
                  </pre>
                )}
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
