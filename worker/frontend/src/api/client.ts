import axios from "axios";

// Empty baseURL: proxied to the worker by Vite in dev (see vite.config.ts),
// same-origin in production (the worker serves this built SPA itself) -
// identical code either way, no CORS configuration needed anywhere. An
// explicit worker_url in localStorage (Settings page) overrides this only
// for the uncommon case of pointing a standalone-served build at a worker
// running on a different host - not needed for the default deployment.
export const api = axios.create({ baseURL: "" });

api.interceptors.request.use((config) => {
  const url = localStorage.getItem("worker_url");
  if (url) config.baseURL = url;
  const token = localStorage.getItem("worker_token");
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Types ──────────────────────────────────────────────────────────────────

export interface AOI {
  west: number;
  south: number;
  east: number;
  north: number;
  /** The drawn polygon when it is not just its bounding box; results are masked to it. */
  geometry?: GeoJSON.Polygon;
}

export interface AuxiliarySpec {
  provider: "era5" | "cams" | "openaq";
  variables?: string[];
  dataset?: string;
  options?: Record<string, unknown>;
}

export interface JobRequest {
  aoi: AOI;
  start: string;
  end: string;
  sensors: string[];
  resolution_m?: number;
  thermal_resolution_m?: number;
  max_products_per_sensor?: number;
  s2_cloud_cover_max?: number;
  auxiliary?: AuxiliarySpec[];
  thermal_overpass?: "any" | "day" | "night";
  terrain_predictors?: boolean;
  on_product_error?: "skip" | "raise";
  downscale?: DownscaleSpec | null;
}

// Mirrors sentinel_analysis.downscale.DownscaleSpec.
export interface DownscaleSpec {
  model?: "linear" | "random_forest" | "xgboost" | "local_trees";
  predictors?: string[];
  coarse_consistent?: boolean;
  min_samples?: number;
}

export type JobStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED";

export const ACTIVE_STATUSES: JobStatus[] = ["PENDING", "RUNNING"];

export interface JobMetrics {
  duration_s?: number;
  peak_rss_mb?: number;
  failed_products?: number | null;
}

export interface JobProgress {
  sensor: string;
  done: number;
  total: number;
}

export interface JobSummary {
  job_id: string;
  name: string | null;
  status: JobStatus;
  progress: JobProgress | null;
  error_message: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  metrics: JobMetrics;
}

export interface JobDetail extends JobSummary {
  request_params: JobRequest | null;
}

// Each provider's shape genuinely differs (OpenAQ has a numeric rate limit,
// CDSE/CDS-ADS only expose a configured flag and a note) - kept as a bag of
// unknowns and rendered generically rather than pretending a common shape.
export type UsageSnapshot = Record<string, Record<string, unknown>>;

// ── API calls ──────────────────────────────────────────────────────────────

export interface ReadinessCheck {
  status: "ok" | "warning" | "error" | "not_configured";
  detail: string;
  expires_at?: string | null;
}

export interface Readiness {
  status: "ok" | "warning" | "error";
  checked_at: string;
  checks: Record<string, ReadinessCheck>;
}

export const workerApi = {
  health: () => api.get<{ status: string; version: string; git_commit: string | null }>("/health").then((r) => r.data),
  // 503 still carries the full report - surface it instead of throwing.
  ready: (refresh = false) =>
    api
      .get<Readiness>("/health/ready", { params: refresh ? { refresh: true } : undefined, validateStatus: (s) => s === 200 || s === 503 })
      .then((r) => r.data),
  usage: () => api.get<UsageSnapshot>("/usage").then((r) => r.data),
};

export const jobsApi = {
  submit: (request: JobRequest, name?: string) =>
    api
      .post<{ job_id: string; status: JobStatus }>("/jobs", request, { params: name ? { name } : undefined })
      .then((r) => r.data),
  list: () => api.get<JobSummary[]>("/jobs").then((r) => r.data),
  get: (jobId: string) => api.get<JobDetail>(`/jobs/${jobId}`).then((r) => r.data),
  cancel: (jobId: string) => api.post(`/jobs/${jobId}/cancel`).then((r) => r.data),
  remove: (jobId: string) => api.delete(`/jobs/${jobId}`).then((r) => r.data),
  log: (jobId: string) => api.get<string>(`/jobs/${jobId}/log`, { responseType: "text" }).then((r) => r.data),
  // The result endpoint requires the bearer token, which a plain <a href>
  // navigation cannot send - fetch it as a blob (picking up the same
  // interceptor-attached header) and hand the browser a local object URL.
  download: async (jobId: string) => {
    const response = await api.get(`/jobs/${jobId}/result`, { responseType: "blob" });
    const url = URL.createObjectURL(response.data as Blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${jobId}.zip`;
    link.click();
    URL.revokeObjectURL(url);
  },
};
