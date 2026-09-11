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
}

export type JobStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED";

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
}

export interface JobDetail extends JobSummary {
  request_params: JobRequest | null;
}

// Each provider's shape genuinely differs (OpenAQ has a numeric rate limit,
// CDSE/CDS-ADS only expose a configured flag and a note) - kept as a bag of
// unknowns and rendered generically rather than pretending a common shape.
export type UsageSnapshot = Record<string, Record<string, unknown>>;

// ── API calls ──────────────────────────────────────────────────────────────

export const workerApi = {
  health: () => api.get<{ status: string; version: string }>("/health").then((r) => r.data),
  usage: () => api.get<UsageSnapshot>("/usage").then((r) => r.data),
};

export const jobsApi = {
  submit: (request: JobRequest, name?: string) =>
    api
      .post<{ job_id: string; status: JobStatus }>("/jobs", request, { params: name ? { name } : undefined })
      .then((r) => r.data),
  list: () => api.get<JobSummary[]>("/jobs").then((r) => r.data),
  get: (jobId: string) => api.get<JobDetail>(`/jobs/${jobId}`).then((r) => r.data),
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
