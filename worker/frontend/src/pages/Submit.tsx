import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { AOIMap } from "../components/Map/AOIMap";
import { jobsApi, type AOI, type AuxiliarySpec } from "../api/client";

// Mirrors sentinel_analysis.workflow.request.SUPPORTED_SENSORS.
const SENSORS = [
  { id: "sentinel1", label: "Sentinel-1 (radar)" },
  { id: "sentinel2", label: "Sentinel-2 (optical)" },
  { id: "sentinel3", label: "Sentinel-3 (thermal, ~1 km)" },
  { id: "sentinel5p", label: "Sentinel-5P (atmospheric)" },
  { id: "landsat", label: "Landsat 8/9 (thermal, ~100 m)" },
  { id: "ecostress", label: "ECOSTRESS (thermal, ~70 m)" },
];

// Mirrors sentinel_analysis.providers.base.AUXILIARY_PROVIDERS.
const AUX_PROVIDERS = [
  { id: "era5", label: "ERA5 (reanalysis meteorology)" },
  { id: "cams", label: "CAMS (atmospheric composition)" },
  { id: "openaq", label: "OpenAQ (ground air-quality stations)" },
];

function todayMinus(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

export function SubmitPage({ onSubmitted }: { onSubmitted: (jobId: string) => void }) {
  const [name, setName] = useState("");
  const [geometry, setGeometry] = useState<GeoJSON.Geometry | null>(null);
  const [start, setStart] = useState(todayMinus(7));
  const [end, setEnd] = useState(todayMinus(0));
  const [sensors, setSensors] = useState<Set<string>>(new Set(["sentinel3"]));
  const [auxProviders, setAuxProviders] = useState<Set<string>>(new Set());
  const [resolutionM, setResolutionM] = useState(100);
  const [maxProducts, setMaxProducts] = useState(20);
  const [dayOnly, setDayOnly] = useState(true);
  const [downscale, setDownscale] = useState(false);
  const [terrain, setTerrain] = useState(false);

  const submitMut = useMutation({
    mutationFn: () => {
      const bounds = boundsOf(geometry!);
      const auxiliary: AuxiliarySpec[] = [...auxProviders].map((provider) => ({ provider: provider as AuxiliarySpec["provider"] }));
      return jobsApi.submit(
        {
          aoi: bounds,
          start,
          end,
          sensors: [...sensors],
          resolution_m: resolutionM,
          max_products_per_sensor: maxProducts,
          auxiliary,
          thermal_overpass: dayOnly ? "day" : "any",
          terrain_predictors: terrain,
          downscale: downscale ? { model: "local_trees" } : null,
        },
        name.trim() || undefined,
      );
    },
    onSuccess: (data) => onSubmitted(data.job_id),
  });

  function toggle(set: Set<string>, setSet: (s: Set<string>) => void, id: string) {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSet(next);
  }

  // Downscaling sharpens Sentinel-3 with Sentinel-2 predictors (mirrors the
  // request's own validation, so the error shows before submitting).
  const downscaleBlocked = downscale && !(sensors.has("sentinel3") && sensors.has("sentinel2"));
  const canSubmit = geometry !== null && sensors.size > 0 && start && end && !downscaleBlocked && !submitMut.isPending;

  return (
    <div className="page">
      <h2 style={{ marginBottom: 4 }}>New analysis run</h2>
      <p style={{ margin: "0 0 24px", color: "var(--text-muted)", fontSize: 13 }}>
        Draw an AOI, pick sensors and a date range, and submit to the sentinel-worker.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 20, alignItems: "start" }}>
        <div className="card" style={{ padding: 16 }}>
          <h3>Area of interest</h3>
          <AOIMap onGeometryChange={setGeometry} />
        </div>

        <div className="card" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="field">
            <label>Name (optional)</label>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Berlin heat, June" />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div className="field">
              <label>Start date</label>
              <input className="input" type="date" value={start} onChange={(e) => setStart(e.target.value)} />
            </div>
            <div className="field">
              <label>End date</label>
              <input className="input" type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
            </div>
          </div>

          <div className="field">
            <label>Sensors</label>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {SENSORS.map(({ id, label }) => (
                <label key={id} className={`check-row${sensors.has(id) ? " checked" : ""}`}>
                  <input type="checkbox" checked={sensors.has(id)} onChange={() => toggle(sensors, setSensors, id)} />
                  {label}
                </label>
              ))}
            </div>
          </div>

          <div className="field">
            <label>Auxiliary data (optional)</label>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {AUX_PROVIDERS.map(({ id, label }) => (
                <label key={id} className={`check-row${auxProviders.has(id) ? " checked" : ""}`}>
                  <input
                    type="checkbox"
                    checked={auxProviders.has(id)}
                    onChange={() => toggle(auxProviders, setAuxProviders, id)}
                  />
                  {label}
                </label>
              ))}
            </div>
          </div>

          <div className="field">
            <label>Options</label>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <label className={`check-row${dayOnly ? " checked" : ""}`}>
                <input type="checkbox" checked={dayOnly} onChange={() => setDayOnly(!dayOnly)} />
                Daytime thermal passes only
              </label>
              <label className={`check-row${terrain ? " checked" : ""}`}>
                <input type="checkbox" checked={terrain} onChange={() => setTerrain(!terrain)} />
                Terrain predictors (elevation, slope, illumination)
              </label>
              <label className={`check-row${downscale ? " checked" : ""}`}>
                <input type="checkbox" checked={downscale} onChange={() => setDownscale(!downscale)} />
                Downscale temperature to the resolution below (needs Sentinel-2 + Sentinel-3)
              </label>
            </div>
            {downscaleBlocked && (
              <div style={{ fontSize: 12, color: "var(--danger)" }}>Downscaling needs both Sentinel-3 and Sentinel-2 selected.</div>
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div className="field">
              <label>Resolution (m)</label>
              <input
                className="input"
                type="number"
                min={10}
                value={resolutionM}
                onChange={(e) => setResolutionM(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>Max products / sensor</label>
              <input
                className="input"
                type="number"
                min={1}
                value={maxProducts}
                onChange={(e) => setMaxProducts(Number(e.target.value))}
              />
            </div>
          </div>

          <button className="btn btn-primary" disabled={!canSubmit} onClick={() => submitMut.mutate()}>
            {submitMut.isPending ? "Submitting…" : "Submit run"}
          </button>

          {!geometry && (
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Draw an AOI on the map first.</div>
          )}
          {submitMut.isError && (
            <div style={{ fontSize: 12, color: "var(--danger)" }}>
              {String((submitMut.error as AxiosError<{ detail?: string }>)?.response?.data?.detail ?? submitMut.error)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function boundsOf(geometry: GeoJSON.Geometry): AOI {
  const coords: [number, number][] =
    geometry.type === "Polygon" ? (geometry.coordinates[0] as [number, number][]) : [];
  const lons = coords.map((c) => c[0]);
  const lats = coords.map((c) => c[1]);
  const bounds: AOI = { west: Math.min(...lons), south: Math.min(...lats), east: Math.max(...lons), north: Math.max(...lats) };
  // Only a non-rectangular drawing carries its polygon: a box needs no mask.
  const onEdge = coords.every(([lon, lat]) => (lon === bounds.west || lon === bounds.east) && (lat === bounds.south || lat === bounds.north));
  if (geometry.type === "Polygon" && !onEdge) bounds.geometry = geometry;
  return bounds;
}
