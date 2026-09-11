import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

type Coord = [number, number];

const OSM_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution:
        '© <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
      maxzoom: 19,
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};

interface AOIMapProps {
  onGeometryChange: (geometry: GeoJSON.Geometry | null) => void;
  initialGeometry?: GeoJSON.Geometry | null;
  height?: number;
}

export function AOIMap({ onGeometryChange, initialGeometry, height = 520 }: AOIMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef       = useRef<maplibregl.Map | null>(null);

  const [ready,    setReady]    = useState(false);
  const [drawing,  setDrawing]  = useState(false);
  const [vertices, setVertices] = useState<Coord[]>([]);
  const [polygon,  setPolygon]  = useState<GeoJSON.Geometry | null>(initialGeometry ?? null);

  // Refs so the single map click handler always sees current values
  const drawingRef  = useRef(false);
  const verticesRef = useRef<Coord[]>([]);
  useEffect(() => { drawingRef.current  = drawing;  }, [drawing]);
  useEffect(() => { verticesRef.current = vertices; }, [vertices]);

  // ── Map initialisation (runs once) ─────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: OSM_STYLE,
      center: [0, 30],
      zoom: 2,
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.addControl(new maplibregl.ScaleControl(), "bottom-left");

    map.on("load", () => {
      // Three GeoJSON sources: completed polygon, in-progress line, vertex dots
      map.addSource("aoi-polygon", { type: "geojson", data: fc([]) });
      map.addSource("aoi-line",    { type: "geojson", data: fc([]) });
      map.addSource("aoi-points",  { type: "geojson", data: fc([]) });

      map.addLayer({
        id: "aoi-polygon-fill",
        type: "fill",
        source: "aoi-polygon",
        paint: { "fill-color": "#2563eb", "fill-opacity": 0.12 },
      });
      map.addLayer({
        id: "aoi-polygon-outline",
        type: "line",
        source: "aoi-polygon",
        paint: { "line-color": "#2563eb", "line-width": 2.5 },
      });
      map.addLayer({
        id: "aoi-line",
        type: "line",
        source: "aoi-line",
        paint: { "line-color": "#2563eb", "line-width": 2, "line-dasharray": [3, 3] },
      });
      map.addLayer({
        id: "aoi-points",
        type: "circle",
        source: "aoi-points",
        paint: {
          "circle-radius": 5,
          "circle-color": "#ffffff",
          "circle-stroke-color": "#2563eb",
          "circle-stroke-width": 2,
        },
      });

      if (initialGeometry) {
        setGeoJSON(map, "aoi-polygon", [{ type: "Feature", geometry: initialGeometry, properties: {} }]);
      }

      setReady(true);
    });

    // Single click listener: reads current state via refs to avoid stale closures
    map.on("click", (e) => {
      if (!drawingRef.current) return;
      const coord: Coord = [e.lngLat.lng, e.lngLat.lat];
      setVertices((prev) => {
        const next = [...prev, coord];
        // Update the live-preview layers immediately
        setGeoJSON(map, "aoi-line", next.length >= 2 ? [lineFeature(next)] : []);
        setGeoJSON(map, "aoi-points", next.map(pointFeature));
        return next;
      });
    });

    mapRef.current = map;
    return () => map.remove();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Cursor ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = mapRef.current?.getCanvas();
    if (canvas) canvas.style.cursor = drawing ? "crosshair" : "";
  }, [drawing]);

  // ── Actions ────────────────────────────────────────────────────────────────
  function startDrawing() {
    clearAll();
    setDrawing(true);
  }

  function closePolygon() {
    const map = mapRef.current;
    const verts = verticesRef.current;
    if (!map || verts.length < 3) return;

    const geometry: GeoJSON.Geometry = {
      type: "Polygon",
      coordinates: [[...verts, verts[0]]],
    };
    setPolygon(geometry);
    setDrawing(false);
    setVertices([]);
    onGeometryChange(geometry);

    setGeoJSON(map, "aoi-polygon", [{ type: "Feature", geometry, properties: {} }]);
    setGeoJSON(map, "aoi-line",    []);
    setGeoJSON(map, "aoi-points",  []);
  }

  function cancelDrawing() {
    const map = mapRef.current;
    setDrawing(false);
    setVertices([]);
    if (map) { setGeoJSON(map, "aoi-line", []); setGeoJSON(map, "aoi-points", []); }
  }

  function clearAll() {
    const map = mapRef.current;
    setPolygon(null);
    setDrawing(false);
    setVertices([]);
    onGeometryChange(null);
    if (map?.loaded()) {
      setGeoJSON(map, "aoi-polygon", []);
      setGeoJSON(map, "aoi-line",    []);
      setGeoJSON(map, "aoi-points",  []);
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  const hint =
    vertices.length === 0 ? "Click on the map to add points" :
    vertices.length === 1 ? "1 point, add at least 2 more" :
    vertices.length === 2 ? "2 points, add 1 more to close" :
    `${vertices.length} points. Click "Close" or keep adding`;

  return (
    <div style={{ position: "relative", borderRadius: "var(--radius-lg)", overflow: "hidden", border: "1px solid var(--border)" }}>
      <div ref={containerRef} style={{ width: "100%", height }} />

      {/* Drawing controls overlay */}
      {ready && (
        <div style={{ position: "absolute", top: 10, left: 10, display: "flex", flexDirection: "column", gap: 6, zIndex: 10 }}>
          {!drawing && !polygon && (
            <button className="btn btn-primary btn-sm" onClick={startDrawing}
              style={{ boxShadow: "0 2px 8px rgba(0,0,0,0.2)" }}>
              ✏ Draw polygon
            </button>
          )}

          {drawing && (
            <>
              <div style={{
                background: "rgba(255,255,255,0.97)",
                padding: "6px 11px",
                borderRadius: "var(--radius)",
                fontSize: 12,
                color: "var(--text-muted)",
                border: "1px solid var(--border)",
                boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
              }}>
                {hint}
              </div>

              {vertices.length >= 3 && (
                <button className="btn btn-primary btn-sm" onClick={closePolygon}
                  style={{ boxShadow: "0 2px 8px rgba(0,0,0,0.2)" }}>
                  ✓ Close polygon
                </button>
              )}

              <button className="btn btn-secondary btn-sm" onClick={cancelDrawing}
                style={{ boxShadow: "0 1px 4px rgba(0,0,0,0.1)" }}>
                ✕ Cancel
              </button>
            </>
          )}

          {polygon && !drawing && (
            <>
              <div style={{
                background: "rgba(239,246,255,0.97)",
                padding: "5px 10px",
                borderRadius: "var(--radius)",
                fontSize: 12,
                color: "var(--primary-text)",
                fontWeight: 500,
                border: "1px solid #bfdbfe",
                boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
              }}>
                ✓ AOI defined
              </div>
              <button className="btn btn-secondary btn-sm" onClick={startDrawing}
                style={{ boxShadow: "0 1px 4px rgba(0,0,0,0.08)" }}>
                ↺ Redraw
              </button>
            </>
          )}
        </div>
      )}

      {/* Loading overlay */}
      {!ready && (
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          background: "#f8fafc", gap: 8,
        }}>
          <div style={{ fontSize: 24 }}>🗺</div>
          <div style={{ fontSize: 13, color: "var(--text-muted)" }}>Loading map…</div>
        </div>
      )}
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fc(features: GeoJSON.Feature[]): GeoJSON.FeatureCollection {
  return { type: "FeatureCollection", features };
}

function lineFeature(coords: Coord[]): GeoJSON.Feature {
  return { type: "Feature", geometry: { type: "LineString", coordinates: coords }, properties: {} };
}

function pointFeature(coord: Coord): GeoJSON.Feature {
  return { type: "Feature", geometry: { type: "Point", coordinates: coord }, properties: {} };
}

function setGeoJSON(map: maplibregl.Map, sourceId: string, features: GeoJSON.Feature[]) {
  (map.getSource(sourceId) as maplibregl.GeoJSONSource | undefined)?.setData(fc(features));
}
