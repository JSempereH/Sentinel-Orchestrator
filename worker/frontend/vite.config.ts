import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dirname = path.dirname(fileURLToPath(import.meta.url));

// Dev-only proxy: the browser only ever talks to this Vite origin, which
// forwards the worker's exact API paths to the real worker process. In
// production the built SPA is served by the worker itself (same origin),
// so no proxy and no CORS are needed either way.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(dirname, "./src") },
  },
  server: {
    proxy: {
      "/health": "http://localhost:8100",
      "/jobs": "http://localhost:8100",
      "/usage": "http://localhost:8100",
    },
  },
});
