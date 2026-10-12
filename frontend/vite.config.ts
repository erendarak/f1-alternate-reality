/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The backend (scripts/serve.py) is authoritative; in development Vite proxies to it.
const BACKEND = process.env.F1AR_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": BACKEND,
      "/ws": { target: BACKEND.replace(/^http/, "ws"), ws: true },
    },
  },
  test: { environment: "node" },
});
