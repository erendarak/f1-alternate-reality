// End-to-end tests (M3 slice 4). The test drives its own backend (e2e/backend.ts) so it can
// kill and restart it mid-test; Playwright starts a separate Vite server pointed at it.
// Ports 8001 / 5174 keep it apart from a running dev setup (8000 / 5173).

import { defineConfig, devices } from "@playwright/test";

import { E2E_BACKEND_PORT } from "./e2e/backend";

const UI_PORT = 5174;

export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${UI_PORT}`,
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: {
    command: `npx vite --port ${UI_PORT} --strictPort`,
    url: `http://localhost:${UI_PORT}`,
    reuseExistingServer: false,
    env: { F1AR_BACKEND: `http://127.0.0.1:${E2E_BACKEND_PORT}` },
    timeout: 60_000,
  },
});
