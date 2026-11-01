import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 30000,
  fullyParallel: false,
  use: {
    baseURL: 'http://127.0.0.1:5174',
    viewport: { width: 1440, height: 1050 },
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: 'uv run --project .. python ../tests/ui_fixture_server.py',
      url: 'http://127.0.0.1:8012/api/health',
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
    {
      command: 'yarn dev --port 5174 --strictPort',
      url: 'http://127.0.0.1:5174',
      env: { INSIGHT_API_URL: 'http://127.0.0.1:8012' },
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
  ],
})
