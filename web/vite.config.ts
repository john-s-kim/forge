import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the dashboard runs on :5173 and proxies API calls to `forge dev`.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/v1": "http://127.0.0.1:8080" } },
});
