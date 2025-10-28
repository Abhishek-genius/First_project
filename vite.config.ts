// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import fs from "fs";
import dotenv from "dotenv";

// Try to load backend/.env (so you can keep a single .env in backend/)
const backendEnvPath = path.resolve(__dirname, "../backend/.env");
if (fs.existsSync(backendEnvPath)) {
  dotenv.config({ path: backendEnvPath });
}

// Fallback URL if env not present
const BACKEND_URL = process.env.VITE_API_URL || "http://127.0.0.1:5000";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ["lucide-react"],
  },
  server: {
    proxy: {
      // Proxy /api/* to backend
      "/api": {
        target: BACKEND_URL,
        changeOrigin: true,
        secure: false,
        rewrite: (p) => p,
      },
      // Proxy /static/* to backend (so matched images served by Flask are reachable)
      "/static": {
        target: BACKEND_URL,
        changeOrigin: true,
        secure: false,
        rewrite: (p) => p,
      },
    },
  },
});
