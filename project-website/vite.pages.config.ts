import react from "@vitejs/plugin-react";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const projectRoot = dirname(fileURLToPath(import.meta.url));
const base = process.env.PAGES_BASE_PATH || "/";

export default defineConfig({
  root: resolve(projectRoot, "static-site"),
  base,
  publicDir: resolve(projectRoot, "public"),
  plugins: [react()],
  build: {
    outDir: resolve(projectRoot, "dist-pages"),
    emptyOutDir: true,
  },
});
