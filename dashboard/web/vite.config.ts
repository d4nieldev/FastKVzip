import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Built straight into the location main.py serves from, so there is one
  // deploy artifact and no CORS.
  build: { outDir: '../server/static', emptyOutDir: true },
  server: {
    port: 5173,
    proxy: { '/api': 'http://localhost:8000', '/healthz': 'http://localhost:8000' },
  },
})
