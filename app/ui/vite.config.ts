import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' so the bundle works when served by the Bernay API at /
// and inside the pywebview desktop shell.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8756',
    },
  },
})
