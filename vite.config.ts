import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['@xenova/transformers', 'onnxruntime-web'],
  },
  resolve: {
    alias: {
      'onnxruntime-web': 'onnxruntime-web/dist/ort-web.min.js',
    },
  },
  define: {
    'global': 'window',
  },
  build: {
    target: 'esnext',
  },
})
