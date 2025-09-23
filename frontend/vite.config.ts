import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import dns from 'node:dns'

// Prevent DNS reordering for localhost on Windows (fixes connection issues)
dns.setDefaultResultOrder('verbatim')

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0', // Bind to all interfaces on Windows
    strictPort: false, // Allow port switching if 5173 is in use
    open: false, // Don't auto-open browser to avoid issues
    cors: true, // Enable CORS
    origin: 'http://127.0.0.1:5173', // Set explicit origin for asset URLs
  },
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
          ui: ['lucide-react', 'recharts'],
        },
      },
    },
  },
})
