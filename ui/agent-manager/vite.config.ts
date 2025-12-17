import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  server: {
    port: 5173,
    strictPort: true,
    host: '127.0.0.1',

    // Proxy API requests to HoloLoom backend
    proxy: {
      // Main HoloLoom API (port 8000)
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
      // Health check endpoint (direct, no /api prefix)
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Agent Manager API (port 8002)
      // Frontend calls /agent-api/* → proxied to localhost:8002/api/*
      '/agent-api': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/agent-api/, '/api'),
        ws: true,
      },
      // Agent Manager health check
      '/agent-health': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        rewrite: () => '/health',
      },
    },
  },

  build: {
    outDir: 'dist',
    sourcemap: true,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
      },
    },
    rollupOptions: {
      output: {
        manualChunks: {
          // Separate vendor chunks for better caching
          'vendor-react': ['react', 'react-dom'],
          'vendor-state': ['zustand'],
        },
      },
    },
  },

  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@stores': path.resolve(__dirname, './src/stores'),
      '@types': path.resolve(__dirname, './src/types'),
      '@utils': path.resolve(__dirname, './src/utils'),
    },
  },

  preview: {
    port: 5173,
    strictPort: true,
    host: '127.0.0.1',
  },

  // Performance optimizations
  define: {
    __DEV__: JSON.stringify(process.env.NODE_ENV === 'development'),
    __PROD__: JSON.stringify(process.env.NODE_ENV === 'production'),
  },
})
