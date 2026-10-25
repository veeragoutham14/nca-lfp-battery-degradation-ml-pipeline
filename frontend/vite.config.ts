import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Keep large third-party libraries in their own long-lived chunks so the
// application chunk stays small and vendor code can be cached across deploys.
function vendorChunk(id: string): string | undefined {
  if (!/[\\/]node_modules[\\/]/.test(id)) return undefined
  if (/[\\/]node_modules[\\/]zrender[\\/]/.test(id)) return 'vendor-zrender'
  if (/[\\/]node_modules[\\/](echarts|echarts-for-react)[\\/]/.test(id)) return 'vendor-echarts'
  if (/[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return 'vendor-react'
  if (/[\\/]node_modules[\\/]lucide-react[\\/]/.test(id)) return 'vendor-icons'
  return 'vendor'
}

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: vendorChunk,
      },
    },
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
