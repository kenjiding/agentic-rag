import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    // 支持 SPA 路由：所有路径都重定向到 index.html
    // 这样 /chat/xxx 路径在开发环境下也能正常工作
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  // 构建时也支持 SPA 路���
  build: {
    rollupOptions: {
      output: {
        // 手动指定 chunk 分割策略
        manualChunks: {},
      },
    },
  },
})

