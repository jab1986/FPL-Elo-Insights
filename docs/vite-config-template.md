# Vite Configuration Template

This document provides the recommended Vite configuration for FPL-Elo-Insights frontend development, particularly addressing Windows DNS resolution issues.

## Recommended vite.config.ts

```typescript
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
    host: '0.0.0.0', // Bind to all network interfaces (Windows compatibility)
    strictPort: false, // Allow port switching if 5173 is in use
    open: false, // Don't auto-open browser to avoid issues
    cors: true, // Enable CORS for backend communication
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
```

## Configuration Explanation

### DNS Resolution Fix
```typescript
import dns from 'node:dns'
dns.setDefaultResultOrder('verbatim')
```
- **Purpose**: Prevents DNS reordering issues on Windows
- **Problem**: Node.js on Windows sometimes reorders DNS results, causing localhost resolution failures
- **Solution**: Forces 'verbatim' order to ensure consistent localhost resolution

### Server Configuration
```typescript
server: {
  port: 5173,
  host: '0.0.0.0',
  strictPort: false,
  open: false,
  cors: true,
  origin: 'http://127.0.0.1:5173',
}
```

- **`host: '0.0.0.0'`**: Binds to all network interfaces, ensuring server is accessible
- **`strictPort: false`**: Allows automatic port switching if 5173 is in use
- **`open: false`**: Prevents auto-opening browser which can cause timing issues
- **`cors: true`**: Enables CORS for backend API communication
- **`origin`**: Sets explicit origin for asset URL generation

### Build Optimization
```typescript
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
}
```

- **`chunkSizeWarningLimit`**: Increases warning threshold for chunk sizes
- **`manualChunks`**: Optimizes bundle splitting for better caching

## Alternative Configurations

### For localhost-only development:
```typescript
server: {
  host: '127.0.0.1', // Localhost only
  port: 5173,
}
```

### For specific port enforcement:
```typescript
server: {
  host: '0.0.0.0',
  port: 5173,
  strictPort: true, // Fail if port is in use
}
```

### Without DNS fix (if not on Windows):
```typescript
// Remove these lines if not needed:
// import dns from 'node:dns'
// dns.setDefaultResultOrder('verbatim')
```

## Troubleshooting

### TypeScript errors about 'node:dns'
This is expected - the import only exists at runtime, not compile time. The error can be ignored.

### Server still unreachable after applying config
1. Restart the development server completely
2. Check Windows Firewall settings
3. Try accessing via the specific IP addresses shown in Vite's startup message
4. Verify Node.js version is 16+ (required for DNS fixes)

### Port conflicts
If you consistently need a different port:
```typescript
server: {
  port: 3000, // or any available port
  // ... other config
}
```

## Testing the Configuration

After applying this configuration, verify it works:

1. **Start the server:**
   ```bash
   npm run dev
   ```

2. **Check startup message shows multiple interfaces:**
   ```
   ➜  Local:   http://localhost:5173/
   ➜  Network: http://192.168.x.x:5173/
   ```

3. **Test accessibility:**
   ```bash
   curl http://localhost:5173
   # Should return HTML, not connection error
   ```

4. **Verify in browser:**
   - Open http://localhost:5173
   - Should load without connection errors
