# FPL-Elo-Insights Frontend

React + TypeScript + Vite frontend for FPL-Elo-Insights, providing an interactive dashboard for Fantasy Premier League data analysis.

## Quick Start

```bash
npm install
npm run dev
```

The development server will start on http://localhost:5173

## Development Server Troubleshooting

### Windows DNS Issues

If the Vite server starts but you can't access it in your browser, you may be experiencing Windows DNS resolution issues. Apply this fix:

**Problem symptoms:**
- Vite shows "ready" message with URLs
- Browser shows "connection refused" or timeouts
- `curl http://localhost:5173` fails

**Solution:**

1. **Update `vite.config.ts`:**

   ```typescript
   import { defineConfig } from 'vite'
   import react from '@vitejs/plugin-react'
   import dns from 'node:dns'

   // Fix Windows DNS resolution issues
   dns.setDefaultResultOrder('verbatim')

   export default defineConfig({
     plugins: [react()],
     server: {
       port: 5173,
       host: '0.0.0.0', // Bind to all network interfaces
       strictPort: false,
       cors: true,
       origin: 'http://127.0.0.1:5173',
     },
   })
   ```

2. **Restart the development server:**

   ```bash
   npm run dev
   ```

3. **Verify the fix:**
   - Server should show multiple network interfaces when starting
   - Access via browser or: `curl http://localhost:5173`

**Alternative approaches:**
- Try `host: '127.0.0.1'` instead of `'0.0.0.0'`
- Use the specific IP addresses shown in Vite's startup message
- Check Windows Firewall settings

### Other Common Issues

#### Port conflicts
```bash
# If port 5173 is in use
npm run dev -- --port 5174
```

#### Build errors
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### API connection issues
Check `src/services/api.ts` - ensure the `API_BASE_URL` matches your backend server:
```typescript
const API_BASE_URL = 'http://localhost:8001/api'
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Architecture

- **Vite**: Build tool and development server
- **React 19**: UI framework with modern features
- **TypeScript**: Type safety and developer experience
- **Tailwind CSS**: Utility-first styling
- **React Router**: Client-side routing
- **Recharts**: Data visualization components

## Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/         # Page components
├── services/      # API communication
├── types/         # TypeScript type definitions
├── hooks/         # Custom React hooks
└── assets/        # Static assets
```

## Backend Integration

The frontend communicates with the FastAPI backend via REST API. Key endpoints:

- `/api/players/top/{limit}` - Top players
- `/api/teams` - Team information
- `/api/health` - Health check

API configuration is in `src/services/api.ts`.

## Development Notes

### ESLint Configuration

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

#### Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
