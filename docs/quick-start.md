# Quick Start Guide

Get FPL-Elo-Insights running in 5 minutes! This guide covers the essential steps and common pitfalls to avoid.

## Prerequisites

- Python 3.12+
- Node.js 18+ and npm
- Git

## 1. Clone and Setup (1 minute)

```bash
git clone <repository-url>
cd FPL-Elo-Insights
```

## Quick Start (All-in-One)

For the fastest setup, use the master startup script:

```bash
# From project root directory
.\start_dev_servers.ps1
```

This will:
1. Start the backend server on http://localhost:8001
2. Start the frontend server on http://localhost:5173
3. Open both in separate PowerShell windows

Skip to [Verification Checklist](#verification-checklist) if using this method.

## Manual Setup (Step-by-Step)

## 2. Backend Setup (2 minutes)

### Navigate to backend directory:
```bash
cd backend
```

### Create and activate virtual environment:
```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows
python3 -m venv .venv
.venv\Scripts\activate
```

### Install Python dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### ⚠️ Critical: Verify package structure
Ensure these `__init__.py` files exist:
- `app/__init__.py`
- `app/services/__init__.py`
- `app/routes/__init__.py`
- `app/models/__init__.py`

Create them if missing:
```bash
touch app/services/__init__.py
```

## 3. Start Backend Server (1 minute)

### Option 1: Use the startup script (Recommended)
From the project root directory:
```bash
.\backend\start_backend.ps1
```

### Option 2: Manual startup
Navigate to backend and start manually:
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### ⚠️ Important: Working Directory
The backend **must** be started from the `backend` directory due to import paths. If you get `ModuleNotFoundError`, ensure you're in the correct directory before running uvicorn.

### ✅ Verify it's working:
```bash
curl http://localhost:8001/health
```
Should return: `{"status":"healthy"}`

### Test API endpoint:
```bash
curl http://localhost:8001/api/players/top/5
```
Should return JSON data.

### Test CLI (alternative verification):
```bash
python -m backend.cli players top --limit 5
```

## 4. Frontend Setup (1 minute)

### Open new terminal, navigate to frontend:
```bash
cd frontend
```

### Install Node dependencies:
```bash
npm install
```

### Start development server:
```bash
npm run dev
```

### ⚠️ Windows Users: DNS Resolution Fix
If the Vite server starts but you can't access it in your browser (connection refused/timeout), apply this fix to `frontend/vite.config.ts`:

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
    host: '0.0.0.0', // Bind to all interfaces
    strictPort: false,
    cors: true,
    origin: 'http://127.0.0.1:5173',
  },
})
```

Then restart the dev server: `npm run dev`

### ✅ Verify:
Open http://localhost:5173 in your browser.

## 5. Connect Frontend to Backend

The frontend expects the API at `http://localhost:8001/api`. If you need to change this:

1. Edit `frontend/src/services/api.ts`
2. Update the `baseURL` to match your backend URL

## Common Issues & Quick Fixes

### "Port already in use"
```bash
# Kill existing processes
pkill -f uvicorn

# Or use different port
python -m uvicorn main:app --port 8002 --reload
```

### "Module not found" errors
```bash
# Ensure you're in the backend directory
cd backend

# Check virtual environment is activated
which python  # Should show .venv path

# Verify __init__.py files exist
ls -la app/services/__init__.py
```

### Server starts but doesn't respond
```bash
# Check for multiple processes
ps aux | grep uvicorn

# Test with absolute imports (should be in main.py)
from app.routes import router  # Not: from .app.routes import router
```

### Frontend can't connect to backend
```bash
# Verify backend is running
curl http://localhost:8001/health

# Check CORS in backend/main.py
# Ensure your frontend URL is in allow_origins
```

## Verification Checklist

- [ ] Backend server starts without errors
- [ ] `curl http://localhost:8001/health` returns `{"status":"healthy"}`
- [ ] `curl http://localhost:8001/api/players/top/5` returns data
- [ ] CLI works: `python -m backend.cli players top --limit 5`
- [ ] Frontend builds: `npm run build` succeeds
- [ ] Frontend dev server runs on http://localhost:5173

## Prediction Artifacts

Store generated projection tables in ml/artifacts/predictions/ so they stay versioned alongside the ML pipeline. Copy fresh exports there with PowerShell: Copy-Item "C:/Users/joebr/Documents/ffs/output/tables/*.csv" "ml/artifacts/predictions" -Recurse. Keep filenames dated or descriptive, and document schema changes here so contributors can consume the data reliably.

## Next Steps

1. **With mock data**: You're ready to develop! The system uses sample data by default.
2. **With live data**: Add Supabase credentials to `backend/.env`:
   ```env
   SUPABASE_URL="https://your-project.supabase.co"
   SUPABASE_KEY="your-service-role-key"
   ```

## Getting Help

If you encounter issues:
1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Verify your setup matches this Quick Start guide
3. Test CLI functionality to isolate web server vs core service issues

## Development Commands Summary

```bash
# Backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
python -m backend.cli players top --limit 5

# Frontend
cd frontend
npm run dev
npm run build

# Testing
pytest
npm run lint
```

You're all set! 🎉
