# Troubleshooting Guide

This guide addresses common issues encountered during development and deployment of FPL-Elo-Insights, based on real problems that have been resolved.

## Backend Server Issues

### Server Not Starting or Responding

#### Problem: Multiple uvicorn processes conflicting on port 8001
**Symptoms:** Server appears to start but doesn't respond to requests, or port already in use errors.

**Solution:**
1. Check for running processes:
   ```bash
   # On Linux/Mac
   ps aux | grep uvicorn
   # On Windows
   netstat -ano | findstr :8001
   ```

2. Kill existing processes:
   ```bash
   # On Linux/Mac
   pkill -f uvicorn
   # On Windows
   taskkill /PID <PID> /F
   ```

3. Verify port is free:
   ```bash
   curl http://localhost:8001/health
   ```

#### Problem: Import path errors with FastAPI
**Symptoms:** `ModuleNotFoundError` or `ImportError` when starting the server.

**Root Cause:** Using relative imports in `backend/main.py` that don't work with `python -m uvicorn main:app`.

**Solution:**
- Use absolute imports in `backend/main.py`:
  ```python
  # ❌ Wrong - relative imports don't work with -m flag
  from .app.routes import router

  # ✅ Correct - absolute imports
  from app.routes import router
  ```

#### Problem: "Could not import module 'main'" error
**Symptoms:** `ERROR: Error loading ASGI app. Could not import module "main".`

**Root Cause:** Working directory issue - uvicorn is not running from the backend directory.

**Solution:**

**Option 1 - Use the provided startup script:**
```bash
# From project root directory
.\backend\start_backend.ps1
```

**Option 2 - Manual command with correct directory:**
```bash
# Ensure you're in the backend directory first
cd C:\Users\joebr\FPL-Elo-Insights\backend
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**Option 3 - Use Start-Process to ensure directory persistence:**
```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location 'C:\Users\joebr\FPL-Elo-Insights\backend'; python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload"
```

#### Problem: Missing package structure
**Symptoms:** `ModuleNotFoundError: No module named 'app'` or similar package import errors.

**Root Cause:** Missing `__init__.py` files in Python packages.

**Solution:**
Ensure these files exist:
- `backend/app/__init__.py`
- `backend/app/services/__init__.py`
- `backend/app/routes/__init__.py`
- `backend/app/models/__init__.py`

Create them if missing:
```bash
touch backend/app/services/__init__.py
```

#### Problem: Server unresponsive to HTTP requests
**Symptoms:** Server starts but doesn't respond to API calls.

**Solution:**
1. Check server startup logs for errors
2. Verify CORS configuration in `backend/main.py`
3. Test health endpoint directly:
   ```bash
   curl http://localhost:8001/health
   ```
4. Check if using correct host/port combination

### Port and Network Issues

#### Problem: Port 8001 already in use
**Solution:**
- Use a different port: `uvicorn main:app --port 8002`
- Or kill the conflicting process (see above)

#### Problem: Connection refused
**Solution:**
1. Verify server is running: `ps aux | grep uvicorn`
2. Check if using correct host (0.0.0.0 for external access, 127.0.0.1 for local only)
3. Test with different ports

## Development Environment Setup

### Python Environment Issues

#### Problem: Dependency conflicts
**Solution:**
1. Use virtual environment:
   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate on Windows
   ```

2. Reinstall requirements:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

#### Problem: Python version issues
**Required:** Python 3.12+
**Solution:**
- Check version: `python --version`
- Use pyenv or conda to manage Python versions
- Update alternatives if using system Python

### Frontend Issues

#### Problem: npm install fails

**Solution:**

1. Clear npm cache: `npm cache clean --force`
2. Delete node_modules: `rm -rf node_modules`
3. Delete package-lock.json: `rm package-lock.json`
4. Reinstall: `npm install`

#### Problem: Vite development server starts but is unreachable (Windows)

**Symptoms:**

- Vite shows "ready" message and lists URLs
- HTTP requests to localhost fail with "Unable to connect to the remote server"
- Browser cannot access the development server

**Root Cause:** Windows DNS resolution issues with Node.js/Vite development server.

**Solution:**

1. **Fix DNS resolution in `frontend/vite.config.ts`:**

   ```typescript
   import { defineConfig } from 'vite'
   import react from '@vitejs/plugin-react'
   import dns from 'node:dns'

   // Prevent DNS reordering for localhost on Windows (fixes connection issues)
   dns.setDefaultResultOrder('verbatim')

   export default defineConfig({
     plugins: [react()],
     server: {
       port: 5173,
       host: '0.0.0.0', // Bind to all interfaces on Windows
       strictPort: false,
       cors: true,
       origin: 'http://127.0.0.1:5173', // Set explicit origin for asset URLs
     },
   })
   ```

2. **Restart the development server:**

   ```bash
   cd frontend
   npm run dev
   ```

3. **Verify server is accessible:**
   - Server should show multiple network interfaces when starting
   - Test with: `curl http://localhost:5173` or access in browser

**Alternative solutions if the above doesn't work:**

- Try `host: '127.0.0.1'` instead of `'0.0.0.0'`
- Use explicit IP addresses shown in Vite startup message
- Check Windows firewall settings
- Verify Node.js version compatibility (requires Node 16+)

#### Problem: Frontend not connecting to backend

**Solution:**

1. Verify backend is running on correct port
2. Check API base URL in `frontend/src/services/api.ts`
3. Ensure CORS is configured correctly in backend

## Database and Data Issues

### Supabase Connection Problems

#### Problem: Cannot connect to Supabase
**Symptoms:** API returns mock data instead of live data.

**Solution:**
1. Check environment variables:
   ```bash
   echo $SUPABASE_URL
   echo $SUPABASE_KEY
   ```

2. Verify credentials are set correctly in `backend/.env`:
   ```env
   SUPABASE_URL="https://your-project.supabase.co"
   SUPABASE_KEY="your-service-role-key"
   ```

3. Test connection manually:
   ```python
   from supabase import create_client
   supabase = create_client(url, key)
   ```

**Note:** Without Supabase credentials, the system gracefully falls back to mock data.

#### Problem: .env is present but backend doesn't see it
**Symptoms:** You see "Warning: SUPABASE_URL and SUPABASE_KEY not found. Using mock data for development." in backend logs even though `.env` exists at repo root.

**Root Cause:** The backend may start from a working directory where default dotenv discovery doesn't reach the repo root before DataService initializes.

**Solution:**
- The backend now loads `.env` robustly from repo root via `DataService` using an explicit path and discovery fallback. To validate:
  1. Run the test script:
     ```bash
     cd backend
     python test_env.py
     ```
     You should see:
     - `.env file exists: True`
     - `✅ Supabase credentials loaded successfully!`
  2. Start backend with the provided script so working directory is correct:
     ```powershell
     .\backend\start_backend.ps1
     ```
     You should see a log line like:
     ```
     ✅ Supabase credentials found. Connecting to: https://<your>.supabase.co
     ```
  3. If you still see the warning, confirm `.env` is at repo root (same folder as `backend/`, `frontend/`) and not inside `backend/`.

### Mock Data Issues

#### Problem: CLI commands not working
**Solution:**
1. Verify CLI installation: `python -m backend.cli --help`
2. Check if in backend directory: `cd backend`
3. Test basic command: `python -m backend.cli players top --limit 5`

## Verification Commands

### Backend Health Check
```bash
# Test server health
curl http://localhost:8001/health

# Test API endpoint
curl http://localhost:8001/api/players/top/5

# Test CLI
python -m backend.cli players top --limit 5
```

### Process Management
```bash
# Check running processes
ps aux | grep uvicorn

# Check port usage
netstat -tlnp | grep :8001

# Kill processes by port
fuser -k 8001/tcp
```

## Common Error Messages and Solutions

### "Address already in use"
- Kill existing process or use different port
- Check for multiple uvicorn instances

### "Module not found"
- Verify __init__.py files exist in all packages
- Check import paths are absolute, not relative
- Ensure virtual environment is activated

### "Connection refused"
- Verify server is running and on correct port
- Check firewall settings
- Ensure using correct host (0.0.0.0 vs 127.0.0.1)

### "CORS error"
- Update allowed origins in `backend/main.py`
- Check frontend API base URL configuration

## Getting Help

1. Check this troubleshooting guide first
2. Review server logs for specific error messages
3. Test with CLI commands to isolate web server vs core service issues
4. Verify environment setup matches the main README.md requirements

If issues persist, check the current work status in `CURRENT_WORK_STATUS.md` for known issues or recent fixes.
