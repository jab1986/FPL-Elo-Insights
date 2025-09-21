# DigitalOcean Deployment Guide

This document walks through setting up the FPL Elo Insights stack (FastAPI backend and Vite frontend) on a DigitalOcean droplet.

---

## 1. Prepare the droplet

1. Provision an Ubuntu droplet (2 GB RAM comfortably supports the stack).
2. Update packages and install core tooling:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y build-essential git python3.12 python3.12-venv python3-pip nodejs npm nginx
   ```
   > Install Node 20 via NodeSource or nvm if the Ubuntu repository version is older.

---

## 2. Fetch the project

```bash
cd /opt
sudo git clone https://github.com/your-account/FPL-Elo-Insights.git
sudo chown -R $USER:$USER FPL-Elo-Insights
cd FPL-Elo-Insights
```

---

## 3. Backend environment

1. Create a virtual environment and install Python dependencies:
   ```bash
   cd backend
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. Create `backend/.env` with Supabase credentials:
   ```env
   SUPABASE_URL="https://<project>.supabase.co"
   SUPABASE_KEY="<service_role_or_anon_key>"
   ```
   The services automatically load these values via `python-dotenv`; if they are missing, the API and CLI fall back to bundled mock data instead of failing hard.

---

## 4. Supabase setup (for live data)

1. In the Supabase dashboard, run `backend/schema.sql` once (SQL Editor → paste → Run) to create tables such as `teams`, `players`, `matches`, `playermatchstats`, and `gameweek_summaries`.
2. Populate with fresh FPL data whenever needed:
   ```bash
   source .venv/bin/activate
   python populate_database.py
   ```
   The script clears existing rows, pulls the official FPL API, and repopulates all tables. Schedule it via cron or a systemd timer to keep data fresh.

---

## 5. Bring up the FastAPI service

1. The Uvicorn entry point is `backend/main.py`, which registers routers under the `/api` prefix and exposes a `/health` endpoint.
2. Start the API (adjust host/port as needed):
   ```bash
   cd backend
   source .venv/bin/activate
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
3. For production, create a `systemd` unit (e.g., `/etc/systemd/system/fpl-api.service`) that runs the above command with `User=fpl`, `Group=fpl`, and `EnvironmentFile=/opt/FPL-Elo-Insights/backend/.env`.
4. Update CORS in `backend/main.py` so your public domain/IP is listed in `allow_origins`.
5. Confirm health:
   ```bash
   curl http://localhost:8000/health
   ```
   This should return `{"status":"healthy"}`.

---

## 6. Validate data access

Use the Typer CLI to verify Supabase connectivity and data integrity:

```bash
python -m backend.cli players top --limit 10
python -m backend.cli dashboard stats
```

The CLI mirrors the API; when Supabase is unreachable, both routes and commands automatically fall back to the offline dataset.

---

## 7. Frontend build

1. Install Node dependencies and create a production build:
   ```bash
   cd /opt/FPL-Elo-Insights/frontend
   npm install
   npm run build
   ```
   (`npm run build` executes `tsc -b` followed by `vite build`.)
2. The API base URL defaults to `http://localhost:8000/api`. Before building, edit `frontend/src/services/api.ts` to point at your droplet domain (e.g., `https://example.com/api`) or refactor to consume an environment variable for flexibility.
3. Production assets are emitted to `frontend/dist/`. Deploy the entire `dist` directory to your web root.

---

## 8. Serve the stack behind Nginx

1. Copy or symlink the built frontend to `/var/www/fpl`.
2. Create `/etc/nginx/sites-available/fpl`:
   ```nginx
   server {
     listen 80;
     server_name your_domain_or_ip;

     root /var/www/fpl;
     index index.html;

     location / {
       try_files $uri /index.html;
     }

     location /api/ {
       proxy_pass http://127.0.0.1:8000/api/;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
       proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       proxy_set_header X-Forwarded-Proto $scheme;
     }
   }
   ```
3. Enable the site, test, and reload:
   ```bash
   sudo ln -s /etc/nginx/sites-available/fpl /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

---

## 9. Background jobs & maintenance

* Re-run `python populate_database.py` periodically to refresh Supabase data.
* Rebuild the frontend after changes to `frontend/src/`, then redeploy the new `dist`.
* Keep dependencies current (`pip install -r backend/requirements.txt` and `npm install`) after pulling updates.
* Use the CLI commands or `/health` endpoint after updates to ensure everything remains connected.

---

## 10. User and permission guidance

For production workloads sharing a droplet, create a dedicated Unix user for this project to isolate file permissions and environment variables:

```bash
sudo adduser --system --group --home /opt/fpl fpl
sudo chown -R fpl:fpl /opt/FPL-Elo-Insights
```

Run deployment commands as that user (e.g., `sudo -u fpl bash`). In a `systemd` unit, set `User=fpl` and combine with hardening directives such as `ProtectSystem` or `ProtectHome` if desired.

