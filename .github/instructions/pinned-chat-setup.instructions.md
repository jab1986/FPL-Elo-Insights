---
applyTo: '**'
---

Repo-scoped Copilot Chat setup (pinned)

- Shell and OS: Windows PowerShell (v5.1). Generate PowerShell-native commands. Use ';' to chain when needed. Avoid Bash-only syntax.
- Primary guidance: Always load and follow both of these files:
  - .github/copilot-instructions.md (FPL-Elo-Insights – AI Coding Assistant Guidelines)
  - AGENTS.md (repository-wide conventions: structure, commands, coding style, tests)
- Data source policy: Backend and CLI use live Supabase only. Verify connectivity via /api/health/data. Do not introduce mock fallbacks in routes/CLI.
- Error handling protocol: Consult Context7 documentation first for any error. Prefer official docs and record fixes in docs/troubleshooting.md.
- Troubleshooting: Prioritize docs/troubleshooting.md for backend start, Vite DNS on Windows, ports, and env loading.
- Historical data integrity: Do not retro-edit locked snapshot players.csv/teams.csv in finished GWs. Follow export/delta rules from .github/copilot-instructions.md.
- Workflow rules for this workspace:
  - Use a structured todo list for multi-step work; keep exactly one item in-progress.
  - After 3–5 actions or >3 file edits, provide a concise progress update (delta only).
  - Run commands yourself when appropriate and summarize outcomes; keep messages skimmable.
  - Avoid repeating unchanged plans; reference only deltas.
  - Never invent file paths or APIs—verify via workspace search/read before editing.
- Backend startup (quick reminder): Prefer backend/start_backend.ps1 so cwd is correct; use absolute imports in backend/main.py. Health check: http://localhost:8001/health.
- Frontend startup: npm install then npm run dev in frontend/. For Windows DNS quirks, see vite.config.ts guidance in docs/troubleshooting.md.

This file intentionally references existing guidance rather than duplicating it. If instructions drift, update .github/copilot-instructions.md first, then reconcile here.
