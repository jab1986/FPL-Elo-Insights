# Repository Guidelines

## Project Structure & Module Organization
FPL-Elo-Insights couples a FastAPI + Typer service with a React 19 SPA. Backend route handlers live in `backend/app/routes/`, data models in `backend/app/models/`, and reusable services in `backend/app/services/`. The CLI entry point sits at `backend/cli.py`. Analytics pipelines and helpers reside in `ml/`, with supporting scripts in `ml/analysis/` and tests in `ml/tests/`. Frontend code is colocated under `frontend/src/`, while shared primitives sit in `frontend/src/components/ui/`. Season exports land in `data/` (read-only), with `docs/` and `scripts/` capturing reference guides and maintenance utilities.

## Build, Test, and Development Commands
- `pip install -r backend/requirements.txt` installs backend and ML dependencies.
- `npm install` inside `frontend/` resolves SPA packages; `npm run dev` launches the Vite dev server.
- `python -m backend.cli players top --limit 5` exercises core services without Supabase credentials.
- `pytest`, `npm run lint`, and `npm run build` are the standard pre-handoff verification trio.

## Coding Style & Naming Conventions
Follow PEP 8 spacing, add type hints, and reuse shared helpers (e.g., `DataService`) rather than duplicating queries. Keep Python identifiers snake_case, React files kebab-case, and prefer functional components with strict props. Tailwind utilities are the default styling layer; do not introduce unused dependencies or non-ASCII characters.

## Testing Guidelines
Pytest drives backend and ML coverage; keep tests colocated (e.g., `ml/tests/test_feature.py`) and fixtures deterministic. Frontend quality gates rely on the lint/build pair; add component tests when behaviour or hooks change. Document any intentionally skipped verification command in `CURRENT_WORK_STATUS.md`.

## Commit & Pull Request Guidelines
Write capitalised, imperative subjects (`Add Supabase health probe`). Group related edits, reference linked issues, and capture behaviour changes plus test evidence. PRs should note environment impacts, include screenshots or CLI output for UX shifts, and confirm a clean `git status`.

## Environment & Error Resolution
Set `SUPABASE_URL` and `SUPABASE_KEY` in `.env` and validate connectivity via `/api/health/data`. Keep secrets out of version control; use the shared vault. Consult Context7 MCP before troubleshooting, apply documented fixes first, and log successful resolutions in `docs/troubleshooting.md`.
