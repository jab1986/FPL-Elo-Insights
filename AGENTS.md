# FPL-Elo-Insights – Agent Handbook

These guidelines apply to the entire repository. Follow them alongside the system instructions in every task.

## 1. Repository map
- **backend/** – FastAPI service plus a Typer-based CLI that proxies the same data access layer. Supabase provides real data, but the services fall back to the rich samples in `backend/app/services/mock_data.py` when credentials are missing.
- **frontend/** – React 19 + TypeScript + Vite single-page application styled with Tailwind utilities and using @tanstack/react-query for data fetching.
- **ml/** – Standalone machine-learning pipeline (NumPy/Pandas) with pytest coverage. Artifacts are written to `ml/artifacts/` by default.
- **data/** – Season CSV exports. Treat these as source data; avoid committing regenerated large files unless explicitly requested.
- **docs/** – Long-form documentation and project plans.
- **scripts/** – Utility scripts for data ingestion/maintenance.

## 2. Environment & dependencies
- Python tooling targets **Python 3.12**. Install dependencies with `pip install -r backend/requirements.txt`; this also brings in the ML stack (pandas, numpy). Add any new Python dependencies to this file.
- Frontend tooling expects **Node 20+**. Install packages with `npm install` inside `frontend/`.
- Supabase access uses the `SUPABASE_URL` and `SUPABASE_KEY` environment variables. When they are absent the backend/CLI should continue to work thanks to the mock data; do not attempt to hard-code credentials.

## 3. Required verification before delivering changes
Always run the relevant checks after modifying code or configuration:
- `pytest` from the repository root (covers the ML pipeline tests). If you only change frontend assets this run is still cheap; skip it only when you are certain no Python code is affected.
- Frontend changes: `npm run lint` and `npm run build` inside `frontend/`.
- If you alter FastAPI routes/services or CLI behaviour, exercise the affected command quickly (e.g. `python -m backend.cli players top --limit 5`) to ensure the mock-mode still works.
- Document any skipped check in the final report with a justification.

## 4. Coding guidelines
### Python (backend/ml)
- Follow PEP 8 spacing and naming. Prefer type hints (already prevalent in `backend/app/services` and `ml/` modules).
- Keep functions pure where possible. Separate I/O concerns (Supabase calls, file writes) from computation to ease testing.
- Reuse existing helpers (`DataService`, `engineer_features`, etc.) instead of duplicating logic. Extend dataclasses/enums cautiously to avoid breaking JSON serialisation to the frontend.
- When adding CLI commands use Typer patterns present in `backend/cli.py` for consistency.

### TypeScript/React
- The app uses function components with hooks; avoid class components.
- Compose UI with the Tailwind-friendly primitives in `frontend/src/components/ui/`. When new shared UI elements are needed, colocate them under `components/ui/` and keep props strongly typed.
- Keep data fetching inside dedicated hooks (see `frontend/src/hooks/useFPLData.ts`). Prefer react-query mutations/queries rather than manual `fetch` calls in components.
- Preserve routing conventions defined in `App.tsx` and `Layout.tsx` and ensure new pages register with the navigation array when appropriate.

## 5. Data handling
- The data directory contains large CSVs; do not rename/move them unless the task requires it. If you must generate new season snapshots, compress or prune them before committing.
- Backend endpoints and ML pipeline assume consistent schema names (`players`, `matches`, `player_gameweek_stats`, etc.). When extending schemas, update both the FastAPI services and the frontend/ML consumers in the same change and document the new fields.

## 6. Documentation & artefacts
- Update README files or inline docstrings when behaviour changes. The top-level README advertises CLI commands and ML instructions—keep them truthful.
- ML experiments persist outputs under `ml/artifacts/`. These files can be ignored for git commits unless the task explicitly asks to surface them.

## 7. Delivery checklist
Before finishing a task:
1. Ensure all modified files are formatted and typed consistently with neighbouring code.
2. Run the mandatory checks (section 3) and capture their outputs for the final report.
3. Verify `git status` is clean after committing.
4. Provide a clear summary of code and documentation changes referencing the relevant paths.

