@echo off
REM FastAPI Server Startup Script for Windows
echo Starting FPL Insights FastAPI Server...

REM Set the project directory
set PROJECT_DIR=C:\Users\joebr\FPL-Elo-Insights\backend
set VENV_DIR=%PROJECT_DIR%\venv\Scripts

REM Change to project directory
cd /d %PROJECT_DIR%

REM Set PYTHONPATH
set PYTHONPATH=%PROJECT_DIR%

REM Activate virtual environment and start server
call "%VENV_DIR%\activate.bat"
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload

pause