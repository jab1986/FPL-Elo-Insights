from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import players, teams, matches, dashboard, user_teams, health
from dotenv import load_dotenv
import os

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Create FastAPI app
app = FastAPI(
    title="FPL Insights API",
    description="Fantasy Premier League data and analytics API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(players.router, prefix="/api", tags=["players"])
app.include_router(teams.router, prefix="/api", tags=["teams"])
app.include_router(matches.router, prefix="/api", tags=["matches"])
app.include_router(dashboard.router, prefix="/api", tags=["dashboard"])
app.include_router(user_teams.router, prefix="/api", tags=["user-teams"])
app.include_router(health.router, prefix="/api", tags=["health"])


@app.get("/")
async def root():
    return {"message": "FPL Insights API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
