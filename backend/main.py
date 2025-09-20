from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import players, teams, matches, dashboard
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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


@app.get("/")
async def root():
    return {"message": "FPL Insights API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}