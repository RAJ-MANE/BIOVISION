"""
BioVision FastAPI Application Entrypoint.
Orchestrates REST endpoints for optical scanning, inventory intelligence,
store action dispatch, and evidence ablation studies.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import scan, inventory, actions, evidence
from database.repository import get_db_repository


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    # Ensure database tables exist and seed demo records
    repo = get_db_repository()
    repo.seed_initial_demo_data()
    print("[BioVision API] SQLite database initialized and seeded successfully.")
    yield
    print("[BioVision API] Shutting down service.")


app = FastAPI(
    title="BioVision Intelligence Platform API",
    description="Multimodal RGB + NIR spectral early-warning produce freshness & retail action engine.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dashboards and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(scan.router, prefix="/api/v1")
app.include_router(inventory.router, prefix="/api/v1")
app.include_router(actions.router, prefix="/api/v1")
app.include_router(evidence.router, prefix="/api/v1")


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "BioVision Intelligence Platform API",
        "status": "ONLINE",
        "version": "1.0.0",
        "docs_url": "/docs",
        "description": "Spectral early-warning system for produce degradation prediction",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
