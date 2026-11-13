import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes.documents import router as documents_router
from app.routes.chat import router as chat_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("documind")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="DocuMind Backend: AI-Powered PDF Question Answering System with Hybrid RAG (BM25 + Semantic Search + RRF + Gemini)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Streamlit frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/", tags=["General"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint to verify backend service availability."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

