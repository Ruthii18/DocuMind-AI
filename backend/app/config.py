import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
ROOT_DIR = BASE_DIR.parent  # DocuMind/

ENV_PATHS = [
    BASE_DIR / ".env",
    ROOT_DIR / ".env",
    Path(".env"),
]
# Select the first .env file that exists, or default to backend/.env
active_env_file = next((str(p) for p in ENV_PATHS if p.exists()), str(BASE_DIR / ".env"))


class Settings(BaseSettings):
    # API & Server Configuration
    APP_NAME: str = "DocuMind Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Gemini AI Configuration
    GEMINI_API_KEY: str = Field(default="", validation_alias="GEMINI_API_KEY")
    LLM_MODEL: str = "gemini-1.5-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 4
    RRF_K: int = 60
    
    # Storage Configuration
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma")
    
    model_config = SettingsConfigDict(
        env_file=active_env_file,
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure data directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
