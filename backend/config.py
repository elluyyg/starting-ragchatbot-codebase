import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


@dataclass
class Config:
    """Configuration settings for the RAG system"""

    # Anthropic API settings (DeepSeek compatible via ANTHROPIC_BASE_URL)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "") or ""
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "") or "claude-sonnet-4-20250514"
    ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "") or ""

    # Ollama embedding settings
    OLLAMA_EMBEDDING_URL: str = os.getenv("OLLAMA_EMBEDDING_URL", "") or "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "") or "nomic-embed-text"

    # Document processing settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2

    # Database paths
    CHROMA_PATH: str = "./chroma_db"


config = Config()
