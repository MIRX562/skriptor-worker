import os
from dotenv import load_dotenv

# Load .env from same directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

class Config:
    # Transcription Provider (groq or local)
    TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "groq")
    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # PostgreSQL Configuration
    POSTGRES_URL = os.getenv("PG_URL", "postgresql://user:pass@localhost:5432/db")
    
    # MinIO Configuration
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "audio")
    
    # API Keys
    HF_TOKEN = os.getenv("HF_TOKEN")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Queue Configuration
    QUEUE_NAME = "transcription:queue"