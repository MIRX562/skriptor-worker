from pydantic import BaseSettings

class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    postgres_dsn: str
    whisperx_api_url: str  # if whisperx is standalone server, else local execution
    whisperx_api_token: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()
