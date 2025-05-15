import asyncio
import logging
from redis.asyncio import Redis
from minio import Minio
import aiohttp
import asyncpg
from config import settings

logging.basicConfig(level=logging.INFO)

redis = Redis.from_url(settings.redis_url)
minio_client = Minio(
    endpoint=settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=False,
)
db_pool: asyncpg.Pool = None

async def connect_db():
    global db_pool
    db_pool = await asyncpg.create_pool(settings.postgres_dsn)

async def fetch_job():
    # Listen or poll your Redis queue, e.g. BRPOP or BullMQ equivalent
    # Placeholder example: BRPOP 'transcription_jobs' queue with 5 sec timeout
    job = await redis.brpop("transcription_jobs", timeout=5)
    return job

async def download_audio(filename: str, local_path: str):
    try:
        minio_client.fget_object(settings.minio_bucket, filename, local_path)
    except Exception as e:
        logging.error(f"Failed to download {filename}: {e}")
        raise

async def upload_transcription_result(id: str, transcription_text: str):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE transcriptions SET status='completed' WHERE id=$1
        """, id)
        # Save transcription text/segments to DB as needed

async def process_job(job_data: dict):
    id = job_data["id"]
    filename = job_data["filename"]
    logging.info(f"Processing transcription job {id} for {filename}")

    local_file = f"/tmp/{filename}"
    await download_audio(filename, local_file)

    # Call whisperx (via API or CLI)
    transcription_text = await call_whisperx(local_file)

    await upload_transcription_result(id, transcription_text)

async def call_whisperx(filepath: str) -> str:
    # Example using REST API of whisperx server
    async with aiohttp.ClientSession() as session:
        with open(filepath, "rb") as f:
            data = {"file": f}
            async with session.post(f"{settings.whisperx_api_url}/transcribe", data=data) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result["text"]

async def worker_loop():
    await connect_db()
    while True:
        job = await fetch_job()
        if not job:
            continue
        # job format example: ['transcription_jobs', '{"id":"uuid", "filename":"file.mp3"}']
        _, raw = job
        job_data = json.loads(raw)
        try:
            await process_job(job_data)
        except Exception as e:
            logging.error(f"Error processing job {job_data['id']}: {e}")
            # Update status to failed in DB, etc.

if __name__ == "__main__":
    asyncio.run(worker_loop())
