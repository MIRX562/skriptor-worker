import json
import time
import redis
from config import Config
from job_processor import JobProcessor

def main():
    """Main worker loop"""
    # Initialize Redis connection
    redis_client = redis.Redis.from_url(Config.REDIS_URL)
    
    # Initialize job processor
    processor = JobProcessor()
    
    print("👂 Worker started")
    
    while True:
        try:
            # Wait for jobs from the queue
            job = redis_client.brpop(Config.QUEUE_NAME, timeout=0)
            if not job:
                continue
            
            # Parse job data
            _, raw = job
            job_data = json.loads(raw.decode())
            
            # Process the transcription job
            processor.process_transcription_job(job_data)
            
        except KeyboardInterrupt:
            print("\n🛑 Worker stopped by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error in main loop: {e}")
            print("🔄 Restarting worker loop after error...")
            time.sleep(2)
            continue

if __name__ == "__main__":
    main()