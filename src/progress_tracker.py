import json
import time
import redis
from config import Config

class ProgressTracker:
    def __init__(self):
        self.redis_client = redis.Redis.from_url(Config.REDIS_URL)
    
    def update_progress(self, transcription_id, status, progress=None, message=None):
        """Update transcription progress and publish to Redis"""
        progress_data = {
            "id": transcription_id,
            "status": status,  # e.g., "downloading", "transcribing", "diarizing", "completed", "error"
            "progress": progress,  # Numerical progress (0-100) if applicable
            "message": message,    # Optional status message
            "timestamp": time.time()
        }
        
        # Publish to a Redis channel for real-time updates
        self.redis_client.publish(f"transcription:progress:{transcription_id}", json.dumps(progress_data))
        
        # Also store the latest status in a Redis key for clients that connect later
        self.redis_client.set(f"transcription:status:{transcription_id}", json.dumps(progress_data))
        
        # Set a reasonable expiration time (e.g., 24 hours)
        self.redis_client.expire(f"transcription:status:{transcription_id}", 86400)
        
        # Log to console for debugging
        print(f"📊 Progress: {status} {progress}% - {message}")
    
    def track_job_start(self, transcription_id):
        """Mark job as started and track in active jobs"""
        job_start_time = time.time()
        self.redis_client.hset(f"transcription:timing:{transcription_id}", "start_time", job_start_time)
        self.redis_client.sadd("transcription:active_jobs", transcription_id)
        return job_start_time
    
    def track_timing(self, transcription_id, event, timestamp=None):
        """Track specific timing events"""
        if timestamp is None:
            timestamp = time.time()
        self.redis_client.hset(f"transcription:timing:{transcription_id}", event, timestamp)
    
    def complete_job(self, transcription_id, summary_data=None):
        """Mark job as complete and store summary"""
        self.redis_client.srem("transcription:active_jobs", transcription_id)
        self.track_timing(transcription_id, "completion_time")
        
        if summary_data:
            self.redis_client.set(f"transcription:summary:{transcription_id}", json.dumps(summary_data))
            self.redis_client.expire(f"transcription:summary:{transcription_id}", 86400 * 7)  # Keep summary for 7 days
    
    def handle_error(self, transcription_id, error_message):
        """Handle job error tracking"""
        self.redis_client.srem("transcription:active_jobs", transcription_id)
        self.track_timing(transcription_id, "error_time")