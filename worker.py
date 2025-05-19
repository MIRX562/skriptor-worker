import os
import json
import time
import redis
import psycopg
import whisperx

from minio import Minio
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

# Load .env from same directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ENV
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("PG_URL", "postgresql://user:pass@localhost:5432/db")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "audio")
HF_TOKEN = os.getenv("HF_TOKEN")

# Progress tracking function
def update_progress(transcription_id, status, progress=None, message=None):
    progress_data = {
        "id": transcription_id,
        "status": status,  # e.g., "downloading", "transcribing", "diarizing", "completed", "error"
        "progress": progress,  # Numerical progress (0-100) if applicable
        "message": message,    # Optional status message
        "timestamp": time.time()
    }
    # Publish to a Redis channel for real-time updates
    r.publish(f"transcription:progress:{transcription_id}", json.dumps(progress_data))
    # Also store the latest status in a Redis key for clients that connect later
    r.set(f"transcription:status:{transcription_id}", json.dumps(progress_data))
    # Set a reasonable expiration time (e.g., 24 hours)
    r.expire(f"transcription:status:{transcription_id}", 86400)
    # Log to console for debugging
    print(f"📊 Progress: {status} {progress}% - {message}")

# Get audio duration helper function (requires ffprobe/ffmpeg)
def get_audio_duration(file_path):
    try:
        import subprocess
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
               "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        output = subprocess.check_output(cmd).decode('utf-8').strip()
        return float(output)
    except Exception as e:
        print(f"⚠️ Could not determine audio duration: {e}")
        return 0  # Return 0 if duration can't be determined

# Redis
r = redis.Redis.from_url(REDIS_URL)

# MinIO
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

# PostgreSQL
pg = psycopg.connect(POSTGRES_URL)
pg.autocommit = True
cursor = pg.cursor()

print("👂 Worker started")

queue_name = "transcription:queue"

while True:
    try:
        job = r.brpop(queue_name, timeout=0)
        if not job:
            continue

        _, raw = job
        job_json = json.loads(raw.decode())

        transcription_id = job_json.get("transcriptionId") or job_json.get("id")
        filename = job_json["filename"]
        language = job_json.get("language") or None

        print(f"📥 Job received: {filename}")
        
        # Record job start time for timing metrics
        job_start_time = time.time()
        r.hset(f"transcription:timing:{transcription_id}", "start_time", job_start_time)
        
        # Update initial progress
        update_progress(transcription_id, "started", 0, f"Starting transcription of {filename}")
        
        # Track this as an active job
        r.sadd("transcription:active_jobs", transcription_id)

        # Download from MinIO
        update_progress(transcription_id, "downloading", 5, f"Downloading audio file {filename}")
        download_start = time.time()
        
        tmpfile = NamedTemporaryFile(suffix=os.path.splitext(filename)[-1], delete=False)
        tmpfile.close()
        minio_client.fget_object(MINIO_BUCKET, filename, tmpfile.name)
        local_audio_path = tmpfile.name
        
        file_size_mb = os.path.getsize(local_audio_path) / (1024 * 1024)
        download_time = time.time() - download_start
        r.hset(f"transcription:timing:{transcription_id}", "download_complete", time.time())
        
        print(f"🔊 Audio downloaded to {local_audio_path}")
        update_progress(transcription_id, "processing", 10, 
                       f"Audio file downloaded ({file_size_mb:.2f} MB in {download_time:.1f}s)")

        # Get audio duration for better progress estimates
        audio_duration = get_audio_duration(local_audio_path)
        estimated_total_time = max(30, audio_duration * 0.5)  # Rough estimate based on audio length
        update_progress(transcription_id, "analyzing", 15, 
                       f"Analyzing audio file ({audio_duration:.1f} seconds)")
        
        # Load model
        model_start = time.time()
        update_progress(transcription_id, "loading_model", 20, "Loading WhisperX transcription model")
        
        model = whisperx.load_model(
            "large-v3",                  # Model size (positional argument)
            device="cpu",                # "cpu" or "cuda"
            compute_type="int8",         # "float16", "int8", "default", etc.
        )
        audio = whisperx.load_audio(local_audio_path)
        
        model_time = time.time() - model_start
        r.hset(f"transcription:timing:{transcription_id}", "model_loaded", time.time())
        update_progress(transcription_id, "transcribing", 25, 
                       f"Beginning speech-to-text transcription (model loaded in {model_time:.1f}s)")

        # Transcription progress monitoring
        transcribe_start = time.time()
        
        # If WhisperX supports progress callbacks, you would use them here
        # Since it doesn't have built-in callbacks, we'll simulate progress updates
        def progress_callback_simulation():
            # Simulate progress during transcription
            expected_duration = min(600, max(60, audio_duration * 0.3))  # Estimate transcription time
            start = time.time()
            while time.time() - start < expected_duration:
                elapsed = time.time() - start
                progress_pct = min(95, int(25 + (elapsed / expected_duration) * 40))
                
                # Only update if transcription is still running
                if time.time() - transcribe_start < expected_duration * 1.5:
                    update_progress(
                        transcription_id, 
                        "transcribing", 
                        progress_pct,
                        f"Transcribing audio ({progress_pct}% complete, {elapsed:.1f}s elapsed)"
                    )
                    time.sleep(3)  # Update every 3 seconds
                else:
                    break
        
        # Start progress simulation in a separate thread if needed
        # In production, you'd use a proper thread
        # import threading
        # progress_thread = threading.Thread(target=progress_callback_simulation)
        # progress_thread.daemon = True
        # progress_thread.start()
        
        # Actual transcription
        result = model.transcribe(
            audio,
            batch_size=16,                   # Number of audio chunks per batch
            language=language,               # Language code (None = auto)
        )
        
        transcribe_time = time.time() - transcribe_start
        r.hset(f"transcription:timing:{transcription_id}", "transcription_complete", time.time())
        detected_language = result.get("language", "unknown")
        
        update_progress(transcription_id, "post_processing", 65, 
                       f"Transcription complete in {transcribe_time:.1f}s ({detected_language})")

        # Diarization (speaker identification)
        update_progress(transcription_id, "diarizing", 70, "Identifying speakers in audio")
        diarize_start = time.time()
        
        try:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device="cpu")
            diarize_segments = diarize_model(audio)
            # diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=4)  # Optionally set speaker count
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Custom segment splitting based on speaker changes
            update_progress(transcription_id, "processing", 80, "Creating speaker-based segments")
            original_segments = result["segments"]
            new_segments = []
            current_speaker = None

            for segment in original_segments:
                words = segment.get("words", [])
                current_chunk = {"text": "", "start": None, "end": None, "speaker": None}
                
                for word in words:
                    speaker = word.get("speaker")
                    
                    # Start new segment when speaker changes
                    if current_speaker is not None and speaker != current_speaker:
                        if current_chunk["text"]:
                            new_segments.append(current_chunk)
                        current_chunk = {"text": "", "start": None, "end": None, "speaker": None}
                    
                    # Add word to current chunk
                    if current_chunk["start"] is None:
                        current_chunk["start"] = word["start"]
                    current_chunk["end"] = word["end"]
                    current_chunk["speaker"] = speaker
                    current_chunk["text"] += " " + word["word"] if current_chunk["text"] else word["word"]
                    current_speaker = speaker
                
                # Add final chunk from segment
                if current_chunk["text"]:
                    new_segments.append(current_chunk)

            result["segments"] = new_segments
            diarize_time = time.time() - diarize_start
            speaker_count = len(set(seg.get("speaker") for seg in new_segments if seg.get("speaker")))
            
            update_progress(transcription_id, "finalizing", 85, 
                          f"Diarization complete in {diarize_time:.1f}s. Detected {speaker_count} speakers.")
            r.hset(f"transcription:timing:{transcription_id}", "diarization_complete", time.time())
            
        except Exception as e:
            update_progress(transcription_id, "diarize_failed", 75, f"Speaker identification failed: {str(e)[:100]}")
            print(f"⚠️ Diarization failed or skipped: {e}")

        segments = result["segments"]
        print(f"✍️ {len(segments)} segments transcribed")
        update_progress(transcription_id, "saving", 90, f"Saving {len(segments)} transcript segments to database")

        # Update transcription record
        db_start = time.time()
        cursor.execute("""
            UPDATE transcriptions SET 
                status = 'completed',
                language = %s,
                duration = %s
            WHERE id = %s
        """, (detected_language, audio_duration, transcription_id))

        # Insert segments
        for idx, seg in enumerate(segments):
            if idx % 20 == 0 and len(segments) > 50:
                # Periodic updates for very large transcripts
                update_progress(transcription_id, "saving", 90 + (idx / len(segments) * 9), 
                              f"Saving segment {idx+1}/{len(segments)}")
            
            cursor.execute("""
                INSERT INTO segments (transcription_id, speaker, text, start, "end")
                VALUES (%s, %s, %s, %s, %s)
            """, (
                transcription_id,
                seg.get("speaker", "Speaker 1"),
                seg["text"],
                float(seg["start"]),
                float(seg["end"]),
            ))

        db_time = time.time() - db_start
        total_job_time = time.time() - job_start_time
        r.hset(f"transcription:timing:{transcription_id}", "completion_time", time.time())
        r.hset(f"transcription:timing:{transcription_id}", "total_duration", total_job_time)
        
        # Generate job summary
        summary = {
            "total_time": total_job_time,
            "audio_duration": audio_duration,
            "segments": len(segments),
            "speakers": speaker_count if 'speaker_count' in locals() else 1,
            "language": detected_language
        }
        r.set(f"transcription:summary:{transcription_id}", json.dumps(summary))
        r.expire(f"transcription:summary:{transcription_id}", 86400 * 7)  # Keep summary for 7 days
        
        # Final completion update
        update_progress(transcription_id, "completed", 100, 
                      f"Transcription complete in {total_job_time:.1f}s. {len(segments)} segments stored.")
        
        # Remove from active jobs
        r.srem("transcription:active_jobs", transcription_id)
        
        print(f"✅ Transcription {transcription_id} saved")
        
        # Clean up temporary files
        try:
            os.remove(local_audio_path)
        except Exception as e:
            print(f"⚠️ Failed to clean up temp file: {e}")
            
    except Exception as e:
        # Error handling
        error_message = str(e)[:200]  # Limit error message length
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        if 'transcription_id' in locals():
            update_progress(transcription_id, "error", None, f"Error: {error_message}")
            r.srem("transcription:active_jobs", transcription_id)
            r.hset(f"transcription:timing:{transcription_id}", "error_time", time.time())
            
            # Try to update the database record
            try:
                cursor.execute("""
                    UPDATE transcriptions SET status = 'error', error = %s
                    WHERE id = %s
                """, (error_message, transcription_id))
            except Exception:
                pass  # If this fails too, just continue
        
        print("🔄 Restarting worker loop after error...")
        time.sleep(2)
        continue