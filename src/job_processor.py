import time
import traceback
import json
from storage import StorageManager
from progress_tracker import ProgressTracker
from transcription_service import TranscriptionService
from audio_utils import get_audio_duration

class JobProcessor:
    def __init__(self):
        self.storage = StorageManager()
        self.progress = ProgressTracker()
        self.transcription = TranscriptionService()
    
    def process_transcription_job(self, job_data):
        """Process a single transcription job"""
        transcription_id = job_data.get("transcriptionId") or job_data.get("id")
        filename = job_data["filename"]
        language = job_data.get("language")
        model_size = job_data.get("model", "tiny")
        is_speaker_diarized = job_data.get("isSpeakerDiarized", False)
        number_of_speakers = job_data.get("numberOfSpeaker")
        
        print(f"📥 Job received: {filename}")
        
        try:
            # Start job tracking
            job_start_time = self.progress.track_job_start(transcription_id)
            self.progress.update_progress(transcription_id, "started", 0, f"Starting transcription of {filename}")
            
            # Download audio file
            local_audio_path = self._download_audio(transcription_id, filename)
            
            # Get audio info
            file_size_mb = self.storage.get_file_size_mb(local_audio_path)
            audio_duration = get_audio_duration(local_audio_path)
            
            self.progress.update_progress(transcription_id, "analyzing", 15, 
                                        f"Analyzing audio file ({audio_duration:.1f} seconds)")
            
            # Transcribe audio
            result = self._transcribe_audio(transcription_id, local_audio_path, language)
            detected_language = result.get("language", language or "unknown")
            
            # Perform diarization if needed
            if is_speaker_diarized:
                result, diarize_error = self._perform_diarization(transcription_id, local_audio_path, result)
            
            # Generate transcription summary
            summary = self._generate_summary(transcription_id, result, detected_language)
            
            # Save results to database (including summary)
            self._save_results(transcription_id, result, detected_language, audio_duration, summary)
            
            # Complete job
            self._complete_job(transcription_id, result, audio_duration, detected_language, job_start_time)
            
            # Cleanup
            self.storage.cleanup_temp_file(local_audio_path)
            
            print(f"✅ Transcription {transcription_id} completed successfully")
            
        except Exception as e:
            self._handle_error(transcription_id, e)
    
    def _download_audio(self, transcription_id, filename):
        """Download audio file from storage"""
        self.progress.update_progress(transcription_id, "downloading", 5, f"Downloading audio file {filename}")
        download_start = time.time()
        
        local_audio_path = self.storage.download_audio_file(filename)
        file_size_mb = self.storage.get_file_size_mb(local_audio_path)
        download_time = time.time() - download_start
        
        self.progress.track_timing(transcription_id, "download_complete")
        self.progress.update_progress(transcription_id, "processing", 10, 
                                    f"Audio file downloaded ({file_size_mb:.2f} MB in {download_time:.1f}s)")
        
        print(f"🔊 Audio downloaded to {local_audio_path}")
        return local_audio_path
    
    def _transcribe_audio(self, transcription_id, local_audio_path, language):
        """Transcribe audio using Groq"""
        self.progress.update_progress(transcription_id, "loading_model", 20, "Preparing Groq Whisper-Large-V3-Turbo")
        self.progress.update_progress(transcription_id, "transcribing", 25, f"Sending audio to Groq for transcription")
        
        transcribe_start = time.time()
        result = self.transcription.transcribe_with_groq(local_audio_path, language)
        transcribe_time = time.time() - transcribe_start
        
        self.progress.track_timing(transcription_id, "transcription_complete")
        detected_language = result.get("language", language or "unknown")
        self.progress.update_progress(transcription_id, "post_processing", 65, 
                                    f"Transcription complete in {transcribe_time:.1f}s ({detected_language})")
        
        return result
    
    def _perform_diarization(self, transcription_id, local_audio_path, result):
        """Perform speaker diarization"""
        self.progress.update_progress(transcription_id, "diarizing", 70, "Identifying speakers in audio")
        diarize_start = time.time()
        
        result, error = self.transcription.perform_diarization(local_audio_path, result)
        
        if error:
            self.progress.update_progress(transcription_id, "diarize_failed", 75, f"Speaker identification failed: {error[:100]}")
            return result, error
        
        diarize_time = time.time() - diarize_start
        speaker_count = self.transcription.get_speaker_count(result["segments"])
        
        self.progress.update_progress(transcription_id, "finalizing", 85, 
                                    f"Diarization complete in {diarize_time:.1f}s. Detected {speaker_count} speakers.")
        self.progress.track_timing(transcription_id, "diarization_complete")
        
        return result, None
    
    def _save_results(self, transcription_id, result, detected_language, audio_duration, summary=None):
        """Save transcription results to Redis for backend consumption"""
        segments = result["segments"]
        self.progress.update_progress(transcription_id, "saving", 95, f"Saving transcription and {len(segments)} segments to Redis")

        # Compose result data
        result_data = {
            "id": transcription_id,
            "language": detected_language,
            "duration": audio_duration,
            "summary": summary,
            "segments": segments
        }
        # Store in Redis (keyed by transcription id)
        self.progress.redis_client.set(f"transcription:result:{transcription_id}", json.dumps(result_data))
        self.progress.redis_client.expire(f"transcription:result:{transcription_id}", 86400 * 7)  # 7 days
        print(f"✍️ {len(segments)} segments transcribed and saved to Redis with summary")
    
    def _generate_summary(self, transcription_id, result, detected_language):
        """Generate transcription summary using Groq API"""
        segments = result["segments"]
        
        # Skip summarization for very short transcripts
        total_text_length = sum(len(seg["text"]) for seg in segments)
        if total_text_length < 500:  # Less than ~500 characters
            self.progress.update_progress(transcription_id, "summary_skipped", 92, 
                                        "Skipping summary - transcript too short")
            return None
        
        self.progress.update_progress(transcription_id, "summarizing", 90, 
                                    f"Generating summary from {len(segments)} segments")
        
        summary_start = time.time()
        summary, error = self.transcription.summarize_transcription(segments, detected_language)
        summary_time = time.time() - summary_start
        
        if error:
            self.progress.update_progress(transcription_id, "summary_failed", 92, 
                                        f"Summary generation failed: {error[:100]}")
            return None
        
        self.progress.track_timing(transcription_id, "summarization_complete")
        self.progress.update_progress(transcription_id, "summary_complete", 94, 
                                    f"Summary generated in {summary_time:.1f}s")
        
        return summary
    
    def _complete_job(self, transcription_id, result, audio_duration, detected_language, job_start_time):
        """Complete job and generate summary"""
        total_job_time = time.time() - job_start_time
        segments = result["segments"]
        speaker_count = self.transcription.get_speaker_count(segments)
        
        self.progress.track_timing(transcription_id, "total_duration", total_job_time)
        
        # Generate summary for job statistics (not transcription summary)
        summary = {
            "total_time": total_job_time,
            "audio_duration": audio_duration,
            "segments": len(segments),
            "speakers": speaker_count,
            "language": detected_language
        }
        
        self.progress.complete_job(transcription_id, summary)
        self.progress.update_progress(transcription_id, "completed", 100, 
                                    f"Transcription complete in {total_job_time:.1f}s. {len(segments)} segments stored.")
    
    def _handle_error(self, transcription_id, error):
        """Handle job errors"""
        error_message = str(error)[:200]  # Limit error message length
        print(f"❌ Error occurred: {error}")
        traceback.print_exc()

        self.progress.update_progress(transcription_id, "error", None, f"Error: {error_message}")
        self.progress.handle_error(transcription_id, error_message)

        # Store error in Redis for backend to consume
        error_data = {
            "id": transcription_id,
            "status": "error",
            "error": error_message
        }
        self.progress.redis_client.set(f"transcription:result:{transcription_id}", json.dumps(error_data))
        self.progress.redis_client.expire(f"transcription:result:{transcription_id}", 86400 * 7)