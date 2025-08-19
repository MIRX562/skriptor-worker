import whisperx
from groq import Groq
from config import Config

class TranscriptionService:
    def __init__(self):
        self.provider = Config.TRANSCRIPTION_PROVIDER.lower()
        if self.provider == "groq":
            self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        elif self.provider == "local":
            self.local_model_cache = {}

    def transcribe(self, audio_file_path, language=None, model_size="large-v3"):
        """Transcribe audio using selected provider and model size"""
        if self.provider == "groq":
            return self._transcribe_with_groq(audio_file_path, language, model_size)
        elif self.provider == "local":
            return self._transcribe_with_local(audio_file_path, language, model_size)
        else:
            raise ValueError(f"Unknown transcription provider: {self.provider}")

    def _transcribe_with_groq(self, audio_file_path, language=None, model_size="large-v3"):
        model_map = {
            "large-v3": "whisper-large-v3-turbo",
            "medium": "whisper-medium",
            "small": "whisper-small",
            "tiny": "whisper-tiny"
        }
        model = model_map.get(model_size, "whisper-large-v3-turbo")
        with open(audio_file_path, "rb") as f:
            transcription = self.groq_client.audio.transcriptions.create(
                file=(audio_file_path, f.read()),
                model=model,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language=language or "id"
            )
            return transcription.model_dump()

    def _transcribe_with_local(self, audio_file_path, language=None, model_size="large-v3"):
        # Cache models by size
        if model_size not in self.local_model_cache:
            self.local_model_cache[model_size] = whisperx.load_model(model_size, device="cpu")
        model = self.local_model_cache[model_size]
        result = model.transcribe(audio_file_path, language=language or "id", batch_size=16, return_segments=True)
        return result
    
    def perform_diarization(self, audio_file_path, transcription_result):
        """Perform speaker diarization using WhisperX"""
        try:
            from whisperx.diarize import DiarizationPipeline
            
            # Load audio
            audio = whisperx.load_audio(audio_file_path)
            
            # Initialize diarization model
            diarize_model = DiarizationPipeline(use_auth_token=Config.HF_TOKEN, device="cpu")
            
            # Perform diarization
            diarize_segments = diarize_model(audio)
            
            # Assign speakers to words
            result = whisperx.assign_word_speakers(diarize_segments, transcription_result)
            
            # Create speaker-based segments
            return self._create_speaker_segments(result)
            
        except Exception as e:
            print(f"⚠️ Diarization failed: {e}")
            return transcription_result, str(e)
    
    def _create_speaker_segments(self, result):
        """Split segments based on speaker changes"""
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
        return result, None  # No error
    
    def get_speaker_count(self, segments):
        """Count unique speakers in segments"""
        return len(set(seg.get("speaker") for seg in segments if seg.get("speaker")))
    
    def summarize_transcription(self, segments, language="id"):
        """Generate summary of transcription using Groq API"""
        try:
            # Combine all segments into full text
            full_text = " ".join([seg["text"] for seg in segments])
            
            # Limit text length for API (roughly 8000 tokens max)
            if len(full_text) > 32000:  # Rough character limit
                full_text = full_text[:32000] + "..."
            
            # Create summarization prompt based on language
            if language.lower() in ['id', 'indonesian']:
                system_prompt = """Anda adalah asisten AI yang ahli dalam merangkum percakapan. Buatlah ringkasan yang komprehensif dari transkrip berikut dalam bahasa Indonesia. Ringkasan harus mencakup:
1. Poin-poin utama yang dibahas
2. Keputusan atau kesimpulan penting
3. Tindakan yang perlu dilakukan (jika ada)
4. Topik atau tema utama

Buatlah ringkasan yang jelas, terstruktur, dan mudah dipahami."""
                user_prompt = f"Rangkum transkrip percakapan berikut:\n\n{full_text}"
            else:
                system_prompt = """You are an AI assistant expert in summarizing conversations. Create a comprehensive summary of the following transcript. The summary should include:
1. Main points discussed
2. Important decisions or conclusions
3. Action items (if any)
4. Key topics or themes

Make the summary clear, structured, and easy to understand."""
                user_prompt = f"Summarize the following conversation transcript:\n\n{full_text}"
            
            # Call Groq API for summarization
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # Good for summarization tasks
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused summaries
                max_tokens=1000,  # Reasonable length for summary
            )
            
            summary = response.choices[0].message.content.strip()
            return summary, None
            
        except Exception as e:
            error_msg = f"Summarization failed: {str(e)}"
            print(f"⚠️ {error_msg}")
            return None, error_msg