import psycopg
from config import Config

class DatabaseManager:
    def __init__(self):
        self.pg = psycopg.connect(Config.POSTGRES_URL)
        self.pg.autocommit = True
        self.cursor = self.pg.cursor()
    
    def update_transcription_status(self, transcription_id, status, language=None, duration=None, error=None):
        """Update transcription record status"""
        if status == 'completed':
            self.cursor.execute("""
                UPDATE transcriptions SET 
                    status = %s,
                    language = %s,
                    duration = %s
                WHERE id = %s
            """, (status, language, duration, transcription_id))
        elif status == 'error':
            self.cursor.execute("""
                UPDATE transcriptions SET status = %s, error = %s
                WHERE id = %s
            """, (status, error, transcription_id))
        else:
            self.cursor.execute("""
                UPDATE transcriptions SET status = %s
                WHERE id = %s
            """, (status, transcription_id))
    
    def save_transcription_complete(self, transcription_id, language, duration, summary=None):
        """Save completed transcription with summary"""
        if summary:
            self.cursor.execute("""
                UPDATE transcriptions SET 
                    status = 'completed',
                    language = %s,
                    duration = %s,
                    summary = %s
                WHERE id = %s
            """, (language, duration, summary, transcription_id))
        else:
            self.cursor.execute("""
                UPDATE transcriptions SET 
                    status = 'completed',
                    language = %s,
                    duration = %s
                WHERE id = %s
            """, (language, duration, transcription_id))
    
    def insert_segments(self, transcription_id, segments, progress_callback=None):
        """Insert transcript segments into database"""
        for idx, seg in enumerate(segments):
            if progress_callback and idx % 20 == 0 and len(segments) > 50:
                progress_callback(idx, len(segments))
            
            self.cursor.execute("""
                INSERT INTO segments (transcription_id, speaker, text, start, "end")
                VALUES (%s, %s, %s, %s, %s)
            """, (
                transcription_id,
                seg.get("speaker", "Speaker 1"),
                seg["text"],
                float(seg["start"]),
                float(seg["end"]),
            ))
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.pg:
            self.pg.close()