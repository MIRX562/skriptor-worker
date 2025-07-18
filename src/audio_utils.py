import subprocess

def get_audio_duration(file_path):
    """Get audio duration using ffprobe (requires ffmpeg)"""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
               "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        output = subprocess.check_output(cmd).decode('utf-8').strip()
        return float(output)
    except Exception as e:
        print(f"⚠️ Could not determine audio duration: {e}")
        return 0  # Return 0 if duration can't be determined