import os
from minio import Minio
from tempfile import NamedTemporaryFile
from config import Config

class StorageManager:
    def __init__(self):
        self.minio_client = Minio(
            Config.MINIO_ENDPOINT,
            access_key=Config.MINIO_ACCESS_KEY,
            secret_key=Config.MINIO_SECRET_KEY,
            secure=False,
        )
    
    def download_audio_file(self, filename):
        """Download audio file from MinIO and return local path"""
        tmpfile = NamedTemporaryFile(suffix=os.path.splitext(filename)[-1], delete=False)
        tmpfile.close()
        self.minio_client.fget_object(Config.MINIO_BUCKET, filename, tmpfile.name)
        return tmpfile.name
    
    def get_file_size_mb(self, file_path):
        """Get file size in MB"""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def cleanup_temp_file(self, file_path):
        """Remove temporary file"""
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"⚠️ Failed to clean up temp file: {e}")