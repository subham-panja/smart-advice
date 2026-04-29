import os
import time
import shutil
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class CacheManager:
    """Utilities for managing and cleaning cached data files."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def clear_old_cache(self, hours: int = 24):
        cutoff = time.time() - (hours * 3600)
        count = 0
        for f in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, f)
            if os.path.getmtime(path) < cutoff:
                os.remove(path); count += 1
        logger.info(f"Cleared {count} files older than {hours}h")

    def clean_corrupted_cache_files(self):
        count = 0
        for f in os.listdir(self.cache_dir):
            if f.endswith('.csv'):
                path = os.path.join(self.cache_dir, f)
                try:
                    if pd.read_csv(path, nrows=1).empty: raise Exception("Empty")
                except:
                    os.remove(path); count += 1
        logger.info(f"Cleaned {count} corrupted files")

    def get_stats(self):
        files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir)]
        size = sum(os.path.getsize(f) for f in files) / (1024*1024)
        return {'files': len(files), 'size_mb': round(size, 2)}

def get_cache_manager(): return CacheManager()
