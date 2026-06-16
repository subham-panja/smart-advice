import logging
import os
import time

import pandas as pd

import config

logger = logging.getLogger(__name__)

CACHE_DIR = config.DATA_CACHE_CONFIG.get("cache_dir", os.path.join(config.BACKEND_DIR, "data", "historical"))


class CacheManager:
    """Utilities for managing and cleaning cached parquet data files."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def clear_old_cache(self, hours: int = 24):
        cutoff = time.time() - (hours * 3600)
        count = 0
        for f in os.listdir(self.cache_dir):
            if not f.endswith(".parquet"):
                continue
            path = os.path.join(self.cache_dir, f)
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                count += 1
        logger.info(f"Cleared {count} parquet files older than {hours}h")

    def clean_corrupted_cache_files(self):
        count = 0
        for f in os.listdir(self.cache_dir):
            if not f.endswith(".parquet"):
                continue
            path = os.path.join(self.cache_dir, f)
            try:
                if pd.read_parquet(path).empty:
                    raise Exception("Empty")
            except Exception:
                os.remove(path)
                count += 1
        if count:
            logger.info(f"Cleaned {count} corrupted parquet files")

    def get_stats(self):
        files = [f for f in os.listdir(self.cache_dir) if f.endswith(".parquet")]
        size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in files) / (1024 * 1024)
        return {"files": len(files), "size_mb": round(size, 2)}


def get_cache_manager():
    return CacheManager()
