#!/usr/bin/env python3
"""
Cache Manager
File: utils/cache_manager.py

Utilities for managing cached data files including cleaning old cache files.
"""

import os
import time
import shutil
from typing import Dict, List
from utils.logger import setup_logging

logger = setup_logging()

class CacheManager:
    """Cache management utilities."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize cache manager."""
        self.cache_dir = cache_dir
        self.ensure_cache_dir()
    
    def ensure_cache_dir(self):
        """Ensure cache directory exists."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Cache directory ready: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Error creating cache directory: {e}")
    
    def clear_old_cache(self, max_age_hours: int = 24):
        """Clear cache files older than specified hours."""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleared_count = 0
            
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        try:
                            os.remove(file_path)
                            cleared_count += 1
                            logger.debug(f"Removed old cache file: {file_path}")
                        except Exception as e:
                            logger.error(f"Error removing cache file {file_path}: {e}")
            
            logger.info(f"Cleared {cleared_count} old cache files (older than {max_age_hours} hours)")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing old cache: {e}")
            return 0
    
    def clear_all_cache(self):
        """Clear all cache files."""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                logger.info("Cleared all cache files")
                self.ensure_cache_dir()
                return True
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        try:
            stats = {
                'total_files': 0,
                'total_size_mb': 0,
                'oldest_file_age_hours': 0,
                'newest_file_age_hours': 0,
                'file_types': {}
            }
            
            current_time = time.time()
            oldest_time = current_time
            newest_time = 0
            
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_stat = os.stat(file_path)
                    
                    stats['total_files'] += 1
                    stats['total_size_mb'] += file_stat.st_size / (1024 * 1024)
                    
                    # Track file types
                    file_ext = os.path.splitext(file)[1]
                    stats['file_types'][file_ext] = stats['file_types'].get(file_ext, 0) + 1
                    
                    # Track age
                    file_time = file_stat.st_mtime
                    if file_time < oldest_time:
                        oldest_time = file_time
                    if file_time > newest_time:
                        newest_time = file_time
            
            if stats['total_files'] > 0:
                stats['oldest_file_age_hours'] = (current_time - oldest_time) / 3600
                stats['newest_file_age_hours'] = (current_time - newest_time) / 3600
            
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            stats['oldest_file_age_hours'] = round(stats['oldest_file_age_hours'], 2)
            stats['newest_file_age_hours'] = round(stats['newest_file_age_hours'], 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def clean_corrupted_cache_files(self):
        """Clean up corrupted cache files that might cause parsing errors."""
        try:
            cleaned_count = 0
            total_checked = 0
            
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        total_checked += 1
                        
                        try:
                            # Try to read the CSV file to check if it's corrupted
                            import pandas as pd
                            test_data = pd.read_csv(file_path, nrows=1)
                            
                            # Check if the file has the expected structure
                            if test_data.empty or len(test_data.columns) < 5:
                                logger.warning(f"Removing corrupted cache file (insufficient columns): {file_path}")
                                os.remove(file_path)
                                cleaned_count += 1
                                
                        except Exception as e:
                            # If we can't read the file, it's likely corrupted
                            logger.warning(f"Removing corrupted cache file: {file_path} - {e}")
                            try:
                                os.remove(file_path)
                                cleaned_count += 1
                            except Exception as remove_error:
                                logger.error(f"Error removing corrupted file {file_path}: {remove_error}")
            
            logger.info(f"Cache cleanup: checked {total_checked} CSV files, cleaned {cleaned_count} corrupted files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning corrupted cache files: {e}")
            return 0
    
    def optimize_cache(self, max_size_mb: int = 1000):
        """Optimize cache by removing oldest files if size exceeds limit."""
        try:
            stats = self.get_cache_stats()
            
            if stats.get('total_size_mb', 0) <= max_size_mb:
                logger.info(f"Cache size {stats['total_size_mb']}MB is within limit ({max_size_mb}MB)")
                return
            
            # Get all files with their ages
            files_with_age = []
            current_time = time.time()
            
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_stat = os.stat(file_path)
                    age = current_time - file_stat.st_mtime
                    size_mb = file_stat.st_size / (1024 * 1024)
                    
                    files_with_age.append({
                        'path': file_path,
                        'age': age,
                        'size_mb': size_mb
                    })
            
            # Sort by age (oldest first)
            files_with_age.sort(key=lambda x: x['age'], reverse=True)
            
            # Remove oldest files until we're under the limit
            removed_count = 0
            current_size = stats['total_size_mb']
            
            for file_info in files_with_age:
                if current_size <= max_size_mb:
                    break
                
                try:
                    os.remove(file_info['path'])
                    current_size -= file_info['size_mb']
                    removed_count += 1
                    logger.debug(f"Removed old cache file: {file_info['path']}")
                except Exception as e:
                    logger.error(f"Error removing cache file {file_info['path']}: {e}")
            
            logger.info(f"Optimized cache: removed {removed_count} files, new size: {current_size:.2f}MB")
            
        except Exception as e:
            logger.error(f"Error optimizing cache: {e}")


def get_cache_manager():
    """Get a singleton cache manager instance."""
    return CacheManager()
