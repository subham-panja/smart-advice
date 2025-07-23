#!/usr/bin/env python3
"""
Cache Cleanup Script
File: clean_cache.py

A utility script to clean corrupted cache files that might be causing errors.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cache_manager import get_cache_manager
from utils.logger import setup_logging

def main():
    """Clean corrupted cache files."""
    logger = setup_logging()
    
    logger.info("Starting cache cleanup...")
    
    try:
        cache_manager = get_cache_manager()
        
        # Clean corrupted cache files
        cleaned_count = cache_manager.clean_corrupted_cache_files()
        
        if cleaned_count > 0:
            logger.info(f"Successfully cleaned {cleaned_count} corrupted cache files")
        else:
            logger.info("No corrupted cache files found")
        
        # Get cache stats
        stats = cache_manager.get_cache_stats()
        
        if stats:
            logger.info(f"Cache stats after cleanup:")
            logger.info(f"  Total files: {stats.get('total_files', 0)}")
            logger.info(f"  Total size: {stats.get('total_size_mb', 0):.2f} MB")
            logger.info(f"  File types: {stats.get('file_types', {})}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
