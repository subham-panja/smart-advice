#!/usr/bin/env python3
"""
Cache Migration Script
======================

Migrates date-stamped cache files to single file per symbol format.

Before: RELIANCE_2026-06-16.parquet, RELIANCE_2026-06-17.parquet, RELIANCE_2026-06-18.parquet
After:  RELIANCE.parquet (most recent data)

Usage:
    cd backend
    python scripts/migrate_cache_to_single_file.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.data_cache import consolidate_cache, get_cache_stats


def main():
    print("=" * 70)
    print("CACHE MIGRATION: Date-Stamped -> Single File Format")
    print("=" * 70)

    # Show stats before migration
    print("\n📊 BEFORE MIGRATION:")
    stats_before = get_cache_stats()
    print(f"  Total files: {stats_before['total_files']}")
    print(f"  Unique symbols: {stats_before['unique_symbols']}")
    print(f"  Total size: {stats_before['total_size_mb']} MB")

    # Run consolidation
    print("\n🔄 RUNNING MIGRATION...")
    results = consolidate_cache()

    print("\n✅ MIGRATION RESULTS:")
    print(f"  Period-based files migrated: {results['period_migration']}")
    print(f"  Date-stamped files migrated: {results['date_stamped_migration']}")

    # Show stats after migration
    print("\n📊 AFTER MIGRATION:")
    stats_after = get_cache_stats()
    print(f"  Total files: {stats_after['total_files']}")
    print(f"  Unique symbols: {stats_after['unique_symbols']}")
    print(f"  Total size: {stats_after['total_size_mb']} MB")
    print(f"  Recent cached: {stats_after['recent_cached']}")

    # Calculate savings
    files_deleted = stats_before["total_files"] - stats_after["total_files"]
    size_saved = stats_before["total_size_mb"] - stats_after["total_size_mb"]

    print("\n💾 SPACE SAVED:")
    print(f"  Files deleted: {files_deleted}")
    print(f"  Size saved: {size_saved:.2f} MB ({size_saved/stats_before['total_size_mb']*100:.1f}%)")

    print("\n" + "=" * 70)
    print("✅ Migration complete! Cache now uses single file per symbol.")
    print("=" * 70)


if __name__ == "__main__":
    main()
