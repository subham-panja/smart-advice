#!/usr/bin/env python3
"""
Temporary fix for memory issues - Forces simplified mode to disable
memory-intensive modules like sector analysis and market regime detection
"""

import sys
import os
import subprocess

def main():
    """Run analysis with memory-intensive modules disabled by setting environment variables"""
    
    # Set environment variables to force simplified mode behavior
    env = os.environ.copy()
    env['FORCE_SIMPLIFIED_MODE'] = '1'
    env['SKIP_SECTOR_ANALYSIS'] = '1'
    env['SKIP_MARKET_REGIME'] = '1'
    env['SKIP_SENTIMENT'] = '1'
    
    # Get command line arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else ['--max-stocks', '1']
    
    # Run the analysis with modified environment
    cmd = ['python', 'run_analysis.py'] + args
    
    print("🔧 Running analysis with memory-intensive modules disabled...")
    print(f"Command: {' '.join(cmd)}")
    print("Environment overrides:")
    print("  - FORCE_SIMPLIFIED_MODE=1")
    print("  - SKIP_SECTOR_ANALYSIS=1") 
    print("  - SKIP_MARKET_REGIME=1")
    print("  - SKIP_SENTIMENT=1")
    print()
    
    try:
        result = subprocess.run(cmd, env=env, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
