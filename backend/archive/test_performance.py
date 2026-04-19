#!/usr/bin/env python3
"""
Performance Test Script
File: test_performance.py

Test script to compare the performance improvements in run_analysis.py
"""

import time
import subprocess
import sys

def run_analysis_test(max_stocks=5, fast_mode=False, test_name=""):
    """Run analysis and measure performance."""
    print(f"\n=== {test_name} ===")
    
    cmd = [
        sys.executable, 
        "run_analysis.py", 
        "--test", 
        f"--max-stocks={max_stocks}"
    ]
    
    if fast_mode:
        cmd.append("--fast")
    
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds")
        
        if result.returncode == 0:
            print("✅ Analysis completed successfully")
            # Print last few lines of output for status
            lines = result.stdout.split('\n')
            for line in lines[-5:]:
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ Analysis failed")
            print(f"Error: {result.stderr}")
            
        return elapsed, result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Analysis timed out (5 minutes)")
        return 300, False
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return 0, False

def main():
    """Main performance test function."""
    print("🚀 Smart Advice Performance Test")
    print("=" * 50)
    
    # Test 1: Standard mode (original)
    time1, success1 = run_analysis_test(
        max_stocks=3, 
        fast_mode=False, 
        test_name="Standard Mode (Original)"
    )
    
    # Test 2: Fast mode (optimized)
    time2, success2 = run_analysis_test(
        max_stocks=3, 
        fast_mode=True, 
        test_name="Fast Mode (Optimized)"
    )
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    
    if success1 and success2:
        improvement = ((time1 - time2) / time1) * 100
        print(f"Standard Mode: {time1:.2f} seconds")
        print(f"Fast Mode:     {time2:.2f} seconds")
        print(f"Improvement:   {improvement:.1f}% faster")
        
        if improvement > 0:
            print(f"🎉 Fast mode is {improvement:.1f}% faster!")
        else:
            print(f"⚠️  Fast mode is {abs(improvement):.1f}% slower")
    else:
        print("⚠️  Unable to compare due to failed tests")
        if not success1:
            print("   - Standard mode failed")
        if not success2:
            print("   - Fast mode failed")
    
    print(f"\n💡 Optimization Summary:")
    print("   ✅ Increased threads from 2 to 4")
    print("   ✅ Reduced batch size from 16 to 8")
    print("   ✅ Reduced delays from 2.0s to 0.5s")
    print("   ✅ Disabled fundamental analysis (network timeouts)")
    print("   ✅ Disabled sentiment analysis (heavy ML)")
    print("   ✅ Optimized garbage collection")
    print("   ✅ Added fast mode option")
    print("   ✅ Cached symbol data")
    print("   ✅ Reduced timeouts")

if __name__ == "__main__":
    main()