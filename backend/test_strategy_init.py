#!/usr/bin/env python3
"""
Test script to isolate strategy initialization issues
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix OpenMP/threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("Starting strategy initialization test...")

try:
    print("Importing StrategyEvaluator...")
    from scripts.strategy_evaluator import StrategyEvaluator
    print("StrategyEvaluator imported successfully")
    
    print("Creating StrategyEvaluator instance...")
    evaluator = StrategyEvaluator()
    print("StrategyEvaluator created successfully")
    
    print("Getting strategy summary...")
    summary = evaluator.get_strategy_summary()
    print(f"Strategy summary: {summary}")
    
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
