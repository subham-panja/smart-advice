#!/usr/bin/env python3
"""
Test script to verify OpenMP threading fixes
"""

# Apply OpenMP fix
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("Testing OpenMP fixes...")

try:
    # Test importing problematic modules
    print("1. Testing numpy import...")
    import numpy as np
    print("   ✓ numpy imported successfully")
    
    print("2. Testing pandas import...")
    import pandas as pd
    print("   ✓ pandas imported successfully")
    
    print("3. Testing sklearn import...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    print("   ✓ sklearn imported successfully")
    
    print("4. Testing market regime detection...")
    from scripts.market_regime_detection import MarketRegimeDetection
    print("   ✓ MarketRegimeDetection imported successfully")
    
    print("5. Testing RL trading agent...")
    from scripts.rl_trading_agent import RLTradingAgent
    print("   ✓ RLTradingAgent imported successfully")
    
    print("6. Testing simple operations...")
    # Test basic operations that might trigger OpenMP
    data = np.random.random((1000, 10))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    print(f"   ✓ Processed {len(data)} samples with KMeans clustering")
    
    print("\n🎉 All tests passed! OpenMP fix is working correctly.")
    print("You can now run the analysis with multiple threads enabled.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("OpenMP fix may not be working correctly.")
    import traceback
    traceback.print_exc()
