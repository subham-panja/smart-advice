#!/usr/bin/env python3
import pymongo
import sys

try:
    # Test MongoDB connection
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['super_advice']
    
    # Test basic operations
    print("✓ MongoDB connection successful")
    collections = db.list_collection_names()
    print(f"✓ Collections: {collections}")
    
    # Test if we can insert/query data
    test_collection = db['test']
    test_collection.insert_one({'test': 'data'})
    result = test_collection.find_one({'test': 'data'})
    if result:
        print("✓ MongoDB read/write operations working")
        test_collection.delete_one({'test': 'data'})
    
    client.close()
    print("✓ MongoDB test completed successfully")
    
except Exception as e:
    print(f"✗ MongoDB test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
