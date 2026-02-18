import pymongo
from datetime import datetime

def check_mongodb():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["super_advice"]
        
        # Check recommended_shares collection
        rec_collection = db["recommended_shares"]
        rec_count = rec_collection.count_documents({})
        print(f"Total documents in 'recommended_shares': {rec_count}")
        
        if rec_count > 0:
            print("\nSample Recommendation:")
            sample = rec_collection.find_one()
            print(f"Symbol: {sample.get('symbol')}")
            print(f"Date: {sample.get('recommendation_date')}")
            print(f"Score: {sample.get('combined_score')}")
        
        # Check backtest_results collection
        bt_collection = db["backtest_results"]
        bt_count = bt_collection.count_documents({})
        print(f"\nTotal documents in 'backtest_results': {bt_count}")
        
        if bt_count > 0:
            print("\nSample Backtest Result:")
            sample_bt = bt_collection.find_one()
            print(f"Symbol: {sample_bt.get('symbol')}")
            print(f"CAGR: {sample_bt.get('cagr')}")
            
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")

if __name__ == "__main__":
    check_mongodb()
