#!/usr/bin/env python3
"""
Script to check the current state of data in MongoDB recommendations collection
"""
import json
from database import get_mongodb
import config

def check_recommendation_data():
    """Check what fields are currently stored in recommendations"""
    try:
        db = get_mongodb()
        collection = db[config.MONGODB_COLLECTIONS['recommended_shares']]
        
        # Get the most recent recommendation to see its structure
        recent_rec = collection.find_one(sort=[('recommendation_date', -1)])
        
        if recent_rec:
            print("Most recent recommendation structure:")
            print("=" * 50)
            # Convert ObjectId to string for JSON serialization
            if '_id' in recent_rec:
                recent_rec['_id'] = str(recent_rec['_id'])
            if 'recommendation_date' in recent_rec:
                recent_rec['recommendation_date'] = recent_rec['recommendation_date'].isoformat()
            
            print(json.dumps(recent_rec, indent=2, default=str))
            
            print("\n" + "=" * 50)
            print("Available fields in this document:")
            for key in recent_rec.keys():
                print(f"- {key}")
                
            # Check if backtest_metrics exists and what it contains
            if 'backtest_metrics' in recent_rec and recent_rec['backtest_metrics']:
                print("\nBacktest metrics structure:")
                print(json.dumps(recent_rec['backtest_metrics'], indent=2, default=str))
        else:
            print("No recommendations found in the database")
            
        # Count total recommendations
        total_count = collection.count_documents({})
        print(f"\nTotal recommendations in database: {total_count}")
        
    except Exception as e:
        print(f"Error checking MongoDB data: {e}")

if __name__ == "__main__":
    check_recommendation_data()
