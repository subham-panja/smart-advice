import logging
from datetime import datetime, timezone

from database import get_mongodb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Migration")


def migrate():
    db = get_mongodb()
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)

    # Find records with recommendation_strength BUY but missing recommendation_date
    query = {"recommendation_strength": "BUY", "recommendation_date": {"$exists": False}}
    update = {"$set": {"recommendation_date": today}}

    res = db.recommended_shares.update_many(query, update)
    logger.info(f"Migrated {res.modified_count} old recommendations for execution today.")

    # Also verify what they are
    all_buys = list(db.recommended_shares.find({"recommendation_strength": "BUY"}))
    for b in all_buys:
        logger.info(f"Found BUY: {b['symbol']} with date {b.get('recommendation_date')}")


if __name__ == "__main__":
    migrate()
