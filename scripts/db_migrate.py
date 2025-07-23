import sqlite3
from flask import current_app
from flask.cli import with_appcontext
import click
import os

# Migration to alter recommended_shares and add backtest_results

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    return column_name in columns

def check_table_exists(cursor, table_name):
    """Check if a table exists."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return cursor.fetchone() is not None

def check_index_exists(cursor, index_name):
    """Check if an index exists."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name=?;", (index_name,))
    return cursor.fetchone() is not None

def migrate_db():
    """Migrate database schema without data loss."""
    db_path = current_app.config['DATABASE_PATH']
    
    # Backup database before migration
    backup_path = f"{db_path}.backup"
    if os.path.exists(db_path):
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"Database backup created at {backup_path}")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check and add missing columns to recommended_shares
        if not check_column_exists(cursor, 'recommended_shares', 'buy_price'):
            cursor.execute("""
                ALTER TABLE recommended_shares
                ADD COLUMN buy_price REAL;
            """)
            print("Added buy_price column to recommended_shares")
        else:
            print("buy_price column already exists")
            
        if not check_column_exists(cursor, 'recommended_shares', 'sell_price'):
            cursor.execute("""
                ALTER TABLE recommended_shares
                ADD COLUMN sell_price REAL;
            """)
            print("Added sell_price column to recommended_shares")
        else:
            print("sell_price column already exists")
            
        if not check_column_exists(cursor, 'recommended_shares', 'est_time_to_target'):
            cursor.execute("""
                ALTER TABLE recommended_shares
                ADD COLUMN est_time_to_target TEXT;
            """)
            print("Added est_time_to_target column to recommended_shares")
        else:
            print("est_time_to_target column already exists")
        
        conn.commit()
        
        # Create backtest_results table if it doesn't exist
        if not check_table_exists(cursor, 'backtest_results'):
            cursor.execute("""
                CREATE TABLE backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    period TEXT NOT NULL,
                    CAGR REAL,
                    win_rate REAL,
                    max_drawdown REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("Created backtest_results table")
        else:
            print("backtest_results table already exists")
        
        # Add index on recommendation_date for faster deletion of old rows
        if not check_index_exists(cursor, 'idx_recommendation_date'):
            cursor.execute("""
                CREATE INDEX idx_recommendation_date 
                ON recommended_shares (recommendation_date);
            """)
            print("Created index on recommendation_date")
        else:
            print("Index on recommendation_date already exists")
        
        conn.commit()
    
    print("Migration complete. Database schema is now up to date.")

@click.command('migrate-db')
@with_appcontext
def migrate_db_command():
    """Run database migration to add missing columns and tables."""
    migrate_db()
    click.echo('Database migration completed successfully!')
