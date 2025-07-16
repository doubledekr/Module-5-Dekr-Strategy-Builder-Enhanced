import sqlite3
import aiosqlite
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

DATABASE_PATH = "dekr_strategy_builder.db"

async def init_db():
    """Initialize the database with required tables"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Users table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                tier INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Strategies table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                name TEXT NOT NULL,
                description TEXT,
                strategy_type TEXT NOT NULL,
                symbols TEXT NOT NULL,
                buy_conditions TEXT NOT NULL,
                sell_conditions TEXT NOT NULL,
                risk_management TEXT NOT NULL,
                tier_required INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                performance_metrics TEXT,
                backtest_results TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Signals table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY,
                strategy_id TEXT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                conditions_met TEXT NOT NULL,
                market_data TEXT NOT NULL,
                sentiment_data TEXT,
                is_processed BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (strategy_id) REFERENCES strategies (id)
            )
        """)
        
        # Backtests table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS backtests (
                id TEXT PRIMARY KEY,
                strategy_id TEXT,
                symbol TEXT NOT NULL,
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP NOT NULL,
                total_return REAL NOT NULL,
                annualized_return REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                win_rate REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                avg_trade_duration REAL NOT NULL,
                profit_factor REAL NOT NULL,
                trades TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES strategies (id)
            )
        """)
        
        # Market data cache table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS market_data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                data TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe)
            )
        """)
        
        await db.commit()
        logger.info("Database initialized successfully")

async def get_db():
    """Get database connection"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db

class DatabaseManager:
    def __init__(self):
        self.db_path = DATABASE_PATH
    
    async def execute_query(self, query: str, params: tuple = None):
        """Execute a query and return results"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params or ())
            return await cursor.fetchall()
    
    async def execute_single(self, query: str, params: tuple = None):
        """Execute a query and return single result"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params or ())
            return await cursor.fetchone()
    
    async def execute_write(self, query: str, params: tuple = None):
        """Execute a write query"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(query, params or ())
            await db.commit()
    
    async def execute_many(self, query: str, params_list: List[tuple]):
        """Execute many queries"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(query, params_list)
            await db.commit()

# Global database manager instance
db_manager = DatabaseManager()
