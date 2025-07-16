from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging
from services.signal_generator import signal_generator
from models import Signal

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def get_signals(
    strategy_id: Optional[str] = Query(None, description="Strategy ID to filter signals"),
    limit: int = Query(50, description="Number of signals to return")
):
    """Get recent signals"""
    
    try:
        signals = await signal_generator.get_recent_signals(strategy_id, limit)
        return {"signals": [signal.to_dict() for signal in signals]}
        
    except Exception as e:
        logger.error(f"Error getting signals: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats")
async def get_signal_stats(
    strategy_id: Optional[str] = Query(None, description="Strategy ID to filter stats")
):
    """Get signal statistics"""
    
    try:
        # Get recent signals for stats
        signals = await signal_generator.get_recent_signals(strategy_id, 1000)
        
        if not signals:
            return {
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "avg_confidence": 0.0,
                "signal_types": {}
            }
        
        # Calculate stats
        total_signals = len(signals)
        buy_signals = sum(1 for s in signals if s.signal_type == "buy")
        sell_signals = sum(1 for s in signals if s.signal_type == "sell")
        avg_confidence = sum(s.confidence for s in signals) / total_signals
        
        # Signal type breakdown
        signal_types = {}
        for signal in signals:
            signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1
        
        return {
            "total_signals": total_signals,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "avg_confidence": avg_confidence,
            "signal_types": signal_types
        }
        
    except Exception as e:
        logger.error(f"Error getting signal stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{signal_id}/process")
async def process_signal(signal_id: str, action_data: Dict[str, Any]):
    """Mark a signal as processed"""
    
    try:
        from database import db_manager
        
        # Update signal in database
        query = "UPDATE signals SET is_processed = TRUE WHERE id = ?"
        await db_manager.execute_write(query, (signal_id,))
        
        return {"message": "Signal processed successfully"}
        
    except Exception as e:
        logger.error(f"Error processing signal: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/recent/{symbol}")
async def get_recent_signals_for_symbol(symbol: str, limit: int = Query(20)):
    """Get recent signals for a specific symbol"""
    
    try:
        from database import db_manager
        import json
        from datetime import datetime
        
        query = """
            SELECT * FROM signals 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        rows = await db_manager.execute_query(query, (symbol, limit))
        
        signals = []
        for row in rows:
            signal_data = {
                "id": row["id"],
                "strategy_id": row["strategy_id"],
                "symbol": row["symbol"],
                "signal_type": row["signal_type"],
                "confidence": row["confidence"],
                "price": row["price"],
                "timestamp": row["timestamp"],
                "conditions_met": json.loads(row["conditions_met"]),
                "market_data": json.loads(row["market_data"]),
                "sentiment_data": json.loads(row["sentiment_data"]) if row["sentiment_data"] else None,
                "is_processed": bool(row["is_processed"])
            }
            signals.append(Signal.from_dict(signal_data).to_dict())
        
        return {"signals": signals}
        
    except Exception as e:
        logger.error(f"Error getting signals for symbol: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
