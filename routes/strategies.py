from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import json
import logging
from models import Strategy, StrategyCondition, StrategyType
from database import db_manager
from services.signal_generator import signal_generator
from services.backtesting import backtesting_engine

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def get_strategies(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(50, description="Number of strategies to return")
):
    """Get user's strategies"""
    
    try:
        query = """
            SELECT id, user_id, name, description, strategy_type, symbols, 
                   buy_conditions, sell_conditions, risk_management, tier_required,
                   is_active, created_at, updated_at, performance_metrics, backtest_results
            FROM strategies 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """
        
        rows = await db_manager.execute_query(query, (user_id, limit))
        
        strategies = []
        for row in rows:
            strategy_data = {
                "id": row["id"],
                "user_id": row["user_id"],
                "name": row["name"],
                "description": row["description"],
                "strategy_type": row["strategy_type"],
                "symbols": json.loads(row["symbols"]),
                "buy_conditions": [StrategyCondition.from_dict(c) for c in json.loads(row["buy_conditions"])],
                "sell_conditions": [StrategyCondition.from_dict(c) for c in json.loads(row["sell_conditions"])],
                "risk_management": json.loads(row["risk_management"]),
                "tier_required": row["tier_required"],
                "is_active": bool(row["is_active"]),
                "created_at": datetime.fromisoformat(row["created_at"]),
                "updated_at": datetime.fromisoformat(row["updated_at"]),
                "performance_metrics": json.loads(row["performance_metrics"]) if row["performance_metrics"] else None,
                "backtest_results": json.loads(row["backtest_results"]) if row["backtest_results"] else None
            }
            strategies.append(Strategy.from_dict(strategy_data).to_dict())
        
        return {"strategies": strategies}
        
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/")
async def create_strategy(strategy_data: Dict[str, Any]):
    """Create a new strategy"""
    
    try:
        # Validate required fields
        required_fields = ["user_id", "name", "strategy_type", "symbols", "buy_conditions", "sell_conditions"]
        for field in required_fields:
            if field not in strategy_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create strategy object
        strategy = Strategy(
            id=str(uuid.uuid4()),
            user_id=strategy_data["user_id"],
            name=strategy_data["name"],
            description=strategy_data.get("description", ""),
            strategy_type=strategy_data["strategy_type"],
            symbols=strategy_data["symbols"],
            buy_conditions=[StrategyCondition.from_dict(c) for c in strategy_data["buy_conditions"]],
            sell_conditions=[StrategyCondition.from_dict(c) for c in strategy_data["sell_conditions"]],
            risk_management=strategy_data.get("risk_management", {"stop_loss": 0.05, "take_profit": 0.10}),
            tier_required=strategy_data.get("tier_required", 1),
            is_active=strategy_data.get("is_active", True),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save to database
        query = """
            INSERT INTO strategies (id, user_id, name, description, strategy_type, symbols,
                                  buy_conditions, sell_conditions, risk_management, tier_required,
                                  is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            strategy.id,
            strategy.user_id,
            strategy.name,
            strategy.description,
            strategy.strategy_type,
            json.dumps(strategy.symbols),
            json.dumps([c.to_dict() for c in strategy.buy_conditions]),
            json.dumps([c.to_dict() for c in strategy.sell_conditions]),
            json.dumps(strategy.risk_management),
            strategy.tier_required,
            strategy.is_active,
            strategy.created_at.isoformat(),
            strategy.updated_at.isoformat()
        )
        
        await db_manager.execute_write(query, params)
        
        # Start monitoring if active
        if strategy.is_active:
            signal_generator.active_strategies[strategy.id] = strategy
            await signal_generator.start_strategy_monitoring(strategy.id)
        
        return {"strategy": strategy.to_dict(), "message": "Strategy created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Get a specific strategy"""
    
    try:
        query = """
            SELECT id, user_id, name, description, strategy_type, symbols, 
                   buy_conditions, sell_conditions, risk_management, tier_required,
                   is_active, created_at, updated_at, performance_metrics, backtest_results
            FROM strategies 
            WHERE id = ?
        """
        
        row = await db_manager.execute_single(query, (strategy_id,))
        
        if not row:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy_data = {
            "id": row["id"],
            "user_id": row["user_id"],
            "name": row["name"],
            "description": row["description"],
            "strategy_type": row["strategy_type"],
            "symbols": json.loads(row["symbols"]),
            "buy_conditions": [StrategyCondition.from_dict(c) for c in json.loads(row["buy_conditions"])],
            "sell_conditions": [StrategyCondition.from_dict(c) for c in json.loads(row["sell_conditions"])],
            "risk_management": json.loads(row["risk_management"]),
            "tier_required": row["tier_required"],
            "is_active": bool(row["is_active"]),
            "created_at": datetime.fromisoformat(row["created_at"]),
            "updated_at": datetime.fromisoformat(row["updated_at"]),
            "performance_metrics": json.loads(row["performance_metrics"]) if row["performance_metrics"] else None,
            "backtest_results": json.loads(row["backtest_results"]) if row["backtest_results"] else None
        }
        
        strategy = Strategy.from_dict(strategy_data)
        return {"strategy": strategy.to_dict()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{strategy_id}")
async def update_strategy(strategy_id: str, strategy_data: Dict[str, Any]):
    """Update a strategy"""
    
    try:
        # Check if strategy exists
        existing = await db_manager.execute_single(
            "SELECT id FROM strategies WHERE id = ?", (strategy_id,)
        )
        
        if not existing:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Update strategy
        update_fields = []
        params = []
        
        if "name" in strategy_data:
            update_fields.append("name = ?")
            params.append(strategy_data["name"])
        
        if "description" in strategy_data:
            update_fields.append("description = ?")
            params.append(strategy_data["description"])
        
        if "symbols" in strategy_data:
            update_fields.append("symbols = ?")
            params.append(json.dumps(strategy_data["symbols"]))
        
        if "buy_conditions" in strategy_data:
            update_fields.append("buy_conditions = ?")
            params.append(json.dumps([c if isinstance(c, dict) else c.to_dict() for c in strategy_data["buy_conditions"]]))
        
        if "sell_conditions" in strategy_data:
            update_fields.append("sell_conditions = ?")
            params.append(json.dumps([c if isinstance(c, dict) else c.to_dict() for c in strategy_data["sell_conditions"]]))
        
        if "risk_management" in strategy_data:
            update_fields.append("risk_management = ?")
            params.append(json.dumps(strategy_data["risk_management"]))
        
        if "is_active" in strategy_data:
            update_fields.append("is_active = ?")
            params.append(strategy_data["is_active"])
        
        update_fields.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        
        params.append(strategy_id)
        
        query = f"UPDATE strategies SET {', '.join(update_fields)} WHERE id = ?"
        await db_manager.execute_write(query, tuple(params))
        
        # Update monitoring
        if "is_active" in strategy_data:
            if strategy_data["is_active"]:
                await signal_generator.load_active_strategies()
                await signal_generator.start_strategy_monitoring(strategy_id)
            else:
                await signal_generator.stop_strategy_monitoring(strategy_id)
        
        return {"message": "Strategy updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete a strategy"""
    
    try:
        # Check if strategy exists
        existing = await db_manager.execute_single(
            "SELECT id FROM strategies WHERE id = ?", (strategy_id,)
        )
        
        if not existing:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Stop monitoring
        await signal_generator.stop_strategy_monitoring(strategy_id)
        
        # Delete from database
        await db_manager.execute_write(
            "DELETE FROM strategies WHERE id = ?", (strategy_id,)
        )
        
        return {"message": "Strategy deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting strategy: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{strategy_id}/backtest")
async def run_backtest(strategy_id: str, backtest_data: Dict[str, Any]):
    """Run a backtest for a strategy"""
    
    try:
        # Get strategy
        strategy_row = await db_manager.execute_single(
            """
            SELECT id, user_id, name, description, strategy_type, symbols, 
                   buy_conditions, sell_conditions, risk_management, tier_required,
                   is_active, created_at, updated_at, performance_metrics, backtest_results
            FROM strategies WHERE id = ?
            """, (strategy_id,)
        )
        
        if not strategy_row:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Create strategy object
        strategy_data = {
            "id": strategy_row["id"],
            "user_id": strategy_row["user_id"],
            "name": strategy_row["name"],
            "description": strategy_row["description"],
            "strategy_type": strategy_row["strategy_type"],
            "symbols": json.loads(strategy_row["symbols"]),
            "buy_conditions": [StrategyCondition.from_dict(c) for c in json.loads(strategy_row["buy_conditions"])],
            "sell_conditions": [StrategyCondition.from_dict(c) for c in json.loads(strategy_row["sell_conditions"])],
            "risk_management": json.loads(strategy_row["risk_management"]),
            "tier_required": strategy_row["tier_required"],
            "is_active": bool(strategy_row["is_active"]),
            "created_at": datetime.fromisoformat(strategy_row["created_at"]),
            "updated_at": datetime.fromisoformat(strategy_row["updated_at"]),
            "performance_metrics": json.loads(strategy_row["performance_metrics"]) if strategy_row["performance_metrics"] else None,
            "backtest_results": json.loads(strategy_row["backtest_results"]) if strategy_row["backtest_results"] else None
        }
        
        strategy = Strategy.from_dict(strategy_data)
        
        # Parse dates
        start_date = datetime.fromisoformat(backtest_data["start_date"])
        end_date = datetime.fromisoformat(backtest_data["end_date"])
        symbol = backtest_data["symbol"]
        
        # Run backtest
        result = await backtesting_engine.run_backtest(strategy, symbol, start_date, end_date)
        
        return {"backtest_result": result.to_dict()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{strategy_id}/backtest-results")
async def get_backtest_results(strategy_id: str):
    """Get backtest results for a strategy"""
    
    try:
        results = await backtesting_engine.get_backtest_results(strategy_id)
        return {"backtest_results": [result.to_dict() for result in results]}
        
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/templates/")
async def get_strategy_templates():
    """Get strategy templates"""
    
    templates = [
        {
            "id": "golden_cross",
            "name": "Golden Cross",
            "description": "Buy when 50-day SMA crosses above 200-day SMA",
            "strategy_type": "technical",
            "buy_conditions": [
                {
                    "indicator": "sma",
                    "operator": "crosses_above",
                    "value": "sma_200",
                    "timeframe": "1D",
                    "parameters": {"period": 50}
                }
            ],
            "sell_conditions": [
                {
                    "indicator": "sma",
                    "operator": "crosses_below",
                    "value": "sma_200",
                    "timeframe": "1D",
                    "parameters": {"period": 50}
                }
            ]
        },
        {
            "id": "rsi_oversold",
            "name": "RSI Oversold",
            "description": "Buy when RSI is oversold (< 30), sell when overbought (> 70)",
            "strategy_type": "technical",
            "buy_conditions": [
                {
                    "indicator": "rsi",
                    "operator": "<",
                    "value": 30,
                    "timeframe": "1D",
                    "parameters": {"period": 14}
                }
            ],
            "sell_conditions": [
                {
                    "indicator": "rsi",
                    "operator": ">",
                    "value": 70,
                    "timeframe": "1D",
                    "parameters": {"period": 14}
                }
            ]
        },
        {
            "id": "bollinger_bounce",
            "name": "Bollinger Bounce",
            "description": "Buy when price touches lower Bollinger Band, sell at upper band",
            "strategy_type": "technical",
            "buy_conditions": [
                {
                    "indicator": "bollinger",
                    "operator": "<=",
                    "value": "bb_lower",
                    "timeframe": "1D",
                    "parameters": {"period": 20, "std": 2}
                }
            ],
            "sell_conditions": [
                {
                    "indicator": "bollinger",
                    "operator": ">=",
                    "value": "bb_upper",
                    "timeframe": "1D",
                    "parameters": {"period": 20, "std": 2}
                }
            ]
        }
    ]
    
    return {"templates": templates}
