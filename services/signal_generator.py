import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import json
from models import Strategy, Signal, SignalType, StrategyCondition
from services.polygon_service import polygon_service
from services.technical_analysis import technical_analysis
from database import db_manager

logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self):
        self.active_strategies = {}
        self.monitoring_tasks = {}
        self.signal_callbacks = []
        
    async def start_monitoring(self):
        """Start monitoring all active strategies"""
        logger.info("Starting signal monitoring")
        
        # Load active strategies
        await self.load_active_strategies()
        
        # Start monitoring tasks
        for strategy_id in self.active_strategies:
            await self.start_strategy_monitoring(strategy_id)
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        logger.info("Stopping signal monitoring")
        
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        self.monitoring_tasks.clear()
    
    async def load_active_strategies(self):
        """Load active strategies from database"""
        try:
            query = """
                SELECT id, user_id, name, description, strategy_type, symbols, 
                       buy_conditions, sell_conditions, risk_management, tier_required,
                       is_active, created_at, updated_at, performance_metrics, backtest_results
                FROM strategies 
                WHERE is_active = TRUE
            """
            
            rows = await db_manager.execute_query(query)
            
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
                
                strategy = Strategy.from_dict(strategy_data)
                self.active_strategies[strategy.id] = strategy
                
            logger.info(f"Loaded {len(self.active_strategies)} active strategies")
            
        except Exception as e:
            logger.error(f"Error loading active strategies: {str(e)}")
    
    async def start_strategy_monitoring(self, strategy_id: str):
        """Start monitoring a specific strategy"""
        if strategy_id not in self.active_strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return
        
        if strategy_id in self.monitoring_tasks:
            logger.warning(f"Strategy {strategy_id} already being monitored")
            return
        
        strategy = self.active_strategies[strategy_id]
        task = asyncio.create_task(self._monitor_strategy(strategy))
        self.monitoring_tasks[strategy_id] = task
        
        logger.info(f"Started monitoring strategy: {strategy.name}")
    
    async def stop_strategy_monitoring(self, strategy_id: str):
        """Stop monitoring a specific strategy"""
        if strategy_id in self.monitoring_tasks:
            self.monitoring_tasks[strategy_id].cancel()
            del self.monitoring_tasks[strategy_id]
            logger.info(f"Stopped monitoring strategy: {strategy_id}")
    
    async def _monitor_strategy(self, strategy: Strategy):
        """Monitor a strategy for signals"""
        while True:
            try:
                for symbol in strategy.symbols:
                    await self._check_strategy_signals(strategy, symbol)
                
                # Wait before next check (adjust based on requirements)
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                logger.info(f"Monitoring cancelled for strategy: {strategy.name}")
                break
            except Exception as e:
                logger.error(f"Error monitoring strategy {strategy.name}: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _check_strategy_signals(self, strategy: Strategy, symbol: str):
        """Check for signals for a specific strategy and symbol"""
        try:
            # Get recent market data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            df = await polygon_service.get_historical_data(
                symbol, "day", start_date, end_date, limit=100
            )
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Get current price
            current_quote = await polygon_service.get_real_time_quote(symbol)
            current_price = current_quote.get('price', df['close'].iloc[-1])
            
            # Check buy conditions
            buy_signals = await self._evaluate_conditions(df, strategy.buy_conditions)
            if buy_signals:
                await self._generate_signal(
                    strategy, symbol, SignalType.BUY, buy_signals, current_price, df
                )
            
            # Check sell conditions
            sell_signals = await self._evaluate_conditions(df, strategy.sell_conditions)
            if sell_signals:
                await self._generate_signal(
                    strategy, symbol, SignalType.SELL, sell_signals, current_price, df
                )
                
        except Exception as e:
            logger.error(f"Error checking signals for {symbol}: {str(e)}")
    
    async def _evaluate_conditions(self, df, conditions: List[StrategyCondition]) -> List[str]:
        """Evaluate strategy conditions"""
        met_conditions = []
        
        try:
            # Get required indicators
            indicators = list(set([c.indicator for c in conditions]))
            df_with_indicators = technical_analysis.calculate_indicators(df, indicators)
            
            # Evaluate each condition
            for condition in conditions:
                condition_result = technical_analysis.evaluate_condition(
                    df_with_indicators, condition.to_dict()
                )
                
                # Check if condition is met in the latest data point
                if not condition_result.empty and condition_result.iloc[-1]:
                    met_conditions.append(f"{condition.indicator} {condition.operator} {condition.value}")
            
            return met_conditions
            
        except Exception as e:
            logger.error(f"Error evaluating conditions: {str(e)}")
            return []
    
    async def _generate_signal(self, strategy: Strategy, symbol: str, signal_type: SignalType, 
                             conditions_met: List[str], price: float, df):
        """Generate a trading signal"""
        try:
            # Calculate confidence score
            confidence = self._calculate_confidence(conditions_met, df)
            
            # Get latest technical indicators
            indicators = ["sma", "ema", "rsi", "macd", "bollinger"]
            market_data = technical_analysis.get_latest_values(df, indicators)
            market_data['current_price'] = price
            market_data['volume'] = df['volume'].iloc[-1] if not df.empty else 0
            
            # Create signal
            signal = Signal(
                id=str(uuid.uuid4()),
                strategy_id=strategy.id,
                symbol=symbol,
                signal_type=signal_type.value,
                confidence=confidence,
                price=price,
                timestamp=datetime.now(),
                conditions_met=conditions_met,
                market_data=market_data,
                sentiment_data=None,  # TODO: Integrate sentiment analysis
                is_processed=False
            )
            
            # Save signal to database
            await self._save_signal(signal)
            
            # Notify callbacks
            await self._notify_signal_callbacks(signal)
            
            logger.info(f"Generated {signal_type.value} signal for {symbol} at ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
    
    def _calculate_confidence(self, conditions_met: List[str], df) -> float:
        """Calculate signal confidence based on conditions met and market data"""
        if not conditions_met:
            return 0.0
        
        # Base confidence from number of conditions met
        base_confidence = min(len(conditions_met) / 3.0, 1.0)  # Max 1.0 for 3+ conditions
        
        # Adjust based on volume (if available)
        volume_factor = 1.0
        if not df.empty and 'volume' in df.columns:
            recent_volume = df['volume'].iloc[-5:].mean()
            avg_volume = df['volume'].mean()
            if recent_volume > avg_volume * 1.5:
                volume_factor = 1.1
            elif recent_volume < avg_volume * 0.5:
                volume_factor = 0.9
        
        confidence = base_confidence * volume_factor
        return min(confidence, 1.0)
    
    async def _save_signal(self, signal: Signal):
        """Save signal to database"""
        try:
            query = """
                INSERT INTO signals (id, strategy_id, symbol, signal_type, confidence, price,
                                   timestamp, conditions_met, market_data, sentiment_data, is_processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                signal.id,
                signal.strategy_id,
                signal.symbol,
                signal.signal_type,
                signal.confidence,
                signal.price,
                signal.timestamp.isoformat(),
                json.dumps(signal.conditions_met),
                json.dumps(signal.market_data),
                json.dumps(signal.sentiment_data) if signal.sentiment_data else None,
                signal.is_processed
            )
            
            await db_manager.execute_write(query, params)
            
        except Exception as e:
            logger.error(f"Error saving signal: {str(e)}")
    
    async def _notify_signal_callbacks(self, signal: Signal):
        """Notify registered callbacks about new signal"""
        for callback in self.signal_callbacks:
            try:
                await callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {str(e)}")
    
    def add_signal_callback(self, callback):
        """Add a callback function for new signals"""
        self.signal_callbacks.append(callback)
    
    def remove_signal_callback(self, callback):
        """Remove a callback function"""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
    
    async def get_recent_signals(self, strategy_id: str = None, limit: int = 50) -> List[Signal]:
        """Get recent signals"""
        try:
            if strategy_id:
                query = """
                    SELECT * FROM signals 
                    WHERE strategy_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                params = (strategy_id, limit)
            else:
                query = """
                    SELECT * FROM signals 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                params = (limit,)
            
            rows = await db_manager.execute_query(query, params)
            
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
                signals.append(Signal.from_dict(signal_data))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {str(e)}")
            return []

# Global instance
signal_generator = SignalGenerator()
