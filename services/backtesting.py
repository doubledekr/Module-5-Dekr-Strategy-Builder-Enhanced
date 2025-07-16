import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import uuid
import json
from models import Strategy, BacktestResult, StrategyCondition
from services.polygon_service import polygon_service
from services.technical_analysis import technical_analysis
from database import db_manager

logger = logging.getLogger(__name__)

class BacktestingEngine:
    def __init__(self):
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0005   # 0.05% slippage
        
    async def run_backtest(self, strategy: Strategy, symbol: str, start_date: datetime, 
                          end_date: datetime) -> BacktestResult:
        """Run a backtest for a strategy"""
        
        try:
            # Get historical data
            df = await polygon_service.get_historical_data(
                symbol, 
                "day", 
                start_date.strftime("%Y-%m-%d"), 
                end_date.strftime("%Y-%m-%d"),
                limit=5000
            )
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Calculate technical indicators
            indicators = self._get_required_indicators(strategy)
            df = technical_analysis.calculate_indicators(df, indicators)
            
            # Run backtest simulation
            trades = await self._simulate_trades(strategy, df)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(trades, df)
            
            # Create backtest result
            result = BacktestResult(
                id=str(uuid.uuid4()),
                strategy_id=strategy.id,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                total_return=metrics['total_return'],
                annualized_return=metrics['annualized_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                total_trades=metrics['total_trades'],
                avg_trade_duration=metrics['avg_trade_duration'],
                profit_factor=metrics['profit_factor'],
                trades=trades,
                created_at=datetime.now()
            )
            
            # Save to database
            await self._save_backtest_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def _get_required_indicators(self, strategy: Strategy) -> List[str]:
        """Get list of required indicators for the strategy"""
        indicators = set()
        
        for condition in strategy.buy_conditions + strategy.sell_conditions:
            indicators.add(condition.indicator)
        
        return list(indicators)
    
    async def _simulate_trades(self, strategy: Strategy, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Simulate trades based on strategy conditions"""
        
        trades = []
        position = None
        
        for i in range(len(df)):
            current_data = df.iloc[:i+1]
            
            if len(current_data) < 50:  # Need enough data for indicators
                continue
            
            # Check for buy signal
            if position is None:
                if self._check_buy_conditions(strategy, current_data):
                    position = {
                        'entry_date': df.index[i],
                        'entry_price': df.iloc[i]['close'] * (1 + self.slippage),
                        'entry_index': i
                    }
            
            # Check for sell signal
            elif position is not None:
                if self._check_sell_conditions(strategy, current_data):
                    exit_price = df.iloc[i]['close'] * (1 - self.slippage)
                    
                    # Calculate trade metrics
                    trade = {
                        'entry_date': position['entry_date'].isoformat(),
                        'exit_date': df.index[i].isoformat(),
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'duration_days': (df.index[i] - position['entry_date']).days,
                        'return_pct': (exit_price - position['entry_price']) / position['entry_price'],
                        'profit_loss': exit_price - position['entry_price']
                    }
                    
                    # Apply commission
                    trade['profit_loss'] -= (trade['entry_price'] + trade['exit_price']) * self.commission
                    
                    trades.append(trade)
                    position = None
        
        return trades
    
    def _check_buy_conditions(self, strategy: Strategy, df: pd.DataFrame) -> bool:
        """Check if buy conditions are met"""
        
        if not strategy.buy_conditions:
            return False
        
        for condition in strategy.buy_conditions:
            condition_result = technical_analysis.evaluate_condition(df, condition.to_dict())
            
            if condition_result.empty or not condition_result.iloc[-1]:
                return False
        
        return True
    
    def _check_sell_conditions(self, strategy: Strategy, df: pd.DataFrame) -> bool:
        """Check if sell conditions are met"""
        
        if not strategy.sell_conditions:
            return False
        
        for condition in strategy.sell_conditions:
            condition_result = technical_analysis.evaluate_condition(df, condition.to_dict())
            
            if condition_result.empty or not condition_result.iloc[-1]:
                return False
        
        return True
    
    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]], df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from trades"""
        
        if not trades:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'avg_trade_duration': 0.0,
                'profit_factor': 0.0
            }
        
        # Calculate returns
        returns = [trade['return_pct'] for trade in trades]
        total_return = np.prod([1 + r for r in returns]) - 1
        
        # Calculate annualized return
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        excess_returns = np.array(returns) - risk_free_rate/252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate win rate
        winning_trades = [trade for trade in trades if trade['return_pct'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Calculate average trade duration
        avg_trade_duration = np.mean([trade['duration_days'] for trade in trades])
        
        # Calculate profit factor
        total_profit = sum([trade['profit_loss'] for trade in trades if trade['profit_loss'] > 0])
        total_loss = abs(sum([trade['profit_loss'] for trade in trades if trade['profit_loss'] < 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_trade_duration': avg_trade_duration,
            'profit_factor': profit_factor
        }
    
    async def _save_backtest_result(self, result: BacktestResult):
        """Save backtest result to database"""
        
        try:
            query = """
                INSERT INTO backtests (id, strategy_id, symbol, start_date, end_date,
                                     total_return, annualized_return, sharpe_ratio, max_drawdown,
                                     win_rate, total_trades, avg_trade_duration, profit_factor,
                                     trades, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                result.id,
                result.strategy_id,
                result.symbol,
                result.start_date.isoformat(),
                result.end_date.isoformat(),
                result.total_return,
                result.annualized_return,
                result.sharpe_ratio,
                result.max_drawdown,
                result.win_rate,
                result.total_trades,
                result.avg_trade_duration,
                result.profit_factor,
                json.dumps(result.trades),
                result.created_at.isoformat()
            )
            
            await db_manager.execute_write(query, params)
            
        except Exception as e:
            logger.error(f"Error saving backtest result: {str(e)}")
    
    async def get_backtest_results(self, strategy_id: str = None, limit: int = 50) -> List[BacktestResult]:
        """Get backtest results"""
        
        try:
            if strategy_id:
                query = """
                    SELECT * FROM backtests 
                    WHERE strategy_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """
                params = (strategy_id, limit)
            else:
                query = """
                    SELECT * FROM backtests 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """
                params = (limit,)
            
            rows = await db_manager.execute_query(query, params)
            
            results = []
            for row in rows:
                result = BacktestResult(
                    id=row["id"],
                    strategy_id=row["strategy_id"],
                    symbol=row["symbol"],
                    start_date=datetime.fromisoformat(row["start_date"]),
                    end_date=datetime.fromisoformat(row["end_date"]),
                    total_return=row["total_return"],
                    annualized_return=row["annualized_return"],
                    sharpe_ratio=row["sharpe_ratio"],
                    max_drawdown=row["max_drawdown"],
                    win_rate=row["win_rate"],
                    total_trades=row["total_trades"],
                    avg_trade_duration=row["avg_trade_duration"],
                    profit_factor=row["profit_factor"],
                    trades=json.loads(row["trades"]),
                    created_at=datetime.fromisoformat(row["created_at"])
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting backtest results: {str(e)}")
            return []

# Global instance
backtesting_engine = BacktestingEngine()
