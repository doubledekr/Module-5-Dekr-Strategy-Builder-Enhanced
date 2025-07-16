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
        
    def calculate_buy_and_hold_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate buy-and-hold performance metrics"""
        if len(df) < 2:
            return {}
            
        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        
        # Calculate total return
        total_return = (end_price - start_price) / start_price
        
        # Calculate daily returns for other metrics
        df['daily_return'] = df['close'].pct_change()
        daily_returns = df['daily_return'].dropna()
        
        # Calculate annualized return (assuming 252 trading days)
        trading_days = len(df)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        
        # Calculate volatility and Sharpe ratio
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'start_price': start_price,
            'end_price': end_price,
            'trading_days': trading_days
        }
        
    async def run_backtest(self, strategy: Strategy, symbol: str, start_date: datetime, 
                          end_date: datetime) -> Dict[str, Any]:
        """Run a backtest for a strategy and compare with buy-and-hold"""
        
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
            
            # Calculate buy-and-hold performance
            buy_hold_performance = self.calculate_buy_and_hold_performance(df.copy())
            
            # Calculate technical indicators
            indicators = self._get_required_indicators(strategy)
            df = technical_analysis.calculate_indicators(df, indicators)
            
            # Run backtest simulation
            trades = await self._simulate_trades(strategy, df)
            
            # Calculate performance metrics
            strategy_metrics = self._calculate_performance_metrics(trades, df)
            
            # Compare strategy vs buy-and-hold
            comparison = self._compare_with_buy_hold(strategy_metrics, buy_hold_performance)
            
            # Create comprehensive result
            result = {
                'strategy_performance': {
                    'id': str(uuid.uuid4()),
                    'strategy_id': strategy.id,
                    'symbol': symbol,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'total_return': strategy_metrics['total_return'],
                    'annualized_return': strategy_metrics['annualized_return'],
                    'sharpe_ratio': strategy_metrics['sharpe_ratio'],
                    'max_drawdown': strategy_metrics['max_drawdown'],
                    'win_rate': strategy_metrics['win_rate'],
                    'total_trades': strategy_metrics['total_trades'],
                    'avg_trade_duration': strategy_metrics['avg_trade_duration'],
                    'profit_factor': strategy_metrics['profit_factor'],
                    'trades': trades,
                    'created_at': datetime.now().isoformat()
                },
                'buy_hold_performance': buy_hold_performance,
                'comparison': comparison,
                'recommendation': self._generate_recommendation(comparison, strategy_metrics, buy_hold_performance)
            }
            
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
                        'entry_date': position['entry_date'].strftime('%Y-%m-%dT%H:%M:%S'),
                        'exit_date': pd.Timestamp(df.index[i]).strftime('%Y-%m-%dT%H:%M:%S'),
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
        
        # Calculate Sharpe ratio using simplified approach for trade-based returns
        # Note: This is a simplified calculation - true Sharpe should use daily returns
        if len(returns) > 1:
            risk_free_rate = 0.02
            # Convert to approximate daily returns (very rough approximation)
            avg_trade_duration = np.mean([trade['duration_days'] for trade in trades])
            daily_trade_returns = [r / trade['duration_days'] for r, trade in zip(returns, trades)]
            excess_returns = np.array(daily_trade_returns) - risk_free_rate/252
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        else:
            sharpe_ratio = 0.0
        
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
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': float(len(trades)),
            'avg_trade_duration': float(avg_trade_duration),
            'profit_factor': float(profit_factor)
        }
    
    async def _save_backtest_result(self, result: Dict[str, Any]):
        """Save backtest result to database"""
        
        try:
            # For now, just log the result since we've restructured the return format
            logger.info(f"Backtest completed for {result.get('strategy_performance', {}).get('symbol', 'Unknown')}")
            
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
    
    def _compare_with_buy_hold(self, strategy_metrics: Dict[str, float], buy_hold_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare strategy performance with buy-and-hold"""
        
        if not buy_hold_metrics:
            return {}
        
        return {
            'total_return_difference': strategy_metrics['total_return'] - buy_hold_metrics['total_return'],
            'annualized_return_difference': strategy_metrics['annualized_return'] - buy_hold_metrics['annualized_return'],
            'sharpe_ratio_difference': strategy_metrics['sharpe_ratio'] - buy_hold_metrics['sharpe_ratio'],
            'max_drawdown_difference': strategy_metrics['max_drawdown'] - buy_hold_metrics['max_drawdown'],
            'strategy_outperforms': strategy_metrics['total_return'] > buy_hold_metrics['total_return'],
            'risk_adjusted_outperforms': strategy_metrics['sharpe_ratio'] > buy_hold_metrics['sharpe_ratio'],
            'lower_drawdown': strategy_metrics['max_drawdown'] > buy_hold_metrics['max_drawdown'],  # Higher is better (less negative)
            'volatility_difference': strategy_metrics.get('volatility', 0) - buy_hold_metrics.get('volatility', 0)
        }
    
    def _generate_recommendation(self, comparison: Dict[str, Any], strategy_metrics: Dict[str, float], buy_hold_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate recommendation based on comparison"""
        
        if not comparison:
            return {'recommendation': 'Unable to compare due to insufficient data'}
        
        # Determine primary recommendation
        if comparison['strategy_outperforms']:
            if comparison['risk_adjusted_outperforms']:
                recommendation = 'STRATEGY_RECOMMENDED'
                message = 'The strategy outperforms buy-and-hold on both returns and risk-adjusted metrics.'
            else:
                recommendation = 'STRATEGY_CAUTIOUS'
                message = 'The strategy has higher returns but worse risk-adjusted performance than buy-and-hold.'
        else:
            recommendation = 'BUY_HOLD_RECOMMENDED'
            message = 'Buy-and-hold is the best course of action for this stock at the moment.'
        
        # Add detailed analysis
        details = []
        
        # Return comparison
        return_diff = comparison['total_return_difference'] * 100
        details.append(f"Strategy return: {strategy_metrics['total_return']*100:.2f}% vs Buy-Hold: {buy_hold_metrics['total_return']*100:.2f}% (Difference: {return_diff:+.2f}%)")
        
        # Risk-adjusted comparison
        details.append(f"Strategy Sharpe: {strategy_metrics['sharpe_ratio']:.2f} vs Buy-Hold: {buy_hold_metrics['sharpe_ratio']:.2f}")
        
        # Drawdown comparison
        details.append(f"Strategy Max Drawdown: {strategy_metrics['max_drawdown']*100:.2f}% vs Buy-Hold: {buy_hold_metrics['max_drawdown']*100:.2f}%")
        
        # Trading activity
        details.append(f"Total trades executed: {strategy_metrics['total_trades']}")
        details.append(f"Win rate: {strategy_metrics['win_rate']*100:.1f}%")
        
        return {
            'recommendation': recommendation,
            'message': message,
            'details': details,
            'confidence': self._calculate_confidence(comparison, strategy_metrics),
            'key_metrics': {
                'return_advantage': comparison['total_return_difference'],
                'risk_adjusted_advantage': comparison['sharpe_ratio_difference'],
                'drawdown_advantage': comparison['max_drawdown_difference'],
                'trade_frequency': strategy_metrics['total_trades']
            }
        }
    
    def _calculate_confidence(self, comparison: Dict[str, Any], strategy_metrics: Dict[str, float]) -> str:
        """Calculate confidence level for the recommendation"""
        
        return_diff = abs(comparison['total_return_difference'])
        sharpe_diff = abs(comparison['sharpe_ratio_difference'])
        trade_count = strategy_metrics['total_trades']
        
        # High confidence: significant performance difference with adequate trades
        if return_diff > 0.10 and sharpe_diff > 0.5 and trade_count >= 10:
            return 'HIGH'
        # Medium confidence: moderate performance difference
        elif return_diff > 0.05 and trade_count >= 5:
            return 'MEDIUM'
        # Low confidence: small differences or insufficient trades
        else:
            return 'LOW'

# Global instance
backtesting_engine = BacktestingEngine()
