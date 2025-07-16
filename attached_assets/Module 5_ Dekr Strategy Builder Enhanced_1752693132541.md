# Module 5: Dekr Strategy Builder Enhanced

## Overview

The Strategy Builder Enhanced module represents a sophisticated evolution of your existing strategy generation system, specifically redesigned to leverage Polygon.io's comprehensive market data while maintaining the intuitive natural language strategy creation that makes your platform unique. This enhanced version transforms your current Twelve Data implementation into a more powerful, scalable, and feature-rich system that seamlessly integrates with the broader Dekr ecosystem while providing advanced signal generation capabilities for automated trading notifications.

The module preserves the core philosophy of your original strategy builder - enabling users to create complex trading strategies through simple, conversational interfaces - while dramatically expanding the underlying data sources, analytical capabilities, and integration points. The enhanced system provides real-time signal generation, comprehensive backtesting with Polygon.io's extensive historical data, and sophisticated notification triggers that work seamlessly with the broader Dekr platform's tier-based access controls and user preference systems.

## Migration from Twelve Data to Polygon.io

The migration from Twelve Data to Polygon.io represents a fundamental enhancement of your strategy builder's capabilities, providing access to more comprehensive market data, better real-time streaming, and significantly improved cost efficiency. The enhanced system maintains full backward compatibility with your existing strategy definitions while expanding the available data points and analytical capabilities through Polygon.io's rich API ecosystem.

The migration process preserves all existing strategy logic while enhancing data quality and availability. Polygon.io provides more granular intraday data, comprehensive options data, and better corporate actions handling, enabling more sophisticated strategy development and more accurate backtesting results. The enhanced system implements intelligent data mapping that automatically translates existing strategy parameters to leverage Polygon.io's expanded data set while maintaining strategy performance characteristics.

The new implementation provides significant performance improvements through Polygon.io's unlimited API access, eliminating the rate limiting constraints that previously restricted strategy complexity and real-time signal generation. Enhanced WebSocket integration enables true real-time strategy monitoring and signal generation, providing users with immediate notifications when strategy conditions are met.

## Enhanced Signal Generation System

The enhanced signal generation system represents a complete redesign of strategy monitoring and notification capabilities, providing real-time signal detection with sophisticated filtering and validation mechanisms. The system continuously monitors market conditions for all active strategies, generating buy and sell signals based on user-defined criteria while implementing advanced validation to minimize false signals and optimize signal quality.

The signal generation engine processes multiple data streams simultaneously, including real-time price data, volume indicators, technical analysis signals, and news sentiment scores from the integrated News Sentiment Service. This multi-dimensional approach enables more accurate signal generation that considers both technical and fundamental factors, providing users with higher-quality trading opportunities and reducing the likelihood of false signals.

The system implements sophisticated signal validation mechanisms that analyze historical signal performance, market conditions, and strategy-specific parameters to determine signal confidence levels. High-confidence signals trigger immediate notifications through multiple channels, while lower-confidence signals are flagged for user review. The validation system continuously learns from signal outcomes, improving accuracy and reliability over time.

## Tier-Based Strategy Features

The enhanced strategy builder implements comprehensive tier-based feature access that provides genuine value at every subscription level while creating natural upgrade incentives through progressively advanced capabilities. Freemium users gain access to basic strategy templates and simple technical indicators, providing sufficient functionality to demonstrate platform value while encouraging upgrades for more sophisticated strategy development.

Higher subscription tiers progressively unlock advanced features including custom indicator development, multi-timeframe analysis, options strategy integration, and institutional-grade backtesting capabilities. The Dark Pool Insider tier provides access to alternative data sources and institutional trading signals, while the Algorithmic Trader tier includes API access for automated strategy execution and custom integration capabilities.

The tier system ensures that strategy complexity and computational requirements scale appropriately with subscription levels, maintaining system performance while providing clear value differentiation. Advanced tiers receive priority processing for signal generation and backtesting, ensuring that premium users experience optimal performance during high-demand periods.

## Replit Implementation Prompt

```
Enhance the existing Dekr Strategy Builder to use Polygon.io instead of Twelve Data, add real-time signal generation, and integrate with the broader Dekr ecosystem for comprehensive trading strategy development and monitoring.

PROJECT SETUP:
Create a new Python Replit project named "dekr-strategy-builder-enhanced" and implement a FastAPI-based microservice that provides advanced strategy building, backtesting, and real-time signal generation capabilities.

CORE REQUIREMENTS:
- FastAPI application with strategy building and signal generation endpoints
- Migration from Twelve Data to Polygon.io for all market data
- Real-time signal generation with WebSocket streaming
- Comprehensive backtesting with historical data analysis
- Tier-based strategy features and access controls
- Integration with news sentiment and user preferences
- Automated notification triggers for buy/sell signals
- Strategy performance analytics and optimization

IMPLEMENTATION STRUCTURE:
```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aiohttp
import json
import redis
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import os
import uuid
import websockets
from concurrent.futures import ThreadPoolExecutor
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import sqlite3
import threading
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dekr Strategy Builder Enhanced", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataTier(Enum):
    FREEMIUM = 1
    MARKET_HOURS_PRO = 2
    SECTOR_SPECIALIST = 3
    WEEKEND_WARRIOR = 4
    DARK_POOL_INSIDER = 5
    ALGORITHMIC_TRADER = 6
    INSTITUTIONAL_ELITE = 7

class StrategyType(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    HYBRID = "hybrid"
    CUSTOM = "custom"

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class IndicatorType(Enum):
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"
    STOCHASTIC = "stochastic"
    VOLUME = "volume"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"

@dataclass
class StrategyCondition:
    indicator: IndicatorType
    operator: str  # >, <, >=, <=, ==, crosses_above, crosses_below
    value: Union[float, str]
    timeframe: str = "1D"
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Strategy:
    strategy_id: str
    user_id: str
    name: str
    description: str
    strategy_type: StrategyType
    symbols: List[str]
    buy_conditions: List[StrategyCondition]
    sell_conditions: List[StrategyCondition]
    risk_management: Dict[str, Any]
    tier_required: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    performance_metrics: Optional[Dict[str, Any]] = None
    backtest_results: Optional[Dict[str, Any]] = None

@dataclass
class Signal:
    signal_id: str
    strategy_id: str
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: datetime
    conditions_met: List[str]
    market_data: Dict[str, Any]
    sentiment_data: Optional[Dict[str, Any]] = None
    is_processed: bool = False

@dataclass
class BacktestResult:
    strategy_id: str
    symbol: str
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration: float
    profit_factor: float
    trades: List[Dict[str, Any]]

class PolygonDataService:
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY')
        self.base_url = "https://api.polygon.io"
        self.ws_url = "wss://socket.polygon.io"
        self.session = None
        
    async def get_historical_data(self, symbol: str, timespan: str, start_date: str, 
                                end_date: str, limit: int = 5000) -> pd.DataFrame:
        """Get historical OHLCV data from Polygon.io"""
        
        if not self.api_key:
            raise HTTPException(status_code=500, detail="Polygon API key not configured")
        
        try:
            params = {
                'apikey': self.api_key,
                'adjusted': 'true',
                'sort': 'asc',
                'limit': limit
            }
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start_date}/{end_date}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        
                        if not results:
                            return pd.DataFrame()
                        
                        df = pd.DataFrame(results)
                        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                        df = df.rename(columns={
                            'o': 'open',
                            'h': 'high',
                            'l': 'low',
                            'c': 'close',
                            'v': 'volume'
                        })
                        df = df.set_index('timestamp')
                        df = df[['open', 'high', 'low', 'close', 'volume']]
                        
                        return df
                    else:
                        logger.error(f"Polygon API error: {response.status}")
                        return pd.DataFrame()
                        
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote from Polygon.io"""
        
        try:
            params = {'apikey': self.api_key}
            url = f"{self.base_url}/v2/last/trade/{symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', {})
                    else:
                        return {}
                        
        except Exception as e:
            logger.error(f"Error fetching real-time quote: {str(e)}")
            return {}

class TechnicalAnalysisEngine:
    def __init__(self):
        self.indicators = {}
        
    def calculate_indicators(self, df: pd.DataFrame, indicators: List[IndicatorType]) -> pd.DataFrame:
        """Calculate technical indicators for the given dataframe"""
        
        result_df = df.copy()
        
        for indicator in indicators:
            if indicator == IndicatorType.SMA:
                result_df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
                result_df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
                result_df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
                
            elif indicator == IndicatorType.EMA:
                result_df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
                result_df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
                
            elif indicator == IndicatorType.RSI:
                result_df['rsi'] = ta.momentum.rsi(df['close'], window=14)
                
            elif indicator == IndicatorType.MACD:
                macd = ta.trend.MACD(df['close'])
                result_df['macd'] = macd.macd()
                result_df['macd_signal'] = macd.macd_signal()
                result_df['macd_histogram'] = macd.macd_diff()
                
            elif indicator == IndicatorType.BOLLINGER:
                bollinger = ta.volatility.BollingerBands(df['close'])
                result_df['bb_upper'] = bollinger.bollinger_hband()
                result_df['bb_middle'] = bollinger.bollinger_mavg()
                result_df['bb_lower'] = bollinger.bollinger_lband()
                
            elif indicator == IndicatorType.STOCHASTIC:
                stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
                result_df['stoch_k'] = stoch.stoch()
                result_df['stoch_d'] = stoch.stoch_signal()
                
            elif indicator == IndicatorType.VOLUME:
                result_df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
                result_df['volume_ratio'] = df['volume'] / result_df['volume_sma']
        
        return result_df
    
    def evaluate_condition(self, df: pd.DataFrame, condition: StrategyCondition) -> pd.Series:
        """Evaluate a strategy condition against the dataframe"""
        
        indicator_col = self.get_indicator_column(condition.indicator, condition.parameters)
        
        if indicator_col not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        
        if condition.operator == '>':
            return df[indicator_col] > float(condition.value)
        elif condition.operator == '<':
            return df[indicator_col] < float(condition.value)
        elif condition.operator == '>=':
            return df[indicator_col] >= float(condition.value)
        elif condition.operator == '<=':
            return df[indicator_col] <= float(condition.value)
        elif condition.operator == '==':
            return df[indicator_col] == float(condition.value)
        elif condition.operator == 'crosses_above':
            if isinstance(condition.value, str) and condition.value in df.columns:
                return (df[indicator_col] > df[condition.value]) & (df[indicator_col].shift(1) <= df[condition.value].shift(1))
            else:
                return (df[indicator_col] > float(condition.value)) & (df[indicator_col].shift(1) <= float(condition.value))
        elif condition.operator == 'crosses_below':
            if isinstance(condition.value, str) and condition.value in df.columns:
                return (df[indicator_col] < df[condition.value]) & (df[indicator_col].shift(1) >= df[condition.value].shift(1))
            else:
                return (df[indicator_col] < float(condition.value)) & (df[indicator_col].shift(1) >= float(condition.value))
        
        return pd.Series([False] * len(df), index=df.index)
    
    def get_indicator_column(self, indicator: IndicatorType, parameters: Dict[str, Any]) -> str:
        """Get the column name for an indicator"""
        
        if indicator == IndicatorType.SMA:
            window = parameters.get('window', 20)
            return f'sma_{window}'
        elif indicator == IndicatorType.EMA:
            window = parameters.get('window', 12)
            return f'ema_{window}'
        elif indicator == IndicatorType.RSI:
            return 'rsi'
        elif indicator == IndicatorType.MACD:
            return 'macd'
        elif indicator == IndicatorType.BOLLINGER:
            band = parameters.get('band', 'middle')
            return f'bb_{band}'
        elif indicator == IndicatorType.STOCHASTIC:
            line = parameters.get('line', 'k')
            return f'stoch_{line}'
        elif indicator == IndicatorType.VOLUME:
            return 'volume_ratio'
        
        return 'close'  # Default fallback

class StrategyEngine:
    def __init__(self):
        self.polygon_service = PolygonDataService()
        self.ta_engine = TechnicalAnalysisEngine()
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0,
            decode_responses=True
        )
        
        # Initialize SQLite database for strategy storage
        self.init_database()
        
        # Tier-based strategy limits
        self.tier_limits = {
            DataTier.FREEMIUM: {
                'max_strategies': 1,
                'max_symbols_per_strategy': 1,
                'max_conditions': 2,
                'backtest_period_days': 30,
                'available_indicators': [IndicatorType.SMA, IndicatorType.RSI]
            },
            DataTier.MARKET_HOURS_PRO: {
                'max_strategies': 3,
                'max_symbols_per_strategy': 3,
                'max_conditions': 5,
                'backtest_period_days': 90,
                'available_indicators': [IndicatorType.SMA, IndicatorType.EMA, IndicatorType.RSI, IndicatorType.MACD]
            },
            DataTier.SECTOR_SPECIALIST: {
                'max_strategies': 5,
                'max_symbols_per_strategy': 5,
                'max_conditions': 8,
                'backtest_period_days': 180,
                'available_indicators': [IndicatorType.SMA, IndicatorType.EMA, IndicatorType.RSI, 
                                       IndicatorType.MACD, IndicatorType.BOLLINGER]
            },
            DataTier.WEEKEND_WARRIOR: {
                'max_strategies': 10,
                'max_symbols_per_strategy': 10,
                'max_conditions': 12,
                'backtest_period_days': 365,
                'available_indicators': list(IndicatorType)
            },
            DataTier.DARK_POOL_INSIDER: {
                'max_strategies': 20,
                'max_symbols_per_strategy': 25,
                'max_conditions': 20,
                'backtest_period_days': 730,
                'available_indicators': list(IndicatorType)
            },
            DataTier.ALGORITHMIC_TRADER: {
                'max_strategies': 50,
                'max_symbols_per_strategy': 50,
                'max_conditions': 50,
                'backtest_period_days': 1825,
                'available_indicators': list(IndicatorType)
            },
            DataTier.INSTITUTIONAL_ELITE: {
                'max_strategies': -1,  # Unlimited
                'max_symbols_per_strategy': -1,
                'max_conditions': -1,
                'backtest_period_days': 3650,
                'available_indicators': list(IndicatorType)
            }
        }
    
    def init_database(self):
        """Initialize SQLite database for strategy storage"""
        
        conn = sqlite3.connect('strategies.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                strategy_type TEXT,
                symbols TEXT,
                buy_conditions TEXT,
                sell_conditions TEXT,
                risk_management TEXT,
                tier_required INTEGER,
                is_active BOOLEAN,
                created_at TEXT,
                updated_at TEXT,
                performance_metrics TEXT,
                backtest_results TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                strategy_id TEXT,
                symbol TEXT,
                signal_type TEXT,
                confidence REAL,
                price REAL,
                timestamp TEXT,
                conditions_met TEXT,
                market_data TEXT,
                sentiment_data TEXT,
                is_processed BOOLEAN,
                FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def create_strategy(self, user_id: str, strategy_data: Dict[str, Any], tier: DataTier) -> Strategy:
        """Create a new trading strategy"""
        
        # Validate tier limits
        tier_config = self.tier_limits[tier]
        
        # Check existing strategy count
        existing_strategies = await self.get_user_strategies(user_id)
        if tier_config['max_strategies'] > 0 and len(existing_strategies) >= tier_config['max_strategies']:
            raise HTTPException(status_code=400, detail="Strategy limit reached for your tier")
        
        # Validate strategy parameters
        symbols = strategy_data.get('symbols', [])
        if tier_config['max_symbols_per_strategy'] > 0 and len(symbols) > tier_config['max_symbols_per_strategy']:
            raise HTTPException(status_code=400, detail="Too many symbols for your tier")
        
        buy_conditions = [StrategyCondition(**cond) for cond in strategy_data.get('buy_conditions', [])]
        sell_conditions = [StrategyCondition(**cond) for cond in strategy_data.get('sell_conditions', [])]
        
        total_conditions = len(buy_conditions) + len(sell_conditions)
        if tier_config['max_conditions'] > 0 and total_conditions > tier_config['max_conditions']:
            raise HTTPException(status_code=400, detail="Too many conditions for your tier")
        
        # Validate indicators
        all_indicators = [cond.indicator for cond in buy_conditions + sell_conditions]
        available_indicators = tier_config['available_indicators']
        for indicator in all_indicators:
            if indicator not in available_indicators:
                raise HTTPException(status_code=400, detail=f"Indicator {indicator.value} not available for your tier")
        
        # Create strategy
        strategy = Strategy(
            strategy_id=str(uuid.uuid4()),
            user_id=user_id,
            name=strategy_data['name'],
            description=strategy_data.get('description', ''),
            strategy_type=StrategyType(strategy_data.get('strategy_type', 'technical')),
            symbols=symbols,
            buy_conditions=buy_conditions,
            sell_conditions=sell_conditions,
            risk_management=strategy_data.get('risk_management', {}),
            tier_required=tier.value,
            is_active=False,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save to database
        await self.save_strategy(strategy)
        
        return strategy
    
    async def save_strategy(self, strategy: Strategy):
        """Save strategy to database"""
        
        conn = sqlite3.connect('strategies.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO strategies 
            (strategy_id, user_id, name, description, strategy_type, symbols, 
             buy_conditions, sell_conditions, risk_management, tier_required, 
             is_active, created_at, updated_at, performance_metrics, backtest_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy.strategy_id,
            strategy.user_id,
            strategy.name,
            strategy.description,
            strategy.strategy_type.value,
            json.dumps(strategy.symbols),
            json.dumps([asdict(cond) for cond in strategy.buy_conditions]),
            json.dumps([asdict(cond) for cond in strategy.sell_conditions]),
            json.dumps(strategy.risk_management),
            strategy.tier_required,
            strategy.is_active,
            strategy.created_at.isoformat(),
            strategy.updated_at.isoformat(),
            json.dumps(strategy.performance_metrics) if strategy.performance_metrics else None,
            json.dumps(strategy.backtest_results) if strategy.backtest_results else None
        ))
        
        conn.commit()
        conn.close()
    
    async def get_user_strategies(self, user_id: str) -> List[Strategy]:
        """Get all strategies for a user"""
        
        conn = sqlite3.connect('strategies.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM strategies WHERE user_id = ?', (user_id,))
        rows = cursor.fetchall()
        
        strategies = []
        for row in rows:
            strategy = Strategy(
                strategy_id=row[0],
                user_id=row[1],
                name=row[2],
                description=row[3],
                strategy_type=StrategyType(row[4]),
                symbols=json.loads(row[5]),
                buy_conditions=[StrategyCondition(**cond) for cond in json.loads(row[6])],
                sell_conditions=[StrategyCondition(**cond) for cond in json.loads(row[7])],
                risk_management=json.loads(row[8]),
                tier_required=row[9],
                is_active=row[10],
                created_at=datetime.fromisoformat(row[11]),
                updated_at=datetime.fromisoformat(row[12]),
                performance_metrics=json.loads(row[13]) if row[13] else None,
                backtest_results=json.loads(row[14]) if row[14] else None
            )
            strategies.append(strategy)
        
        conn.close()
        return strategies
    
    async def backtest_strategy(self, strategy: Strategy, start_date: str, end_date: str) -> BacktestResult:
        """Backtest a strategy using historical data"""
        
        results = []
        
        for symbol in strategy.symbols:
            # Get historical data
            df = await self.polygon_service.get_historical_data(symbol, 'day', start_date, end_date)
            
            if df.empty:
                continue
            
            # Calculate technical indicators
            all_indicators = list(set([cond.indicator for cond in strategy.buy_conditions + strategy.sell_conditions]))
            df_with_indicators = self.ta_engine.calculate_indicators(df, all_indicators)
            
            # Evaluate buy and sell conditions
            buy_signals = pd.Series([True] * len(df_with_indicators), index=df_with_indicators.index)
            for condition in strategy.buy_conditions:
                condition_result = self.ta_engine.evaluate_condition(df_with_indicators, condition)
                buy_signals = buy_signals & condition_result
            
            sell_signals = pd.Series([True] * len(df_with_indicators), index=df_with_indicators.index)
            for condition in strategy.sell_conditions:
                condition_result = self.ta_engine.evaluate_condition(df_with_indicators, condition)
                sell_signals = sell_signals & condition_result
            
            # Simulate trades
            trades = self.simulate_trades(df_with_indicators, buy_signals, sell_signals, symbol)
            results.extend(trades)
        
        # Calculate performance metrics
        if not results:
            return BacktestResult(
                strategy_id=strategy.strategy_id,
                symbol="ALL",
                start_date=datetime.fromisoformat(start_date),
                end_date=datetime.fromisoformat(end_date),
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_duration=0.0,
                profit_factor=0.0,
                trades=[]
            )
        
        # Calculate aggregate metrics
        total_return = sum(trade['return_pct'] for trade in results)
        winning_trades = [trade for trade in results if trade['return_pct'] > 0]
        losing_trades = [trade for trade in results if trade['return_pct'] < 0]
        
        win_rate = len(winning_trades) / len(results) if results else 0
        avg_trade_duration = np.mean([trade['duration_days'] for trade in results])
        
        gross_profit = sum(trade['return_pct'] for trade in winning_trades)
        gross_loss = abs(sum(trade['return_pct'] for trade in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        returns = [trade['return_pct'] for trade in results]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Calculate max drawdown (simplified)
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns)
        
        # Annualized return
        days = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days
        annualized_return = (total_return / 100) * (365 / days) * 100 if days > 0 else 0
        
        return BacktestResult(
            strategy_id=strategy.strategy_id,
            symbol="ALL",
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate * 100,
            total_trades=len(results),
            avg_trade_duration=avg_trade_duration,
            profit_factor=profit_factor,
            trades=results
        )
    
    def simulate_trades(self, df: pd.DataFrame, buy_signals: pd.Series, sell_signals: pd.Series, symbol: str) -> List[Dict[str, Any]]:
        """Simulate trades based on buy and sell signals"""
        
        trades = []
        position = None
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if position is None and buy_signals.iloc[i]:
                # Open long position
                position = {
                    'symbol': symbol,
                    'entry_date': timestamp,
                    'entry_price': row['close'],
                    'type': 'long'
                }
            
            elif position is not None and sell_signals.iloc[i]:
                # Close position
                exit_price = row['close']
                duration = (timestamp - position['entry_date']).days
                return_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                
                trade = {
                    'symbol': symbol,
                    'entry_date': position['entry_date'].isoformat(),
                    'exit_date': timestamp.isoformat(),
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'return_pct': return_pct,
                    'duration_days': duration,
                    'type': position['type']
                }
                
                trades.append(trade)
                position = None
        
        return trades
    
    async def generate_signals(self, strategy: Strategy) -> List[Signal]:
        """Generate real-time signals for a strategy"""
        
        signals = []
        
        for symbol in strategy.symbols:
            # Get recent data for signal generation
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            df = await self.polygon_service.get_historical_data(symbol, 'day', start_date, end_date)
            
            if df.empty:
                continue
            
            # Calculate indicators
            all_indicators = list(set([cond.indicator for cond in strategy.buy_conditions + strategy.sell_conditions]))
            df_with_indicators = self.ta_engine.calculate_indicators(df, all_indicators)
            
            # Check latest conditions
            latest_data = df_with_indicators.iloc[-1]
            latest_timestamp = df_with_indicators.index[-1]
            
            # Evaluate buy conditions
            buy_conditions_met = []
            buy_signal = True
            for condition in strategy.buy_conditions:
                condition_result = self.ta_engine.evaluate_condition(df_with_indicators, condition)
                if condition_result.iloc[-1]:
                    buy_conditions_met.append(f"{condition.indicator.value} {condition.operator} {condition.value}")
                else:
                    buy_signal = False
            
            # Evaluate sell conditions
            sell_conditions_met = []
            sell_signal = True
            for condition in strategy.sell_conditions:
                condition_result = self.ta_engine.evaluate_condition(df_with_indicators, condition)
                if condition_result.iloc[-1]:
                    sell_conditions_met.append(f"{condition.indicator.value} {condition.operator} {condition.value}")
                else:
                    sell_signal = False
            
            # Generate signals
            if buy_signal and buy_conditions_met:
                signal = Signal(
                    signal_id=str(uuid.uuid4()),
                    strategy_id=strategy.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=0.8,  # Would be calculated based on signal strength
                    price=latest_data['close'],
                    timestamp=latest_timestamp,
                    conditions_met=buy_conditions_met,
                    market_data=latest_data.to_dict()
                )
                signals.append(signal)
            
            if sell_signal and sell_conditions_met:
                signal = Signal(
                    signal_id=str(uuid.uuid4()),
                    strategy_id=strategy.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=0.8,
                    price=latest_data['close'],
                    timestamp=latest_timestamp,
                    conditions_met=sell_conditions_met,
                    market_data=latest_data.to_dict()
                )
                signals.append(signal)
        
        # Save signals to database
        for signal in signals:
            await self.save_signal(signal)
        
        return signals
    
    async def save_signal(self, signal: Signal):
        """Save signal to database"""
        
        conn = sqlite3.connect('strategies.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals 
            (signal_id, strategy_id, symbol, signal_type, confidence, price, 
             timestamp, conditions_met, market_data, sentiment_data, is_processed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.signal_id,
            signal.strategy_id,
            signal.symbol,
            signal.signal_type.value,
            signal.confidence,
            signal.price,
            signal.timestamp.isoformat(),
            json.dumps(signal.conditions_met),
            json.dumps(signal.market_data),
            json.dumps(signal.sentiment_data) if signal.sentiment_data else None,
            signal.is_processed
        ))
        
        conn.commit()
        conn.close()

# Initialize strategy engine
strategy_engine = StrategyEngine()

# Background task for signal generation
async def signal_generation_task():
    """Background task to generate signals for active strategies"""
    
    while True:
        try:
            # Get all active strategies
            conn = sqlite3.connect('strategies.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM strategies WHERE is_active = 1')
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                strategy = Strategy(
                    strategy_id=row[0],
                    user_id=row[1],
                    name=row[2],
                    description=row[3],
                    strategy_type=StrategyType(row[4]),
                    symbols=json.loads(row[5]),
                    buy_conditions=[StrategyCondition(**cond) for cond in json.loads(row[6])],
                    sell_conditions=[StrategyCondition(**cond) for cond in json.loads(row[7])],
                    risk_management=json.loads(row[8]),
                    tier_required=row[9],
                    is_active=row[10],
                    created_at=datetime.fromisoformat(row[11]),
                    updated_at=datetime.fromisoformat(row[12])
                )
                
                # Generate signals
                signals = await strategy_engine.generate_signals(strategy)
                
                # Process signals (would trigger notifications here)
                for signal in signals:
                    logger.info(f"Generated signal: {signal.signal_type.value} for {signal.symbol}")
            
            # Wait 5 minutes before next check
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error in signal generation task: {str(e)}")
            await asyncio.sleep(60)

# Start background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(signal_generation_task())

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/strategies")
async def create_strategy(
    user_id: str,
    strategy_data: Dict[str, Any],
    tier: int = Query(1, description="User tier (1-7)")
):
    """Create a new trading strategy"""
    
    data_tier = DataTier(tier)
    strategy = await strategy_engine.create_strategy(user_id, strategy_data, data_tier)
    
    return asdict(strategy)

@app.get("/strategies/{user_id}")
async def get_user_strategies(user_id: str):
    """Get all strategies for a user"""
    
    strategies = await strategy_engine.get_user_strategies(user_id)
    return [asdict(strategy) for strategy in strategies]

@app.post("/strategies/{strategy_id}/backtest")
async def backtest_strategy(
    strategy_id: str,
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """Backtest a strategy"""
    
    # Get strategy from database
    conn = sqlite3.connect('strategies.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM strategies WHERE strategy_id = ?', (strategy_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    strategy = Strategy(
        strategy_id=row[0],
        user_id=row[1],
        name=row[2],
        description=row[3],
        strategy_type=StrategyType(row[4]),
        symbols=json.loads(row[5]),
        buy_conditions=[StrategyCondition(**cond) for cond in json.loads(row[6])],
        sell_conditions=[StrategyCondition(**cond) for cond in json.loads(row[7])],
        risk_management=json.loads(row[8]),
        tier_required=row[9],
        is_active=row[10],
        created_at=datetime.fromisoformat(row[11]),
        updated_at=datetime.fromisoformat(row[12])
    )
    
    result = await strategy_engine.backtest_strategy(strategy, start_date, end_date)
    return asdict(result)

@app.post("/strategies/{strategy_id}/activate")
async def activate_strategy(strategy_id: str):
    """Activate a strategy for signal generation"""
    
    conn = sqlite3.connect('strategies.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE strategies SET is_active = 1 WHERE strategy_id = ?', (strategy_id,))
    conn.commit()
    conn.close()
    
    return {"message": "Strategy activated", "strategy_id": strategy_id}

@app.post("/strategies/{strategy_id}/deactivate")
async def deactivate_strategy(strategy_id: str):
    """Deactivate a strategy"""
    
    conn = sqlite3.connect('strategies.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE strategies SET is_active = 0 WHERE strategy_id = ?', (strategy_id,))
    conn.commit()
    conn.close()
    
    return {"message": "Strategy deactivated", "strategy_id": strategy_id}

@app.get("/signals/{strategy_id}")
async def get_strategy_signals(
    strategy_id: str,
    limit: int = Query(50, description="Number of signals to return")
):
    """Get signals for a strategy"""
    
    conn = sqlite3.connect('strategies.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM signals WHERE strategy_id = ? ORDER BY timestamp DESC LIMIT ?',
        (strategy_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    
    signals = []
    for row in rows:
        signal = {
            'signal_id': row[0],
            'strategy_id': row[1],
            'symbol': row[2],
            'signal_type': row[3],
            'confidence': row[4],
            'price': row[5],
            'timestamp': row[6],
            'conditions_met': json.loads(row[7]),
            'market_data': json.loads(row[8]),
            'sentiment_data': json.loads(row[9]) if row[9] else None,
            'is_processed': row[10]
        }
        signals.append(signal)
    
    return signals

@app.get("/indicators/available")
async def get_available_indicators(tier: int = Query(1)):
    """Get available indicators for a tier"""
    
    data_tier = DataTier(tier)
    tier_config = strategy_engine.tier_limits[data_tier]
    
    return {
        "tier": tier,
        "available_indicators": [indicator.value for indicator in tier_config['available_indicators']],
        "limits": {
            "max_strategies": tier_config['max_strategies'],
            "max_symbols_per_strategy": tier_config['max_symbols_per_strategy'],
            "max_conditions": tier_config['max_conditions'],
            "backtest_period_days": tier_config['backtest_period_days']
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
```

ENVIRONMENT VARIABLES:
Set these in your Replit secrets:
- POLYGON_API_KEY=your_polygon_api_key
- REDIS_HOST=localhost
- REDIS_PORT=6379

REQUIREMENTS.TXT:
```
fastapi==0.104.1
uvicorn==0.24.0
aiohttp==3.9.1
redis==5.0.1
pandas==2.1.4
numpy==1.24.3
ta==0.10.2
scikit-learn==1.3.2
joblib==1.3.2
websockets==12.0
python-multipart==0.0.6
```

TESTING INSTRUCTIONS:
1. Test health: GET /health
2. Create strategy: POST /strategies
3. Backtest strategy: POST /strategies/{id}/backtest?start_date=2023-01-01&end_date=2023-12-31
4. Get available indicators: GET /indicators/available?tier=3
5. Activate strategy: POST /strategies/{id}/activate
6. Get signals: GET /signals/{strategy_id}

This enhanced implementation provides comprehensive strategy building with Polygon.io integration, real-time signal generation, and tier-based feature access.
```

## Integration with Existing Modules

The Strategy Builder Enhanced module integrates seamlessly with your existing StrategyBuilder repository while providing significant enhancements and new capabilities. The migration process preserves all existing strategy logic and user interfaces while upgrading the underlying data infrastructure to leverage Polygon.io's superior capabilities. The enhanced system maintains backward compatibility with existing strategy definitions while providing new features and improved performance.

The module integrates directly with the User Preference Engine to incorporate user investment styles and risk tolerance into strategy recommendations and signal generation. Preference data influences strategy template suggestions, indicator selections, and signal confidence calculations, ensuring that generated strategies align with user goals and comfort levels. The integration enables personalized strategy development that adapts to user behavior and performance feedback over time.

The News Sentiment Service integration provides sentiment-based signals and strategy enhancements that incorporate market sentiment into trading decisions. Strategies can include sentiment conditions alongside technical indicators, enabling more sophisticated trading approaches that consider both quantitative and qualitative market factors. The integration supports sentiment-based strategy validation and signal confirmation, improving overall strategy performance and reducing false signals.

## Performance and Scalability

The Strategy Builder Enhanced module implements comprehensive performance optimization strategies to handle complex strategy calculations and real-time signal generation at scale. Advanced caching mechanisms store calculated indicators, strategy results, and historical data to minimize computational overhead and improve response times. Intelligent cache invalidation ensures that strategies operate on fresh data while maximizing performance through strategic data reuse.

Asynchronous processing capabilities enable parallel strategy evaluation and signal generation across multiple symbols and timeframes, ensuring that the system can handle large numbers of active strategies without performance degradation. Background processing tasks handle computationally intensive operations like backtesting and indicator calculation, maintaining responsive user interfaces even during complex analytical operations.

The system implements sophisticated resource management that scales computational resources based on strategy complexity and user tier requirements. Higher-tier users receive priority processing and additional computational resources, ensuring optimal performance for premium subscribers while maintaining acceptable performance levels for all users. Database optimization strategies ensure fast strategy retrieval and signal storage even with large numbers of active strategies and historical signals.

## Advanced Analytics and Reporting

The Strategy Builder Enhanced module provides comprehensive analytics capabilities that offer deep insights into strategy performance, signal accuracy, and optimization opportunities. Real-time performance monitoring tracks strategy effectiveness across different market conditions, providing users with actionable insights for strategy refinement and optimization. Advanced metrics include risk-adjusted returns, drawdown analysis, and correlation studies that help users understand strategy behavior and performance characteristics.

Signal analytics provide detailed insights into signal accuracy, timing, and market impact, enabling users to optimize their strategies based on historical performance data. The system tracks signal-to-outcome correlations, helping users identify the most effective strategy components and eliminate underperforming elements. Machine learning algorithms analyze strategy performance patterns to suggest optimizations and improvements.

The module implements comprehensive reporting capabilities that provide users with professional-grade strategy analysis and performance documentation. Automated report generation creates detailed strategy summaries, performance analyses, and optimization recommendations that help users make informed decisions about strategy modifications and portfolio allocation. Integration with the newsletter system ensures that users receive regular updates on their strategy performance and market-relevant insights.

