from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import json
import uuid

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
    indicator: str
    operator: str  # >, <, >=, <=, ==, crosses_above, crosses_below
    value: Union[float, str]
    timeframe: str = "1D"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

@dataclass
class Strategy:
    id: str
    user_id: int
    name: str
    description: str
    strategy_type: str
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
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type,
            "symbols": self.symbols,
            "buy_conditions": [c.to_dict() for c in self.buy_conditions],
            "sell_conditions": [c.to_dict() for c in self.sell_conditions],
            "risk_management": self.risk_management,
            "tier_required": self.tier_required,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "performance_metrics": self.performance_metrics,
            "backtest_results": self.backtest_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            name=data["name"],
            description=data["description"],
            strategy_type=data["strategy_type"],
            symbols=data["symbols"],
            buy_conditions=[StrategyCondition.from_dict(c) for c in data["buy_conditions"]],
            sell_conditions=[StrategyCondition.from_dict(c) for c in data["sell_conditions"]],
            risk_management=data["risk_management"],
            tier_required=data["tier_required"],
            is_active=data["is_active"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            performance_metrics=data.get("performance_metrics"),
            backtest_results=data.get("backtest_results")
        )

@dataclass
class Signal:
    id: str
    strategy_id: str
    symbol: str
    signal_type: str
    confidence: float
    price: float
    timestamp: datetime
    conditions_met: List[str]
    market_data: Dict[str, Any]
    sentiment_data: Optional[Dict[str, Any]] = None
    is_processed: bool = False
    
    def to_dict(self):
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "conditions_met": self.conditions_met,
            "market_data": self.market_data,
            "sentiment_data": self.sentiment_data,
            "is_processed": self.is_processed
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            id=data["id"],
            strategy_id=data["strategy_id"],
            symbol=data["symbol"],
            signal_type=data["signal_type"],
            confidence=data["confidence"],
            price=data["price"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            conditions_met=data["conditions_met"],
            market_data=data["market_data"],
            sentiment_data=data.get("sentiment_data"),
            is_processed=data["is_processed"]
        )

@dataclass
class BacktestResult:
    id: str
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
    created_at: datetime
    
    def to_dict(self):
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "avg_trade_duration": self.avg_trade_duration,
            "profit_factor": self.profit_factor,
            "trades": self.trades,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class User:
    id: int
    username: str
    email: str
    tier: int
    created_at: datetime
    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "tier": self.tier,
            "created_at": self.created_at.isoformat()
        }
