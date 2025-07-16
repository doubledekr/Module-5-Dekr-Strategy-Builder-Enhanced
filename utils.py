import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

def get_env_var(name: str, default: str = None) -> str:
    """Get environment variable with default"""
    value = os.getenv(name, default)
    if value is None:
        logger.warning(f"Environment variable {name} not set")
    return value

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount"""
    return f"${amount:,.2f}" if currency == "USD" else f"{amount:,.2f} {currency}"

def format_percentage(value: float) -> str:
    """Format percentage value"""
    return f"{value:.2%}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change"""
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value

def validate_symbol(symbol: str) -> bool:
    """Validate stock symbol format"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation - letters and numbers only, reasonable length
    symbol = symbol.upper().strip()
    if not symbol.isalnum() or len(symbol) > 10:
        return False
    
    return True

def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe format"""
    valid_timeframes = ["1min", "5min", "15min", "30min", "1hour", "1day", "1week", "1month"]
    return timeframe in valid_timeframes

def parse_date_range(start_date: str, end_date: str) -> tuple:
    """Parse and validate date range"""
    try:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if start >= end:
            raise ValueError("Start date must be before end date")
        
        # Limit to reasonable range (e.g., 5 years)
        if (end - start).days > 1825:
            raise ValueError("Date range too large (max 5 years)")
        
        return start, end
        
    except ValueError as e:
        raise ValueError(f"Invalid date format: {str(e)}")

def sanitize_input(input_str: str, max_length: int = 255) -> str:
    """Sanitize user input"""
    if not input_str:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = input_str.strip()
    sanitized = sanitized.replace('<', '&lt;')
    sanitized = sanitized.replace('>', '&gt;')
    sanitized = sanitized.replace('"', '&quot;')
    sanitized = sanitized.replace("'", '&#x27;')
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def calculate_risk_metrics(returns: List[float]) -> Dict[str, float]:
    """Calculate risk metrics from returns"""
    if not returns:
        return {}
    
    import numpy as np
    
    returns_array = np.array(returns)
    
    # Calculate metrics
    volatility = np.std(returns_array) * np.sqrt(252)  # Annualized volatility
    mean_return = np.mean(returns_array)
    sharpe_ratio = mean_return / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    return {
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "mean_return": mean_return
    }

def calculate_position_size(account_balance: float, risk_per_trade: float, 
                          entry_price: float, stop_loss: float) -> int:
    """Calculate position size based on risk management"""
    if stop_loss >= entry_price:
        return 0
    
    risk_amount = account_balance * risk_per_trade
    price_diff = entry_price - stop_loss
    
    if price_diff <= 0:
        return 0
    
    shares = int(risk_amount / price_diff)
    return max(0, shares)

def generate_unique_id() -> str:
    """Generate unique ID"""
    import uuid
    return str(uuid.uuid4())

def time_until_market_open() -> Optional[timedelta]:
    """Calculate time until market opens"""
    now = datetime.now()
    
    # Market opens at 9:30 AM ET on weekdays
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # If it's weekend, calculate until Monday
    if now.weekday() >= 5:  # Saturday or Sunday
        days_until_monday = 7 - now.weekday()
        market_open = market_open + timedelta(days=days_until_monday)
    
    # If market already opened today, calculate until tomorrow
    elif now.time() > market_open.time():
        market_open = market_open + timedelta(days=1)
        # Skip weekend
        if market_open.weekday() >= 5:
            market_open = market_open + timedelta(days=7 - market_open.weekday())
    
    return market_open - now

def is_market_hours() -> bool:
    """Check if market is currently open"""
    now = datetime.now()
    
    # Market is closed on weekends
    if now.weekday() >= 5:
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def parse_indicators_from_conditions(conditions: List[Dict[str, Any]]) -> List[str]:
    """Parse required indicators from strategy conditions"""
    indicators = set()
    
    for condition in conditions:
        indicator = condition.get("indicator")
        if indicator:
            indicators.add(indicator)
    
    return list(indicators)

def validate_strategy_conditions(conditions: List[Dict[str, Any]]) -> List[str]:
    """Validate strategy conditions and return errors"""
    errors = []
    
    if not conditions:
        errors.append("At least one condition is required")
        return errors
    
    valid_operators = [">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"]
    valid_indicators = ["sma", "ema", "rsi", "macd", "bollinger", "stochastic", "volume"]
    
    for i, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            errors.append(f"Condition {i+1}: Must be a dictionary")
            continue
        
        # Check required fields
        if "indicator" not in condition:
            errors.append(f"Condition {i+1}: Missing 'indicator' field")
        elif condition["indicator"] not in valid_indicators:
            errors.append(f"Condition {i+1}: Invalid indicator '{condition['indicator']}'")
        
        if "operator" not in condition:
            errors.append(f"Condition {i+1}: Missing 'operator' field")
        elif condition["operator"] not in valid_operators:
            errors.append(f"Condition {i+1}: Invalid operator '{condition['operator']}'")
        
        if "value" not in condition:
            errors.append(f"Condition {i+1}: Missing 'value' field")
        
        # Validate value based on indicator
        if condition.get("indicator") == "rsi":
            value = condition.get("value")
            if not isinstance(value, (int, float)) or value < 0 or value > 100:
                errors.append(f"Condition {i+1}: RSI value must be between 0 and 100")
    
    return errors

async def fetch_with_retry(url: str, params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """Fetch data with retry logic"""
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"HTTP {response.status} on attempt {attempt + 1}")
                        
        except Exception as e:
            logger.warning(f"Request failed on attempt {attempt + 1}: {str(e)}")
            
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"Failed to fetch data after {max_retries} attempts")
    return {}

def log_performance(func_name: str, start_time: datetime, end_time: datetime):
    """Log performance metrics"""
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Function {func_name} took {duration:.3f} seconds")

class PerformanceTimer:
    """Context manager for performance timing"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        log_performance(self.name, self.start_time, end_time)
