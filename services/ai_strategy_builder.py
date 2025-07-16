"""
AI-powered strategy builder that converts natural language descriptions
into structured trading strategies using OpenAI
"""

import os
import json
import logging
from typing import Dict, List, Any
from openai import OpenAI
from models import Strategy, StrategyCondition, StrategyType
from datetime import datetime

logger = logging.getLogger(__name__)

class AIStrategyBuilder:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    async def parse_strategy_description(self, description: str, user_id: int) -> Dict[str, Any]:
        """
        Parse natural language strategy description and convert to structured format
        """
        try:
            # Enhanced system prompt for better strategy interpretation
            system_prompt = """You are an expert quantitative analyst and trading system architect. 
            Convert natural language trading strategy descriptions into structured JSON format.
            
            Your task is to interpret trading strategies and output them in the following JSON structure:
            {
                "name": "Strategy Name",
                "description": "Clear description of the strategy",
                "strategy_type": "technical|fundamental|sentiment|hybrid",
                "symbols": ["LIST", "OF", "SYMBOLS"],
                "buy_conditions": [
                    {
                        "indicator": "rsi|sma|ema|macd|bollinger|stochastic|volume|price",
                        "operator": ">|<|>=|<=|==|crosses_above|crosses_below",
                        "value": numeric_value_or_indicator_name,
                        "timeframe": "1m|5m|15m|1h|4h|1d",
                        "parameters": {"period": 14, "other_params": "as_needed"}
                    }
                ],
                "sell_conditions": [
                    {
                        "indicator": "rsi|sma|ema|macd|bollinger|stochastic|volume|price",
                        "operator": ">|<|>=|<=|==|crosses_above|crosses_below", 
                        "value": numeric_value_or_indicator_name,
                        "timeframe": "1m|5m|15m|1h|4h|1d",
                        "parameters": {"period": 14, "other_params": "as_needed"}
                    }
                ],
                "risk_management": {
                    "stop_loss": 0.02,
                    "take_profit": 0.06,
                    "position_size": 0.1,
                    "max_positions": 3
                },
                "tier_required": 1
            }
            
            Common trading terms and their mappings:
            - "oversold" = RSI < 30
            - "overbought" = RSI > 70
            - "golden cross" = SMA(50) crosses above SMA(200)
            - "death cross" = SMA(50) crosses below SMA(200)
            - "breakout" = price crosses above resistance or below support
            - "momentum" = price > SMA(20) and volume > average volume
            - "mean reversion" = price returns to moving average
            - "bollinger squeeze" = Bollinger band width is narrow
            - "MACD bullish" = MACD line crosses above signal line
            - "MACD bearish" = MACD line crosses below signal line
            
            If symbols are not specified, use ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"] as defaults.
            If timeframe is not specified, use "1d" as default.
            Always include reasonable risk management parameters.
            
            Return only valid JSON without any additional text or explanation."""
            
            user_prompt = f"""Convert this trading strategy description into structured JSON format:

            "{description}"
            
            Please interpret the strategy and return the complete JSON structure."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using latest OpenAI model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            strategy_data = json.loads(response.choices[0].message.content)
            
            # Add user_id and timestamps
            strategy_data["user_id"] = user_id
            strategy_data["id"] = f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            strategy_data["created_at"] = datetime.now().isoformat()
            strategy_data["updated_at"] = datetime.now().isoformat()
            strategy_data["is_active"] = True
            
            # Validate and sanitize the strategy
            validated_strategy = self._validate_strategy(strategy_data)
            
            return validated_strategy
            
        except Exception as e:
            logger.error(f"Error parsing strategy description: {e}")
            raise Exception(f"Failed to parse strategy: {str(e)}")
    
    def _validate_strategy(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize strategy data"""
        
        # Ensure required fields exist
        required_fields = ["name", "description", "strategy_type", "symbols", "buy_conditions", "sell_conditions"]
        for field in required_fields:
            if field not in strategy_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate strategy type
        valid_types = ["technical", "fundamental", "sentiment", "hybrid"]
        if strategy_data["strategy_type"] not in valid_types:
            strategy_data["strategy_type"] = "technical"
        
        # Ensure symbols is a list
        if isinstance(strategy_data["symbols"], str):
            strategy_data["symbols"] = [strategy_data["symbols"]]
        
        # Validate conditions
        strategy_data["buy_conditions"] = self._validate_conditions(strategy_data.get("buy_conditions", []))
        strategy_data["sell_conditions"] = self._validate_conditions(strategy_data.get("sell_conditions", []))
        
        # Ensure risk management exists
        if "risk_management" not in strategy_data:
            strategy_data["risk_management"] = {
                "stop_loss": 0.02,
                "take_profit": 0.06,
                "position_size": 0.1,
                "max_positions": 3
            }
        
        # Set default tier
        if "tier_required" not in strategy_data:
            strategy_data["tier_required"] = 1
            
        return strategy_data
    
    def _validate_conditions(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate individual strategy conditions"""
        valid_conditions = []
        
        valid_indicators = ["rsi", "sma", "ema", "macd", "bollinger", "stochastic", "volume", "price"]
        valid_operators = [">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"]
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
                
            # Validate indicator
            if condition.get("indicator") not in valid_indicators:
                continue
                
            # Validate operator
            if condition.get("operator") not in valid_operators:
                continue
                
            # Validate timeframe
            if condition.get("timeframe") not in valid_timeframes:
                condition["timeframe"] = "1d"
                
            # Ensure parameters exist
            if "parameters" not in condition:
                condition["parameters"] = {}
                
            valid_conditions.append(condition)
            
        return valid_conditions
    
    async def generate_strategy_examples(self) -> List[Dict[str, str]]:
        """Generate example strategy prompts for users"""
        examples = [
            {
                "title": "RSI Mean Reversion",
                "description": "Buy when RSI drops below 30 (oversold) and sell when it rises above 70 (overbought)",
                "category": "Technical Analysis"
            },
            {
                "title": "Golden Cross Momentum",
                "description": "Buy when 50-day moving average crosses above 200-day moving average, sell when it crosses below",
                "category": "Moving Averages"
            },
            {
                "title": "Bollinger Band Squeeze",
                "description": "Buy when price breaks above upper Bollinger Band with high volume, sell when it touches the lower band",
                "category": "Volatility"
            },
            {
                "title": "MACD Trend Following",
                "description": "Buy when MACD line crosses above signal line and histogram is positive, sell when MACD crosses below signal line",
                "category": "Momentum"
            },
            {
                "title": "Tech Stock Momentum",
                "description": "Buy technology stocks when price is above 20-day moving average and volume is 50% above average, sell when price drops below 20-day MA",
                "category": "Sector Strategy"
            },
            {
                "title": "Breakout Strategy",
                "description": "Buy when price breaks above resistance with volume confirmation, sell when it breaks below support",
                "category": "Breakout"
            }
        ]
        
        return examples
    
    async def explain_strategy(self, strategy_data: Dict[str, Any]) -> str:
        """Generate human-readable explanation of a strategy"""
        try:
            explanation_prompt = f"""Explain this trading strategy in simple, clear language that a beginner can understand:

            Strategy Data: {json.dumps(strategy_data, indent=2)}

            Please provide:
            1. A brief overview of what the strategy does
            2. When it triggers buy signals
            3. When it triggers sell signals
            4. The risk management approach
            5. What type of market conditions it works best in

            Keep the explanation concise and avoid technical jargon."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful trading strategy educator. Explain trading strategies in simple, clear language."},
                    {"role": "user", "content": explanation_prompt}
                ],
                temperature=0.5
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error explaining strategy: {e}")
            return "Unable to generate strategy explanation at this time."