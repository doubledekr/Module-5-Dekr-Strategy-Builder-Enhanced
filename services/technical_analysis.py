import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Any, Optional
import logging
from models import IndicatorType

logger = logging.getLogger(__name__)

class TechnicalAnalysisEngine:
    def __init__(self):
        self.indicators = {}
        
    def calculate_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Calculate technical indicators for the given dataframe"""
        
        if df.empty:
            return df
        
        result_df = df.copy()
        
        for indicator in indicators:
            try:
                if indicator == "sma":
                    result_df = self._calculate_sma(result_df)
                elif indicator == "ema":
                    result_df = self._calculate_ema(result_df)
                elif indicator == "rsi":
                    result_df = self._calculate_rsi(result_df)
                elif indicator == "macd":
                    result_df = self._calculate_macd(result_df)
                elif indicator == "bollinger":
                    result_df = self._calculate_bollinger_bands(result_df)
                elif indicator == "stochastic":
                    result_df = self._calculate_stochastic(result_df)
                elif indicator == "volume":
                    result_df = self._calculate_volume_indicators(result_df)
                else:
                    logger.warning(f"Unknown indicator: {indicator}")
                    
            except Exception as e:
                logger.error(f"Error calculating {indicator}: {str(e)}")
        
        return result_df
    
    def _calculate_sma(self, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Calculate Simple Moving Averages"""
        
        for period in periods:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
        
        return df
    
    def _calculate_ema(self, df: pd.DataFrame, periods: List[int] = [12, 26, 50]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages"""
        
        for period in periods:
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        
        df['rsi'] = ta.momentum.rsi(df['close'], window=window)
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicators"""
        
        macd = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        
        bollinger = ta.volatility.BollingerBands(df['close'], window=window, window_dev=std)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()
        df['bb_percent'] = bollinger.bollinger_pband()
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 
                                               window=k_window, smooth_window=d_window)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume indicators"""
        
        # Volume SMA
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
        
        # On Balance Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=1)
        
        return df
    
    def evaluate_condition(self, df: pd.DataFrame, condition: Dict[str, Any]) -> pd.Series:
        """Evaluate a strategy condition against the dataframe"""
        
        try:
            indicator = condition['indicator']
            operator = condition['operator']
            value = condition['value']
            parameters = condition.get('parameters', {})
            
            # Calculate indicator if not already present (skip for basic price data)
            if indicator not in df.columns and indicator not in ['price', 'close', 'open', 'high', 'low', 'volume']:
                df = self.calculate_indicators(df, [indicator])
            
            # Handle different indicator types
            if indicator == "sma":
                period = parameters.get('period', 20)
                column = f'sma_{period}'
            elif indicator == "ema":
                period = parameters.get('period', 12)
                column = f'ema_{period}'
            elif indicator == "rsi":
                column = 'rsi'
            elif indicator == "macd":
                column = 'macd'
            elif indicator == "bollinger":
                band = parameters.get('band', 'upper')
                column = f'bb_{band}'
            elif indicator == "stochastic":
                line = parameters.get('line', 'k')
                column = f'stoch_{line}'
            elif indicator == "volume":
                column = 'volume'
            elif indicator == "price":
                column = 'close'  # Price refers to closing price
            else:
                column = indicator
            
            if column not in df.columns:
                logger.warning(f"Column {column} not found in dataframe")
                return pd.Series([False] * len(df), index=df.index)
            
            # Handle special value types
            comparison_series = None
            if isinstance(value, str):
                if value == "average_volume":
                    # Calculate 20-period average volume for each row
                    if 'volume' in df.columns:
                        comparison_series = df['volume'].rolling(window=20).mean()
                    else:
                        logger.warning("Volume column not found for average_volume comparison")
                        return pd.Series([False] * len(df), index=df.index)
                elif value == "close_price":
                    value = df['close'].iloc[-1]
                elif value == "sma_20" or value == "sma":
                    if 'sma_20' not in df.columns:
                        df = self.calculate_indicators(df, ['sma'])
                    comparison_series = df['sma_20'] if 'sma_20' in df.columns else df['close']
                elif value == "sma_50":
                    if 'sma_50' not in df.columns:
                        df = self.calculate_indicators(df, ['sma'])
                    comparison_series = df['sma_50'] if 'sma_50' in df.columns else df['close']
                elif value == "sma_long" or value == "sma_200":
                    if 'sma_200' not in df.columns:
                        df = self.calculate_indicators(df, ['sma'])
                    comparison_series = df['sma_200'] if 'sma_200' in df.columns else df['close']
                elif value == "sma_short" or value == "sma_10":
                    if 'sma_10' not in df.columns:
                        df = self.calculate_indicators(df, ['sma'])
                    comparison_series = df['sma_10'] if 'sma_10' in df.columns else df['close']
                elif value == "signal" or value == "macd_signal" or value == "signal_line":
                    # Handle MACD signal line comparison
                    if 'macd_signal' not in df.columns:
                        df = self.calculate_indicators(df, ['macd'])
                    comparison_series = df['macd_signal'] if 'macd_signal' in df.columns else pd.Series([0] * len(df), index=df.index)
                elif value == "histogram" or value == "macd_histogram":
                    # Handle MACD histogram comparison  
                    if 'macd_histogram' not in df.columns:
                        df = self.calculate_indicators(df, ['macd'])
                    comparison_series = df['macd_histogram'] if 'macd_histogram' in df.columns else pd.Series([0] * len(df), index=df.index)
                elif value == "bb_upper" or value == "upper_band" or value == "upper":
                    # Handle Bollinger Band upper band comparison
                    if 'bb_upper' not in df.columns:
                        df = self.calculate_indicators(df, ['bollinger'])
                    comparison_series = df['bb_upper'] if 'bb_upper' in df.columns else df['close']
                elif value == "bb_lower" or value == "lower_band" or value == "lower":
                    # Handle Bollinger Band lower band comparison
                    if 'bb_lower' not in df.columns:
                        df = self.calculate_indicators(df, ['bollinger'])
                    comparison_series = df['bb_lower'] if 'bb_lower' in df.columns else df['close']
                elif value == "bb_middle" or value == "middle_band" or value == "middle":
                    # Handle Bollinger Band middle band comparison
                    if 'bb_middle' not in df.columns:
                        df = self.calculate_indicators(df, ['bollinger'])
                    comparison_series = df['bb_middle'] if 'bb_middle' in df.columns else df['close']
                elif value == "zero" or value == "0":
                    # Handle zero line crossovers
                    value = 0
                else:
                    # Try to convert string to float
                    try:
                        value = float(value)
                    except ValueError:
                        logger.warning(f"Cannot convert value '{value}' to numeric")
                        return pd.Series([False] * len(df), index=df.index)
            
            # Evaluate condition
            series = df[column]
            
            # Handle comparisons with dynamic series (like average_volume)
            if comparison_series is not None:
                if operator == ">":
                    return series > comparison_series
                elif operator == "<":
                    return series < comparison_series
                elif operator == ">=":
                    return series >= comparison_series
                elif operator == "<=":
                    return series <= comparison_series
                elif operator == "==":
                    return series == comparison_series
                elif operator == "crosses_above":
                    return (series > comparison_series) & (series.shift(1) <= comparison_series.shift(1))
                elif operator == "crosses_below":
                    return (series < comparison_series) & (series.shift(1) >= comparison_series.shift(1))
                else:
                    logger.warning(f"Unknown operator: {operator}")
                    return pd.Series([False] * len(df), index=df.index)
            else:
                # Regular numeric comparisons
                if operator == ">":
                    return series > value
                elif operator == "<":
                    return series < value
                elif operator == ">=":
                    return series >= value
                elif operator == "<=":
                    return series <= value
                elif operator == "==":
                    return series == value
                elif operator == "crosses_above":
                    return (series > value) & (series.shift(1) <= value)
                elif operator == "crosses_below":
                    return (series < value) & (series.shift(1) >= value)
                else:
                    logger.warning(f"Unknown operator: {operator}")
                    return pd.Series([False] * len(df), index=df.index)
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {str(e)}")
            return pd.Series([False] * len(df), index=df.index)
    
    def get_latest_values(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, float]:
        """Get the latest values for specified indicators"""
        
        if df.empty:
            return {}
        
        # Calculate indicators if not present
        df = self.calculate_indicators(df, indicators)
        
        latest_values = {}
        
        for indicator in indicators:
            try:
                if indicator == "sma":
                    latest_values['sma_20'] = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else None
                    latest_values['sma_50'] = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else None
                elif indicator == "ema":
                    latest_values['ema_12'] = df['ema_12'].iloc[-1] if 'ema_12' in df.columns else None
                    latest_values['ema_26'] = df['ema_26'].iloc[-1] if 'ema_26' in df.columns else None
                elif indicator == "rsi":
                    latest_values['rsi'] = df['rsi'].iloc[-1] if 'rsi' in df.columns else None
                elif indicator == "macd":
                    latest_values['macd'] = df['macd'].iloc[-1] if 'macd' in df.columns else None
                    latest_values['macd_signal'] = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else None
                elif indicator == "bollinger":
                    latest_values['bb_upper'] = df['bb_upper'].iloc[-1] if 'bb_upper' in df.columns else None
                    latest_values['bb_lower'] = df['bb_lower'].iloc[-1] if 'bb_lower' in df.columns else None
                    latest_values['bb_percent'] = df['bb_percent'].iloc[-1] if 'bb_percent' in df.columns else None
                elif indicator == "stochastic":
                    latest_values['stoch_k'] = df['stoch_k'].iloc[-1] if 'stoch_k' in df.columns else None
                    latest_values['stoch_d'] = df['stoch_d'].iloc[-1] if 'stoch_d' in df.columns else None
                
            except Exception as e:
                logger.error(f"Error getting latest value for {indicator}: {str(e)}")
        
        return latest_values

# Global instance
technical_analysis = TechnicalAnalysisEngine()
