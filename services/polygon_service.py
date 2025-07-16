import aiohttp
import asyncio
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import websockets
import ssl

logger = logging.getLogger(__name__)

class PolygonDataService:
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable is required")
        self.base_url = "https://api.polygon.io"
        self.ws_url = "wss://socket.polygon.io"
        self.session = None
        self.ws_connection = None
        
    async def get_session(self):
        """Get or create aiohttp session"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def get_historical_data(self, symbol: str, timespan: str, start_date: str, 
                                end_date: str, limit: int = 5000) -> pd.DataFrame:
        """Get historical OHLCV data from Polygon.io"""
        
        try:
            params = {
                'apikey': self.api_key,
                'adjusted': 'true',
                'sort': 'asc',
                'limit': limit
            }
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start_date}/{end_date}"
            
            # Create a new session for each request to avoid event loop issues
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        
                        if not results:
                            logger.warning(f"No data returned for {symbol}")
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
                        
                        logger.info(f"Retrieved {len(df)} data points for {symbol}")
                        return df
                    else:
                        logger.error(f"Polygon API error for {symbol}: {response.status}")
                        return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            # Close and recreate session if there's an error
            await self.close_session()
            return pd.DataFrame()
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote from Polygon.io"""
        
        try:
            params = {'apikey': self.api_key}
            url = f"{self.base_url}/v2/last/trade/{symbol}"
            
            session = await self.get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', {})
                else:
                    logger.error(f"Error fetching real-time quote for {symbol}: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {str(e)}")
            return {}
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        
        try:
            params = {'apikey': self.api_key}
            url = f"{self.base_url}/v1/marketstatus/now"
            
            session = await self.get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Error fetching market status: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching market status: {str(e)}")
            return {}
    
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information"""
        
        try:
            params = {'apikey': self.api_key}
            url = f"{self.base_url}/v3/reference/tickers/{symbol}"
            
            session = await self.get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', {})
                else:
                    logger.error(f"Error fetching company info for {symbol}: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return {}
    
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get real-time quotes for multiple symbols"""
        
        results = {}
        tasks = []
        
        for symbol in symbols:
            task = asyncio.create_task(self.get_real_time_quote(symbol))
            tasks.append((symbol, task))
        
        for symbol, task in tasks:
            try:
                quote = await task
                results[symbol] = quote
            except Exception as e:
                logger.error(f"Error getting quote for {symbol}: {e}")
                results[symbol] = {}
        
        return results
    
    async def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for symbols"""
        
        try:
            params = {
                'apikey': self.api_key,
                'search': query,
                'limit': limit,
                'active': 'true'
            }
            url = f"{self.base_url}/v3/reference/tickers"
            
            session = await self.get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', [])
                else:
                    logger.error(f"Error searching symbols: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching symbols: {str(e)}")
            return []
    
    async def get_gainers_losers(self, direction: str = "gainers") -> List[Dict[str, Any]]:
        """Get top gainers or losers"""
        
        try:
            params = {'apikey': self.api_key}
            url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/{direction}"
            
            session = await self.get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', [])
                else:
                    logger.error(f"Error fetching {direction}: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching {direction}: {str(e)}")
            return []

# Global instance
polygon_service = PolygonDataService()
