from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set, List
import asyncio
import json
import logging
from services.signal_generator import signal_generator
from models import Signal

logger = logging.getLogger(__name__)

router = APIRouter()

class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.strategy_subscribers: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"New WebSocket connection: {websocket.client}")
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        self.active_connections.discard(websocket)
        # Remove from strategy subscriptions
        for strategy_id, subscribers in self.strategy_subscribers.items():
            subscribers.discard(websocket)
        logger.info(f"WebSocket disconnected: {websocket.client}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to a specific WebSocket"""
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSockets"""
        if not self.active_connections:
            return
        
        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.add(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def broadcast_to_strategy_subscribers(self, strategy_id: str, message: str):
        """Broadcast message to subscribers of a specific strategy"""
        if strategy_id not in self.strategy_subscribers:
            return
        
        subscribers = self.strategy_subscribers[strategy_id].copy()
        disconnected = set()
        
        for websocket in subscribers:
            try:
                await websocket.send_text(message)
            except:
                disconnected.add(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)
    
    def subscribe_to_strategy(self, websocket: WebSocket, strategy_id: str):
        """Subscribe a WebSocket to strategy signals"""
        if strategy_id not in self.strategy_subscribers:
            self.strategy_subscribers[strategy_id] = set()
        self.strategy_subscribers[strategy_id].add(websocket)
        logger.info(f"WebSocket subscribed to strategy {strategy_id}")
    
    def unsubscribe_from_strategy(self, websocket: WebSocket, strategy_id: str):
        """Unsubscribe a WebSocket from strategy signals"""
        if strategy_id in self.strategy_subscribers:
            self.strategy_subscribers[strategy_id].discard(websocket)
            if not self.strategy_subscribers[strategy_id]:
                del self.strategy_subscribers[strategy_id]
        logger.info(f"WebSocket unsubscribed from strategy {strategy_id}")

# Global WebSocket manager
ws_manager = WebSocketManager()

async def signal_callback(signal: Signal):
    """Callback function for new signals"""
    try:
        message = {
            "type": "signal",
            "data": signal.to_dict()
        }
        
        # Broadcast to all connections
        await ws_manager.broadcast(json.dumps(message))
        
        # Broadcast to strategy subscribers
        await ws_manager.broadcast_to_strategy_subscribers(
            signal.strategy_id, json.dumps(message)
        )
        
    except Exception as e:
        logger.error(f"Error in signal callback: {str(e)}")

# Register signal callback
signal_generator.add_signal_callback(signal_callback)

@router.websocket("/signals")
async def websocket_signals_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time signals"""
    await ws_manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "subscribe":
                    strategy_id = message.get("strategy_id")
                    if strategy_id:
                        ws_manager.subscribe_to_strategy(websocket, strategy_id)
                        await ws_manager.send_personal_message(
                            json.dumps({"type": "subscription_confirmed", "strategy_id": strategy_id}),
                            websocket
                        )
                
                elif message_type == "unsubscribe":
                    strategy_id = message.get("strategy_id")
                    if strategy_id:
                        ws_manager.unsubscribe_from_strategy(websocket, strategy_id)
                        await ws_manager.send_personal_message(
                            json.dumps({"type": "unsubscription_confirmed", "strategy_id": strategy_id}),
                            websocket
                        )
                
                elif message_type == "ping":
                    await ws_manager.send_personal_message(
                        json.dumps({"type": "pong"}),
                        websocket
                    )
                
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")
                await ws_manager.send_personal_message(
                    json.dumps({"type": "error", "message": "Invalid JSON format"}),
                    websocket
                )
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        ws_manager.disconnect(websocket)

@router.websocket("/market-data")
async def websocket_market_data_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await ws_manager.connect(websocket)
    
    try:
        # Send initial market status
        from services.polygon_service import polygon_service
        market_status = await polygon_service.get_market_status()
        await ws_manager.send_personal_message(
            json.dumps({"type": "market_status", "data": market_status}),
            websocket
        )
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(30)  # Send updates every 30 seconds
            
            # Send market status update
            market_status = await polygon_service.get_market_status()
            await ws_manager.send_personal_message(
                json.dumps({"type": "market_status", "data": market_status}),
                websocket
            )
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Market data WebSocket error: {str(e)}")
        ws_manager.disconnect(websocket)
