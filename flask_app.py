#!/usr/bin/env python3
"""
Flask-based Dekr Strategy Builder Enhanced
Compatible with existing Gunicorn workflow
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
import asyncio
from contextlib import asynccontextmanager
from services.ai_strategy_builder import AIStrategyBuilder
from services.backtesting import BacktestingEngine
from services.polygon_service import PolygonDataService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Initialize database on startup
def initialize_database():
    """Initialize database tables"""
    try:
        from database import init_db
        asyncio.run(init_db())
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

# Initialize database when app starts
with app.app_context():
    initialize_database()

# Initialize AI Strategy Builder and Backtesting Engine
ai_builder = AIStrategyBuilder()
backtesting_engine = BacktestingEngine()
polygon_service = PolygonDataService()

# Root route
@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/strategy-builder')
def strategy_builder():
    """Strategy builder page"""
    return render_template('strategy_builder.html')

@app.route('/backtesting')
def backtesting():
    """Backtesting page"""
    return render_template('backtesting.html')

@app.route('/signals')
def signals():
    """Signals page"""
    return render_template('signals.html')

# API Routes
@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/strategies/')
def get_strategies():
    """Get strategies endpoint"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        limit = request.args.get('limit', 50, type=int)
        
        # Mock data for now - will be replaced with actual database calls
        strategies = [
            {
                "id": "1",
                "name": "Sample RSI Strategy",
                "description": "A sample RSI-based trading strategy",
                "strategy_type": "technical",
                "symbols": ["AAPL", "GOOGL"],
                "is_active": True,
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "2",
                "name": "Moving Average Strategy",
                "description": "Simple moving average crossover strategy",
                "strategy_type": "technical",
                "symbols": ["MSFT", "TSLA"],
                "is_active": False,
                "created_at": datetime.now().isoformat()
            }
        ]
        return jsonify({"strategies": strategies})
    except Exception as e:
        logger.error(f"Error fetching strategies: {e}")
        return jsonify({"error": "Failed to fetch strategies"}), 500

@app.route('/api/strategies', methods=['POST'])
def create_strategy():
    """Create new strategy endpoint"""
    try:
        data = request.get_json()
        # TODO: Implement actual strategy creation
        return jsonify({
            "id": "new-strategy-id",
            "message": "Strategy created successfully"
        })
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        return jsonify({"error": "Failed to create strategy"}), 500

@app.route('/api/signals/')
def get_signals():
    """Get signals endpoint"""
    try:
        strategy_id = request.args.get('strategy_id', None)
        limit = request.args.get('limit', 50, type=int)
        
        # Mock data for now
        signals = [
            {
                "id": "1",
                "symbol": "AAPL",
                "signal_type": "buy",
                "price": 150.00,
                "confidence": 0.85,
                "timestamp": datetime.now().isoformat(),
                "conditions_met": ["RSI < 30", "Price above SMA"],
                "market_data": {
                    "current_price": 150.00,
                    "volume": 45678900,
                    "change": 2.45,
                    "change_percent": 1.65
                }
            },
            {
                "id": "2",
                "symbol": "GOOGL",
                "signal_type": "sell",
                "price": 2750.00,
                "confidence": 0.72,
                "timestamp": datetime.now().isoformat(),
                "conditions_met": ["RSI > 70", "MACD bearish crossover"],
                "market_data": {
                    "current_price": 2750.00,
                    "volume": 25432100,
                    "change": -15.30,
                    "change_percent": -0.55
                }
            }
        ]
        return jsonify({"signals": signals})
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        return jsonify({"error": "Failed to fetch signals"}), 500

@app.route('/api/signals/stats')
def get_signal_stats():
    """Get signal statistics"""
    try:
        stats = {
            "total_signals": 10,
            "buy_signals": 6,
            "sell_signals": 4,
            "avg_confidence": 0.75
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error fetching signal stats: {e}")
        return jsonify({"error": "Failed to fetch signal stats"}), 500

@app.route('/api/strategies/<strategy_id>/backtest', methods=['POST'])
def run_backtest(strategy_id):
    """Run backtest for strategy with buy-and-hold comparison"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2024-01-01')
        
        # Parse dates
        from datetime import datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Create a mock strategy for testing (replace with actual strategy retrieval)
        from models import Strategy, StrategyCondition
        strategy = Strategy(
            id=strategy_id,
            user_id=1,
            name="Test Strategy",
            description="Test strategy for backtesting",
            strategy_type="technical",
            symbols=[symbol],
            buy_conditions=[
                StrategyCondition(
                    indicator="rsi",
                    operator="<",
                    value=30,
                    timeframe="1d",
                    parameters={"period": 14}
                )
            ],
            sell_conditions=[
                StrategyCondition(
                    indicator="rsi",
                    operator=">",
                    value=70,
                    timeframe="1d",
                    parameters={"period": 14}
                )
            ],
            risk_management={
                "stop_loss": 0.02,
                "take_profit": 0.06,
                "position_size": 0.1,
                "max_positions": 3
            },
            tier_required=1,
            is_active=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Run backtest with buy-and-hold comparison
        result = asyncio.run(backtesting_engine.run_backtest(strategy, symbol, start_dt, end_dt))
        
        # Convert any numpy/pandas types to Python native types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif isinstance(obj, (bool, int, float, str)):
                return obj
            else:
                return str(obj)
        
        serializable_result = convert_to_json_serializable(result)
        return jsonify(serializable_result)
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return jsonify({"error": str(e)}), 500

# AI Strategy Builder endpoints
@app.route('/api/ai/strategy-examples')
def get_strategy_examples():
    """Get AI strategy examples"""
    try:
        examples = asyncio.run(ai_builder.generate_strategy_examples())
        return jsonify(examples)
    except Exception as e:
        logger.error(f"Error fetching strategy examples: {e}")
        return jsonify({"error": "Failed to fetch strategy examples"}), 500

@app.route('/api/ai/parse-strategy', methods=['POST'])
def parse_strategy():
    """Parse natural language strategy description"""
    try:
        data = request.get_json()
        description = data.get('description', '')
        user_id = data.get('user_id', 1)  # Default user ID
        
        if not description:
            return jsonify({"error": "Description is required"}), 400
        
        strategy = asyncio.run(ai_builder.parse_strategy_description(description, user_id))
        return jsonify(strategy)
    except Exception as e:
        logger.error(f"Error parsing strategy: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/explain-strategy', methods=['POST'])
def explain_strategy():
    """Explain a strategy in plain language"""
    try:
        data = request.get_json()
        strategy_data = data.get('strategy', {})
        
        if not strategy_data:
            return jsonify({"error": "Strategy data is required"}), 400
        
        explanation = asyncio.run(ai_builder.explain_strategy(strategy_data))
        return jsonify({"explanation": explanation})
    except Exception as e:
        logger.error(f"Error explaining strategy: {e}")
        return jsonify({"error": "Failed to explain strategy"}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)