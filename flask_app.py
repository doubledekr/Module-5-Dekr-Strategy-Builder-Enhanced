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
                "conditions_met": ["RSI < 30", "Price above SMA"]
            },
            {
                "id": "2",
                "symbol": "GOOGL",
                "signal_type": "sell",
                "price": 2750.00,
                "confidence": 0.72,
                "timestamp": datetime.now().isoformat(),
                "conditions_met": ["RSI > 70", "MACD bearish crossover"]
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
    """Run backtest for strategy"""
    try:
        data = request.get_json()
        # TODO: Implement actual backtesting
        result = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "win_rate": 0.65,
            "total_trades": 25
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return jsonify({"error": "Failed to run backtest"}), 500

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