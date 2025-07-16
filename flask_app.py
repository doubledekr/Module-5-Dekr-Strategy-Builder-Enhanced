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
        
        # Get strategies from database
        import sqlite3
        import json
        
        conn = sqlite3.connect('dekr_strategy_builder.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, user_id, name, description, strategy_type, symbols, 
                   buy_conditions, sell_conditions, risk_management, tier_required,
                   is_active, created_at, updated_at
            FROM strategies 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        strategies = []
        for row in rows:
            strategy = {
                "id": row[0],
                "user_id": row[1],
                "name": row[2],
                "description": row[3],
                "strategy_type": row[4],
                "symbols": json.loads(row[5]) if row[5] else [],
                "buy_conditions": json.loads(row[6]) if row[6] else [],
                "sell_conditions": json.loads(row[7]) if row[7] else [],
                "risk_management": json.loads(row[8]) if row[8] else {},
                "tier_required": row[9],
                "is_active": bool(row[10]),
                "created_at": row[11],
                "updated_at": row[12]
            }
            strategies.append(strategy)
        
        return jsonify({"strategies": strategies})
    except Exception as e:
        logger.error(f"Error fetching strategies: {e}")
        return jsonify({"error": "Failed to fetch strategies"}), 500

@app.route('/api/strategies/<strategy_id>')
def get_strategy(strategy_id):
    """Get a specific strategy by ID"""
    try:
        import sqlite3
        import json
        
        conn = sqlite3.connect('dekr_strategy_builder.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, user_id, name, description, strategy_type, symbols, 
                   buy_conditions, sell_conditions, risk_management, tier_required,
                   is_active, created_at, updated_at
            FROM strategies 
            WHERE id = ?
        """, (strategy_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return jsonify({"error": "Strategy not found"}), 404
        
        strategy = {
            "id": row[0],
            "user_id": row[1],
            "name": row[2],
            "description": row[3],
            "strategy_type": row[4],
            "symbols": json.loads(row[5]) if row[5] else [],
            "buy_conditions": json.loads(row[6]) if row[6] else [],
            "sell_conditions": json.loads(row[7]) if row[7] else [],
            "risk_management": json.loads(row[8]) if row[8] else {},
            "tier_required": row[9],
            "is_active": bool(row[10]),
            "created_at": row[11],
            "updated_at": row[12]
        }
        
        return jsonify({"strategy": strategy})
    except Exception as e:
        logger.error(f"Error fetching strategy: {e}")
        return jsonify({"error": "Failed to fetch strategy"}), 500

@app.route('/api/strategies', methods=['POST'])
def create_strategy():
    """Create new strategy endpoint"""
    try:
        data = request.get_json()
        
        # Create strategy using AI strategy builder
        from services.ai_strategy_builder import AIStrategyBuilder
        ai_builder = AIStrategyBuilder()
        
        if 'description' in data:
            # Natural language strategy creation
            strategy_data = asyncio.run(ai_builder.parse_strategy_description(
                data['description'], 
                data.get('user_id', 1)
            ))
        else:
            # Direct strategy creation
            strategy_data = data
            strategy_data['user_id'] = data.get('user_id', 1)
            strategy_data['id'] = f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            strategy_data['created_at'] = datetime.now().isoformat()
            strategy_data['updated_at'] = datetime.now().isoformat()
            strategy_data['is_active'] = True
        
        # Save to database
        from models import Strategy, StrategyCondition
        import json
        import uuid
        
        strategy = Strategy(
            id=strategy_data['id'],
            user_id=strategy_data['user_id'],
            name=strategy_data['name'],
            description=strategy_data.get('description', ''),
            strategy_type=strategy_data['strategy_type'],
            symbols=strategy_data['symbols'],
            buy_conditions=[StrategyCondition.from_dict(c) for c in strategy_data['buy_conditions']],
            sell_conditions=[StrategyCondition.from_dict(c) for c in strategy_data['sell_conditions']],
            risk_management=strategy_data.get('risk_management', {"stop_loss": 0.05, "take_profit": 0.10}),
            tier_required=strategy_data.get('tier_required', 1),
            is_active=strategy_data.get('is_active', True),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save to SQLite database
        query = """
            INSERT INTO strategies (id, user_id, name, description, strategy_type, symbols,
                                  buy_conditions, sell_conditions, risk_management, tier_required,
                                  is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            strategy.id,
            strategy.user_id,
            strategy.name,
            strategy.description,
            strategy.strategy_type,
            json.dumps(strategy.symbols),
            json.dumps([c.to_dict() for c in strategy.buy_conditions]),
            json.dumps([c.to_dict() for c in strategy.sell_conditions]),
            json.dumps(strategy.risk_management),
            strategy.tier_required,
            strategy.is_active,
            strategy.created_at.isoformat(),
            strategy.updated_at.isoformat()
        )
        
        # Execute database write
        import sqlite3
        conn = sqlite3.connect('dekr_strategy_builder.db')
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        conn.close()
        
        return jsonify({
            "strategy": strategy.to_dict(),
            "message": "Strategy created and saved successfully"
        })
        
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        return jsonify({"error": f"Failed to create strategy: {str(e)}"}), 500

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
        strategy_config = data.get('strategy_config', {})
        
        # Parse dates
        from datetime import datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Create strategy from config or use default RSI strategy
        from models import Strategy, StrategyCondition
        
        if strategy_config and 'buy_conditions' in strategy_config:
            buy_conditions = [
                StrategyCondition(
                    indicator=cond.get('indicator', 'rsi'),
                    operator=cond.get('operator', '<'),
                    value=cond.get('value', 30),
                    timeframe=cond.get('timeframe', '1d'),
                    parameters=cond.get('parameters', {})
                )
                for cond in strategy_config['buy_conditions']
            ]
            
            sell_conditions = [
                StrategyCondition(
                    indicator=cond.get('indicator', 'rsi'),
                    operator=cond.get('operator', '>'),
                    value=cond.get('value', 70),
                    timeframe=cond.get('timeframe', '1d'),
                    parameters=cond.get('parameters', {})
                )
                for cond in strategy_config['sell_conditions']
            ]
        else:
            # Default RSI strategy
            buy_conditions = [
                StrategyCondition(
                    indicator="rsi",
                    operator="<",
                    value=30,
                    timeframe="1d",
                    parameters={"period": 14}
                )
            ]
            
            sell_conditions = [
                StrategyCondition(
                    indicator="rsi",
                    operator=">",
                    value=70,
                    timeframe="1d",
                    parameters={"period": 14}
                )
            ]
        
        strategy = Strategy(
            id=strategy_id,
            user_id=1,
            name=strategy_config.get('name', 'Test Strategy'),
            description=strategy_config.get('description', 'Test strategy for backtesting'),
            strategy_type="technical",
            symbols=[symbol],
            buy_conditions=buy_conditions,
            sell_conditions=sell_conditions,
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
def ai_parse_strategy():
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