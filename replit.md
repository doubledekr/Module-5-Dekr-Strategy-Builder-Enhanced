# Dekr Strategy Builder Enhanced

## Overview

The Dekr Strategy Builder Enhanced is a sophisticated FastAPI-based microservice designed for algorithmic trading strategy development, backtesting, and real-time signal generation. The application has been enhanced to use Polygon.io for comprehensive market data instead of the previous Twelve Data integration, providing unlimited API access and more granular intraday data.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: FastAPI with asynchronous request handling
- **Database**: SQLite with aiosqlite for asynchronous operations
- **Real-time Communication**: WebSocket connections for live signal streaming
- **Data Processing**: Pandas and NumPy for technical analysis and backtesting
- **Caching**: Redis for session management and performance optimization

### Frontend Architecture
- **Templates**: Jinja2 templating with Bootstrap dark theme
- **Static Assets**: CSS and JavaScript for interactive user interface
- **Real-time Updates**: WebSocket client for live signal reception
- **Charting**: Chart.js for data visualization

## Key Components

### 1. Strategy Management System
- **Purpose**: Handle creation, modification, and storage of trading strategies
- **Implementation**: RESTful APIs with tier-based access control
- **Features**: Support for technical, fundamental, sentiment, and hybrid strategies

### 2. Signal Generation Engine
- **Purpose**: Monitor active strategies and generate real-time trading signals
- **Implementation**: Asynchronous background tasks with WebSocket broadcasting
- **Features**: Multi-timeframe analysis, confidence scoring, and historical validation

### 3. Backtesting Engine
- **Purpose**: Validate strategy performance using historical data
- **Implementation**: Pandas-based simulation with realistic trading costs
- **Features**: Performance metrics calculation, trade simulation, and risk analysis

### 4. Technical Analysis Engine
- **Purpose**: Calculate technical indicators for strategy evaluation
- **Implementation**: TA-Lib integration with custom indicator support
- **Features**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, and volume indicators

### 5. Polygon.io Integration
- **Purpose**: Provide comprehensive market data for analysis and backtesting
- **Implementation**: Asynchronous HTTP client with WebSocket streaming
- **Features**: Historical data, real-time quotes, and unlimited API access

## Data Flow

### Strategy Creation Flow
1. User creates strategy through web interface
2. Strategy validation and tier-based feature checks
3. Database storage with JSON serialization
4. Signal generator registration for active strategies

### Signal Generation Flow
1. Background tasks monitor active strategies
2. Real-time market data processing from Polygon.io
3. Technical indicator calculation and condition evaluation
4. Signal generation with confidence scoring
5. WebSocket broadcasting to connected clients
6. Database storage for historical analysis

### Backtesting Flow
1. User selects strategy and time period
2. Historical data retrieval from Polygon.io
3. Technical indicator calculation on historical data
4. Trade simulation with realistic costs and slippage
5. Performance metrics calculation and result storage

## External Dependencies

### Primary Dependencies
- **Polygon.io API**: Market data provider with unlimited access
- **Redis**: Optional caching layer for performance optimization
- **Chart.js**: Frontend charting library for data visualization
- **Bootstrap**: UI framework with dark theme support

### Python Libraries
- **FastAPI**: Web framework with automatic API documentation
- **aiosqlite**: Asynchronous SQLite database operations
- **pandas**: Data manipulation and analysis
- **ta**: Technical analysis library
- **websockets**: WebSocket server implementation

## Deployment Strategy

### Development Environment
- **Platform**: Replit with automatic dependency management
- **Database**: SQLite file-based storage
- **Environment Variables**: Polygon.io API key configuration
- **Port**: Application runs on port 5000

### Production Considerations
- **Database Migration**: SQLite can be easily migrated to PostgreSQL
- **Caching**: Redis integration for improved performance
- **Monitoring**: Comprehensive logging throughout the application
- **Scalability**: Async architecture supports high concurrent usage

### Tier-Based Features
- **Freemium (Tier 1)**: Basic strategy creation and limited backtesting
- **Market Hours Pro (Tier 2)**: Extended market data and real-time signals
- **Sector Specialist (Tier 3)**: Advanced technical indicators and multi-timeframe analysis
- **Higher Tiers (4-7)**: Custom indicators, options strategies, and institutional features

## Recent Changes

### July 16, 2025 - Major AI and Backtesting Enhancement
- **AI-Powered Strategy Creation**: Implemented OpenAI GPT-4o integration for natural language strategy creation
- **Prompt-Based Interface**: Users can now describe strategies in plain language (e.g., "Buy when RSI is below 30")
- **Real-Time Backtesting**: Integrated Polygon.io API for authentic historical market data
- **Buy-and-Hold Comparison**: Added comprehensive comparison showing when buy-and-hold outperforms strategies
- **Intelligent Recommendations**: System now provides clear recommendations with confidence levels
- **Enhanced Performance Metrics**: Added Sharpe ratio, volatility, and detailed trade analysis
- **Comprehensive UI**: New backtesting interface with visual performance comparison

The application is designed to be fully functional in a Replit environment while maintaining the flexibility to scale to production infrastructure with minimal modifications.