from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from database import init_db
from routes import strategies, signals, websocket
import redis
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis connection
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection successful")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Dekr Strategy Builder Enhanced")
    await init_db()
    yield
    # Shutdown
    logger.info("Shutting down Dekr Strategy Builder Enhanced")

app = FastAPI(
    title="Dekr Strategy Builder Enhanced",
    version="2.0.0",
    description="Enhanced trading strategy builder with Polygon.io integration",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Root route
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/strategy-builder", response_class=HTMLResponse)
async def strategy_builder(request: Request):
    return templates.TemplateResponse("strategy_builder.html", {"request": request})

@app.get("/backtesting", response_class=HTMLResponse)
async def backtesting(request: Request):
    return templates.TemplateResponse("backtesting.html", {"request": request})

@app.get("/signals", response_class=HTMLResponse)
async def signals_page(request: Request):
    return templates.TemplateResponse("signals.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}
