from fastapi import FastAPI
from .middleware import setup_middleware
from .routes import setup_routes

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="DeepSeek API",
        description="High-performance API for DeepSeek model inference",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Setup middleware and routes
    setup_middleware(app)
    setup_routes(app)
    
    return app

