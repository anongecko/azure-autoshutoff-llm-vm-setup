from fastapi import FastAPI
from .middleware import setup_middleware
from .routes import setup_routes
import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

_config_cache: Dict = {}


def get_config(refresh: bool = False) -> Dict:
    """Get cached API configuration"""
    if not _config_cache or refresh:
        try:
            with open(Path("config") / "api_config.json") as f:
                _config_cache.update(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load API configuration: {e}")
            raise
    return _config_cache


def create_app(config_override: Dict = None) -> FastAPI:
    """Create and configure FastAPI application with enhanced settings"""
    config = {**get_config(), **(config_override or {})}

    app = FastAPI(
        title="DeepSeek API",
        description="High-performance API for DeepSeek model inference",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        debug=False,
    )

    return app


__all__ = ["create_app", "get_config"]

