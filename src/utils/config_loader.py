import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader for API and model settings"""

    _instance = None
    _api_config = None
    _model_config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._api_config is None:
            self.load_configs()

    def load_configs(self):
        """Load both configuration files"""
        try:
            config_dir = Path("config")

            # Load API config
            api_config_path = config_dir / "api_config.json"
            with open(api_config_path) as f:
                self._api_config = json.load(f)

            # Load model config
            model_config_path = config_dir / "model_config.json"
            with open(model_config_path) as f:
                self._model_config = json.load(f)

            logger.info("Configuration files loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise

    @property
    def api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self._api_config

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self._model_config

    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference-specific configuration"""
        return self._model_config.get("inference", {})

    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation-specific configuration"""
        return self._model_config.get("generation", {})

    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware-specific configuration"""
        return self._model_config.get("hardware", {})
