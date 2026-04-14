"""
Config manager for the long-term-memory-mcp plugin.
Loads config.json with environment variable overrides.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

VALID_BACKENDS = frozenset({"sentence-transformers", "ollama", "fallback"})
DEFAULT_CONFIG_PATH = Path.home() / ".lmstudio/extensions/plugins/installed/long-term-memory-mcp" / "config.json"


class Config:
    """
    Configuration manager that loads from config.json with env var overrides.

    Environment variables always override file config values for CI/dev flexibility.

    Env vars (backward compatible with existing names):
      EMBEDDING_BACKEND, SENTENCE_TRANSFORMER_MODEL, EMBEDDING_OFFLINE,
      OLLAMA_MODEL, OLLAMA_BASE_URL, FALLBACK_DIMENSIONS
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: Dict[str, Any] = {}
        self._loaded = False
        self._load()

    def _load(self) -> None:
        """Load config from JSON file, applying env var overrides."""
        # Start with defaults derived from env vars
        self._config = {
            "embedding": {
                "backend": os.environ.get("EMBEDDING_BACKEND", "sentence-transformers"),
                "model": None,
                "offline": os.environ.get("EMBEDDING_OFFLINE", "true").lower() in ("true", "1", "yes"),
                "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            },
            "fallback_dimensions": int(os.environ.get("FALLBACK_DIMENSIONS", "384")),
        }

        # Override from config file if present
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                self._apply_file_config(file_config)
                logger.info(f"Config loaded from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")

        # Env vars override file config (always, for CI/dev flexibility)
        self._apply_env_overrides()

        # Validate
        self._validate()
        self._loaded = True

    def _apply_file_config(self, file_config: Dict[str, Any]) -> None:
        """Merge file config into self._config."""
        if "embedding" in file_config:
            emb = file_config["embedding"]
            if "backend" in emb:
                self._config["embedding"]["backend"] = emb["backend"]
            if "model" in emb:
                self._config["embedding"]["model"] = emb["model"]
            if "offline" in emb:
                self._config["embedding"]["offline"] = emb["offline"]
            if "base_url" in emb:
                self._config["embedding"]["base_url"] = emb["base_url"]
        if "fallback_dimensions" in file_config:
            self._config["fallback_dimensions"] = file_config["fallback_dimensions"]

    def _apply_env_overrides(self) -> None:
        """Environment variables always win over file config."""
        self._config["embedding"]["backend"] = os.environ.get(
            "EMBEDDING_BACKEND", self._config["embedding"]["backend"]
        )
        self._config["embedding"]["offline"] = os.environ.get(
            "EMBEDDING_OFFLINE", str(self._config["embedding"]["offline"])
        ).lower() in ("true", "1", "yes")
        self._config["embedding"]["base_url"] = os.environ.get(
            "OLLAMA_BASE_URL", self._config["embedding"]["base_url"]
        )
        # SENTENCE_TRANSFORMER_MODEL and OLLAMA_MODEL come from file or are backend-dependent defaults

    def _validate(self) -> None:
        """Validate the loaded configuration."""
        backend = self._config["embedding"]["backend"]
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid embedding backend: '{backend}'. "
                f"Supported: {', '.join(sorted(VALID_BACKENDS))}"
            )

    def get_embedding_backend_type(self) -> str:
        """Returns the configured backend type."""
        return self._config["embedding"]["backend"]

    def get_model_name(self) -> Optional[str]:
        """Returns the configured model name, or None for backend-dependent default."""
        return self._config["embedding"].get("model")

    def get_offline(self) -> bool:
        """Returns whether offline mode is enabled (sentence-transformers only)."""
        return self._config["embedding"]["offline"]

    def get_base_url(self) -> str:
        """Returns the base URL for Ollama/LM Studio."""
        return self._config["embedding"]["base_url"]

    def get_dimensions(self) -> int:
        """Returns the embedding dimensions for the fallback backend."""
        return self._config["fallback_dimensions"]

    def get_all(self) -> Dict[str, Any]:
        """Return the full config dict (for debugging/inspection)."""
        return dict(self._config)

    def save(self, config_data: Dict[str, Any]) -> None:
        """
        Save config_data to config.json.
        Called by the GUI after user edits settings.
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Config saved to {self.config_path}")
