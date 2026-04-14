"""
Embedding backends for the Long-Term Memory MCP system.
Supports multiple embedding providers with a unified interface.
"""

import os
import logging
from typing import List, Optional, Union
from abc import ABC, abstractmethod
import random
import hashlib


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (batch)."""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model/backend."""
        pass


class SentenceTransformersBackend(EmbeddingBackend):
    """Sentence Transformers backend using HuggingFace models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", offline: bool = True):
        self.model_name = model_name
        self.offline = offline
        self.model = None
        self.dimensions = 384  # Default for all-MiniLM-L6-v2

        try:
            from sentence_transformers import SentenceTransformer

            if offline:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"

            self.model = SentenceTransformer(model_name, local_files_only=offline)

            # Get actual dimensions from the model
            test_embedding = self.model.encode("test")
            self.dimensions = len(test_embedding)

            logging.info(
                f"SentenceTransformers backend initialized with model '{model_name}' "
                f"({self.dimensions} dimensions, offline={offline})"
            )

        except ImportError:
            raise RuntimeError("sentence-transformers package not installed")
        except Exception as e:
            logging.error(
                f"Failed to load SentenceTransformer model '{model_name}': {e}"
            )
            raise

    def get_embedding(self, text: str) -> List[float]:
        if self.model is None:
            raise RuntimeError("SentenceTransformer model not loaded")
        return self.model.encode(text).tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            raise RuntimeError("SentenceTransformer model not loaded")
        return [embedding.tolist() for embedding in self.model.encode(texts)]

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_model_name(self) -> str:
        return f"sentence-transformers:{self.model_name}"


class OllamaBackend(EmbeddingBackend):
    """Ollama backend for local model inference."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.dimensions = 768  # Default for nomic-embed-text

        try:
            import ollama

            # Test connection and get model info
            client = ollama.Client(host=base_url)

            print(f"[OllamaBackend] Connecting to {base_url}, testing model '{model_name}'...")

            # Try to get embedding dimensions
            try:
                test_response = client.embeddings(model=model_name, prompt="test")
                self.dimensions = len(test_response.get("embedding", []))
                if self.dimensions == 0:
                    logging.warning(
                        f"Could not determine dimensions for model '{model_name}', using default"
                    )
                else:
                    print(f"[OllamaBackend] Model '{model_name}' loaded, {self.dimensions} dimensions")
            except Exception as emb_e:
                print(f"[OllamaBackend] Failed to get embeddings for '{model_name}': {emb_e}")
                logging.warning(
                    f"Could not determine dimensions for model '{model_name}', using default"
                )

            logging.info(
                f"Ollama backend initialized with model '{model_name}' "
                f"({self.dimensions} dimensions, base_url={base_url})"
            )

        except ImportError:
            raise RuntimeError("ollama package not installed")
        except Exception as e:
            logging.error(f"Failed to initialize Ollama backend: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        try:
            import ollama

            client = ollama.Client(host=self.base_url)
            response = client.embeddings(model=self.model_name, prompt=text)
            return response.get("embedding", [])
        except Exception as e:
            logging.error(f"Failed to get embedding from Ollama: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Ollama doesn't support batch embeddings natively, so we do them sequentially
        embeddings = []
        for text in texts:
            embeddings.append(self.get_embedding(text))
        return embeddings

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_model_name(self) -> str:
        return f"ollama:{self.model_name}"


class FallbackBackend(EmbeddingBackend):
    """Simple fallback backend for when no other backend is available.
    Uses TF-IDF like approach with random projections for demo/testing."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.vocab = {}
        self.next_word_id = 0
        logging.warning(
            f"Using FallbackBackend with {dimensions} dimensions "
            "(not suitable for production)"
        )

    def _text_to_vector(self, text: str) -> List[float]:
        """Convert text to a simple bag-of-words like vector with random projection."""
        words = text.lower().split()

        # Build vocabulary and count words
        word_counts = {}
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self.next_word_id
                self.next_word_id += 1
            word_id = self.vocab[word]
            word_counts[word_id] = word_counts.get(word_id, 0) + 1

        # Create a deterministic random projection based on word IDs
        vector = [0.0] * self.dimensions
        for word_id, count in word_counts.items():
            # Use hash of word_id for deterministic "random" values
            seed = hashlib.md5(str(word_id).encode()).hexdigest()
            random.seed(seed)

            # Distribute the count across dimensions
            for i in range(min(self.dimensions, 10)):  # Use up to 10 dimensions
                idx = random.randint(0, self.dimensions - 1)
                value = random.uniform(-1.0, 1.0) * (count / len(words))
                vector[idx] += value

        # Normalize
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def get_embedding(self, text: str) -> List[float]:
        return self._text_to_vector(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(text) for text in texts]

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_model_name(self) -> str:
        return f"fallback:{self.dimensions}d"


from dataclasses import dataclass
from typing import Optional


@dataclass
class OllamaModelInfo:
    """Information about a discovered Ollama/OpenAI-compatible model."""
    name: str
    dimensions: int  # 0 if unknown
    modified_at: Optional[str] = None


class OllamaDiscovery:
    """
    Discover embedding models from Ollama or LM Studio (OpenAI-compatible).
    Uses the `ollama` Python library for Ollama, falls back to httpx for OpenAI-compatible APIs.
    """

    # Model name patterns strongly associated with embedding models in Ollama/LM Studio
    EMBEDDING_NAME_PATTERNS = (
        "embed", "nomic", "e5", "bge", "gemma", "mistral", "qwen2",
    )

    @staticmethod
    def _is_likely_embedding_model(model_name: str) -> bool:
        """Heuristic: return True if model name suggests an embedding model."""
        name_lower = model_name.lower()
        return any(pattern in name_lower for pattern in OllamaDiscovery.EMBEDDING_NAME_PATTERNS)

    @staticmethod
    def _fetch_ollama_models(base_url: str) -> list[OllamaModelInfo]:
        """
        Use the `ollama` Python library to list models from an Ollama server.
        Does NOT probe dimensions — just returns names filtered by embedding likelihood.
        """
        import ollama

        client = ollama.Client(host=base_url)

        try:
            response = client.list()
        except Exception:
            response = ollama.list()

        # response.models is a list of Model objects
        raw_models = getattr(response, "models", []) or []
        if not raw_models and hasattr(response, "model_names"):
            raw_models = response.model_names

        embedding_models = []

        for m in raw_models:
            model_name = getattr(m, "model", None) or getattr(m, "name", "")
            if not model_name:
                continue

            modified_at = getattr(m, "modified_at", None)

            # Only include models that look like embedding models
            if OllamaDiscovery._is_likely_embedding_model(model_name):
                embedding_models.append(OllamaModelInfo(
                    name=model_name,
                    dimensions=0,  # dimensions probed lazily on actual use
                    modified_at=modified_at,
                ))

        # If no embedding models found, return all available (user may have custom names)
        if not embedding_models:
            for m in raw_models:
                model_name = getattr(m, "model", None) or getattr(m, "name", "")
                if model_name:
                    embedding_models.append(OllamaModelInfo(
                        name=model_name,
                        dimensions=0,
                        modified_at=getattr(m, "modified_at", None),
                    ))

        return embedding_models

    @staticmethod
    def _fetch_openai_compatible_models(base_url: str, timeout: float = 5.0) -> list[OllamaModelInfo]:
        """Fetch from an OpenAI-compatible /v1/models endpoint (LM Studio, etc.)."""
        import httpx

        models: list[OllamaModelInfo] = []
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(f"{base_url.rstrip('/')}/v1/models")
                resp.raise_for_status()
                data = resp.json()
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    if model_id and OllamaDiscovery._is_likely_embedding_model(model_id):
                        models.append(OllamaModelInfo(
                            name=model_id,
                            dimensions=0,
                            modified_at=None,
                        ))
        except Exception:
            pass

        return models

    @staticmethod
    def list_models(base_url: str, timeout: float = 5.0) -> list[OllamaModelInfo]:
        """
        List available embedding models from an Ollama or LM Studio server.

        Args:
            base_url: e.g. "http://localhost:11434"
            timeout: seconds to wait per request

        Returns:
            List of OllamaModelInfo, empty list on any failure (never raises)
        """
        # Try Ollama library first (only works for native Ollama, not LM Studio)
        try:
            models = OllamaDiscovery._fetch_ollama_models(base_url)
            if models:
                return models
        except Exception:
            pass

        # Fall back to OpenAI-compatible /v1/models (LM Studio, ollama with proxy, etc.)
        try:
            models = OllamaDiscovery._fetch_openai_compatible_models(base_url, timeout)
            if models:
                return models
        except Exception:
            pass

        return []


class SentenceTransformersDiscovery:
    """
    Discover locally available Sentence Transformers models.
    """

    # Well-known high-quality embedding models with known dimensions
    KNOWN_MODELS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }

    @classmethod
    def list_local_models(cls) -> list[tuple[str, int]]:
        """
        List available local Sentence Transformer models.

        Returns:
            List of (model_name, dimensions) tuples.
            Tries HuggingFace cache inspection, falls back to KNOWN_MODELS.
        """
        # Try to list from HuggingFace cache
        try:
            from huggingface_hub import list_models
            local_models = []
            for model in list_models(search="sentence-transformers", full=True):
                if hasattr(model, "id") and hasattr(model, "dims"):
                    local_models.append((model.id, model.dims or 0))
            if local_models:
                return local_models
        except Exception:
            pass

        # Fall back to known models that work well for embeddings
        return list(cls.KNOWN_MODELS.items())


def create_embedding_backend(
    backend_type: str = "sentence-transformers",
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    offline: bool = True,
    dimensions: Optional[int] = None,
) -> EmbeddingBackend:
    """
    Factory function to create an embedding backend.

    Args:
        backend_type: "sentence-transformers", "ollama", or "fallback"
        model_name: Model name for the backend
        base_url: Base URL for Ollama (default: http://localhost:11434)
        offline: Whether to work offline (for sentence-transformers)
        dimensions: Dimensions for fallback backend

    Returns:
        An EmbeddingBackend instance
    """
    backend_type = backend_type.lower()

    if backend_type == "sentence-transformers":
        if model_name is None:
            model_name = "all-MiniLM-L6-v2"
        return SentenceTransformersBackend(model_name=model_name, offline=offline)

    elif backend_type == "ollama":
        if model_name is None:
            model_name = "nomic-embed-text:latest"
        if base_url is None:
            base_url = "http://localhost:11434"
        return OllamaBackend(model_name=model_name, base_url=base_url)

    elif backend_type == "fallback":
        if dimensions is None:
            dimensions = 384
        return FallbackBackend(dimensions=dimensions)

    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported: 'sentence-transformers', 'ollama', 'fallback'"
        )
