# Configuration Guide

The long-term-memory-mcp plugin can be configured via a JSON config file or environment variables.

## Config File Location

```
~/.lmstudio/extensions/plugins/installed/long-term-memory-mcp/config.json
```

If the config file does not exist, defaults are used (or values derived from environment variables).

---

## Config File Schema

```json
{
  "embedding": {
    "backend": "sentence-transformers",
    "model": null,
    "offline": true,
    "base_url": "http://localhost:11434"
  },
  "fallback_dimensions": 384
}
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embedding.backend` | string | `"sentence-transformers"` | Embedding backend: `"sentence-transformers"`, `"ollama"`, or `"fallback"` |
| `embedding.model` | string\|null | `null` | Model name. `null` uses the backend's default. Examples: `"all-MiniLM-L6-v2"`, `"nomic-embed-text:latest"` |
| `embedding.offline` | boolean | `true` | For `sentence-transformers` only: if `true`, only use locally cached models |
| `embedding.base_url` | string | `"http://localhost:11434"` | For `ollama` backend only: base URL of the Ollama/LM Studio server |
| `fallback_dimensions` | integer | `384` | Embedding dimensions for the `fallback` backend |

---

## Environment Variables

Environment variables **always override** config file values. This is useful for CI, Docker, or quick testing.

| Variable | Backend | Default | Description |
|----------|---------|---------|-------------|
| `EMBEDDING_BACKEND` | all | `"sentence-transformers"` | Backend type |
| `SENTENCE_TRANSFORMER_MODEL` | sentence-transformers | `"all-MiniLM-L6-v2"` | HF model name |
| `EMBEDDING_OFFLINE` | sentence-transformers | `"true"` | Offline mode (`"true"`/`"false"`) |
| `OLLAMA_MODEL` | ollama | `"nomic-embed-text:latest"` | Ollama model name |
| `OLLAMA_BASE_URL` | ollama | `"http://localhost:11434"` | Ollama server URL |
| `FALLBACK_DIMENSIONS` | fallback | `"384"` | Fallback embedding dimensions |

---

## Embedding Backends

### Sentence Transformers (default)

Uses HuggingFace's `sentence-transformers` library to run embedding models locally.

**Pros:** Fast, private, works offline
**Cons:** Requires downloaded models on disk

**Recommended models:**
- `all-MiniLM-L6-v2` — 384 dimensions, fast, good quality
- `all-mpnet-base-v2` — 768 dimensions, slower, higher quality
- `paraphrase-multilingual-MiniLM-L12-v2` — 384 dimensions, multilingual support

**Offline mode:** Set `embedding.offline: true` to ensure only locally cached models are used.

### Ollama / LM Studio

Connects to a local Ollama or LM Studio server that exposes an OpenAI-compatible API.

**Pros:** GPU-accelerated, supports a wide range of models
**Cons:** Requires running server

**Setup:**
1. Install [Ollama](https://ollama.ai/) or [LM Studio](https://lmstudio.ai/)
2. Start the server (Ollama: `ollama serve`, LM Studio: built-in server)
3. Pull a model: `ollama pull nomic-embed-text`
4. Configure the URL and model name in Settings

LM Studio uses an OpenAI-compatible endpoint — the discovery will automatically detect it.

### Fallback

A simple random-projection backend for testing when no other backend is available. **Not suitable for production** — embeddings are not meaningful and cannot be used for semantic search across different sessions.

---

## Example Configs

### Sentence Transformers (offline)

```json
{
  "embedding": {
    "backend": "sentence-transformers",
    "model": "all-MiniLM-L6-v2",
    "offline": true,
    "base_url": "http://localhost:11434"
  }
}
```

### Ollama

```json
{
  "embedding": {
    "backend": "ollama",
    "model": "nomic-embed-text:latest",
    "offline": false,
    "base_url": "http://localhost:11434"
  }
}
```

### LM Studio (OpenAI-compatible)

```json
{
  "embedding": {
    "backend": "ollama",
    "model": "nomic-embed-text:latest",
    "offline": false,
    "base_url": "http://localhost:1234"
  }
}
```

---

## GUI Settings

The Memory Manager GUI (accessible from the system tray or via `memory_manager_gui.py`) includes a **Settings tab** where you can:

1. Select the embedding backend from a dropdown
2. Configure backend-specific options
3. Discover available models from Ollama/LM Studio servers
4. Save settings to `config.json`

**Note:** Changes to the embedding backend require an MCP server restart to take effect. Restart LM Studio or the MCP server after saving.

---

## Dimension Mismatch Warning

If you change the embedding model to one with a different number of dimensions, your existing vector database (ChromaDB) will be incompatible — existing embeddings were generated with a different dimensionality.

The MCP server will warn you about this on startup. You may need to re-index your memories after changing the embedding model.
