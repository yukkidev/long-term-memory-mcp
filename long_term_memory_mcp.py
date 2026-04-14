"""\
Robust Long-Term Memory System for AI Companions\
Hybrid approach: ChromaDB (vector) + SQLite (structured) + File backup

Features:

* Semantic search via embeddings

* Structured metadata queries

* Cross-platform compatibility (Windows/Ubuntu/macOS)

* Automatic backups and data integrity

* Migration-friendly exports

* Scalable to decades of conversations\
  """

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import shutil
import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
import logging
from logging.handlers import TimedRotatingFileHandler
import asyncio
import sys
import atexit
from fastmcp import FastMCP

# Local imports
try:
    from .embedding_backends import create_embedding_backend
except ImportError:
    # Fallback for direct execution
    from embedding_backends import create_embedding_backend

# Config loading (must happen before RobustMemorySystem is instantiated)
print("[MCP] Loading config...")
try:
    from .config_manager import Config
except ImportError:
    from config_manager import Config

config = Config()  # Module-level singleton
print(f"[MCP] Config loaded: backend={config.get_embedding_backend_type()}, model={config.get_model_name()}, base_url={config.get_base_url()}")

# Status file path — written by MCP server, read by GUI
DEFAULT_CONFIG_DIR = config.config_path.parent
STATUS_FILE = DEFAULT_CONFIG_DIR / "status.json"

# Third-party imports (will be installed)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print("Missing required packages. Install with: pip install chromadb")
    print(f"Error: {e}")
    sys.exit(1)

# Configuration
# You can override the data folder by setting the AI_COMPANION_DATA_DIR environment variable.
# Example (PowerShell): $env:AI_COMPANION_DATA_DIR = "D:\a.i. apps\long_term_memory_mcp\data"

DATA_FOLDER = Path(
    os.environ.get(
        "AI_COMPANION_DATA_DIR", str(Path.home() / "Documents" / "ai_companion_memory")
    )
)

# -------- Lazy Decay Configuration --------
DECAY_ENABLED = True

# Half-life in days by memory_type (how fast each type fades)
DECAY_HALF_LIFE_DAYS_BY_TYPE = {
    "conversation": 45,
    "fact": 120,
    "preference": 90,
    "task": 30,
    "ephemeral": 10,
}
DECAY_HALF_LIFE_DAYS_DEFAULT = 60

# Minimum floor per type (never decay below this)
DECAY_MIN_IMPORTANCE_BY_TYPE = {
    "conversation": 2,
    "fact": 3,
    "preference": 2,
    "task": 1,
    "ephemeral": 1,
}
DECAY_MIN_IMPORTANCE_DEFAULT = 1

# Tags that prevent decay entirely
DECAY_PROTECT_TAGS = {"core", "identity", "pinned"}

# Writeback policy (to avoid churn)
DECAY_WRITEBACK_STEP = 0.5  # only persist if change >= 0.5
DECAY_MIN_INTERVAL_HOURS = 12  # don't write decay more often than this
# -----------------------------------------

# -------- Reinforcement Configuration --------
REINFORCEMENT_ENABLED = True
REINFORCEMENT_STEP = 0.1  # amount per retrieval
REINFORCEMENT_WRITEBACK_STEP = 0.5  # write to DB when accumulated ≥ 0.5
REINFORCEMENT_MAX = 10  # cap importance
# ---------------------------------------------

# -------- Embedding Configuration --------
# All config now flows through the `config` object (loaded from config.json + env vars).
# For backward compatibility, we read from `config` but keep the module-level
# constants as derived values so existing code paths (e.g., _init_embeddings)
# don't need to change their variable names.

EMBEDDING_BACKEND = config.get_embedding_backend_type()
_EMBEDDING_MODEL_RAW = config.get_model_name()  # May be None or a cross-backend stale value
EMBEDDING_OFFLINE = config.get_offline()
OLLAMA_BASE_URL = config.get_base_url()
FALLBACK_DIMENSIONS = config.get_dimensions()

# Validate model names to prevent cross-backend contamination.
# Ollama model names (e.g. "nomic-embed-text:latest") are NOT valid HuggingFace
# sentence-transformer IDs, and vice versa. If a stale model name from a different
# backend is detected, we fall back to the default for that backend.
_ST_VALID_PATTERNS = (":", "/", "nomic", "embed", "ollama")
_OLLAMA_VALID_PATTERNS = (":", "/")


def _make_st_model(model: Optional[str]) -> str:
    if model and not any(p in model for p in _ST_VALID_PATTERNS):
        return model
    return "all-MiniLM-L6-v2"


def _make_ollama_model(model: Optional[str]) -> str:
    if model and any(p in model for p in _OLLAMA_VALID_PATTERNS):
        return model
    return "nomic-embed-text:latest"


SENTENCE_TRANSFORMER_MODEL = (
    _make_st_model(_EMBEDDING_MODEL_RAW)
    if EMBEDDING_BACKEND == "sentence-transformers"
    else "all-MiniLM-L6-v2"
)
OLLAMA_MODEL = (
    _make_ollama_model(_EMBEDDING_MODEL_RAW)
    if EMBEDDING_BACKEND == "ollama"
    else "nomic-embed-text:latest"
)
# -----------------------------------------


@dataclass
class MemoryRecord:
    """
    Structured memory record.

    Attributes:
        id (str)
        title (str)
        content (str)
        timestamp (datetime)
        tags (List[str])
        importance (int): 1–10 scale
        memory_type (str): conversation, fact, preference, event, etc.
        metadata (Dict[str, Any])
    """

    id: str
    title: str
    content: str
    timestamp: datetime
    tags: List[str]
    importance: int
    memory_type: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """
    Search result with relevance score.

    Attributes:
        record (MemoryRecord): The matched memory record.
        relevance_score (float): Similarity or match score.
        match_type (str): Type of match (e.g., "semantic", "exact", "metadata").
    """

    record: MemoryRecord
    relevance_score: float
    match_type: str  # semantic, exact, metadata


@dataclass
class Result:
    """
    Standard result container for memory operations.

    Attributes:
        success (bool): Whether the operation succeeded.
        reason (str, optional): Explanation when the operation fails.
        data (list of dict, optional): Operation-specific data, such as
            memory objects, statistics, or search results.
    """

    success: bool
    reason: Optional[str] = None
    data: Optional[List[Dict]] = None


class RobustMemorySystem:
    """
    Hybrid memory system combining:
    1. ChromaDB for semantic/vector search
    2. SQLite for structured queries and metadata
    3. JSON backup files for portability
    """

    def __init__(self, data_folder: Path = DATA_FOLDER):
        self.data_folder = Path(data_folder)
        self.db_folder = self.data_folder / "memory_db"
        self.backup_folder = self.data_folder / "memory_backups"
        self.sqlite_path = self.db_folder / "memories.db"

        # Create directories
        self.db_folder.mkdir(parents=True, exist_ok=True)
        self.backup_folder.mkdir(parents=True, exist_ok=True)

        # Predeclare attributes for linters/type checkers
        self.logger = None  # will be set in _setup_logging
        self.sqlite_conn = None  # will be set in _init_sqlite
        self.chroma_client: Optional[object] = None
        self.chroma_collection: Optional[object] = None
        self.embedding_backend = None  # Will be set in _init_embeddings

        # Setup logging
        self._setup_logging()

        # Initialize components
        self._init_sqlite()
        self._init_chromadb()
        self._init_embeddings()

        # Perform integrity check on startup
        self._integrity_check()

        # Check for embedding model changes that might require reindexing
        self._check_embedding_model_change()

    def _setup_logging(self):
        """Setup logging for debugging and monitoring"""
        log_file = self.data_folder / "memory_system.log"

        # Daily rotation, keep 30 days
        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=30, utc=False
        )
        file_handler.setLevel(logging.INFO)  # Keep full INFO in the file

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only WARNING+ to console/stderr

        # Set format for both
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,  # Overall minimum level
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[file_handler, console_handler],
        )

        self.logger = logging.getLogger(__name__)

    def _init_sqlite(self):
        """Initialize SQLite database for structured data"""
        try:
            self.sqlite_conn = sqlite3.connect(
                str(self.sqlite_path), check_same_thread=False, timeout=30.0
            )
            self.sqlite_conn.row_factory = sqlite3.Row

            self.sqlite_conn.execute("PRAGMA journal_mode=WAL;")
            self.sqlite_conn.execute(
                "PRAGMA wal_autocheckpoint=500;"
            )  # checkpoint every 500 pgs
            self.sqlite_conn.commit()

            # Create base tables (OK to have DEFAULT CURRENT_TIMESTAMP here for fresh DBs)
            self.sqlite_conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,  
                    timestamp TEXT NOT NULL,  
                    tags TEXT,  -- JSON array  
                    importance INTEGER DEFAULT 5,  
                    memory_type TEXT DEFAULT 'conversation',  
                    metadata TEXT,  -- JSON object  
                    content_hash TEXT,  
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,  
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,  
                    last_accessed TEXT DEFAULT CURRENT_TIMESTAMP  
                );  
      
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp);  
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);  
                CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);  
                CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash);  
                CREATE TABLE IF NOT EXISTS memory_stats (  
                    key TEXT PRIMARY KEY,  
                    value TEXT,  
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP  
                );  
                CREATE TABLE IF NOT EXISTS system_config (  
                    key TEXT PRIMARY KEY,  
                    value TEXT,  
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP  
                );  
            """
            )

            # Migration: ensure last_accessed exists in existing DBs
            cursor = self.sqlite_conn.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]

            if "last_accessed" not in columns:
                self.logger.info(
                    "Adding last_accessed column to existing memories table"
                )
                # 1) Add column WITHOUT default (avoids 'non-constant default' error)
                self.sqlite_conn.execute(
                    "ALTER TABLE memories ADD COLUMN last_accessed TEXT"
                )

                # 2) Backfill existing rows
                now_iso = datetime.now(timezone.utc).isoformat()
                self.sqlite_conn.execute(
                    "UPDATE memories SET last_accessed = COALESCE(created_at, ?)",
                    (now_iso,),
                )

                self.sqlite_conn.commit()

            # Now that the column exists, create its index
            self.sqlite_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed)"
            )

            # Schema version bookkeeping
            self.sqlite_conn.execute(
                "INSERT OR REPLACE INTO memory_stats (key, value, updated_at)"
                "VALUES ('schema_version', '1.1', CURRENT_TIMESTAMP)"
            )

            # Normalize last_backup to an ISO UTC string
            now_iso = datetime.now(timezone.utc).isoformat()
            self.sqlite_conn.execute(
                "INSERT OR REPLACE INTO memory_stats (key, value, updated_at)"
                "VALUES ('last_backup', ?, ?)",
                (now_iso, now_iso),
            )

            self.sqlite_conn.commit()
            self.logger.info("SQLite database initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize SQLite: %s", e)
            raise

    def _init_chromadb(self):
        """Initialize ChromaDB for vector storage"""
        try:
            # Use persistent storage
            chroma_path = str(self.db_folder / "chroma_db")

            self.chroma_client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Use a stable collection name and cosine space
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="ai_companion_memories",
                metadata={
                    "description": "Long-term memory for AI companion",
                    "hnsw:space": "cosine",  # important for sentence embeddings
                },
                embedding_function=None,  # we pass embeddings manually
            )

            self.logger.info("ChromaDB initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize ChromaDB: %s", e)
            raise

    def _init_embeddings(self):
        """Initialize embedding backend with fallback support"""
        backend_type = EMBEDDING_BACKEND
        backend_attempts = []

        # Try to create the requested backend
        try:
            if backend_type == "sentence-transformers":
                print(f"[MCP] Loading sentence-transformers backend: model={SENTENCE_TRANSFORMER_MODEL}, offline={EMBEDDING_OFFLINE}")
                self.embedding_backend = create_embedding_backend(
                    backend_type=backend_type,
                    model_name=SENTENCE_TRANSFORMER_MODEL,
                    offline=EMBEDDING_OFFLINE,
                )
            elif backend_type == "ollama":
                print(f"[MCP] Loading Ollama backend: model={OLLAMA_MODEL}, base_url={OLLAMA_BASE_URL}")
                self.embedding_backend = create_embedding_backend(
                    backend_type=backend_type,
                    model_name=OLLAMA_MODEL,
                    base_url=OLLAMA_BASE_URL,
                )
            else:
                # Use fallback for any other type
                print(f"[MCP] Loading fallback backend: dimensions={FALLBACK_DIMENSIONS}")
                self.embedding_backend = create_embedding_backend(
                    backend_type="fallback", dimensions=FALLBACK_DIMENSIONS
                )

            backend_attempts.append(f"{backend_type} (primary)")
            print(f"[MCP] Embedding backend initialized: {self.embedding_backend.get_model_name()} ({self.embedding_backend.get_dimensions()} dimensions)")
            self.logger.info(
                f"Embedding backend initialized: {self.embedding_backend.get_model_name()} "
                f"({self.embedding_backend.get_dimensions()} dimensions)"
            )
            self._write_status()

        except Exception as e:
            self.logger.warning(f"Failed to initialize {backend_type} backend: {e}")
            backend_attempts.append(f"{backend_type} (failed: {e})")

            # Try fallback options
            fallback_order = ["sentence-transformers", "ollama", "fallback"]
            fallback_order.remove(backend_type)  # Remove already attempted

            for fallback_type in fallback_order:
                try:
                    if fallback_type == "sentence-transformers":
                        print(f"[MCP] Falling back to sentence-transformers: model=all-MiniLM-L6-v2, offline=True")
                        self.embedding_backend = create_embedding_backend(
                            backend_type=fallback_type,
                            model_name="all-MiniLM-L6-v2",
                            offline=True,
                        )
                    elif fallback_type == "ollama":
                        print(f"[MCP] Falling back to ollama: model=nomic-embed-text:latest, base_url={OLLAMA_BASE_URL}")
                        self.embedding_backend = create_embedding_backend(
                            backend_type=fallback_type,
                            model_name="nomic-embed-text:latest",
                            base_url=OLLAMA_BASE_URL,
                        )
                    else:  # fallback
                        print(f"[MCP] Falling back to fallback backend: dimensions={FALLBACK_DIMENSIONS}")
                        self.embedding_backend = create_embedding_backend(
                            backend_type="fallback", dimensions=FALLBACK_DIMENSIONS
                        )

                    backend_attempts.append(f"{fallback_type} (fallback success)")
                    print(f"[MCP] Fallback backend initialized: {self.embedding_backend.get_model_name()} ({self.embedding_backend.get_dimensions()} dimensions)")
                    self.logger.info(
                        f"Using fallback embedding backend: {self.embedding_backend.get_model_name()} "
                        f"({self.embedding_backend.get_dimensions()} dimensions)"
                    )
                    break

                except Exception as fallback_e:
                    backend_attempts.append(f"{fallback_type} (failed: {fallback_e})")

            # If all backends failed, raise an error
            if self.embedding_backend is None:
                attempts_str = ", ".join(backend_attempts)
                error_msg = f"All embedding backends failed. Attempted: {attempts_str}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Write status after successful fallback
            self._write_status()

    def _write_status(self):
        """Write current backend status to status.json so the GUI can read it."""
        import json
        status = {
            "loaded_backend": self.embedding_backend.get_model_name(),
            "loaded_dimensions": self.embedding_backend.get_dimensions(),
            "loaded_backend_type": (
                "sentence-transformers"
                if "sentence-transformers" in self.embedding_backend.get_model_name()
                else "ollama"
                if "ollama" in self.embedding_backend.get_model_name()
                else "fallback"
            ),
            "config_backend": EMBEDDING_BACKEND,
            "config_model": (
                SENTENCE_TRANSFORMER_MODEL
                if EMBEDDING_BACKEND == "sentence-transformers"
                else OLLAMA_MODEL
            ),
            "config_base_url": OLLAMA_BASE_URL,
            "status": "loaded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            status_path = DEFAULT_CONFIG_DIR / "status.json"
            status_path.parent.mkdir(parents=True, exist_ok=True)
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)
            print(f"[MCP] Status written to {status_path}")
        except Exception as e:
            print(f"[MCP] Failed to write status file: {e}")

    def _generate_id(self, content: str, timestamp: datetime) -> str:
        """Generate unique ID for memory record"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        time_hash = hashlib.sha256(timestamp.isoformat().encode()).hexdigest()[:8]
        return f"mem_{time_hash}_{content_hash}"

    def _content_hash(self, content: str) -> str:
        """Generate content hash for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _integrity_check(self):
        """Check data integrity between SQLite and ChromaDB"""
        try:
            # Count records in both systems
            cursor = self.sqlite_conn.execute("SELECT COUNT(*) FROM memories")
            sqlite_count = cursor.fetchone()[0]

            chroma_count = self.chroma_collection.count()

            self.logger.info(
                "Integrity check: SQLite=%s, ChromaDB=%s",
                sqlite_count,
                chroma_count,
            )

            if sqlite_count != chroma_count:
                self.logger.warning("Record count mismatch between SQLite and ChromaDB")
                # Could implement auto-repair here

        except Exception as e:
            self.logger.error("Integrity check failed: %s", e)

    def _check_embedding_model_change(self):
        """Check if embedding model has changed and warn if reindexing might be needed.

        Compares BOTH model name and dimensions against stored values.
        A dimension mismatch is fatal by default (to prevent silent corruption).
        Set EMBEDDING_FORCE_LOAD=1 to override and proceed with a warning — this
        allows the server to start so rebuild_vectors() can be called.
        """
        # Allow override via env var so the server can start even on mismatch,
        # enabling rebuild_vectors() to be called for recovery.
        force_load = os.environ.get("EMBEDDING_FORCE_LOAD", "").lower() in ("1", "true", "yes")

        try:
            current_model = self.embedding_backend.get_model_name()
            current_dimensions = self.embedding_backend.get_dimensions()

            try:
                cursor = self.sqlite_conn.execute(
                    "SELECT value FROM system_config WHERE key = 'embedding_model'"
                )
                row = cursor.fetchone()
                stored_model = row[0] if row else None

                cursor2 = self.sqlite_conn.execute(
                    "SELECT value FROM system_config WHERE key = 'embedding_dimensions'"
                )
                row2 = cursor2.fetchone()
                stored_dims = int(row2[0]) if row2 else None
            except Exception:
                stored_model = None
                stored_dims = None

            # Check for dimension mismatch
            if stored_dims is not None and stored_dims != current_dimensions:
                msg = (
                    f"Embedding dimension mismatch! Collection was indexed with "
                    f"{stored_dims}d vectors, but the current backend produces "
                    f"{current_dimensions}d embeddings. "
                    + ("EMBEDDING_FORCE_LOAD is set — proceeding anyway; call rebuild_vectors() to fix."
                        if force_load else
                        "Set EMBEDDING_FORCE_LOAD=1 to override and run rebuild_vectors().")
                )
                if force_load:
                    self.logger.warning(f"WARNING: {msg}")
                else:
                    self.logger.error(msg)
                    raise ValueError(msg)

            # Warn if model name changed (dimensions are fine, but embeddings may differ)
            if stored_model is not None and stored_model != current_model:
                self.logger.warning(
                    f"Embedding model changed from '{stored_model}' to '{current_model}'. "
                    f"Dimensions match ({current_dimensions}d) but semantic similarity "
                    f"scores may be different. Run rebuild_vectors() if search quality degrades."
                )

            # Store current model info for future checks
            try:
                self.sqlite_conn.execute(
                    "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
                    ("embedding_model", current_model),
                )
                self.sqlite_conn.execute(
                    "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
                    ("embedding_dimensions", str(current_dimensions)),
                )
                self.sqlite_conn.commit()
            except Exception as e:
                self.logger.debug(f"Could not store model config: {e}")

        except ValueError:
            # Re-raise fatal dimension-mismatch errors
            raise
        except Exception as e:
            self.logger.debug(f"Error checking embedding model change: {e}")

    def remember(
        self,
        title: str,
        content: str,
        tags: List[str] = None,
        importance: int = 5,
        memory_type: str = "conversation",
        metadata: Dict[str, Any] = None,
    ) -> Result:
        """
        Store a new memory with both vector and structured storage
        """
        try:
            if not title or not content:
                return Result(success=False, reason="Title and content are required")

            # Validate importance
            importance = max(1, min(10, importance))

            # Prepare data
            timestamp = datetime.now(timezone.utc)
            tags = tags or []
            metadata = metadata or {}

            # Generate ID and hash
            memory_id = self._generate_id(content, timestamp)
            content_hash = self._content_hash(content)

            # Check for duplicates
            cursor = self.sqlite_conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?", (content_hash,)
            )
            if cursor.fetchone():
                return Result(success=False, reason="Duplicate content detected")

            # Create memory record
            record = MemoryRecord(
                id=memory_id,
                title=title,
                content=content,
                timestamp=timestamp,
                tags=tags,
                importance=importance,
                memory_type=memory_type,
                metadata=metadata,
            )

            # Store in SQLite
            self.sqlite_conn.execute(
                """
                INSERT INTO memories 
                (id, title, content, timestamp, tags, importance,
                 memory_type, metadata, content_hash, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.id,
                    record.title,
                    record.content,
                    record.timestamp.isoformat(),
                    json.dumps(record.tags),
                    record.importance,
                    record.memory_type,
                    json.dumps(record.metadata),
                    content_hash,
                    record.timestamp.isoformat(),  # Set last_accessed to creation time
                ),
            )

            # Generate embedding and store in ChromaDB
            # Combine title and content for better semantic search
            text_for_embedding = f"{title}\n{content}"
            embedding = self.embedding_backend.get_embedding(text_for_embedding)

            self.chroma_collection.add(
                ids=[record.id],
                embeddings=[embedding],
                documents=[text_for_embedding],
                metadatas=[
                    {
                        "title": title,
                        "timestamp": record.timestamp.isoformat(),
                        "importance": importance,
                        "memory_type": memory_type,
                        "tags": json.dumps(tags),
                    }
                ],
            )

            try:
                if hasattr(self.chroma_client, "persist"):
                    self.chroma_client.persist()
            except Exception as pe:
                self.logger.warning("Chroma persist warning: %s", pe)

            # Debug: check what Chroma actually contains after add
            self.logger.info("Chroma after add: %s", self._debug_vector_index())

            self.sqlite_conn.commit()

            # Trigger backup if needed
            self._maybe_backup()

            self.logger.info("Memory stored successfully: %s", memory_id)
            rec = asdict(record)
            rec["timestamp"] = record.timestamp.isoformat()
            return Result(success=True, data=[rec])

        except Exception as e:
            self.logger.error("Failed to store memory: %s", e)
            self.sqlite_conn.rollback()
            return Result(success=False, reason=f"Storage error: {str(e)}")

    def search_semantic(
        self, query: str, limit: int = 10, min_relevance: float = 0.15
    ) -> Result:
        """
        Semantic search using vector similarity with adaptive thresholding + top-1 fallback.
        """
        try:
            if not query.strip():
                return Result(success=False, reason="Query cannot be empty")

            # Debug: check what Chroma contains before query
            self.logger.info(
                "Chroma before query: %s",
                self._debug_vector_index(),
            )

            # Generate query embedding
            query_embedding = self.embedding_backend.get_embedding(query)
            print(f"[MCP] search_semantic: query='{query}', backend={EMBEDDING_BACKEND}, embedding_dims={len(query_embedding)}")

            # Search ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            if not results["ids"][0]:
                print(f"[MCP] search_semantic: no results from ChromaDB")
                return Result(success=True, data=[])

            ids = results["ids"][0]
            distances = results["distances"][0]
            similarities = [1.0 - d for d in distances]

            print(f"[MCP] search_semantic: top similarities={[f'{s:.3f}' for s in similarities[:5]]}")

            # Adaptive threshold anchored on top match, with clamps
            # Larger embedding models (Ollama 768d+) produce lower similarities,
            # so use a lower floor for them vs sentence-transformers (384d)
            is_ollama = EMBEDDING_BACKEND == "ollama"
            floor = 0.05 if is_ollama else 0.12
            if similarities:
                top_sim = similarities[0]
                adaptive = max(floor, min(0.35, top_sim - 0.05))
                threshold = max(min_relevance, adaptive)
            else:
                threshold = min_relevance

            print(f"[MCP] search_semantic: top_sim={top_sim:.3f}, adaptive={adaptive:.3f}, threshold={threshold:.3f}")

            self.logger.info("Adaptive threshold computed: %.3f", threshold)

            search_results = []
            now_iso = datetime.now(timezone.utc).isoformat()

            # First pass: collect those meeting threshold
            selected = [
                (mid, sim) for mid, sim in zip(ids, similarities) if sim >= threshold
            ]

            # FALLBACK: if none pass threshold, keep the top-1 candidate anyway
            if not selected and ids:
                fallback_floor = 0.05 if is_ollama else 0.08
                if similarities[0] >= fallback_floor:  # only fallback if it's not total garbage
                    print(f"[MCP] search_semantic: no candidates passed threshold, using top-1 fallback (sim={similarities[0]:.3f})")
                    selected = [(ids[0], similarities[0])]
                else:
                    print(f"[MCP] search_semantic: no candidates passed and top-1 sim {similarities[0]:.3f} < 0.05, skipping fallback")

            # Fetch selected rows and reinforce
            for i, (memory_id, relevance) in enumerate(selected):
                if i < 3:
                    self.logger.info(
                        "Candidate %d: relevance=%.3f, threshold=%.3f",
                        i,
                        relevance,
                        threshold,
                    )

                cursor = self.sqlite_conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()

                if not row:
                    continue

                # Lazy decay before reinforcement
                self._maybe_decay(row)
                self._maybe_reinforce(row)

                # Reinforcement
                self.sqlite_conn.execute(
                    "UPDATE memories SET last_accessed = ? WHERE id = ?",
                    (now_iso, memory_id),
                )

                record = MemoryRecord(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    tags=json.loads(row["tags"]),
                    importance=row["importance"],
                    memory_type=row["memory_type"],
                    metadata=json.loads(row["metadata"]),
                )

                search_results.append(
                    SearchResult(
                        record=record,
                        relevance_score=relevance,
                        match_type="semantic"
                        if relevance >= threshold
                        else "semantic_fallback",
                    )
                )

            # Commit the last_accessed updates
            self.sqlite_conn.commit()

            # Sort by relevance
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)

            # Convert to dict format
            result_data = []
            for sr in search_results:
                result_dict = asdict(sr.record)
                result_dict["timestamp"] = sr.record.timestamp.isoformat()
                result_dict["relevance_score"] = sr.relevance_score
                result_dict["match_type"] = sr.match_type
                result_data.append(result_dict)

            self.logger.info(
                "Semantic search returned %d results (threshold=%.3f)",
                len(result_data),
                threshold,
            )
            return Result(success=True, data=result_data)

        except Exception as e:
            self.logger.error("Semantic search failed: %s", e)
            return Result(success=False, reason=f"Search error: {str(e)}")

    def search_structured(
        self,
        memory_type: str = None,
        tags: List[str] = None,
        importance_min: int = None,
        date_from: str = None,
        date_to: str = None,
        limit: int = 50,
    ) -> Result:
        """
        Structured search using SQL queries
        """
        try:
            conditions = []
            params = []

            if memory_type:
                conditions.append("memory_type = ?")
                params.append(memory_type)

            if importance_min:
                conditions.append("importance >= ?")
                params.append(importance_min)

            if date_from:
                conditions.append("timestamp >= ?")
                params.append(date_from)

            if date_to:
                conditions.append("timestamp <= ?")
                params.append(date_to)

            if tags:
                # Search for any of the provided tags
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')
                conditions.append(f"({' OR '.join(tag_conditions)})")

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            query = f"""
                SELECT * FROM memories
                WHERE {where_clause}
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """
            params.append(limit)

            cursor = self.sqlite_conn.execute(query, params)
            rows = cursor.fetchall()

            # REINFORCEMENT: Update last_accessed for retrieved memories
            now_iso = datetime.now(timezone.utc).isoformat()
            memory_ids = [row["id"] for row in rows]

            if memory_ids:
                placeholders = ",".join(["?" for _ in memory_ids])
                self.sqlite_conn.execute(
                    f"""UPDATE memories SET last_accessed = ? WHERE id IN ({
                        placeholders
                    })""",
                    [now_iso] + memory_ids,
                )
                self.sqlite_conn.commit()

            # Convert to MemoryRecord objects
            results = []
            for row in rows:
                self._maybe_decay(row)
                self._maybe_reinforce(row)
                record = MemoryRecord(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    tags=json.loads(row["tags"]),
                    importance=row["importance"],
                    memory_type=row["memory_type"],
                    metadata=json.loads(row["metadata"]),
                )

                result_dict = asdict(record)
                result_dict["timestamp"] = record.timestamp.isoformat()
                result_dict["match_type"] = "structured"
                results.append(result_dict)

            self.logger.info(
                "Structured search returned %d results",
                len(results),
            )
            return Result(success=True, data=results)

        except Exception as e:
            self.logger.error("Structured search failed: %s", e)
            return Result(success=False, reason=f"Search error: {str(e)}")

    def get_recent(self, limit: int = 20) -> Result:
        """Get most recent memories"""
        return self.search_structured(limit=limit)

    def update_memory(
        self,
        memory_id: str,
        title: str = None,
        content: str = None,
        tags: List[str] = None,
        importance: int = None,
        memory_type: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Result:
        """
        Update or modify an existing memory by its unique ID.
        Also updates updated_at and last_accessed (treating edits as an access).
        """
        try:
            # Get existing record
            cursor = self.sqlite_conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()

            if not row:
                return Result(success=False, reason="Memory not found")

            # Prepare updates
            updates = []
            params = []

            if title is not None:
                updates.append("title = ?")
                params.append(title)

            if content is not None:
                updates.append("content = ?")
                params.append(content)
                updates.append("content_hash = ?")
                params.append(self._content_hash(content))

            if tags is not None:
                updates.append("tags = ?")
                params.append(json.dumps(tags))

            if importance is not None:
                importance = max(1, min(10, importance))
                updates.append("importance = ?")
                params.append(importance)

            if memory_type is not None:
                updates.append("memory_type = ?")
                params.append(memory_type)

            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))

            now_iso = datetime.now(timezone.utc).isoformat()
            updates.append("updated_at = ?")
            params.append(now_iso)
            updates.append("last_accessed = ?")
            params.append(now_iso)

            params.append(memory_id)

            # Update SQLite
            update_query = f"""UPDATE memories SET {", ".join(updates)} WHERE id = ?"""
            self.sqlite_conn.execute(update_query, params)

            # Update ChromaDB if content changed
            if content is not None or title is not None:
                # Get updated record
                cursor = self.sqlite_conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                )
                updated_row = cursor.fetchone()

                # Re-generate embedding
                text_for_embedding = (
                    f"""{updated_row["title"]}\n{updated_row["content"]}"""
                )
                embedding = self.embedding_backend.get_embedding(text_for_embedding)

                # Update ChromaDB
                self.chroma_collection.update(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[text_for_embedding],
                    metadatas=[
                        {
                            "title": updated_row["title"],
                            "timestamp": updated_row["timestamp"],
                            "importance": updated_row["importance"],
                            "memory_type": updated_row["memory_type"],
                            "tags": updated_row["tags"],
                        }
                    ],
                )

                try:
                    if hasattr(self.chroma_client, "persist"):
                        self.chroma_client.persist()
                except Exception as pe:
                    self.logger.warning("Chroma persist warning: %s", pe)

            self.sqlite_conn.commit()

            self.logger.info("Memory updated successfully: %s", memory_id)
            return Result(success=True, data=[{"id": memory_id, "updated": True}])

        except Exception as e:
            self.logger.error("Failed to update memory: %s", e)
            self.sqlite_conn.rollback()
            return Result(success=False, reason=f"Update error: {str(e)}")

    def delete_memory(self, memory_id: str) -> Result:
        """
        Delete a memory from both systems
        """
        try:
            # Check if exists
            cursor = self.sqlite_conn.execute(
                "SELECT title, content FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()

            if not row:
                return Result(success=False, reason="Memory not found")

            # Delete from SQLite
            self.sqlite_conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

            # Delete from ChromaDB
            self.chroma_collection.delete(ids=[memory_id])

            self.sqlite_conn.commit()

            self.logger.info("Memory deleted successfully: %s", memory_id)
            return Result(
                success=True,
                data=[{"id": memory_id, "title": row["title"], "deleted": True}],
            )

        except Exception as e:
            self.logger.error("Failed to delete memory: %s", e)
            self.sqlite_conn.rollback()
            return Result(success=False, reason=f"Delete error: {str(e)}")

    def get_statistics(self) -> Result:
        """Get memory system statistics"""
        try:
            cursor = self.sqlite_conn.execute(
                """
                SELECT 
                    COUNT(*) as total_memories,
                    COUNT(DISTINCT memory_type) as memory_types,
                    AVG(importance) as avg_importance,
                    MIN(timestamp) as oldest_memory,
                    MAX(timestamp) as newest_memory
                FROM memories
            """
            )
            stats = cursor.fetchone()

            # Get memory type breakdown
            cursor = self.sqlite_conn.execute(
                """
                SELECT memory_type, COUNT(*) as count
                FROM memories
                GROUP BY memory_type
                ORDER BY count DESC
            """
            )
            type_breakdown = {
                row["memory_type"]: row["count"] for row in cursor.fetchall()
            }

            # Get database sizes
            sqlite_size = (
                self.sqlite_path.stat().st_size if self.sqlite_path.exists() else 0
            )
            chroma_size = sum(
                f.stat().st_size
                for f in (self.db_folder / "chroma_db").rglob("*")
                if f.is_file()
            )

            result_data = {
                "total_memories": stats["total_memories"],
                "memory_types": stats["memory_types"],
                "avg_importance": round(stats["avg_importance"] or 0, 2),
                "oldest_memory": stats["oldest_memory"],
                "newest_memory": stats["newest_memory"],
                "type_breakdown": type_breakdown,
                "storage_size_mb": round((sqlite_size + chroma_size) / 1024 / 1024, 2),
                "sqlite_size_mb": round(sqlite_size / 1024 / 1024, 2),
                "chroma_size_mb": round(chroma_size / 1024 / 1024, 2),
            }

            return Result(success=True, data=[result_data])

        except Exception as e:
            self.logger.error("Failed to get statistics: %s", e)
            return Result(success=False, reason=f"Statistics error: {str(e)}")

    def _maybe_backup(self):
        """Trigger backup if conditions are met"""
        try:
            # Get last backup time (string)
            cursor = self.sqlite_conn.execute(
                "SELECT value FROM memory_stats WHERE key = 'last_backup'"
            )
            row = cursor.fetchone()

            last_backup = None
            if row and row["value"]:
                raw = row["value"]
                try:
                    # Handle common ISO formats and 'Z'
                    val = raw.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(val)
                except Exception:
                    dt = None

                if dt is not None:
                    # Normalize to timezone-aware (assume UTC if naive)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    last_backup = dt

            # If we couldn't parse, default far in the past to force a backup later
            if last_backup is None:
                last_backup = datetime(1970, 1, 1, tzinfo=timezone.utc)

            hours_since_backup = (
                datetime.now(timezone.utc) - last_backup
            ).total_seconds() / 3600.0

            # Backup every 24 hours or every 100 new memories
            cursor = self.sqlite_conn.execute("SELECT COUNT(*) FROM memories")
            total_memories = cursor.fetchone()[0]

            if hours_since_backup > 24 or (
                total_memories > 0 and total_memories % 100 == 0
            ):
                self.create_backup()

        except Exception as e:
            self.logger.error("Backup check failed: %s", e)

    def create_backup(self) -> Result:
        """Create a complete backup of the memory system"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"memory_backup_{timestamp}"
            backup_path = self.backup_folder / backup_name
            backup_path.mkdir(exist_ok=True)

            # ===== CHECKPOINT BEFORE BACKUP =====
            # This merges the WAL into the main .db file and truncates the WAL
            self.logger.warning("Starting backup: checkpointing WAL...")
            try:
                self.sqlite_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self.sqlite_conn.commit()
                self.logger.warning("WAL checkpoint completed")
            except Exception as e:
                self.logger.error("Checkpoint failed: %s", e)
                # Continue anyway; we'll copy all files as fallback

            # Backup SQLite database + WAL files
            # Copy main .db
            sqlite_backup = backup_path / "memories.db"
            shutil.copy2(self.sqlite_path, sqlite_backup)

            # Copy WAL and SHM if they exist (belt-and-suspenders)
            wal_path = Path(str(self.sqlite_path) + "-wal")
            shm_path = Path(str(self.sqlite_path) + "-shm")

            if wal_path.exists():
                shutil.copy2(wal_path, backup_path / "memories.db-wal")
                self.logger.info("Copied WAL file to backup")

            if shm_path.exists():
                shutil.copy2(shm_path, backup_path / "memories.db-shm")
                self.logger.info("Copied SHM file to backup")

            # Backup ChromaDB
            chroma_backup = backup_path / "chroma_db"
            if (self.db_folder / "chroma_db").exists():
                shutil.copytree(self.db_folder / "chroma_db", chroma_backup)

            # Export to JSON for portability
            cursor = self.sqlite_conn.execute(
                "SELECT * FROM memories ORDER BY timestamp"
            )
            memories = []
            for row in cursor.fetchall():
                memory_dict = dict(row)
                memory_dict["tags"] = json.loads(memory_dict["tags"])
                memory_dict["metadata"] = json.loads(memory_dict["metadata"])
                memories.append(memory_dict)

            json_backup = backup_path / "memories_export.json"
            with open(json_backup, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "export_timestamp": datetime.now(timezone.utc).isoformat(),
                        "total_memories": len(memories),
                        "memories": memories,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # Update backup timestamp
            now_iso = datetime.now(timezone.utc).isoformat()
            self.sqlite_conn.execute(
                "UPDATE memory_stats SET value = ?, updated_at = ? WHERE key = 'last_backup'",
                (now_iso, now_iso),
            )
            self.sqlite_conn.commit()

            # Clean old backups (keep last 10)
            backups = sorted(
                [d for d in self.backup_folder.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            for old_backup in backups[10:]:
                shutil.rmtree(old_backup)

            self.logger.warning(
                "Backup created successfully: %s (memories: %d)",
                backup_name,
                len(memories),
            )
            return Result(
                success=True,
                data=[
                    {
                        "backup_name": backup_name,
                        "backup_path": str(backup_path),
                        "memories_backed_up": len(memories),
                    }
                ],
            )

        except Exception as e:
            self.logger.error("Backup failed: %s", e)
            return Result(success=False, reason=f"Backup error: {str(e)}")

    def close(self):
        """Clean shutdown of the memory system"""
        try:
            if hasattr(self, "sqlite_conn"):
                try:
                    self.sqlite_conn.commit()
                except Exception:
                    pass
                try:
                    self.sqlite_conn.close()
                except Exception:
                    pass
        except Exception:
            pass

        # ChromaDB client doesn't need explicit close; make GC-friendly
        try:
            if hasattr(self, "chroma_client"):
                self.chroma_client = None
        except Exception:
            pass

        # Optional: free embedding backend reference
        try:
            if hasattr(self, "embedding_backend"):
                self.embedding_backend = None
            # Also clean up old attribute name for backward compatibility
            if hasattr(self, "embedding_model"):
                self.embedding_model = None
        except Exception:
            pass

        try:
            if hasattr(self, "logger"):
                self.logger.info("Memory system closed successfully")
        except Exception:
            pass

    def _debug_vector_index(self, sample: int = 5):
        """
        Rebuilds the ChromaDB vector index from all SQLite memories.

        Clears the existing vector collection and re-embeds all memories
        from the SQLite database in batches to avoid memory issues.

        Args:
            batch_size (int, optional): Number of memories to process per
                batch. Defaults to 128.

        Returns:
            Result: Dictionary with the following keys:
                - success (bool): Whether the rebuild succeeded.
                - reason (str, optional): Error message if rebuild failed.
                - data (list, optional): Contains reindexed status and
                  total count of memories indexed.
        """
        try:
            count = self.chroma_collection.count()
            data = self.chroma_collection.get(
                include=["documents", "metadatas"], limit=sample
            )
            return {
                "count": count,
                "ids": data.get("ids", []),
                "have_documents": bool(data.get("documents")),
                "have_metadatas": bool(data.get("metadatas")),
            }
        except Exception as e:
            return {"error": str(e)}

    def rebuild_vector_index(self, batch_size: int = 128) -> Result:
        """
        Rebuilds the ChromaDB vector index from all SQLite memories.

        Clears the existing vector collection and re-embeds all memories
        from the SQLite database in batches to avoid memory issues.

        Args:
            batch_size (int, optional): Number of memories to process per
                batch. Defaults to 128.

        Returns:
            Result: Dictionary with the following keys:
                - success (bool): Whether the rebuild succeeded.
                - reason (str, optional): Error message if rebuild failed.
                - data (list, optional): Contains reindexed status and
                  total count of memories indexed.
        """
        try:
            # wipe collection to avoid duplicates
            try:
                self.chroma_collection.delete(where={})
            except Exception as e:
                self.logger.warning("Chroma wipe warning: %s", e)

            rows = self.sqlite_conn.execute(
                "SELECT id, title, content, timestamp, importance, memory_type, "
                "tags FROM memories ORDER BY timestamp ASC"
            ).fetchall()

            ids, embs, docs, metas = [], [], [], []
            # Use batch embeddings if available for better performance
            texts = [f"{row['title']}\n{row['content']}" for row in rows]
            embeddings = self.embedding_backend.get_embeddings(texts)

            for i, row in enumerate(rows):
                ids.append(row["id"])
                embs.append(embeddings[i])
                docs.append(texts[i])
                metas.append(
                    {
                        "title": row["title"],
                        "timestamp": row["timestamp"],
                        "importance": row["importance"],
                        "memory_type": row["memory_type"],
                        "tags": row["tags"],
                    }
                )

                if len(ids) >= batch_size:
                    self.chroma_collection.add(
                        ids=ids, embeddings=embs, documents=docs, metadatas=metas
                    )
                    ids, embs, docs, metas = [], [], [], []

            if ids:
                self.chroma_collection.add(
                    ids=ids, embeddings=embs, documents=docs, metadatas=metas
                )

            try:
                if hasattr(self.chroma_client, "persist"):
                    self.chroma_client.persist()
            except Exception as pe:
                self.logger.warning("Chroma persist warning: %s", pe)

            return Result(
                success=True,
                data=[{"reindexed": True, "count": self.chroma_collection.count()}],
            )
        except Exception as e:
            self.logger.error("Reindex failed: %s", e)
            return Result(success=False, reason=str(e))

    def _parse_iso(self, iso_str: str) -> datetime:
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _days_since(self, iso_str: Optional[str]) -> float:
        if not iso_str:
            return 0.0
        try:
            then = self._parse_iso(iso_str)
            return max(
                0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0
            )
        except Exception:
            return 0.0

    def _get_half_life_days(self, memory_type: Optional[str]) -> float:
        return DECAY_HALF_LIFE_DAYS_BY_TYPE.get(
            memory_type or "", DECAY_HALF_LIFE_DAYS_DEFAULT
        )

    def _get_floor(self, memory_type: Optional[str]) -> int:
        return DECAY_MIN_IMPORTANCE_BY_TYPE.get(
            memory_type or "", DECAY_MIN_IMPORTANCE_DEFAULT
        )

    def _should_protect(self, tags_field) -> bool:
        try:
            tags = json.loads(tags_field) if isinstance(tags_field, str) else tags_field
            tags = tags or []
            return any(t in DECAY_PROTECT_TAGS for t in tags)
        except Exception:
            return False

    def _compute_decay_importance(
        self, importance: float, days_idle: float, half_life_days: float
    ) -> float:
        if half_life_days <= 0:
            return importance
        factor = 0.5 ** (days_idle / half_life_days)
        return importance * factor

    def _round_to_half(self, value: float) -> float:
        return round(value * 2.0) / 2.0

    def _maybe_decay(self, row) -> Optional[float]:
        """
        Lazily decays importance for a single row if conditions are met.
        Returns the new importance if updated, else None.
        Safe: never drops below type floor; respects protected tags.
        """
        if not DECAY_ENABLED:
            return None

        try:
            mem_id = row["id"]
            mem_type = row["memory_type"]
            importance = float(row["importance"])
            floor = self._get_floor(mem_type)

            # Skip if protected or already at/below floor
            if self._should_protect(row["tags"]) or importance <= floor:
                self.logger.info(
                    "Decay check: id=%s type=%s skipped (protected or floor reached)",
                    mem_id,
                    mem_type,
                )
                return None

            # Anchor idle time on last_accessed; fallback to timestamp
            last_accessed = row["last_accessed"] if "last_accessed" in row else None
            if not last_accessed:
                last_accessed = row["timestamp"]

            days_idle = self._days_since(last_accessed)
            half_life = self._get_half_life_days(mem_type)

            decayed = self._compute_decay_importance(importance, days_idle, half_life)
            decayed = max(floor, self._round_to_half(decayed))

            # If no meaningful change, just log and bail
            change = importance - decayed
            if change < DECAY_WRITEBACK_STEP:
                self.logger.info(
                    "Decay check: id=%s type=%s old=%s -> new=%s "
                    "(idle=%.1fd, half_life=%sd) [no decay applied]",
                    mem_id,
                    mem_type,
                    importance,
                    decayed,
                    days_idle,
                    half_life,
                )
                return None

            # Rate limit writes
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except Exception:
                meta = {}

            last_decay_at = meta.get("last_decay_at")
            hours_since_last = (
                self._days_since(last_decay_at) * 24 if last_decay_at else 1e9
            )
            if hours_since_last < DECAY_MIN_INTERVAL_HOURS:
                self.logger.info(
                    "Decay check: id=%s type=%s old=%s → would become %s "
                    "(idle=%.1fd), but last decay %.1fh ago [rate‑limited]",
                    mem_id,
                    mem_type,
                    importance,
                    decayed,
                    days_idle,
                    hours_since_last,
                )
                return None

            # Persist decay to DB
            meta["last_decay_at"] = datetime.now(timezone.utc).isoformat()
            self.sqlite_conn.execute(
                "UPDATE memories SET importance = ?, metadata = ? WHERE id = ?",
                (decayed, json.dumps(meta), mem_id),
            )
            self.sqlite_conn.commit()

            self.logger.info(
                "Lazy decay: id=%s type=%s old=%s new=%s idle_days=%.1f half_life=%s",
                mem_id,
                mem_type,
                importance,
                decayed,
                days_idle,
                half_life,
            )
            return decayed

        except Exception as e:
            self.logger.warning(
                "Lazy decay skipped for id=%s: %s",
                row.get("id", "UNKNOWN"),
                e,
            )
            return None

    def _maybe_reinforce(self, row) -> Optional[float]:
        """
        Apply reinforcement bump on access.

        Returns:
            float | None: New importance if updated (writeback occurred),
            otherwise None.
        """
        if not REINFORCEMENT_ENABLED:
            return None

        try:
            mem_id = row["id"]
            mem_type = row["memory_type"]
            importance = float(row["importance"])

            # Load metadata safely
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except Exception:
                meta = {}

            accum = meta.get("reinforcement_accum", 0.0) + REINFORCEMENT_STEP

            # If accumulated boost reaches threshold, persist
            if accum >= REINFORCEMENT_WRITEBACK_STEP:
                new_importance = min(
                    REINFORCEMENT_MAX, self._round_to_half(importance + accum)
                )
                meta["reinforcement_accum"] = 0.0  # reset accumulator

                self.sqlite_conn.execute(
                    "UPDATE memories SET importance = ?, metadata = ? WHERE id = ?",
                    (new_importance, json.dumps(meta), mem_id),
                )
                self.sqlite_conn.commit()

                self.logger.info(
                    "Reinforcement: id=%s type=%s old=%s new=%s (+%s)",
                    mem_id,
                    mem_type,
                    importance,
                    new_importance,
                    accum,
                )
                return new_importance

            # No writeback yet: just save accumulator
            meta["reinforcement_accum"] = accum
            self.sqlite_conn.execute(
                "UPDATE memories SET metadata = ? WHERE id = ?",
                (json.dumps(meta), mem_id),
            )
            self.sqlite_conn.commit()

            self.logger.info(
                "Reinforcement accum: id=%s +%s, total=%.2f (not written yet)",
                mem_id,
                REINFORCEMENT_STEP,
                accum,
            )
            return None

        except Exception as e:
            self.logger.warning(
                "Reinforcement skipped for id=%s: %s", row.get("id", "UNKNOWN"), e
            )
            return None


# Initialize the memory system
memory_system = RobustMemorySystem()

# FastMCP setup
mcp = FastMCP("RobustMemory")


def _jsonify_result(res: Result) -> dict:
    out = {"success": res.success}
    if res.reason is not None:
        out["reason"] = res.reason
    if res.data is not None:
        data = []
        for item in res.data:
            # Ensure we have a plain dict to mutate safely
            obj = dict(item)
            # Normalize timestamp fields (top-level)
            ts = obj.get("timestamp")
            if isinstance(ts, datetime):
                obj["timestamp"] = ts.isoformat()
            # Normalize nested fields you might have added in searches
            # (e.g., 'relevance_score', 'match_type' already JSON-safe)
            data.append(obj)
        out["data"] = data
    return out


@mcp.tool
def remember(
    title: str,
    content: str,
    tags: str = "",
    importance: int = 5,
    memory_type: str = "conversation",
) -> dict:
    """
    Store a new memory (fact, preference, event, or conversation snippet).

    When to use:
    - The user shares something to keep or says “remember this.”
    - New personal details, preferences, events, instructions.

    Args:
    - title (str): Short title for the memory.
    - content (str): Full text to store.
    - tags (str, optional): Comma-separated tags, e.g., "personal, preference".
    - importance (int, optional): 1–10 (default 5). Higher = more important.
    - memory_type (str, optional): e.g., "conversation", "fact", "preference", "event".

    Returns:
        dict: Dictionary with the following keys:
            - success (bool): Whether the operation succeeded.
            - reason (str, optional): Explanation when the operation fails.
            - data (list, optional): List of memory objects. Each object includes:
                - id
                - title
                - content
                - timestamp
                - tags
                - importance
                - memory_type
                - ... (additional fields as needed)

    Example triggers:
    - “My birthday is July 4th.”
    - “Remember that I prefer tea over coffee.”
    - “Please save this: truck camping next weekend.”
    """
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
    res = memory_system.remember(title, content, tag_list, importance, memory_type)
    return _jsonify_result(res)


@mcp.tool
def search_memories(query: str, search_type: str = "semantic", limit: int = 10) -> dict:
    """
    Search memories using natural language queries for general recall.

    When to use:
    - User asks about a specific fact, event, or detail from the past.
    - General "what did you tell me about..." or "when is my..." queries.
    - Default search when no specific category, tags, or dates are mentioned.

    Args:
    - query (str): Natural language search query.
    - search_type (str, optional): "semantic" (default). Other types not fully implemented.
    - limit (int, optional): Max results to return (default 10).

    Returns:
        dict: Dictionary with the following keys:
            - success (bool): Whether the operation succeeded.
            - reason (str, optional): Explanation when the operation fails.
            - data (list, optional): List of match results. Each result includes:
                - id
                - title
                - content
                - timestamp
                - tags
                - relevance_score
                - match_type
                - ... (additional fields as needed)

    Example triggers:
    - "When is my birthday?"
    - "What did I tell you about my favorite color?"
    - "Do you remember what I said about camping?"
    """
    if search_type == "semantic":
        res = memory_system.search_semantic(query, limit)
    else:
        res = memory_system.search_structured(limit=limit)
    return _jsonify_result(res)


@mcp.tool
def search_by_type(memory_type: str, limit: int = 20) -> dict:
    """
    Retrieve memories by category/type for organized recall.

    When to use:
    - User asks for a specific category of memories.
    - Requests like "show me all my preferences" or "list my facts."
    - When they want to see everything in a particular memory type.

    Args:
    - memory_type (str): Category to search for, e.g., "conversation", "fact",
    "preference", "event".
    - limit (int, optional): Max results to return (default 20).

    Returns:
        dict: Dictionary with the following keys:
            - success (bool): Whether the operation succeeded.
            - reason (str, optional): Explanation when the operation fails.
            - data (list, optional): List of memory objects. Each object includes:
                - id
                - title
                - content
                - timestamp
                - tags
                - memory_type
                - ... (additional fields as needed)

    Example triggers:
    - "Show me all my preferences so far."
    - "List the facts you know about me."
    - "What events have we discussed?"
    """
    res = memory_system.search_structured(memory_type=memory_type, limit=limit)
    return _jsonify_result(res)


@mcp.tool
def search_by_tags(tags: str, limit: int = 20) -> dict:
    """
    Find memories associated with specific tags for thematic recall.

    When to use:
    - User mentions specific tags or themes they want to find.
    - Requests like "find everything tagged X" or "show me camping memories."
    - When they want memories grouped by topic/theme rather than type.

    Args:
    - tags (str): Comma-separated tags to search for, e.g., "camping, truck" or "music, guitar".
    - limit (int, optional): Max results to return (default 20).

    Returns:
        dict: Dictionary with the following keys:
            - success (bool): Whether the operation succeeded.
            - reason (str, optional): Explanation when operation fails.
            - data (list, optional): List of memory objects. Each object
              includes id, title, content, timestamp, tags, memory_type,
              and other fields as needed.

    Example triggers:
    - "Find everything tagged camping and truck."
    - "Show me memories about music."
    - "What do you have tagged as personal?"
    """
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    res = memory_system.search_structured(tags=tag_list, limit=limit)
    return _jsonify_result(res)


@mcp.tool
def get_recent_memories(limit: int = 20) -> dict:
    """
    Retrieve the most recently stored memories for timeline-based recall.

    When to use:
    - User asks about recent interactions or conversations.
    - Time-based queries like "today," "last night," "recently," "yesterday."
    - When they want to review what was discussed in the current or recent sessions.
    - Use this instead of date ranges when no specific dates are mentioned.

    Args:
    - limit (int, optional): Max results to return (default 20).

    Returns:
        dict: Dictionary with the following keys:
            - success (bool): Whether the operation succeeded
            - reason (str, optional): Error message if failed
            - data (list, optional): List of memory objects, each with
              id, title, content, timestamp, tags, memory_type, etc.

    Example triggers:
    - "What did we talk about today?"
    - "What have we discussed recently?"
    - "Remind me what we covered last night."
    - "What's been happening lately?"
    """
    res = memory_system.get_recent(limit)
    return _jsonify_result(res)


@mcp.tool
def update_memory(
    memory_id: str,
    title: str = None,
    content: str = None,
    tags: str = None,
    importance: int = None,
    memory_type: str = None,
) -> dict:
    """
    Update or modify an existing memory by its unique ID.

    When to use:
    - User wants to correct, change, or add details to a stored memory.
    - Requests like "update that memory" or "change my favorite color to blue."
    - Use this to change content, tags, importance, or type.

    Args:
    - memory_id (str): Unique ID of the memory to update.
    - title (str, optional): New title.
    - content (str, optional): New content.
    - tags (str, optional): New comma-separated tags.
    - importance (int, optional): New importance 1–10.
    - memory_type (str, optional): New category, e.g., "fact", "preference", "event",
    "conversation".

    Returns:
    - dict: { "success": bool, "reason"?: str, "data"?: [ {id, ...} ] }

    Example triggers:
    - "Change that to type 'preference' and tag it 'personal'."
    - "Update the camping note to type 'event'."
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    res = memory_system.update_memory(
        memory_id=memory_id,
        title=title,
        content=content,
        tags=tag_list,
        importance=importance,
        memory_type=memory_type,
    )
    return _jsonify_result(res)


@mcp.tool
def delete_memory(memory_id: str) -> dict:
    """
    Permanently delete a memory by its unique ID.

    When to use:
    - User explicitly asks you to forget or erase something.
    - Requests like "forget my old phone number" or "delete that memory."
    - Use for permanent removal rather than updating or downgrading importance.

    Args:
    - memory_id (str): Unique ID of the memory to delete.

    Returns:
    - dict: { "success": bool, "reason"?: str }

    Example triggers:
    - "Please forget my old address."
    - "Delete that memory about my ex."
    - "Erase what I told you earlier about my school."
    """
    res = memory_system.delete_memory(memory_id)
    return _jsonify_result(res)


@mcp.tool
def get_memory_stats() -> dict:
    """
    Retrieve statistics and information about the memory system.

    When to use:
    - User asks about memory system capacity, totals, or status.
    - Questions about "how many memories" or system health.
    - When they want to know storage details or usage metrics.

    Args:
    - None

    Returns:
    - dict: {
        "success": bool,
        "reason"?: str,
        "data"?: {
            "total_memories": int,
            "by_type": {...},
            "by_importance": {...},
            "storage_info": {...},
            ...
            }
        }

    Example triggers:
    - "How many memories do you have?"
    - "What's your memory system status?"
    - "Show me your storage stats."
    - "How much have you remembered so far?"
    """
    res = memory_system.get_statistics()
    return _jsonify_result(res)


@mcp.tool
def create_backup() -> dict:
    """
    Create a complete backup of the memory system right now.

    When to use:
    - User explicitly requests a backup or save operation.
    - Before major changes or when they want to preserve current state.
    - Only use when directly asked - automatic backups happen regularly.

    Args:
    - None

    Returns:
        dict: Dictionary with the following keys:
            - success (bool): Whether the operation succeeded.
            - reason (str, optional): Explanation when the operation fails.
            - data (dict, optional): Backup details, including:
                - backup_path (str): Filesystem path to the backup.
                - timestamp (str): ISO 8601 timestamp of when the backup was created.
                - files_backed_up (list): List of file paths included in the backup.
                - ...: Additional fields as needed.

    Example triggers:
    - "Make a backup now."
    - "Save everything to backup."
    - "Create a backup of my memories."
    - "Back up the system."
    """
    res = memory_system.create_backup()
    return _jsonify_result(res)


@mcp.tool
def search_by_date_range(date_from: str, date_to: str = None, limit: int = 50) -> dict:
    """
    Find memories stored within a specific date or date range.

    When to use:
    - User asks about discussions or events during a particular time window.
    - Queries mentioning explicit dates ("on Sept 10th") or ranges ("between Sept 1 and Sept 15").
    - Use this instead of recent-memory search when precise dates are provided.

    Args:
    - date_from (str): Start date/time in ISO format (e.g., "2025-09-01" or "2025-09-01T10:30:00Z").
    - date_to (str, optional): End date/time in ISO format. Defaults to current UTC time if omitted.
    - limit (int, optional): Max results to return (default 50).

    Returns:
        dict: Dictionary with the following keys:
            - success (bool): Whether the operation succeeded.
            - reason (str, optional): Explanation when the operation fails.
            - data (list, optional): List of memory objects. Each object includes:
                - id
                - title
                - content
                - timestamp
                - tags
                - memory_type
                - ... (additional fields as needed)

    Example triggers:
    - "What did we discuss on September 10th?"
    - "Show me everything between September 1 and 15."
    - "What memories are there from last week?"
    - "Pull up our conversations from August."
    """
    if date_to is None:
        date_to = datetime.now(timezone.utc).isoformat()
    res = memory_system.search_structured(
        date_from=date_from, date_to=date_to, limit=limit
    )
    return _jsonify_result(res)


@mcp.tool
def rebuild_vectors() -> dict:
    """
    One-time repair: rebuild vector index from SQLite memories.
    Use if semantic search isn't working but structured search is.
    """
    res = memory_system.rebuild_vector_index()
    return _jsonify_result(res)


# Cleanup on exit


atexit.register(memory_system.close)

if __name__ == "__main__":
    try:
        asyncio.run(mcp.run_stdio_async(show_banner=False))
    except KeyboardInterrupt:
        print("\nShutting down memory system...")
        memory_system.close()
    except Exception as e:
        print(f"Error running MCP server: {e}")
        memory_system.close()
