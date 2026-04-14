"""
Memory Manager - GUI Application
A beautiful, easy-to-use interface for managing AI companion memories

Features:
- Search by all fields (title, content, tags, type, date, importance)
- View, edit, and delete memories
- Backup and restore functionality
- Statistics dashboard
- Export/import capabilities
- Corruption-safe database operations
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from pathlib import Path
import sqlite3
import json
import shutil
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import os
import threading

# chromadb may not be installed in the GUI environment
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    ChromaSettings = None  # type: ignore

# Configuration
DATA_FOLDER = Path(
    os.environ.get(
        "AI_COMPANION_DATA_DIR", str(Path.home() / "Documents" / "ai_companion_memory")
    )
)
DB_PATH = DATA_FOLDER / "memory_db" / "memories.db"
BACKUP_FOLDER = DATA_FOLDER / "memory_backups"
CHROMA_PATH = DATA_FOLDER / "memory_db" / "chroma_db"

# Backend dimension map for detecting dimension changes on backend switch
_BACKEND_DIMENSIONS = {
    "sentence-transformers": 384,  # all-MiniLM-L6-v2 default
    "ollama": 768,  # nomic-embed-text default
    "fallback": 384,
}


class MemoryManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Memory Manager")
        self.root.geometry("1400x900")

        # Set modern theme
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Custom colors
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#4a9eff"
        self.secondary_bg = "#3a3a3a"

        # Configure styles
        self.configure_styles()

        # Database connection
        self.db_conn = None
        self._rebuild_in_progress = False  # blocks window close during vector rebuild
        self.connect_database()

        # Current selection
        self.selected_memory_id = None

        # Load config for settings tab
        self._load_config()

        # Build UI
        self.create_tabs()

        # Load initial data
        self.refresh_memories()
        self.update_statistics()

    def _load_config(self):
        """Load existing config to pre-select values in the Settings tab."""
        try:
            from config_manager import Config

            self.config = Config()
        except Exception:
            # Config not available (MCP not running), use defaults
            class DummyConfig:
                def get_embedding_backend_type(self):
                    return "sentence-transformers"

                def get_model_name(self):
                    return None

                def get_offline(self):
                    return True

                def get_base_url(self):
                    return "http://localhost:11434"

                def get_dimensions(self):
                    return 384

            self.config = DummyConfig()

    def configure_styles(self):
        """Configure custom styles for widgets"""
        self.style.configure(
            "Title.TLabel", font=("Segoe UI", 24, "bold"), foreground=self.accent_color
        )
        self.style.configure(
            "Subtitle.TLabel", font=("Segoe UI", 12), foreground="#cccccc"
        )
        self.style.configure(
            "Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=self.fg_color
        )
        self.style.configure(
            "Normal.TLabel", font=("Segoe UI", 10), foreground=self.fg_color
        )
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def connect_database(self):
        """Connect to the SQLite database with corruption protection"""
        try:
            if not DB_PATH.exists():
                messagebox.showerror(
                    "Error",
                    f"Database not found at:\n{DB_PATH}\n\nPlease ensure the memory system is initialized.",
                )
                self.root.quit()
                return

            self.db_conn = sqlite3.connect(
                str(DB_PATH),
                check_same_thread=False,
                timeout=30.0,
                isolation_level="DEFERRED",
            )
            self.db_conn.row_factory = sqlite3.Row

            # Enable WAL mode for better concurrency
            self.db_conn.execute("PRAGMA journal_mode=WAL")
            self.db_conn.execute("PRAGMA synchronous=FULL")

            # Verify database integrity
            cursor = self.db_conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            if result and result[0] != "ok":
                messagebox.showwarning(
                    "Warning",
                    "Database integrity check failed. Some data may be corrupted.",
                )

        except Exception as e:
            messagebox.showerror(
                "Database Error", f"Failed to connect to database:\n{str(e)}"
            )
            self.root.quit()

    def create_tabs(self):
        """Create tabbed interface: Memories tab + Settings tab."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Memories tab
        self.memories_tab = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(self.memories_tab, text="Memories")
        self._build_memories_tab()

        # Settings tab
        self.settings_tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.settings_tab, text="Settings")
        self._build_settings_tab()

    def _build_memories_tab(self):
        """Build the Memories tab content."""
        parent = self.memories_tab
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        # ===== HEADER =====
        header_frame = ttk.Frame(parent)
        header_frame.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        title_label = ttk.Label(
            header_frame, text="Memory Manager", style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, sticky=tk.W)

        subtitle_label = ttk.Label(
            header_frame,
            text="View and manage AI companion memories",
            style="Subtitle.TLabel",
        )
        subtitle_label.grid(row=1, column=0, sticky=tk.W)

        # ===== LEFT PANEL - Search and List =====
        left_panel = ttk.Frame(parent, padding="5")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(2, weight=1)

        # Search section
        search_frame = ttk.LabelFrame(left_panel, text="Search Memories", padding="10")
        search_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)

        # Search by text
        ttk.Label(search_frame, text="Search:", style="Normal.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *args: self.on_search_changed())
        search_entry = ttk.Entry(
            search_frame, textvariable=self.search_var, font=("Segoe UI", 10)
        )
        search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))

        # Filter by type
        ttk.Label(search_frame, text="Type:", style="Normal.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.type_var = tk.StringVar(value="All")
        type_combo = ttk.Combobox(
            search_frame,
            textvariable=self.type_var,
            state="readonly",
            font=("Segoe UI", 10),
        )
        type_combo["values"] = [
            "All",
            "conversation",
            "fact",
            "preference",
            "event",
            "task",
            "ephemeral",
        ]
        type_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        type_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_memories())

        # Filter by importance
        ttk.Label(search_frame, text="Min Importance:", style="Normal.TLabel").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.importance_var = tk.StringVar(value="1")
        importance_spin = ttk.Spinbox(
            search_frame,
            from_=1,
            to=10,
            textvariable=self.importance_var,
            width=10,
            font=("Segoe UI", 10),
        )
        importance_spin.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        importance_spin.bind("<Return>", lambda e: self.refresh_memories())

        # Filter by tags
        ttk.Label(search_frame, text="Tags:", style="Normal.TLabel").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        self.tags_filter_var = tk.StringVar()
        tags_entry = ttk.Entry(
            search_frame, textvariable=self.tags_filter_var, font=("Segoe UI", 10)
        )
        tags_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        tags_entry.bind("<Return>", lambda e: self.refresh_memories())

        # Search button
        search_btn = ttk.Button(
            search_frame,
            text="Search",
            command=self.refresh_memories,
            style="Accent.TButton",
        )
        search_btn.grid(row=4, column=0, columnspan=2, pady=(10, 0))

        # Statistics section
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.stats_label = ttk.Label(
            stats_frame, text="Loading...", style="Normal.TLabel", justify=tk.LEFT
        )
        self.stats_label.grid(row=0, column=0, sticky=tk.W)

        # Memory list
        list_frame = ttk.LabelFrame(left_panel, text="Memories", padding="5")
        list_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Treeview for memories
        columns = ("ID", "Title", "Type", "Importance", "Date")
        self.tree = ttk.Treeview(
            list_frame, columns=columns, show="tree headings", selectmode="browse"
        )

        # Configure columns
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("ID", width=0, stretch=tk.NO)
        self.tree.column("Title", width=300)
        self.tree.column("Type", width=100)
        self.tree.column("Importance", width=80, anchor=tk.CENTER)
        self.tree.column("Date", width=150)

        # Configure headings
        self.tree.heading("Title", text="Title")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Importance", text="Importance")
        self.tree.heading("Date", text="Date")

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self.on_memory_selected)

        # ===== RIGHT PANEL - Details and Actions =====
        right_panel = ttk.Frame(parent, padding="5")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)

        # Action buttons
        action_frame = ttk.Frame(right_panel)
        action_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(action_frame, text="New Memory", command=self.new_memory).grid(
            row=0, column=0, padx=2
        )
        ttk.Button(action_frame, text="Save Changes", command=self.save_memory).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(action_frame, text="Delete", command=self.delete_memory).grid(
            row=0, column=2, padx=2
        )
        ttk.Button(action_frame, text="Refresh", command=self.refresh_memories).grid(
            row=0, column=3, padx=2
        )
        ttk.Button(action_frame, text="Backup", command=self.create_backup).grid(
            row=0, column=4, padx=2
        )
        ttk.Button(action_frame, text="Export", command=self.export_memories).grid(
            row=0, column=5, padx=2
        )

        # Details section
        details_frame = ttk.LabelFrame(right_panel, text="Memory Details", padding="10")
        details_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_frame.columnconfigure(1, weight=1)
        details_frame.rowconfigure(5, weight=1)

        # ID (hidden, for reference)
        self.id_var = tk.StringVar()

        # Title
        ttk.Label(details_frame, text="Title:", style="Header.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.title_var = tk.StringVar()
        title_entry = ttk.Entry(
            details_frame, textvariable=self.title_var, font=("Segoe UI", 11)
        )
        title_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))

        # Type
        ttk.Label(details_frame, text="Type:", style="Header.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.detail_type_var = tk.StringVar()
        type_detail_combo = ttk.Combobox(
            details_frame,
            textvariable=self.detail_type_var,
            state="readonly",
            font=("Segoe UI", 10),
        )
        type_detail_combo["values"] = [
            "conversation",
            "fact",
            "preference",
            "event",
            "task",
            "ephemeral",
        ]
        type_detail_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 0))

        # Importance
        ttk.Label(details_frame, text="Importance:", style="Header.TLabel").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.detail_importance_var = tk.StringVar()
        importance_detail_spin = ttk.Spinbox(
            details_frame,
            from_=1,
            to=10,
            textvariable=self.detail_importance_var,
            width=10,
            font=("Segoe UI", 10),
        )
        importance_detail_spin.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(10, 0))

        # Tags
        ttk.Label(details_frame, text="Tags:", style="Header.TLabel").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.tags_var = tk.StringVar()
        tags_detail_entry = ttk.Entry(
            details_frame, textvariable=self.tags_var, font=("Segoe UI", 10)
        )
        tags_detail_entry.grid(
            row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0)
        )
        ttk.Label(
            details_frame,
            text="(comma-separated)",
            font=("Segoe UI", 8),
            foreground="#888888",
        ).grid(row=4, column=1, sticky=tk.W, padx=(10, 0))

        # Content
        ttk.Label(details_frame, text="Content:", style="Header.TLabel").grid(
            row=5, column=0, sticky=(tk.W, tk.N), pady=5
        )
        self.content_text = scrolledtext.ScrolledText(
            details_frame, wrap=tk.WORD, font=("Segoe UI", 10), height=15
        )
        self.content_text.grid(
            row=5, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(10, 0)
        )

        # Metadata display
        metadata_frame = ttk.LabelFrame(right_panel, text="Metadata", padding="10")
        metadata_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        metadata_frame.columnconfigure(0, weight=1)

        self.metadata_label = ttk.Label(
            metadata_frame,
            text="Select a memory to view details",
            style="Normal.TLabel",
            justify=tk.LEFT,
        )
        self.metadata_label.grid(row=0, column=0, sticky=tk.W)

    def on_search_changed(self):
        """Handle search text changes with debouncing"""
        # Cancel previous scheduled search
        if hasattr(self, "_search_after_id"):
            self.root.after_cancel(self._search_after_id)

        # Schedule new search after 500ms
        self._search_after_id = self.root.after(500, self.refresh_memories)

    def refresh_memories(self):
        """Refresh the memory list based on current filters"""
        try:
            # Clear current items
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Build query
            conditions = []
            params = []

            # Search text (searches in title and content)
            search_text = self.search_var.get().strip()
            if search_text:
                conditions.append("(title LIKE ? OR content LIKE ?)")
                params.extend([f"%{search_text}%", f"%{search_text}%"])

            # Type filter
            if self.type_var.get() != "All":
                conditions.append("memory_type = ?")
                params.append(self.type_var.get())

            # Importance filter
            try:
                min_importance = int(self.importance_var.get())
                conditions.append("importance >= ?")
                params.append(min_importance)
            except:
                pass

            # Tags filter
            tags_filter = self.tags_filter_var.get().strip()
            if tags_filter:
                tag_list = [t.strip() for t in tags_filter.split(",") if t.strip()]
                if tag_list:
                    tag_conditions = []
                    for tag in tag_list:
                        tag_conditions.append("tags LIKE ?")
                        params.append(f'%"{tag}"%')
                    conditions.append(f"({' OR '.join(tag_conditions)})")

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            query = f"""
                SELECT id, title, memory_type, importance, timestamp, tags, content, metadata, created_at, updated_at, last_accessed
                FROM memories
                WHERE {where_clause}
                ORDER BY importance DESC, timestamp DESC
                LIMIT 1000
            """

            cursor = self.db_conn.execute(query, params)
            rows = cursor.fetchall()

            # Populate tree
            for row in rows:
                # Format date
                try:
                    dt = datetime.fromisoformat(row["timestamp"])
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    date_str = row["timestamp"][:16] if row["timestamp"] else "Unknown"

                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        row["id"],
                        row["title"][:50] + ("..." if len(row["title"]) > 50 else ""),
                        row["memory_type"],
                        row["importance"],
                        date_str,
                    ),
                )

            # Update status
            self.update_statistics()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh memories:\n{str(e)}")

    def on_memory_selected(self, event):
        """Handle memory selection in the tree"""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        memory_id = item["values"][0]

        try:
            cursor = self.db_conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()

            if row:
                self.selected_memory_id = row["id"]
                self.id_var.set(row["id"])
                self.title_var.set(row["title"])
                self.detail_type_var.set(row["memory_type"])
                self.detail_importance_var.set(row["importance"])

                # Parse and display tags
                try:
                    tags = json.loads(row["tags"])
                    self.tags_var.set(", ".join(tags))
                except:
                    self.tags_var.set("")

                # Display content
                self.content_text.delete("1.0", tk.END)
                self.content_text.insert("1.0", row["content"])

                # Display metadata
                try:
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                except:
                    metadata = {}

                metadata_text = f"ID: {row['id']}\n"
                metadata_text += f"Created: {row['created_at']}\n"
                metadata_text += f"Updated: {row['updated_at']}\n"
                metadata_text += f"Last Accessed: {row['last_accessed']}\n"

                if metadata:
                    metadata_text += f"\nCustom Metadata:\n"
                    for key, value in metadata.items():
                        metadata_text += f"  {key}: {value}\n"

                self.metadata_label.config(text=metadata_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load memory details:\n{str(e)}")

    def new_memory(self):
        """Create a new memory"""
        self.selected_memory_id = None
        self.id_var.set("")
        self.title_var.set("")
        self.detail_type_var.set("conversation")
        self.detail_importance_var.set("5")
        self.tags_var.set("")
        self.content_text.delete("1.0", tk.END)
        self.metadata_label.config(text="New memory - fill in details and click Save")

    def save_memory(self):
        """Save the current memory (create or update)"""
        try:
            title = self.title_var.get().strip()
            content = self.content_text.get("1.0", tk.END).strip()

            if not title or not content:
                messagebox.showwarning(
                    "Validation Error", "Title and content are required."
                )
                return

            memory_type = self.detail_type_var.get()
            importance = int(self.detail_importance_var.get())

            # Parse tags
            tags_str = self.tags_var.get().strip()
            tags = (
                [t.strip() for t in tags_str.split(",") if t.strip()]
                if tags_str
                else []
            )

            now_iso = datetime.now(timezone.utc).isoformat()

            if self.selected_memory_id:
                # Update existing memory
                self.db_conn.execute(
                    """
                    UPDATE memories
                    SET title = ?, content = ?, memory_type = ?, importance = ?, tags = ?, updated_at = ?
                    WHERE id = ?
                """,
                    (
                        title,
                        content,
                        memory_type,
                        importance,
                        json.dumps(tags),
                        now_iso,
                        self.selected_memory_id,
                    ),
                )

                messagebox.showinfo("Success", "Memory updated successfully!")
            else:
                # Create new memory
                import hashlib

                content_hash = hashlib.sha256(content.encode()).hexdigest()
                time_hash = hashlib.sha256(now_iso.encode()).hexdigest()[:8]
                memory_id = f"mem_{time_hash}_{content_hash[:16]}"

                self.db_conn.execute(
                    """
                    INSERT INTO memories (id, title, content, timestamp, tags, importance, memory_type, metadata, content_hash, created_at, updated_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        memory_id,
                        title,
                        content,
                        now_iso,
                        json.dumps(tags),
                        importance,
                        memory_type,
                        "{}",
                        content_hash,
                        now_iso,
                        now_iso,
                        now_iso,
                    ),
                )

                self.selected_memory_id = memory_id
                messagebox.showinfo(
                    "Success",
                    "Memory created successfully!\n\nNote: Vector embeddings will be updated when the MCP server is restarted.",
                )

            self.db_conn.commit()
            self.refresh_memories()

        except Exception as e:
            self.db_conn.rollback()
            messagebox.showerror("Error", f"Failed to save memory:\n{str(e)}")

    def delete_memory(self):
        """Delete the selected memory"""
        if not self.selected_memory_id:
            messagebox.showwarning("No Selection", "Please select a memory to delete.")
            return

        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete this memory?\n\nTitle: {self.title_var.get()}\n\nThis action cannot be undone.",
        )

        if result:
            try:
                self.db_conn.execute(
                    "DELETE FROM memories WHERE id = ?", (self.selected_memory_id,)
                )
                self.db_conn.commit()

                messagebox.showinfo(
                    "Success",
                    "Memory deleted successfully!\n\nNote: Vector embeddings will be cleaned up when the MCP server is restarted.",
                )

                self.new_memory()
                self.refresh_memories()

            except Exception as e:
                self.db_conn.rollback()
                messagebox.showerror("Error", f"Failed to delete memory:\n{str(e)}")

    def create_backup(self):
        """Create a backup of the database"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"memory_backup_{timestamp}"
            backup_path = BACKUP_FOLDER / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup SQLite database
            sqlite_backup = backup_path / "memories.db"
            shutil.copy2(DB_PATH, sqlite_backup)

            # Export to JSON
            cursor = self.db_conn.execute("SELECT * FROM memories ORDER BY timestamp")
            memories = []
            for row in cursor.fetchall():
                memory_dict = dict(row)
                memory_dict["tags"] = json.loads(memory_dict["tags"])
                memory_dict["metadata"] = (
                    json.loads(memory_dict["metadata"])
                    if memory_dict["metadata"]
                    else {}
                )
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

            messagebox.showinfo(
                "Success",
                f"Backup created successfully!\n\nLocation: {backup_path}\n\nFiles:\n- memories.db\n- memories_export.json",
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create backup:\n{str(e)}")

    def export_memories(self):
        """Export memories to a JSON file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"memories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            if not file_path:
                return

            cursor = self.db_conn.execute("SELECT * FROM memories ORDER BY timestamp")
            memories = []
            for row in cursor.fetchall():
                memory_dict = dict(row)
                memory_dict["tags"] = json.loads(memory_dict["tags"])
                memory_dict["metadata"] = (
                    json.loads(memory_dict["metadata"])
                    if memory_dict["metadata"]
                    else {}
                )
                memories.append(memory_dict)

            with open(file_path, "w", encoding="utf-8") as f:
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

            messagebox.showinfo(
                "Success", f"Exported {len(memories)} memories to:\n{file_path}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export memories:\n{str(e)}")

    def update_statistics(self):
        """Update the statistics display"""
        try:
            cursor = self.db_conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT memory_type) as types,
                    AVG(importance) as avg_importance
                FROM memories
            """)
            stats = cursor.fetchone()

            cursor = self.db_conn.execute("""
                SELECT memory_type, COUNT(*) as count
                FROM memories
                GROUP BY memory_type
                ORDER BY count DESC
            """)
            type_breakdown = cursor.fetchall()

            stats_text = f"Total Memories: {stats['total']}\n"
            stats_text += f"Memory Types: {stats['types']}\n"
            stats_text += f"Avg Importance: {stats['avg_importance']:.1f}\n\n"
            stats_text += "Breakdown:\n"
            for row in type_breakdown:
                stats_text += f"  {row['memory_type']}: {row['count']}\n"

            self.stats_label.config(text=stats_text)

        except Exception as e:
            self.stats_label.config(text=f"Error loading stats:\n{str(e)}")

    def _build_settings_tab(self):
        """Build the Settings tab with embedding backend configuration."""
        parent = self.settings_tab
        parent.columnconfigure(0, weight=1)

        # Current Status section
        status_frame = ttk.LabelFrame(
            parent, text="Current Loaded Backend", padding="15"
        )
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        status_frame.columnconfigure(1, weight=1)

        self.status_indicator = tk.Label(
            status_frame, text="Loading...", font=("Segoe UI", 11), anchor="w"
        )
        self.status_indicator.grid(row=0, column=0, sticky="w", pady=5)

        ttk.Button(
            status_frame, text="Refresh Status", command=self._refresh_status
        ).grid(row=0, column=1, sticky="e", padx=(10, 0))

        self.status_detail = tk.Label(
            status_frame,
            text="",
            font=("Segoe UI", 9),
            foreground="#888888",
            anchor="w",
        )
        self.status_detail.grid(row=1, column=0, columnspan=2, sticky="w")

        # Load initial status
        self._refresh_status()

        # Embedding Configuration section
        embed_frame = ttk.LabelFrame(
            parent, text="Embedding Configuration", padding="15"
        )
        embed_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        embed_frame.columnconfigure(1, weight=1)

        # Backend type selector
        ttk.Label(embed_frame, text="Backend Type:", style="Header.TLabel").grid(
            row=0, column=0, sticky="w", pady=8
        )
        self.settings_backend_var = tk.StringVar(
            value=self.config.get_embedding_backend_type()
        )
        backend_combo = ttk.Combobox(
            embed_frame,
            textvariable=self.settings_backend_var,
            values=["sentence-transformers", "ollama", "fallback"],
            state="readonly",
            font=("Segoe UI", 10),
        )
        backend_combo.grid(row=0, column=1, sticky="ew", pady=8, padx=(10, 0))
        backend_combo.bind("<<ComboboxSelected>>", self._on_backend_changed)

        # Separator
        ttk.Separator(embed_frame, orient="horizontal").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=10
        )

        # Backend-specific panels (stacked vertically, one shown at a time)
        self.backend_panels = {}

        # --- Sentence Transformers panel ---
        st_panel = ttk.Frame(embed_frame)

        ttk.Label(st_panel, text="Model:", style="Normal.TLabel").grid(
            row=0, column=0, sticky="w", pady=5
        )
        st_model_value = self.config.get_model_name() or "all-MiniLM-L6-v2"
        self.st_model_var = tk.StringVar(value=st_model_value)
        st_model_combo = ttk.Combobox(
            st_panel, textvariable=self.st_model_var, font=("Segoe UI", 10), width=40
        )
        st_model_combo.grid(row=0, column=1, sticky="w", pady=5, padx=(10, 0))

        self.st_offline_var = tk.BooleanVar(value=self.config.get_offline())
        ttk.Checkbutton(
            st_panel,
            text="Offline mode (use only locally cached models)",
            variable=self.st_offline_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=5)

        ttk.Button(
            st_panel,
            text="Refresh Local Models",
            command=lambda: self._refresh_st_models(st_model_combo),
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))

        self.backend_panels["sentence-transformers"] = st_panel

        # --- Ollama / LM Studio panel ---
        ollama_panel = ttk.Frame(embed_frame)

        ttk.Label(ollama_panel, text="Base URL:", style="Normal.TLabel").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.ollama_url_var = tk.StringVar(value=self.config.get_base_url())
        ttk.Entry(
            ollama_panel,
            textvariable=self.ollama_url_var,
            font=("Segoe UI", 10),
            width=40,
        ).grid(row=0, column=1, sticky="w", pady=5, padx=(10, 0))

        ttk.Label(ollama_panel, text="Model:", style="Normal.TLabel").grid(
            row=1, column=0, sticky="w", pady=5
        )
        ollama_model_value = self.config.get_model_name() or "nomic-embed-text:latest"
        self.ollama_model_var = tk.StringVar(value=ollama_model_value)
        self.ollama_model_combo = ttk.Combobox(
            ollama_panel,
            textvariable=self.ollama_model_var,
            font=("Segoe UI", 10),
            width=40,
        )
        self.ollama_model_combo.grid(row=1, column=1, sticky="w", pady=5, padx=(10, 0))

        self.ollama_status_label = ttk.Label(
            ollama_panel, text="", foreground="#888888", font=("Segoe UI", 9)
        )
        self.ollama_status_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=5)

        ttk.Button(
            ollama_panel, text="Refresh Model List", command=self._refresh_ollama_models
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

        self.backend_panels["ollama"] = ollama_panel

        # --- Fallback panel ---
        fallback_panel = ttk.Frame(embed_frame)

        ttk.Label(fallback_panel, text="Dimensions:", style="Normal.TLabel").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.fallback_dim_var = tk.IntVar(value=self.config.get_dimensions())
        ttk.Spinbox(
            fallback_panel,
            from_=64,
            to=4096,
            increment=64,
            textvariable=self.fallback_dim_var,
            width=10,
            font=("Segoe UI", 10),
        ).grid(row=0, column=1, sticky="w", pady=5, padx=(10, 0))

        ttk.Label(
            fallback_panel,
            text="Note: Fallback uses a simple random-projection backend. Not suitable for production.",
            style="Normal.TLabel",
            foreground="#ff8800",
            font=("Segoe UI", 8),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

        self.backend_panels["fallback"] = fallback_panel

        # Place all panels in the embed_frame (row=2), hidden by default
        for panel in self.backend_panels.values():
            panel.grid(row=2, column=0, columnspan=2, sticky="w")
            panel.grid_remove()

        # Show the correct panel for the current backend
        self._on_backend_changed()

        # Action buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=2, column=0, sticky="w", pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="Save Settings",
            command=self._save_settings,
            style="Accent.TButton",
        ).pack(side="left", padx=5)
        ttk.Button(
            btn_frame, text="Reset to Defaults", command=self._reset_settings
        ).pack(side="left", padx=5)

        ttk.Label(
            btn_frame,
            text="Note: Changes require MCP server restart to take effect.",
            foreground="#ff8800",
            font=("Segoe UI", 9),
        ).pack(side="left", padx=20)

    def _on_backend_changed(self, *args):
        """Show/hide the correct backend-specific panel and reset backend-appropriate defaults."""
        selected = self.settings_backend_var.get()
        for backend, panel in self.backend_panels.items():
            if backend == selected:
                panel.grid()
            else:
                panel.grid_remove()

        # Reset model to backend-appropriate default when switching
        if selected == "sentence-transformers":
            self.st_model_var.set("all-MiniLM-L6-v2")
        elif selected == "ollama":
            self.ollama_model_var.set("nomic-embed-text:latest")
        elif selected == "fallback":
            self.fallback_dim_var.set(384)

    def _refresh_status(self):
        """Read the status.json written by the MCP server to show what's actually loaded."""
        try:
            status_path = (
                Path.home()
                / ".lmstudio/extensions/plugins/installed/long-term-memory-mcp"
                / "status.json"
            )
            if status_path.exists():
                with open(status_path) as f:
                    status = json.load(f)

                loaded = status.get("loaded_backend", "unknown")
                dims = status.get("loaded_dimensions", "?")
                cfg_backend = status.get("config_backend", "?")
                cfg_model = status.get("config_model", "?")
                ts = status.get("timestamp", "")

                self.status_indicator.config(
                    text=f"✓ {loaded} ({dims} dims)", foreground="green"
                )
                self.status_detail.config(
                    text=f"Config: {cfg_backend} / {cfg_model} | Loaded: {ts}"
                )
            else:
                self.status_indicator.config(
                    text="✗ No status file — MCP server may not be running",
                    foreground="red",
                )
                self.status_detail.config(
                    text="Start the MCP server to see loaded backend status"
                )
        except Exception as e:
            self.status_indicator.config(
                text=f"✗ Error reading status: {e}", foreground="red"
            )
            self.status_detail.config(text="")

    def _refresh_ollama_models(self):
        """Refresh Ollama model list in a background thread."""
        self.ollama_status_label.config(text="Refreshing...", foreground="#888888")
        base_url = self.ollama_url_var.get().strip()

        def do_discovery():
            try:
                from embedding_backends import OllamaDiscovery

                models = OllamaDiscovery.list_models(base_url)
                model_names = [m.name for m in models] if models else []
                self.root.after(
                    0, lambda: self._update_ollama_models(model_names, base_url)
                )
            except Exception as e:
                self.root.after(
                    0, lambda: self._update_ollama_models([], base_url, error=str(e))
                )

        thread = threading.Thread(target=do_discovery, daemon=True)
        thread.start()

    def _update_ollama_models(
        self, model_names: list, base_url: str, error: str = None
    ):
        """Called on main thread after model discovery completes."""
        if error or not model_names:
            self.ollama_model_combo["values"] = []
            if error:
                self.ollama_status_label.config(
                    text=f"Error: {error}", foreground="red"
                )
            else:
                self.ollama_status_label.config(
                    text=f"Could not reach {base_url} — check server is running",
                    foreground="red",
                )
        else:
            self.ollama_model_combo["values"] = model_names
            self.ollama_status_label.config(
                text=f"Connected to {base_url} — {len(model_names)} models found",
                foreground="green",
            )

    def _refresh_st_models(self, combobox):
        """Refresh Sentence Transformers model list."""
        try:
            from embedding_backends import SentenceTransformersDiscovery

            models = SentenceTransformersDiscovery.list_local_models()
            combobox["values"] = [f"{name} ({dims}d)" for name, dims in models]
            if models:
                combobox.current(0)
        except Exception as e:
            messagebox.showwarning("Discovery Failed", f"Could not list models: {e}")

    def _rebuild_vectors(self, backend: str, backend_cfg: Dict[str, Any]):
        """
        Rebuild the ChromaDB vector index with a new embedding backend.

        Uses the same logic as the MCP server's rebuild_vector_index method.
        """
        self._rebuild_in_progress = True
        self._rebuild_status = "running"
        self._rebuild_error = None
        self._pending_config = {"backend": backend, "backend_cfg": backend_cfg}
        self._rebuild_overlay = None

        # Launch rebuild in background thread
        def _run_rebuild():
            try:
                print(
                    f"[REBUILD] Starting rebuild with backend={backend}, cfg={backend_cfg}"
                )
                from embedding_backends import create_embedding_backend
                import os

                if backend == "sentence-transformers" and backend_cfg.get(
                    "offline", True
                ):
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    os.environ["HF_HUB_OFFLINE"] = "1"

                emb_backend = create_embedding_backend(
                    backend_type=backend,
                    model_name=backend_cfg.get("model"),
                    base_url=backend_cfg.get("base_url", "http://localhost:11434"),
                    offline=backend_cfg.get("offline", True),
                    dimensions=backend_cfg.get("dimensions", 384),
                )
                print(f"[REBUILD] Created embedding backend: {emb_backend}")

                model_name = emb_backend.get_model_name()
                dims = emb_backend.get_dimensions()
                print(f"[REBUILD] Model: {model_name}, dimensions: {dims}")

                # Read memories
                print(f"[REBUILD] Reading memories from {DB_PATH}")
                conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT id, title, content FROM memories ORDER BY timestamp ASC"
                ).fetchall()
                total = len(rows)
                print(f"[REBUILD] Found {total} memories to re-index")
                conn.close()

                if total == 0:
                    self._rebuild_status = "success"
                    return

                # Wipe and recreate ChromaDB collection (needed when dimensions change)
                print(f"[REBUILD] Wiping and recreating ChromaDB at {CHROMA_PATH}")
                import chromadb as chroma_lib

                chroma_client = chroma_lib.PersistentClient(
                    path=str(CHROMA_PATH),
                    settings=chroma_lib.config.Settings(
                        anonymized_telemetry=False, allow_reset=True
                    ),
                )
                # Delete entire collection to handle dimension change
                try:
                    chroma_client.delete_collection(name="ai_companion_memories")
                    print("[REBUILD] Deleted old collection")
                except Exception as e:
                    print(f"[REBUILD] Collection deletion warning: {e}")
                # Create fresh collection with new dimensions
                collection = chroma_client.create_collection(
                    name="ai_companion_memories",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None,
                )
                print(f"[REBUILD] Created new collection for {dims}d vectors")

                # Re-embed and store
                batch_size = 64
                for i in range(0, total, batch_size):
                    batch = rows[i : i + batch_size]
                    texts = [f"{r['title']}\n{r['content']}" for r in batch]
                    print(
                        f"[REBUILD] Embedding batch {i // batch_size + 1} ({len(texts)} texts)..."
                    )
                    embeddings = emb_backend.get_embeddings(texts)
                    print(
                        f"[REBUILD] Got {len(embeddings)} embeddings, shape check: {len(embeddings[0]) if embeddings else 'none'}"
                    )

                    print(
                        f"[REBUILD] Adding batch {i // batch_size + 1} to collection..."
                    )
                    collection.add(
                        ids=[r["id"] for r in batch],
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=[{"title": r["title"]} for r in batch],
                    )
                    print(f"[REBUILD] Batch {i // batch_size + 1} added successfully")

                print(
                    f"[REBUILD] SUCCESS: {total} memories re-indexed with {dims}d vectors"
                )
                # Update SQLite system_config with new dimensions (MCP server reads this on startup)
                try:
                    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                    conn.execute(
                        "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
                        ("embedding_dimensions", str(dims)),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
                        ("embedding_model", model_name),
                    )
                    conn.commit()
                    conn.close()
                    print(
                        f"[REBUILD] Updated SQLite system_config: dimensions={dims}, model={model_name}"
                    )
                except Exception as sqle:
                    print(
                        f"[REBUILD] Warning: Failed to update SQLite system_config: {sqle}"
                    )
                # Update pending config with actual dimensions
                self._pending_config["backend_cfg"]["dimensions"] = dims
                print(f"[REBUILD] Updated config dimensions to {dims}")
                self._rebuild_status = "success"

            except Exception as e:
                import traceback

                err = f"{e}\n{traceback.format_exc()}"
                print(f"[REBUILD] ERROR: {err}")
                self._rebuild_status = "error"
                self._rebuild_error = str(e)

        # Start rebuild thread
        rebuild_thread = threading.Thread(target=_run_rebuild, daemon=True)
        rebuild_thread.start()

        # Create overlay AFTER thread starts
        self._rebuild_overlay = tk.Toplevel(self.root)
        self._rebuild_overlay.title("Rebuilding Vectors")
        self._rebuild_overlay.geometry("500x180")
        self._rebuild_overlay.resizable(False, False)
        self._rebuild_overlay.grab_set()
        self._rebuild_overlay.transient(self.root)
        self._rebuild_overlay.geometry(
            f"+{self.root.winfo_x() + 300}+{self.root.winfo_y() + 300}"
        )

        tk.Label(
            self._rebuild_overlay,
            text="Rebuilding vector index...",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=20)

        self._rebuild_status_var = tk.StringVar(value="Initializing rebuild...")
        tk.Label(
            self._rebuild_overlay,
            textvariable=self._rebuild_status_var,
            font=("Segoe UI", 10),
            foreground="#888888",
        ).pack(pady=5)

        progress = ttk.Progressbar(
            self._rebuild_overlay, mode="indeterminate", length=400
        )
        progress.pack(pady=15)
        progress.start()

        # Poll for completion using main thread's event loop
        def _check_rebuild():
            status = self._rebuild_status
            print(f"[REBUILD] Poll check: status={status}")
            if status == "running":
                self._rebuild_status_var.set("Running... (check console for progress)")
                self.root.after(500, _check_rebuild)
            elif status == "success":
                progress.stop()
                self._rebuild_overlay.destroy()
                self._rebuild_overlay = None
                self._rebuild_in_progress = False
                # Save config
                be = self._pending_config.get("backend")
                cfg = self._pending_config.get("backend_cfg")
                print(f"[REBUILD] Saving config: backend={be}, cfg={cfg}")
                self._save_settings_complete(be, cfg)
            elif status == "error":
                progress.stop()
                err = self._rebuild_error or "Unknown error"
                self._rebuild_status_var.set(f"Error: {err}")
                tk.Button(
                    self._rebuild_overlay,
                    text="Close",
                    command=lambda: [
                        self._rebuild_overlay.destroy(),
                        self._rebuild_on_close(),
                    ],
                ).pack(pady=10)

        # Start polling
        self.root.after(500, _check_rebuild)

    def _rebuild_on_close(self):
        """Called when user closes the failed-init overlay without completing rebuild."""
        self._rebuild_in_progress = False

    def _save_settings(self):
        """Entry point for Save Settings button.

        Detects whether a backend switch (and thus dimension change) is in progress.
        If so, triggers an inline rebuild before saving.
        """
        backend = self.settings_backend_var.get()

        # Determine model name based on backend
        if backend == "sentence-transformers":
            model_raw = self.st_model_var.get()
            model = model_raw.split(" (")[0]
        elif backend == "ollama":
            model = self.ollama_model_var.get()
        else:
            model = None

        # Guard: if the model name looks like it belongs to a different backend,
        # reset to None so the backend's default is used instead.
        if backend == "sentence-transformers" and any(p in model for p in (":", "/")):
            model = None
        elif backend == "ollama" and model and not any(p in model for p in (":", "/")):
            model = None

        offline = (
            self.st_offline_var.get() if backend == "sentence-transformers" else True
        )
        base_url = (
            self.ollama_url_var.get().strip()
            if backend == "ollama"
            else "http://localhost:11434"
        )

        self._pending_config = None  # clear any stale state

        # Detect if this is a backend switch that would change embedding dimensions.
        # Compare current stored dimensions (from status.json or config) with the
        # target backend's default dimensions. We use status.json if available so we
        # know what's actually loaded, not just what's in config.
        needs_rebuild = False
        # Get current backend from config to detect if it's changing
        try:
            current_backend = self.config.get_embedding_backend_type()
            print(f"[SAVE] current_backend from config: {current_backend}")
        except Exception as e:
            print(f"[SAVE] Error getting current backend: {e}")
            current_backend = ""

        # Always rebuild when switching backends - status.json isn't updated until MCP restarts
        needs_rebuild = current_backend != "" and backend != current_backend
        print(
            f"[SAVE] current_backend={current_backend}, target_backend={backend}, needs_rebuild={needs_rebuild}"
        )

        # Also check dimensions for fallback backend (uses custom dimensions)
        if not needs_rebuild and backend == "fallback":
            try:
                status_path = (
                    Path.home()
                    / ".lmstudio/extensions/plugins/installed/long-term-memory-mcp"
                    / "status.json"
                )
                if status_path.exists():
                    with open(status_path) as f:
                        status = json.load(f)
                    current_dims = status.get("loaded_dimensions")
                    target_dims = self.fallback_dim_var.get()
                    needs_rebuild = current_dims != target_dims
                    print(
                        f"[SAVE] fallback: current_dims={current_dims}, target_dims={target_dims}, needs_rebuild={needs_rebuild}"
                    )
            except Exception as e:
                print(f"[SAVE] Error checking dimensions: {e}")

        backend_cfg = {
            "model": model,
            "offline": offline,
            "base_url": base_url,
            "dimensions": self.fallback_dim_var.get() if backend == "fallback" else 384,
        }

        if needs_rebuild:
            # Trigger inline rebuild; _rebuild_finish will call _save_settings_complete on success
            self._pending_config = {"backend": backend, "backend_cfg": backend_cfg}
            self._rebuild_vectors(backend, backend_cfg)
        else:
            # No dimension change — save directly
            self._save_settings_complete(backend, backend_cfg)

    def _save_settings_complete(
        self, backend: str = None, backend_cfg: Dict[str, Any] = None
    ):
        """Actually write config.json. Called directly or after rebuild via _rebuild_finish."""
        if backend is None:
            backend = self._pending_config["backend"]
            backend_cfg = self._pending_config["backend_cfg"]

        config_data = {
            "embedding": {
                "backend": backend,
                "model": backend_cfg["model"],
                "offline": backend_cfg["offline"],
                "base_url": backend_cfg["base_url"],
            },
            "fallback_dimensions": backend_cfg.get("dimensions", 384),
        }

        try:
            from config_manager import Config

            cfg = Config()
            cfg.save(config_data)
            # Update in-memory config so subsequent reads see the new values
            # Note: we must update _config directly because Config() constructor
            # applies env var overrides which may differ from what's saved to file
            cfg._config = config_data
            self.config = cfg
            print(
                f"[SAVE] Config saved and reloaded: backend={cfg.get_embedding_backend_type()}"
            )
            messagebox.showinfo(
                "Settings Saved",
                "Configuration saved successfully.\n\n"
                "Please restart the MCP server for changes to take effect.",
            )
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save config: {e}")

    def _reset_settings(self):
        """Reset settings to defaults."""
        self.settings_backend_var.set("sentence-transformers")
        self.st_model_var.set("all-MiniLM-L6-v2")
        self.st_offline_var.set(True)
        self.ollama_url_var.set("http://localhost:11434")
        self.ollama_model_var.set("nomic-embed-text:latest")
        self.fallback_dim_var.set(384)
        self._on_backend_changed()

    def on_closing(self):
        """Handle window closing — blocked while vector rebuild is in progress."""
        if self._rebuild_in_progress:
            messagebox.showwarning(
                "Cannot Close",
                "Vector rebuild is still in progress.\n"
                "Please wait for it to complete before closing.",
            )
            return
        if self.db_conn:
            try:
                self.db_conn.commit()
                self.db_conn.close()
            except Exception:
                pass
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = MemoryManagerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
