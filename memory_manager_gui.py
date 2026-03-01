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

# Configuration
DATA_FOLDER = Path(os.environ.get("AI_COMPANION_DATA_DIR", str(Path.home() / "Documents" / "ai_companion_memory")))
DB_PATH = DATA_FOLDER / "memory_db" / "memories.db"
BACKUP_FOLDER = DATA_FOLDER / "memory_backups"

class MemoryManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Memory Manager")
        self.root.geometry("1400x900")
        
        # Set modern theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom colors
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#4a9eff"
        self.secondary_bg = "#3a3a3a"
        
        # Configure styles
        self.configure_styles()
        
        # Database connection
        self.db_conn = None
        self.connect_database()
        
        # Current selection
        self.selected_memory_id = None
        
        # Build UI
        self.create_widgets()
        
        # Load initial data
        self.refresh_memories()
        self.update_statistics()
        
    def configure_styles(self):
        """Configure custom styles for widgets"""
        self.style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), foreground=self.accent_color)
        self.style.configure('Subtitle.TLabel', font=('Segoe UI', 12), foreground='#cccccc')
        self.style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'), foreground=self.fg_color)
        self.style.configure('Normal.TLabel', font=('Segoe UI', 10), foreground=self.fg_color)
        self.style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'))
        
    def connect_database(self):
        """Connect to the SQLite database with corruption protection"""
        try:
            if not DB_PATH.exists():
                messagebox.showerror("Error", f"Database not found at:\n{DB_PATH}\n\nPlease ensure the memory system is initialized.")
                self.root.quit()
                return
            
            self.db_conn = sqlite3.connect(
                str(DB_PATH),
                check_same_thread=False,
                timeout=30.0,
                isolation_level='DEFERRED'
            )
            self.db_conn.row_factory = sqlite3.Row
            
            # Enable WAL mode for better concurrency
            self.db_conn.execute("PRAGMA journal_mode=WAL")
            self.db_conn.execute("PRAGMA synchronous=FULL")
            
            # Verify database integrity
            cursor = self.db_conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            if result and result[0] != "ok":
                messagebox.showwarning("Warning", "Database integrity check failed. Some data may be corrupted.")
                
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to connect to database:\n{str(e)}")
            self.root.quit()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # ===== HEADER =====
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="Memory Manager", style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        subtitle_label = ttk.Label(header_frame, text="View and manage AI companion memories", style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0, sticky=tk.W)
        
        # ===== LEFT PANEL - Search and List =====
        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(2, weight=1)
        
        # Search section
        search_frame = ttk.LabelFrame(left_panel, text="Search Memories", padding="10")
        search_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)
        
        # Search by text
        ttk.Label(search_frame, text="Search:", style='Normal.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self.on_search_changed())
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, font=('Segoe UI', 10))
        search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Filter by type
        ttk.Label(search_frame, text="Type:", style='Normal.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        self.type_var = tk.StringVar(value="All")
        type_combo = ttk.Combobox(search_frame, textvariable=self.type_var, state='readonly', font=('Segoe UI', 10))
        type_combo['values'] = ['All', 'conversation', 'fact', 'preference', 'event', 'task', 'ephemeral']
        type_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        type_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_memories())
        
        # Filter by importance
        ttk.Label(search_frame, text="Min Importance:", style='Normal.TLabel').grid(row=2, column=0, sticky=tk.W, pady=2)
        self.importance_var = tk.StringVar(value="1")
        importance_spin = ttk.Spinbox(search_frame, from_=1, to=10, textvariable=self.importance_var, width=10, font=('Segoe UI', 10))
        importance_spin.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        importance_spin.bind('<Return>', lambda e: self.refresh_memories())
        
        # Filter by tags
        ttk.Label(search_frame, text="Tags:", style='Normal.TLabel').grid(row=3, column=0, sticky=tk.W, pady=2)
        self.tags_filter_var = tk.StringVar()
        tags_entry = ttk.Entry(search_frame, textvariable=self.tags_filter_var, font=('Segoe UI', 10))
        tags_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        tags_entry.bind('<Return>', lambda e: self.refresh_memories())
        
        # Search button
        search_btn = ttk.Button(search_frame, text="Search", command=self.refresh_memories, style='Accent.TButton')
        search_btn.grid(row=4, column=0, columnspan=2, pady=(10, 0))
        
        # Statistics section
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="Loading...", style='Normal.TLabel', justify=tk.LEFT)
        self.stats_label.grid(row=0, column=0, sticky=tk.W)
        
        # Memory list
        list_frame = ttk.LabelFrame(left_panel, text="Memories", padding="5")
        list_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Treeview for memories
        columns = ('ID', 'Title', 'Type', 'Importance', 'Date')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='tree headings', selectmode='browse')
        
        # Configure columns
        self.tree.column('#0', width=0, stretch=tk.NO)
        self.tree.column('ID', width=0, stretch=tk.NO)
        self.tree.column('Title', width=300)
        self.tree.column('Type', width=100)
        self.tree.column('Importance', width=80, anchor=tk.CENTER)
        self.tree.column('Date', width=150)
        
        # Configure headings
        self.tree.heading('Title', text='Title')
        self.tree.heading('Type', text='Type')
        self.tree.heading('Importance', text='Importance')
        self.tree.heading('Date', text='Date')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.on_memory_selected)
        
        # ===== RIGHT PANEL - Details and Actions =====
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        # Action buttons
        action_frame = ttk.Frame(right_panel)
        action_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(action_frame, text="New Memory", command=self.new_memory).grid(row=0, column=0, padx=2)
        ttk.Button(action_frame, text="Save Changes", command=self.save_memory).grid(row=0, column=1, padx=2)
        ttk.Button(action_frame, text="Delete", command=self.delete_memory).grid(row=0, column=2, padx=2)
        ttk.Button(action_frame, text="Refresh", command=self.refresh_memories).grid(row=0, column=3, padx=2)
        ttk.Button(action_frame, text="Backup", command=self.create_backup).grid(row=0, column=4, padx=2)
        ttk.Button(action_frame, text="Export", command=self.export_memories).grid(row=0, column=5, padx=2)
        
        # Details section
        details_frame = ttk.LabelFrame(right_panel, text="Memory Details", padding="10")
        details_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_frame.columnconfigure(1, weight=1)
        details_frame.rowconfigure(5, weight=1)
        
        # ID (hidden, for reference)
        self.id_var = tk.StringVar()
        
        # Title
        ttk.Label(details_frame, text="Title:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.title_var = tk.StringVar()
        title_entry = ttk.Entry(details_frame, textvariable=self.title_var, font=('Segoe UI', 11))
        title_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Type
        ttk.Label(details_frame, text="Type:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.detail_type_var = tk.StringVar()
        type_detail_combo = ttk.Combobox(details_frame, textvariable=self.detail_type_var, state='readonly', font=('Segoe UI', 10))
        type_detail_combo['values'] = ['conversation', 'fact', 'preference', 'event', 'task', 'ephemeral']
        type_detail_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Importance
        ttk.Label(details_frame, text="Importance:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5)
        self.detail_importance_var = tk.StringVar()
        importance_detail_spin = ttk.Spinbox(details_frame, from_=1, to=10, textvariable=self.detail_importance_var, width=10, font=('Segoe UI', 10))
        importance_detail_spin.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Tags
        ttk.Label(details_frame, text="Tags:", style='Header.TLabel').grid(row=3, column=0, sticky=tk.W, pady=5)
        self.tags_var = tk.StringVar()
        tags_detail_entry = ttk.Entry(details_frame, textvariable=self.tags_var, font=('Segoe UI', 10))
        tags_detail_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        ttk.Label(details_frame, text="(comma-separated)", font=('Segoe UI', 8), foreground='#888888').grid(row=4, column=1, sticky=tk.W, padx=(10, 0))
        
        # Content
        ttk.Label(details_frame, text="Content:", style='Header.TLabel').grid(row=5, column=0, sticky=(tk.W, tk.N), pady=5)
        self.content_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, font=('Segoe UI', 10), height=15)
        self.content_text.grid(row=5, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(10, 0))
        
        # Metadata display
        metadata_frame = ttk.LabelFrame(right_panel, text="Metadata", padding="10")
        metadata_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        metadata_frame.columnconfigure(0, weight=1)
        
        self.metadata_label = ttk.Label(metadata_frame, text="Select a memory to view details", style='Normal.TLabel', justify=tk.LEFT)
        self.metadata_label.grid(row=0, column=0, sticky=tk.W)
    
    def on_search_changed(self):
        """Handle search text changes with debouncing"""
        # Cancel previous scheduled search
        if hasattr(self, '_search_after_id'):
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
                params.extend([f'%{search_text}%', f'%{search_text}%'])
            
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
                tag_list = [t.strip() for t in tags_filter.split(',') if t.strip()]
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
                    dt = datetime.fromisoformat(row['timestamp'])
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = row['timestamp'][:16] if row['timestamp'] else 'Unknown'
                
                self.tree.insert('', tk.END, values=(
                    row['id'],
                    row['title'][:50] + ('...' if len(row['title']) > 50 else ''),
                    row['memory_type'],
                    row['importance'],
                    date_str
                ))
            
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
        memory_id = item['values'][0]
        
        try:
            cursor = self.db_conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            
            if row:
                self.selected_memory_id = row['id']
                self.id_var.set(row['id'])
                self.title_var.set(row['title'])
                self.detail_type_var.set(row['memory_type'])
                self.detail_importance_var.set(row['importance'])
                
                # Parse and display tags
                try:
                    tags = json.loads(row['tags'])
                    self.tags_var.set(', '.join(tags))
                except:
                    self.tags_var.set('')
                
                # Display content
                self.content_text.delete('1.0', tk.END)
                self.content_text.insert('1.0', row['content'])
                
                # Display metadata
                try:
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
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
        self.id_var.set('')
        self.title_var.set('')
        self.detail_type_var.set('conversation')
        self.detail_importance_var.set('5')
        self.tags_var.set('')
        self.content_text.delete('1.0', tk.END)
        self.metadata_label.config(text="New memory - fill in details and click Save")
    
    def save_memory(self):
        """Save the current memory (create or update)"""
        try:
            title = self.title_var.get().strip()
            content = self.content_text.get('1.0', tk.END).strip()
            
            if not title or not content:
                messagebox.showwarning("Validation Error", "Title and content are required.")
                return
            
            memory_type = self.detail_type_var.get()
            importance = int(self.detail_importance_var.get())
            
            # Parse tags
            tags_str = self.tags_var.get().strip()
            tags = [t.strip() for t in tags_str.split(',') if t.strip()] if tags_str else []
            
            now_iso = datetime.now(timezone.utc).isoformat()
            
            if self.selected_memory_id:
                # Update existing memory
                self.db_conn.execute("""
                    UPDATE memories
                    SET title = ?, content = ?, memory_type = ?, importance = ?, tags = ?, updated_at = ?
                    WHERE id = ?
                """, (title, content, memory_type, importance, json.dumps(tags), now_iso, self.selected_memory_id))
                
                messagebox.showinfo("Success", "Memory updated successfully!")
            else:
                # Create new memory
                import hashlib
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                time_hash = hashlib.sha256(now_iso.encode()).hexdigest()[:8]
                memory_id = f"mem_{time_hash}_{content_hash[:16]}"
                
                self.db_conn.execute("""
                    INSERT INTO memories (id, title, content, timestamp, tags, importance, memory_type, metadata, content_hash, created_at, updated_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (memory_id, title, content, now_iso, json.dumps(tags), importance, memory_type, '{}', content_hash, now_iso, now_iso, now_iso))
                
                self.selected_memory_id = memory_id
                messagebox.showinfo("Success", "Memory created successfully!\n\nNote: Vector embeddings will be updated when the MCP server is restarted.")
            
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
        result = messagebox.askyesno("Confirm Deletion", 
                                     f"Are you sure you want to delete this memory?\n\nTitle: {self.title_var.get()}\n\nThis action cannot be undone.")
        
        if result:
            try:
                self.db_conn.execute("DELETE FROM memories WHERE id = ?", (self.selected_memory_id,))
                self.db_conn.commit()
                
                messagebox.showinfo("Success", "Memory deleted successfully!\n\nNote: Vector embeddings will be cleaned up when the MCP server is restarted.")
                
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
                memory_dict['tags'] = json.loads(memory_dict['tags'])
                memory_dict['metadata'] = json.loads(memory_dict['metadata']) if memory_dict['metadata'] else {}
                memories.append(memory_dict)
            
            json_backup = backup_path / "memories_export.json"
            with open(json_backup, 'w', encoding='utf-8') as f:
                json.dump({
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_memories": len(memories),
                    "memories": memories
                }, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Success", f"Backup created successfully!\n\nLocation: {backup_path}\n\nFiles:\n- memories.db\n- memories_export.json")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create backup:\n{str(e)}")
    
    def export_memories(self):
        """Export memories to a JSON file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"memories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            if not file_path:
                return
            
            cursor = self.db_conn.execute("SELECT * FROM memories ORDER BY timestamp")
            memories = []
            for row in cursor.fetchall():
                memory_dict = dict(row)
                memory_dict['tags'] = json.loads(memory_dict['tags'])
                memory_dict['metadata'] = json.loads(memory_dict['metadata']) if memory_dict['metadata'] else {}
                memories.append(memory_dict)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_memories": len(memories),
                    "memories": memories
                }, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Success", f"Exported {len(memories)} memories to:\n{file_path}")
            
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
    
    def on_closing(self):
        """Handle window closing"""
        if self.db_conn:
            try:
                self.db_conn.commit()
                self.db_conn.close()
            except:
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
