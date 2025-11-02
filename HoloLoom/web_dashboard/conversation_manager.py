"""
Conversation Manager - SQLite-based persistent chat storage
============================================================

Manages conversation threads and message history for the agentic chat interface.

Database Schema:
- conversations: Thread metadata (id, title, created_at, updated_at)
- messages: Individual messages (id, conversation_id, role, content, metadata, timestamp)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single chat message"""
    id: Optional[int]
    conversation_id: int
    role: str  # 'user' or 'assistant'
    content: str
    metadata: Dict  # reasoning_steps, confidence, duration, etc.
    timestamp: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'role': self.role,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


@dataclass
class Conversation:
    """A conversation thread"""
    id: Optional[int]
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    project_id: Optional[int] = None
    is_favorite: bool = False
    tags: str = ""  # Comma-separated tags

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'message_count': self.message_count,
            'project_id': self.project_id,
            'is_favorite': self.is_favorite,
            'tags': self.tags.split(',') if self.tags else []
        }


@dataclass
class Project:
    """A project/folder for organizing conversations"""
    id: Optional[int]
    name: str
    color: str  # Hex color code
    icon: str  # Emoji or icon name
    created_at: str
    conversation_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'color': self.color,
            'icon': self.icon,
            'created_at': self.created_at,
            'conversation_count': self.conversation_count
        }


@dataclass
class ConversationTemplate:
    """A conversation template"""
    id: Optional[int]
    name: str
    description: str
    icon: str
    messages: str  # JSON string of message list
    created_at: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon,
            'messages': json.loads(self.messages),
            'created_at': self.created_at
        }


class ConversationManager:
    """Manages conversation persistence using SQLite"""

    def __init__(self, db_path: str = "./data/conversations.db"):
        """
        Initialize conversation manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Add new columns to existing conversations table (safe for existing DBs)
            try:
                cursor.execute("ALTER TABLE conversations ADD COLUMN project_id INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute("ALTER TABLE conversations ADD COLUMN is_favorite INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass

            try:
                cursor.execute("ALTER TABLE conversations ADD COLUMN tags TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass

            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    color TEXT NOT NULL DEFAULT '#667eea',
                    icon TEXT NOT NULL DEFAULT '📁',
                    created_at TEXT NOT NULL
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)

            # Templates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    icon TEXT DEFAULT '📄',
                    messages TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_project
                ON conversations(project_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at)
            """)

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def create_conversation(self, title: Optional[str] = None) -> Conversation:
        """
        Create a new conversation thread.

        Args:
            title: Optional conversation title. Auto-generated if not provided.

        Returns:
            Conversation object with assigned ID
        """
        now = datetime.now().isoformat()
        if not title:
            title = f"Conversation {now[:10]}"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now)
            )
            conversation_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Created conversation {conversation_id}: {title}")
        return Conversation(
            id=conversation_id,
            title=title,
            created_at=now,
            updated_at=now,
            message_count=0
        )

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Get conversation by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Get message count
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            message_count = cursor.fetchone()[0]

        return Conversation(
            id=row[0],
            title=row[1],
            created_at=row[2],
            updated_at=row[3],
            message_count=message_count
        )

    def list_conversations(self, limit: int = 50, offset: int = 0) -> Tuple[List[Conversation], bool]:
        """
        List conversations with pagination, most recent first.

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip (for pagination)

        Returns:
            Tuple of (conversations list, has_more flag)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get conversations with LIMIT and OFFSET
            cursor.execute("""
                SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) as msg_count,
                       c.project_id, c.is_favorite, c.tags
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            conversations = []
            for row in cursor.fetchall():
                conversations.append(Conversation(
                    id=row[0],
                    title=row[1],
                    created_at=row[2],
                    updated_at=row[3],
                    message_count=row[4],
                    project_id=row[5],
                    is_favorite=bool(row[6]),
                    tags=row[7] or ""
                ))

            # Check if there are more conversations
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total = cursor.fetchone()[0]
            has_more = (offset + limit) < total

        return conversations, has_more

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Message:
        """
        Add a message to a conversation.

        Args:
            conversation_id: ID of conversation to add message to
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (reasoning_steps, confidence, etc.)

        Returns:
            Message object with assigned ID
        """
        if metadata is None:
            metadata = {}

        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert message
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (conversation_id, role, content, metadata_json, now))
            message_id = cursor.lastrowid

            # Update conversation updated_at
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id)
            )

            conn.commit()

        return Message(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata,
            timestamp=now
        )

    def get_messages(self, conversation_id: int) -> List[Message]:
        """
        Get all messages for a conversation, chronologically ordered.

        Args:
            conversation_id: ID of conversation

        Returns:
            List of Message objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, conversation_id, role, content, metadata, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """, (conversation_id,))

            messages = []
            for row in cursor.fetchall():
                messages.append(Message(
                    id=row[0],
                    conversation_id=row[1],
                    role=row[2],
                    content=row[3],
                    metadata=json.loads(row[4]),
                    timestamp=row[5]
                ))

        return messages

    def update_conversation_title(self, conversation_id: int, title: str):
        """Update conversation title"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, datetime.now().isoformat(), conversation_id)
            )
            conn.commit()

        logger.info(f"Updated conversation {conversation_id} title to: {title}")

    def delete_conversation(self, conversation_id: int):
        """Delete a conversation and all its messages"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()

        logger.info(f"Deleted conversation {conversation_id}")

    def search_messages(self, query: str, limit: int = 20) -> List[Tuple[Conversation, Message]]:
        """
        Search for messages containing query text.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of (Conversation, Message) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    c.id, c.title, c.created_at, c.updated_at,
                    m.id, m.conversation_id, m.role, m.content, m.metadata, m.timestamp
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.content LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT ?
            """, (f"%{query}%", limit))

            results = []
            for row in cursor.fetchall():
                conversation = Conversation(
                    id=row[0],
                    title=row[1],
                    created_at=row[2],
                    updated_at=row[3]
                )
                message = Message(
                    id=row[4],
                    conversation_id=row[5],
                    role=row[6],
                    content=row[7],
                    metadata=json.loads(row[8]),
                    timestamp=row[9]
                )
                results.append((conversation, message))

        return results

    def get_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM messages WHERE role = 'user'")
            user_messages = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM messages WHERE role = 'assistant'")
            assistant_messages = cursor.fetchone()[0]

        return {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'db_path': str(self.db_path)
        }

    # ===== PROJECT MANAGEMENT =====

    def create_project(self, name: str, color: str = '#667eea', icon: str = '📁') -> Project:
        """Create a new project"""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO projects (name, color, icon, created_at) VALUES (?, ?, ?, ?)",
                (name, color, icon, now)
            )
            project_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Created project {project_id}: {name}")
        return Project(
            id=project_id,
            name=name,
            color=color,
            icon=icon,
            created_at=now,
            conversation_count=0
        )

    def list_projects(self) -> List[Project]:
        """List all projects with conversation counts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.id, p.name, p.color, p.icon, p.created_at, COUNT(c.id)
                FROM projects p
                LEFT JOIN conversations c ON p.id = c.project_id
                GROUP BY p.id
                ORDER BY p.name
            """)

            projects = []
            for row in cursor.fetchall():
                projects.append(Project(
                    id=row[0],
                    name=row[1],
                    color=row[2],
                    icon=row[3],
                    created_at=row[4],
                    conversation_count=row[5]
                ))

        return projects

    def get_project(self, project_id: int) -> Optional[Project]:
        """Get a single project by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.id, p.name, p.color, p.icon, p.created_at, COUNT(c.id)
                FROM projects p
                LEFT JOIN conversations c ON p.id = c.project_id
                WHERE p.id = ?
                GROUP BY p.id
            """, (project_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return Project(
                id=row[0],
                name=row[1],
                color=row[2],
                icon=row[3],
                created_at=row[4],
                conversation_count=row[5]
            )

    def update_project(self, project_id: int, name: Optional[str] = None,
                      color: Optional[str] = None, icon: Optional[str] = None):
        """Update project details"""
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if color is not None:
            updates.append("color = ?")
            params.append(color)
        if icon is not None:
            updates.append("icon = ?")
            params.append(icon)

        if not updates:
            return

        params.append(project_id)
        sql = f"UPDATE projects SET {', '.join(updates)} WHERE id = ?"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()

        logger.info(f"Updated project {project_id}")

    def delete_project(self, project_id: int, delete_conversations: bool = False):
        """
        Delete project.
        If delete_conversations=False, conversations become uncategorized.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if delete_conversations:
                # Delete all conversations in project (and their messages via CASCADE)
                cursor.execute("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE project_id = ?)", (project_id,))
                cursor.execute("DELETE FROM conversations WHERE project_id = ?", (project_id,))
            else:
                # Move conversations to uncategorized
                cursor.execute("UPDATE conversations SET project_id = NULL WHERE project_id = ?", (project_id,))

            # Delete project
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()

        logger.info(f"Deleted project {project_id} (delete_conversations={delete_conversations})")

    def move_to_project(self, conversation_id: int, project_id: Optional[int]):
        """Move conversation to project (None = uncategorized)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE conversations SET project_id = ? WHERE id = ?",
                (project_id, conversation_id)
            )
            conn.commit()

        logger.info(f"Moved conversation {conversation_id} to project {project_id}")

    # ===== SEARCH & FILTERING =====

    def search_conversations(self, query: str, limit: int = 20) -> List[Conversation]:
        """Search conversations by title or content"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id),
                       c.project_id, c.is_favorite, c.tags
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.title LIKE ? OR m.content LIKE ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
            """, (f'%{query}%', f'%{query}%', limit))

            conversations = []
            for row in cursor.fetchall():
                conversations.append(Conversation(
                    id=row[0],
                    title=row[1],
                    created_at=row[2],
                    updated_at=row[3],
                    message_count=row[4],
                    project_id=row[5],
                    is_favorite=bool(row[6]),
                    tags=row[7] or ""
                ))

        return conversations

    def toggle_favorite(self, conversation_id: int):
        """Toggle conversation favorite status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations
                SET is_favorite = NOT is_favorite
                WHERE id = ?
            """, (conversation_id,))
            conn.commit()

        logger.info(f"Toggled favorite for conversation {conversation_id}")

    def update_tags(self, conversation_id: int, tags: List[str]):
        """Update conversation tags"""
        tags_str = ','.join(tags)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations
                SET tags = ?
                WHERE id = ?
            """, (tags_str, conversation_id))
            conn.commit()

        logger.info(f"Updated tags for conversation {conversation_id}: {tags}")

    def rename_conversation(self, conversation_id: int, new_title: str):
        """Rename a conversation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
            """, (new_title, datetime.now().isoformat(), conversation_id))
            conn.commit()

        logger.info(f"Renamed conversation {conversation_id} to: {new_title}")

    # ===== TEMPLATE METHODS =====

    def create_template(self, name: str, description: str, messages: List[Dict], icon: str = '📄') -> ConversationTemplate:
        """Save conversation as template"""
        now = datetime.now().isoformat()
        messages_json = json.dumps(messages)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversation_templates (name, description, icon, messages, created_at) VALUES (?, ?, ?, ?, ?)",
                (name, description, icon, messages_json, now)
            )
            template_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Created template {template_id}: {name}")
        return ConversationTemplate(
            id=template_id,
            name=name,
            description=description,
            icon=icon,
            messages=messages_json,
            created_at=now
        )

    def list_templates(self) -> List[ConversationTemplate]:
        """List all templates"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, description, icon, messages, created_at FROM conversation_templates ORDER BY created_at DESC")

            templates = []
            for row in cursor.fetchall():
                templates.append(ConversationTemplate(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    icon=row[3],
                    messages=row[4],
                    created_at=row[5]
                ))

        return templates

    def load_template(self, template_id: int) -> Optional[ConversationTemplate]:
        """Load a template"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, description, icon, messages, created_at FROM conversation_templates WHERE id = ?",
                (template_id,)
            )
            row = cursor.fetchone()

            if row:
                return ConversationTemplate(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    icon=row[3],
                    messages=row[4],
                    created_at=row[5]
                )

        return None

    def delete_template(self, template_id: int):
        """Delete a template"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversation_templates WHERE id = ?", (template_id,))
            conn.commit()

        logger.info(f"Deleted template {template_id}")

    def create_project(self, name: str, icon: str = '📁', color: str = '#667eea') -> Project:
        """Create a new project"""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO projects (name, color, icon, created_at) VALUES (?, ?, ?, ?)",
                (name, color, icon, now)
            )
            project_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Created project {project_id}: {name}")
        return Project(
            id=project_id,
            name=name,
            color=color,
            icon=icon,
            created_at=now,
            conversation_count=0
        )

    def get_analytics(self) -> Dict:
        """Get analytics data for dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Messages per day (last 7 days)
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM messages
                WHERE DATE(timestamp) >= DATE('now', '-7 days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            messages_per_day = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]

            # Confidence distribution (mock data for now - would need actual confidence data in DB)
            confidence_distribution = [
                {'range': '0.0-0.2', 'count': 5},
                {'range': '0.2-0.4', 'count': 12},
                {'range': '0.4-0.6', 'count': 25},
                {'range': '0.6-0.8', 'count': 45},
                {'range': '0.8-1.0', 'count': 30}
            ]

            # Mode usage (mock data for now - would need mode tracking in DB)
            mode_usage = [
                {'mode': 'DIRECT', 'count': 50},
                {'mode': 'VERIFY', 'count': 30},
                {'mode': 'RESEARCH', 'count': 15},
                {'mode': 'PLAN_EXECUTE', 'count': 10}
            ]

            # Response times (mock data for now - would need timing data in DB)
            response_times = [
                {'timestamp': '2025-11-02 10:00', 'time': 150},
                {'timestamp': '2025-11-02 11:00', 'time': 200},
                {'timestamp': '2025-11-02 12:00', 'time': 180},
                {'timestamp': '2025-11-02 13:00', 'time': 220},
                {'timestamp': '2025-11-02 14:00', 'time': 190}
            ]

            return {
                'messages_per_day': messages_per_day,
                'confidence_distribution': confidence_distribution,
                'mode_usage': mode_usage,
                'response_times': response_times
            }
