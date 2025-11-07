#!/usr/bin/env python3
"""Quick script to check database contents"""

import sqlite3
from pathlib import Path

db_path = Path("data/conversations.db")

if not db_path.exists():
    print(f"[X] Database not found at {db_path}")
    exit(1)

print(f"[OK] Database found at {db_path}\n")

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()

    # Check projects
    cursor.execute("SELECT COUNT(*) FROM projects")
    project_count = cursor.fetchone()[0]
    print(f"Projects: {project_count}")

    if project_count > 0:
        cursor.execute("SELECT id, name, icon, color FROM projects")
        print("\nExisting projects:")
        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]} ({row[3]})")
    else:
        print("  No projects found in database")

    # Check conversations
    cursor.execute("SELECT COUNT(*) FROM conversations")
    conv_count = cursor.fetchone()[0]
    print(f"\nConversations: {conv_count}")

    # Check messages
    cursor.execute("SELECT COUNT(*) FROM messages")
    msg_count = cursor.fetchone()[0]
    print(f"Messages: {msg_count}")

print("\n[OK] Database check complete")
