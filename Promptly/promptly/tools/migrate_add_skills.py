#!/usr/bin/env python3
"""
Migrate existing Promptly database to add skills tables
Run this if you have an existing .promptly repository
"""

import sqlite3
from pathlib import Path

def migrate_db(db_path):
    """Add skills tables to existing database"""
    print(f"Migrating database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if skills table already exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='skills'")
    if cursor.fetchone():
        print("✓ Skills tables already exist")
        conn.close()
        return
    
    print("Adding skills tables...")
    
    # Skills table (versioned)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            branch TEXT NOT NULL DEFAULT 'main',
            version INTEGER NOT NULL DEFAULT 1,
            parent_id INTEGER,
            commit_hash TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (parent_id) REFERENCES skills(id)
        )
    """)
    
    # Skill files table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skill_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            filetype TEXT,
            filepath TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (skill_id) REFERENCES skills(id)
        )
    """)
    
    conn.commit()
    conn.close()
    
    print("✓ Migration complete!")
    print("\nNew tables added:")
    print("  • skills - Versioned skill storage")
    print("  • skill_files - File attachments for skills")


def main():
    # Find .promptly directory
    promptly_dir = Path.cwd() / ".promptly"
    
    if not promptly_dir.exists():
        print("Error: No .promptly directory found in current directory")
        print("Run 'python promptly_cli.py init' first or cd to your Promptly repository")
        return
    
    db_path = promptly_dir / "promptly.db"
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return
    
    migrate_db(str(db_path))
    
    # Create skills directory if it doesn't exist
    skills_dir = promptly_dir / "skills"
    if not skills_dir.exists():
        skills_dir.mkdir()
        print(f"\n✓ Created skills directory: {skills_dir}")
    
    print("\nYou can now use skill commands:")
    print("  python promptly_cli.py skill add <name> [description]")
    print("  python promptly_cli.py skill list")
    print("\nSee SKILLS.md for full documentation!")


if __name__ == '__main__':
    main()
