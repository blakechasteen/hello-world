"""
Ernest Zero-G Integration - Wave 5
===================================
Zero-copy access to creative writing projects for collaborative workflows.

Enables:
- Multi-author manuscript collaboration
- Chapter versioning and rollback
- Character database sync (names, traits, arcs)
- World-building knowledge graphs
- Style guide enforcement across team

Philosophy: "Great stories need many voices. Zero-G coordinates without copying."
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json


@dataclass
class CreativeProject:
    """
    A creative writing project (novel, screenplay, etc.)

    Zero-copy structure - references files in place, no duplication.
    """
    project_id: str
    title: str
    project_type: str  # novel, screenplay, short_story, etc.
    authors: List[str]
    chapters_path: Path  # Path to chapters folder (zero-copy)
    characters_path: Optional[Path] = None  # Character database
    worldbuilding_path: Optional[Path] = None  # World-building notes
    style_guide_path: Optional[Path] = None  # Team style guide
    created: Optional[datetime] = None
    last_modified: Optional[datetime] = None


@dataclass
class CollaborativeSession:
    """Active collaborative writing session"""
    session_id: str
    project: CreativeProject
    active_authors: List[str]
    locked_chapters: Dict[str, str]  # chapter -> author
    session_start: datetime


class CreativeProjectLoader:
    """
    Zero-copy loader for creative writing projects.

    Provides read-only access to project files without duplication.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CreativeProjectLoader")
        self.loaded_projects: Dict[str, CreativeProject] = {}

    def load_project(self, project_path: str) -> CreativeProject:
        """
        Load creative project with zero-copy access.

        Args:
            project_path: Path to project root folder

        Returns:
            CreativeProject with file references (no copying)
        """
        project_root = Path(project_path)

        if not project_root.exists():
            raise ValueError(f"Project path not found: {project_path}")

        # Load project metadata (if exists)
        metadata_file = project_root / "project.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "project_id": project_root.name,
                "title": project_root.name,
                "project_type": "novel",
                "authors": ["unknown"]
            }

        # Discover project structure (zero-copy - just references)
        chapters_path = project_root / "chapters"
        characters_path = project_root / "characters.json"
        worldbuilding_path = project_root / "worldbuilding"
        style_guide_path = project_root / "style_guide.md"

        project = CreativeProject(
            project_id=metadata["project_id"],
            title=metadata["title"],
            project_type=metadata["project_type"],
            authors=metadata["authors"],
            chapters_path=chapters_path if chapters_path.exists() else None,
            characters_path=characters_path if characters_path.exists() else None,
            worldbuilding_path=worldbuilding_path if worldbuilding_path.exists() else None,
            style_guide_path=style_guide_path if style_guide_path.exists() else None,
            created=datetime.fromtimestamp(project_root.stat().st_ctime),
            last_modified=datetime.fromtimestamp(project_root.stat().st_mtime)
        )

        # Cache (zero-copy - just references)
        self.loaded_projects[project.project_id] = project

        self.logger.info(f"Loaded project: {project.title} (zero-copy)")
        return project

    def get_chapters(self, project: CreativeProject) -> List[Dict[str, Any]]:
        """
        Get chapters list (zero-copy - file references only).

        Args:
            project: Creative project

        Returns:
            List of chapter metadata (paths, not content)
        """
        if not project.chapters_path:
            return []

        chapters = []
        for chapter_file in sorted(project.chapters_path.glob("chapter*.md")):
            chapters.append({
                "chapter_number": self._extract_chapter_number(chapter_file.name),
                "path": str(chapter_file),
                "title": chapter_file.stem,
                "size_bytes": chapter_file.stat().st_size,
                "last_modified": datetime.fromtimestamp(chapter_file.stat().st_mtime)
            })

        return chapters

    def get_characters(self, project: CreativeProject) -> Dict[str, Any]:
        """
        Load character database (minimal copy - just metadata).

        Args:
            project: Creative project

        Returns:
            Character database dict
        """
        if not project.characters_path or not project.characters_path.exists():
            return {}

        with open(project.characters_path, 'r') as f:
            return json.load(f)

    def _extract_chapter_number(self, filename: str) -> int:
        """Extract chapter number from filename"""
        import re
        match = re.search(r'chapter(\d+)', filename.lower())
        return int(match.group(1)) if match else 0


class CollaborativeWorkflowManager:
    """
    Manages collaborative creative writing workflows.

    Coordinates multiple authors working on same project with:
    - Chapter locking (one author per chapter)
    - Style consistency enforcement
    - Character trait synchronization
    """

    def __init__(self, project_loader: CreativeProjectLoader):
        self.project_loader = project_loader
        self.active_sessions: Dict[str, CollaborativeSession] = {}
        self.logger = logging.getLogger(f"{__name__}.CollaborativeWorkflowManager")

    async def start_session(
        self,
        project_id: str,
        author: str
    ) -> CollaborativeSession:
        """
        Start collaborative writing session.

        Args:
            project_id: Project ID
            author: Author name

        Returns:
            Active session
        """
        project = self.project_loader.loaded_projects.get(project_id)
        if not project:
            raise ValueError(f"Project not loaded: {project_id}")

        session_id = f"{project_id}_{datetime.now().timestamp()}"

        session = CollaborativeSession(
            session_id=session_id,
            project=project,
            active_authors=[author],
            locked_chapters={},
            session_start=datetime.now()
        )

        self.active_sessions[session_id] = session
        self.logger.info(f"Started session: {session_id} for author: {author}")

        return session

    async def lock_chapter(
        self,
        session_id: str,
        chapter_number: int,
        author: str
    ) -> bool:
        """
        Lock chapter for editing (prevents conflicts).

        Args:
            session_id: Session ID
            chapter_number: Chapter to lock
            author: Author requesting lock

        Returns:
            True if locked successfully
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        chapter_key = f"chapter{chapter_number}"

        # Check if already locked
        if chapter_key in session.locked_chapters:
            current_owner = session.locked_chapters[chapter_key]
            if current_owner != author:
                self.logger.warning(
                    f"Chapter {chapter_number} locked by {current_owner}, "
                    f"{author} request denied"
                )
                return False

        # Lock chapter
        session.locked_chapters[chapter_key] = author
        self.logger.info(f"Chapter {chapter_number} locked by {author}")
        return True

    async def unlock_chapter(
        self,
        session_id: str,
        chapter_number: int,
        author: str
    ) -> bool:
        """Unlock chapter after editing"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        chapter_key = f"chapter{chapter_number}"

        if chapter_key in session.locked_chapters:
            if session.locked_chapters[chapter_key] == author:
                del session.locked_chapters[chapter_key]
                self.logger.info(f"Chapter {chapter_number} unlocked by {author}")
                return True

        return False

    async def get_team_style_guide(self, project: CreativeProject) -> Optional[str]:
        """
        Load team style guide for consistency.

        Args:
            project: Creative project

        Returns:
            Style guide content (or None)
        """
        if not project.style_guide_path or not project.style_guide_path.exists():
            return None

        with open(project.style_guide_path, 'r') as f:
            return f.read()


# Quick helpers
def load_creative_project(project_path: str) -> CreativeProject:
    """Quick project loader"""
    loader = CreativeProjectLoader()
    return loader.load_project(project_path)


async def start_collaborative_session(
    project_path: str,
    author: str
) -> CollaborativeSession:
    """Quick session starter"""
    loader = CreativeProjectLoader()
    project = loader.load_project(project_path)

    manager = CollaborativeWorkflowManager(loader)
    session = await manager.start_session(project.project_id, author)

    return session
