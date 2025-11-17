#!/usr/bin/env python3
"""
Template Registry - Template versioning, discovery, and management
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import yaml
import json
import hashlib


class RegistryError(Exception):
    """Base exception for registry errors"""
    pass


class TemplateExistsError(RegistryError):
    """Raised when attempting to register a template that already exists"""
    pass


class TemplateNotFoundError(RegistryError):
    """Raised when a template is not found in the registry"""
    pass


@dataclass
class TemplateVersion:
    """Represents a specific version of a template"""

    version: int
    content: str
    author: Optional[str] = None
    description: Optional[str] = None
    changelog: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    hash: str = field(default='')
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA256 hash of template content"""
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'content': self.content,
            'author': self.author,
            'description': self.description,
            'changelog': self.changelog,
            'created_at': self.created_at,
            'hash': self.hash,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateVersion':
        """Create from dictionary"""
        return cls(
            version=data.get('version', 1),
            content=data.get('content', ''),
            author=data.get('author'),
            description=data.get('description'),
            changelog=data.get('changelog'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            hash=data.get('hash', ''),
            metadata=data.get('metadata', {})
        )


@dataclass
class TemplateEntry:
    """Represents a template in the registry"""

    name: str
    category: str
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    versions: List[TemplateVersion] = field(default_factory=list)
    current_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_version(self, version: Optional[int] = None) -> Optional[TemplateVersion]:
        """Get a specific version or the current version"""
        if version is None:
            version = self.current_version

        for v in self.versions:
            if v.version == version:
                return v

        return None

    def get_latest_version(self) -> Optional[TemplateVersion]:
        """Get the latest version"""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.version)

    def add_version(self, version: TemplateVersion):
        """Add a new version"""
        self.versions.append(version)
        self.current_version = version.version
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'tags': list(self.tags),
            'versions': [v.to_dict() for v in self.versions],
            'current_version': self.current_version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateEntry':
        """Create from dictionary"""
        return cls(
            name=data.get('name', ''),
            category=data.get('category', 'general'),
            description=data.get('description'),
            tags=set(data.get('tags', [])),
            versions=[TemplateVersion.from_dict(v) for v in data.get('versions', [])],
            current_version=data.get('current_version', 1),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            metadata=data.get('metadata', {})
        )


class TemplateRegistry:
    """
    Template registry with versioning support

    Features:
    - Template registration and versioning
    - Category-based organization
    - Tag-based filtering
    - Template discovery
    - Import/export capabilities
    """

    def __init__(self, registry_file: Optional[str] = None):
        self.registry_file = registry_file
        self.templates: Dict[str, TemplateEntry] = {}
        self.categories: Set[str] = set()
        self.tags: Set[str] = set()

        if registry_file and Path(registry_file).exists():
            self.load()

    def register(
        self,
        name: str,
        content: str,
        category: str = 'general',
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        allow_overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TemplateEntry:
        """Register a new template"""
        if name in self.templates and not allow_overwrite:
            raise TemplateExistsError(f"Template '{name}' already exists")

        if name in self.templates:
            # Add new version to existing template
            entry = self.templates[name]
            version_num = entry.current_version + 1
            version = TemplateVersion(
                version=version_num,
                content=content,
                author=author,
                description=description,
                metadata=metadata or {}
            )
            entry.add_version(version)
        else:
            # Create new template entry
            version = TemplateVersion(
                version=1,
                content=content,
                author=author,
                description=description,
                metadata=metadata or {}
            )
            entry = TemplateEntry(
                name=name,
                category=category,
                description=description,
                tags=set(tags or []),
                versions=[version],
                metadata=metadata or {}
            )
            self.templates[name] = entry

        # Update category and tag indices
        self.categories.add(category)
        if tags:
            self.tags.update(tags)

        return entry

    def get(
        self,
        name: str,
        version: Optional[int] = None
    ) -> Optional[TemplateVersion]:
        """Get a template by name and version"""
        entry = self.templates.get(name)
        if not entry:
            return None

        return entry.get_version(version)

    def get_entry(self, name: str) -> Optional[TemplateEntry]:
        """Get the full template entry"""
        return self.templates.get(name)

    def list_templates(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None
    ) -> List[TemplateEntry]:
        """
        List templates with optional filtering

        Args:
            category: Filter by category
            tags: Filter by tags (AND logic - template must have all tags)
            search: Search in name and description

        Returns:
            List of matching template entries
        """
        templates = list(self.templates.values())

        # Filter by category
        if category:
            templates = [t for t in templates if t.category == category]

        # Filter by tags
        if tags:
            tag_set = set(tags)
            templates = [t for t in templates if tag_set.issubset(t.tags)]

        # Search filter
        if search:
            search_lower = search.lower()
            templates = [
                t for t in templates
                if search_lower in t.name.lower() or
                (t.description and search_lower in t.description.lower())
            ]

        return sorted(templates, key=lambda t: t.name)

    def list_categories(self) -> List[str]:
        """List all categories"""
        return sorted(self.categories)

    def list_tags(self) -> List[str]:
        """List all tags"""
        return sorted(self.tags)

    def delete(self, name: str) -> bool:
        """Delete a template from the registry"""
        if name in self.templates:
            del self.templates[name]
            return True
        return False

    def update_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """Update template metadata"""
        entry = self.templates.get(name)
        if not entry:
            return False

        entry.metadata.update(metadata)
        entry.updated_at = datetime.now().isoformat()
        return True

    def save(self, filepath: Optional[str] = None):
        """Save registry to file"""
        filepath = filepath or self.registry_file
        if not filepath:
            raise RegistryError("No registry file specified")

        data = {
            'version': '1.0',
            'templates': {
                name: entry.to_dict()
                for name, entry in self.templates.items()
            },
            'metadata': {
                'categories': list(self.categories),
                'tags': list(self.tags),
                'updated_at': datetime.now().isoformat()
            }
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load(self, filepath: Optional[str] = None):
        """Load registry from file"""
        filepath = filepath or self.registry_file
        if not filepath:
            raise RegistryError("No registry file specified")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        templates_data = data.get('templates', {})
        for name, entry_data in templates_data.items():
            entry = TemplateEntry.from_dict(entry_data)
            self.templates[name] = entry

        metadata = data.get('metadata', {})
        self.categories = set(metadata.get('categories', []))
        self.tags = set(metadata.get('tags', []))

    def export_template(self, name: str, filepath: str, version: Optional[int] = None):
        """Export a single template to a file"""
        entry = self.templates.get(name)
        if not entry:
            raise TemplateNotFoundError(f"Template '{name}' not found")

        version_obj = entry.get_version(version)
        if not version_obj:
            raise TemplateNotFoundError(f"Version {version} of template '{name}' not found")

        export_data = {
            'name': name,
            'category': entry.category,
            'description': entry.description,
            'tags': list(entry.tags),
            'version': version_obj.version,
            'content': version_obj.content,
            'author': version_obj.author,
            'metadata': entry.metadata
        }

        with open(filepath, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False)

    def import_template(self, filepath: str) -> TemplateEntry:
        """Import a template from a file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return self.register(
            name=data.get('name', ''),
            content=data.get('content', ''),
            category=data.get('category', 'general'),
            description=data.get('description'),
            tags=data.get('tags', []),
            author=data.get('author'),
            metadata=data.get('metadata', {})
        )


class TemplateDiscovery:
    """
    Template discovery system

    Discovers templates from:
    - File system directories
    - YAML/JSON files
    - Built-in library
    """

    def __init__(self, registry: TemplateRegistry):
        self.registry = registry

    def discover_from_directory(
        self,
        directory: str,
        pattern: str = '*.yaml',
        category: Optional[str] = None,
        recursive: bool = True
    ) -> int:
        """
        Discover templates from a directory

        Returns:
            Number of templates discovered
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise RegistryError(f"Directory not found: {directory}")

        count = 0
        if recursive:
            files = dir_path.rglob(pattern)
        else:
            files = dir_path.glob(pattern)

        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)

                # Support both single template and template list
                if isinstance(data, list):
                    templates = data
                elif 'templates' in data:
                    templates = data['templates']
                else:
                    templates = [data]

                for template_data in templates:
                    name = template_data.get('name') or filepath.stem
                    content = template_data.get('content', '')
                    tmpl_category = template_data.get('category') or category or 'discovered'

                    self.registry.register(
                        name=name,
                        content=content,
                        category=tmpl_category,
                        description=template_data.get('description'),
                        tags=template_data.get('tags', []),
                        metadata=template_data.get('metadata', {}),
                        allow_overwrite=True
                    )
                    count += 1

            except Exception as e:
                print(f"Warning: Failed to load template from {filepath}: {e}")

        return count

    def discover_builtin_templates(self, library_path: str) -> int:
        """
        Discover built-in templates from library directory

        Expected structure:
        library/
        ├── base/
        ├── roles/
        ├── domains/
        └── patterns/
        """
        count = 0

        library = Path(library_path)
        if not library.exists():
            return count

        # Discover from each category directory
        for category_dir in library.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                discovered = self.discover_from_directory(
                    str(category_dir),
                    pattern='*.yaml',
                    category=category,
                    recursive=False
                )
                count += discovered

        return count

    def scan_for_templates(
        self,
        paths: List[str],
        categories: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Scan multiple paths for templates

        Args:
            paths: List of directory paths to scan
            categories: Optional mapping of path to category name

        Returns:
            Total number of templates discovered
        """
        total = 0
        categories = categories or {}

        for path in paths:
            category = categories.get(path)
            count = self.discover_from_directory(
                path,
                category=category,
                recursive=True
            )
            total += count

        return total
