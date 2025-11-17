"""
Tool Registry

Manages custom tool definitions with versioning, categorization, and search.
"""

from typing import Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path
import json
import shutil
from .definition import ToolDefinition


class ToolRegistryError(Exception):
    """Tool registry error exception"""
    pass


class ToolRegistry:
    """Registry for custom tool definitions"""

    def __init__(self, storage_path: str = "./tools"):
        """
        Initialize tool registry

        Args:
            storage_path: Directory to store tool definitions
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.definitions_path = self.storage_path / "definitions"
        self.versions_path = self.storage_path / "versions"
        self.generated_path = self.storage_path / "generated"

        self.definitions_path.mkdir(exist_ok=True)
        self.versions_path.mkdir(exist_ok=True)
        self.generated_path.mkdir(exist_ok=True)

        # In-memory cache
        self._cache: Dict[str, ToolDefinition] = {}
        self._load_all()

    def _load_all(self):
        """Load all tool definitions into cache"""
        for definition_file in self.definitions_path.glob("*.json"):
            try:
                with open(definition_file, 'r') as f:
                    data = json.load(f)
                    tool_def = ToolDefinition.from_dict(data)
                    self._cache[tool_def.id] = tool_def
            except Exception as e:
                print(f"Error loading {definition_file}: {e}")

    def register(self, tool_def: ToolDefinition, overwrite: bool = False) -> str:
        """
        Register a new tool

        Args:
            tool_def: Tool definition to register
            overwrite: Whether to overwrite existing tool

        Returns:
            Tool ID

        Raises:
            ToolRegistryError: If tool already exists and overwrite=False
        """
        # Check if tool already exists
        if tool_def.id in self._cache and not overwrite:
            raise ToolRegistryError(
                f"Tool {tool_def.id} already exists. Use overwrite=True to replace."
            )

        # Set timestamps
        now = datetime.utcnow().isoformat()
        if not tool_def.created_at:
            tool_def.created_at = now
        tool_def.updated_at = now

        # Save to disk
        definition_file = self.definitions_path / f"{tool_def.id}.json"
        with open(definition_file, 'w') as f:
            json.dump(tool_def.to_dict(), f, indent=2)

        # Update cache
        self._cache[tool_def.id] = tool_def

        return tool_def.id

    def get(self, tool_id: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by ID

        Args:
            tool_id: Tool ID

        Returns:
            Tool definition or None if not found
        """
        return self._cache.get(tool_id)

    def list_all(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[ToolDefinition]:
        """
        List all tools with optional filtering

        Args:
            category: Filter by category
            tag: Filter by tag
            enabled_only: Only return enabled tools

        Returns:
            List of tool definitions
        """
        tools = list(self._cache.values())

        # Apply filters
        if enabled_only:
            tools = [t for t in tools if t.enabled]

        if category:
            tools = [t for t in tools if t.category == category]

        if tag:
            tools = [t for t in tools if tag in t.tags]

        return tools

    def search(self, query: str) -> List[ToolDefinition]:
        """
        Search tools by name or description

        Args:
            query: Search query

        Returns:
            List of matching tool definitions
        """
        query_lower = query.lower()
        results = []

        for tool in self._cache.values():
            if not tool.enabled:
                continue

            # Search in name, description, and tags
            if (query_lower in tool.name.lower() or
                query_lower in tool.description.lower() or
                any(query_lower in tag.lower() for tag in tool.tags)):
                results.append(tool)

        return results

    def delete(self, tool_id: str, keep_versions: bool = True) -> bool:
        """
        Delete a tool

        Args:
            tool_id: Tool ID to delete
            keep_versions: Whether to keep version history

        Returns:
            True if deleted, False if not found
        """
        if tool_id not in self._cache:
            return False

        # Remove from cache
        del self._cache[tool_id]

        # Remove definition file
        definition_file = self.definitions_path / f"{tool_id}.json"
        if definition_file.exists():
            definition_file.unlink()

        # Remove generated code
        generated_file = self.generated_path / f"{tool_id}.py"
        if generated_file.exists():
            generated_file.unlink()

        # Optionally remove versions
        if not keep_versions:
            version_dir = self.versions_path / tool_id
            if version_dir.exists():
                shutil.rmtree(version_dir)

        return True

    def enable(self, tool_id: str) -> bool:
        """
        Enable a tool

        Args:
            tool_id: Tool ID

        Returns:
            True if enabled, False if not found
        """
        tool = self.get(tool_id)
        if not tool:
            return False

        tool.enabled = True
        self.register(tool, overwrite=True)
        return True

    def disable(self, tool_id: str) -> bool:
        """
        Disable a tool

        Args:
            tool_id: Tool ID

        Returns:
            True if disabled, False if not found
        """
        tool = self.get(tool_id)
        if not tool:
            return False

        tool.enabled = False
        self.register(tool, overwrite=True)
        return True

    def save_version(self, tool_id: str) -> str:
        """
        Save a version snapshot of a tool

        Args:
            tool_id: Tool ID

        Returns:
            Version identifier

        Raises:
            ToolRegistryError: If tool not found
        """
        tool = self.get(tool_id)
        if not tool:
            raise ToolRegistryError(f"Tool {tool_id} not found")

        # Create version directory
        version_dir = self.versions_path / tool_id
        version_dir.mkdir(exist_ok=True)

        # Generate version identifier
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        version_id = f"{tool.version}_{timestamp}"

        # Save version snapshot
        version_file = version_dir / f"{version_id}.json"
        with open(version_file, 'w') as f:
            json.dump(tool.to_dict(), f, indent=2)

        return version_id

    def get_versions(self, tool_id: str) -> List[str]:
        """
        Get all versions of a tool

        Args:
            tool_id: Tool ID

        Returns:
            List of version identifiers
        """
        version_dir = self.versions_path / tool_id
        if not version_dir.exists():
            return []

        versions = []
        for version_file in sorted(version_dir.glob("*.json"), reverse=True):
            versions.append(version_file.stem)

        return versions

    def restore_version(self, tool_id: str, version_id: str) -> ToolDefinition:
        """
        Restore a tool from a version snapshot

        Args:
            tool_id: Tool ID
            version_id: Version identifier

        Returns:
            Restored tool definition

        Raises:
            ToolRegistryError: If version not found
        """
        version_file = self.versions_path / tool_id / f"{version_id}.json"
        if not version_file.exists():
            raise ToolRegistryError(
                f"Version {version_id} not found for tool {tool_id}"
            )

        with open(version_file, 'r') as f:
            data = json.load(f)

        tool_def = ToolDefinition.from_dict(data)
        self.register(tool_def, overwrite=True)

        return tool_def

    def get_categories(self) -> List[str]:
        """
        Get all unique categories

        Returns:
            List of category names
        """
        categories = set()
        for tool in self._cache.values():
            categories.add(tool.category)
        return sorted(list(categories))

    def get_tags(self) -> List[str]:
        """
        Get all unique tags

        Returns:
            List of tags
        """
        tags = set()
        for tool in self._cache.values():
            tags.update(tool.tags)
        return sorted(list(tags))

    def save_generated_code(self, tool_id: str, code: str) -> Path:
        """
        Save generated Python code for a tool

        Args:
            tool_id: Tool ID
            code: Generated Python code

        Returns:
            Path to saved file
        """
        code_file = self.generated_path / f"{tool_id}.py"
        with open(code_file, 'w') as f:
            f.write(code)
        return code_file

    def get_generated_code(self, tool_id: str) -> Optional[str]:
        """
        Get generated Python code for a tool

        Args:
            tool_id: Tool ID

        Returns:
            Generated code or None if not found
        """
        code_file = self.generated_path / f"{tool_id}.py"
        if not code_file.exists():
            return None

        with open(code_file, 'r') as f:
            return f.read()

    def export_tool(self, tool_id: str, export_path: str) -> Path:
        """
        Export a tool definition to a file

        Args:
            tool_id: Tool ID
            export_path: Path to export to

        Returns:
            Path to exported file

        Raises:
            ToolRegistryError: If tool not found
        """
        tool = self.get(tool_id)
        if not tool:
            raise ToolRegistryError(f"Tool {tool_id} not found")

        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)

        with open(export_file, 'w') as f:
            json.dump(tool.to_dict(), f, indent=2)

        return export_file

    def import_tool(self, import_path: str, overwrite: bool = False) -> str:
        """
        Import a tool definition from a file

        Args:
            import_path: Path to import from
            overwrite: Whether to overwrite existing tool

        Returns:
            Tool ID

        Raises:
            ToolRegistryError: If import fails
        """
        import_file = Path(import_path)
        if not import_file.exists():
            raise ToolRegistryError(f"Import file not found: {import_path}")

        try:
            with open(import_file, 'r') as f:
                data = json.load(f)

            tool_def = ToolDefinition.from_dict(data)
            return self.register(tool_def, overwrite=overwrite)

        except Exception as e:
            raise ToolRegistryError(f"Failed to import tool: {e}")

    def get_stats(self) -> Dict[str, any]:
        """
        Get registry statistics

        Returns:
            Dictionary of statistics
        """
        total_tools = len(self._cache)
        enabled_tools = len([t for t in self._cache.values() if t.enabled])
        disabled_tools = total_tools - enabled_tools

        categories = {}
        for tool in self._cache.values():
            categories[tool.category] = categories.get(tool.category, 0) + 1

        return {
            "total_tools": total_tools,
            "enabled_tools": enabled_tools,
            "disabled_tools": disabled_tools,
            "categories": categories,
            "total_categories": len(categories),
            "total_tags": len(self.get_tags()),
        }
