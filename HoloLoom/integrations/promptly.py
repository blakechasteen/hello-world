#!/usr/bin/env python3
"""
Promptly Integration for HoloLoom
==================================
Enables HoloLoom to load and use prompts from Promptly repositories.

Features:
- PromptlyLoader: Loads prompts from Promptly repositories
- PromptlyMemoryAdapter: Converts Promptly prompts to HoloLoom MemoryShards
- Integration with HoloLoom's memory and orchestrator systems
"""

import sys
import warnings
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

# Import HoloLoom types (these should always be available in HoloLoom)
from HoloLoom.Documentation.types import MemoryShard, Query, Context

# Try to import Promptly
PROMPTLY_AVAILABLE = False
try:
    # Add Promptly to path if not already present
    repo_root = Path(__file__).parent.parent.parent
    promptly_path = repo_root / 'Promptly' / 'promptly'
    if str(promptly_path) not in sys.path:
        sys.path.insert(0, str(promptly_path))

    from promptly import Promptly, PromptlyDB
    PROMPTLY_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Promptly not available: {e}. Promptly integration disabled.")


@dataclass
class PromptMetadata:
    """Metadata about a loaded Promptly prompt."""
    name: str
    version: int
    branch: str
    commit_hash: str
    created_at: str
    tags: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class PromptlyMemoryAdapter:
    """
    Converts Promptly prompts into HoloLoom MemoryShards.

    This adapter bridges the gap between Promptly's prompt storage
    and HoloLoom's memory system, allowing prompts to be used as
    context in HoloLoom's decision-making.

    Usage:
        adapter = PromptlyMemoryAdapter()
        shards = adapter.prompts_to_shards(promptly_prompts)
    """

    def __init__(
        self,
        extract_entities: bool = True,
        extract_motifs: bool = True
    ):
        """
        Initialize the memory adapter.

        Args:
            extract_entities: Whether to extract entities from prompt metadata
            extract_motifs: Whether to extract motifs from prompt content
        """
        self.extract_entities = extract_entities
        self.extract_motifs = extract_motifs

    def prompt_to_shard(self, prompt_data: Dict[str, Any]) -> MemoryShard:
        """
        Convert a single Promptly prompt to a HoloLoom MemoryShard.

        Args:
            prompt_data: Dict containing prompt information from Promptly.get()

        Returns:
            MemoryShard ready for HoloLoom ingestion
        """
        # Extract basic info
        name = prompt_data.get('name', 'unnamed')
        content = prompt_data.get('content', '')
        version = prompt_data.get('version', 1)
        branch = prompt_data.get('branch', 'main')
        commit_hash = prompt_data.get('commit_hash', '')

        # Extract entities from metadata tags
        entities = []
        metadata_dict = prompt_data.get('metadata', {})
        if self.extract_entities and isinstance(metadata_dict, dict):
            # Extract tags as entities
            tags = metadata_dict.get('tags', [])
            if isinstance(tags, list):
                entities.extend(tags)

            # Extract model info as entity
            if 'model' in metadata_dict:
                entities.append(f"model:{metadata_dict['model']}")

        # Extract motifs from prompt name and content
        motifs = []
        if self.extract_motifs:
            # Use prompt name as primary motif
            motifs.append(name)

            # Extract keywords from content (simple word extraction)
            # In production, use HoloLoom's motif detector
            words = content.split()
            # Look for format placeholders as motifs
            placeholders = [w for w in words if '{' in w and '}' in w]
            motifs.extend(placeholders[:5])  # Limit to avoid noise

        # Create comprehensive metadata
        shard_metadata = {
            'source': 'promptly',
            'prompt_name': name,
            'version': version,
            'branch': branch,
            'commit_hash': commit_hash,
            'created_at': prompt_data.get('created_at', ''),
            'original_metadata': metadata_dict
        }

        # Create unique ID
        shard_id = f"promptly_{name}_v{version}_{commit_hash[:8]}"

        return MemoryShard(
            id=shard_id,
            text=content,
            episode=f"promptly_branch_{branch}",
            entities=list(set(entities)),  # Deduplicate
            motifs=list(set(motifs)),  # Deduplicate
            metadata=shard_metadata
        )

    def prompts_to_shards(
        self,
        prompts: List[Dict[str, Any]]
    ) -> List[MemoryShard]:
        """
        Convert multiple Promptly prompts to MemoryShards.

        Args:
            prompts: List of prompt dicts from Promptly

        Returns:
            List of MemoryShards
        """
        shards = []
        for prompt_data in prompts:
            try:
                shard = self.prompt_to_shard(prompt_data)
                shards.append(shard)
            except Exception as e:
                warnings.warn(f"Failed to convert prompt to shard: {e}")
                continue

        return shards


class PromptlyLoader:
    """
    Loads prompts from Promptly repositories into HoloLoom.

    This loader provides high-level methods to:
    - Load all prompts from a Promptly repo
    - Filter prompts by branch, tags, or version
    - Convert prompts to MemoryShards for HoloLoom
    - Integrate with HoloLoom's memory systems

    Usage:
        loader = PromptlyLoader(promptly_repo_path='/path/to/repo')
        shards = loader.load_all_prompts()

        # Or load into HoloLoom memory directly
        loader.load_into_memory(memory_manager)
    """

    def __init__(
        self,
        promptly_repo_path: Optional[str] = None,
        promptly_instance: Optional[Any] = None
    ):
        """
        Initialize Promptly loader.

        Args:
            promptly_repo_path: Path to Promptly repository root
            promptly_instance: Existing Promptly instance (alternative to repo_path)
        """
        if not PROMPTLY_AVAILABLE:
            raise RuntimeError(
                "Promptly is not available. Please ensure Promptly is installed."
            )

        # Initialize Promptly instance
        if promptly_instance:
            self.promptly = promptly_instance
        elif promptly_repo_path:
            self.promptly = Promptly(root_dir=promptly_repo_path)
        else:
            # Use current directory
            self.promptly = Promptly()

        # Initialize memory adapter
        self.adapter = PromptlyMemoryAdapter()

    def load_prompts(
        self,
        branch: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_version: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load prompts from Promptly repository with optional filtering.

        Args:
            branch: Branch to load from (default: current branch)
            tags: Filter prompts by tags in metadata
            min_version: Only load prompts at or above this version

        Returns:
            List of prompt dicts
        """
        # Get list of prompts
        prompt_list = self.promptly.list_prompts(branch=branch)

        # Load full prompt data
        prompts = []
        for prompt_meta in prompt_list:
            # Get full prompt
            prompt_data = self.promptly.get(
                prompt_meta['name'],
                version=prompt_meta.get('version')
            )

            if not prompt_data:
                continue

            # Apply filters
            if min_version and prompt_data.get('version', 0) < min_version:
                continue

            if tags:
                prompt_tags = prompt_data.get('metadata', {}).get('tags', [])
                if not any(tag in prompt_tags for tag in tags):
                    continue

            prompts.append(prompt_data)

        return prompts

    def load_all_prompts(
        self,
        branch: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load all prompts from Promptly repository.

        Args:
            branch: Branch to load from (default: current branch)

        Returns:
            List of all prompt dicts
        """
        return self.load_prompts(branch=branch)

    def prompts_to_shards(
        self,
        prompts: Optional[List[Dict[str, Any]]] = None,
        branch: Optional[str] = None
    ) -> List[MemoryShard]:
        """
        Convert Promptly prompts to HoloLoom MemoryShards.

        Args:
            prompts: List of prompt dicts (if None, loads all from current branch)
            branch: Branch to load from if prompts is None

        Returns:
            List of MemoryShards
        """
        if prompts is None:
            prompts = self.load_all_prompts(branch=branch)

        return self.adapter.prompts_to_shards(prompts)

    def load_into_memory(
        self,
        memory_manager: Any,
        branch: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Load Promptly prompts directly into HoloLoom MemoryManager.

        Args:
            memory_manager: HoloLoom MemoryManager instance
            branch: Branch to load from
            tags: Filter by tags

        Returns:
            Number of shards loaded
        """
        # Load and convert prompts
        prompts = self.load_prompts(branch=branch, tags=tags)
        shards = self.adapter.prompts_to_shards(prompts)

        # Add to memory manager
        # Assuming MemoryManager has add_shard or similar method
        count = 0
        for shard in shards:
            try:
                if hasattr(memory_manager, 'add_shard'):
                    memory_manager.add_shard(shard)
                elif hasattr(memory_manager, 'add'):
                    memory_manager.add(shard)
                else:
                    warnings.warn(
                        "MemoryManager doesn't have add_shard or add method. "
                        "Please add shards manually."
                    )
                    break
                count += 1
            except Exception as e:
                warnings.warn(f"Failed to add shard to memory: {e}")

        return count

    def get_prompt_metadata(
        self,
        prompt_name: str,
        version: Optional[int] = None
    ) -> Optional[PromptMetadata]:
        """
        Get metadata for a specific prompt.

        Args:
            prompt_name: Name of the prompt
            version: Specific version (default: latest)

        Returns:
            PromptMetadata or None if not found
        """
        prompt_data = self.promptly.get(prompt_name, version=version)

        if not prompt_data:
            return None

        metadata_dict = prompt_data.get('metadata', {})

        return PromptMetadata(
            name=prompt_data['name'],
            version=prompt_data['version'],
            branch=prompt_data['branch'],
            commit_hash=prompt_data['commit_hash'],
            created_at=prompt_data['created_at'],
            tags=metadata_dict.get('tags', []),
            performance_metrics=metadata_dict.get('performance', {})
        )


def load_integration_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load integration configuration from YAML file.

    Args:
        config_path: Path to integration_config.yaml (default: repo root)

    Returns:
        Dict with integration configuration
    """
    import yaml

    if config_path is None:
        # Look for config in repo root
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / 'integration_config.yaml'

    config_path = Path(config_path)

    if not config_path.exists():
        warnings.warn(f"Integration config not found at {config_path}")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        warnings.warn(f"Failed to load integration config: {e}")
        return {}


# Convenience function for quick integration
def quick_load_prompts(
    promptly_repo_path: Optional[str] = None,
    branch: str = 'main'
) -> List[MemoryShard]:
    """
    Quick function to load Promptly prompts as HoloLoom MemoryShards.

    Args:
        promptly_repo_path: Path to Promptly repo (default: current dir)
        branch: Branch to load from

    Returns:
        List of MemoryShards

    Usage:
        shards = quick_load_prompts('/path/to/promptly/repo', branch='main')
    """
    if not PROMPTLY_AVAILABLE:
        warnings.warn("Promptly not available, returning empty list")
        return []

    try:
        loader = PromptlyLoader(promptly_repo_path=promptly_repo_path)
        return loader.prompts_to_shards(branch=branch)
    except Exception as e:
        warnings.warn(f"Failed to load prompts: {e}")
        return []
