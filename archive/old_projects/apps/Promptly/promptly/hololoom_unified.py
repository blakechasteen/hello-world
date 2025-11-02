#!/usr/bin/env python3
"""
HoloLoom + Promptly Unified Integration
========================================
Complete integration connecting Promptly's prompts, skills, and analytics
with HoloLoom's knowledge graph, memory system, and multi-modal capabilities.

Features:
- Store prompts in Neo4j knowledge graph
- Semantic search across prompts
- Link prompts to concepts and entities
- Multi-modal memory (text + images + audio)
- Shared memory between Promptly and HoloLoom
- Unified analytics across both systems
"""

import json
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Try to import HoloLoom
HOLOLOOM_AVAILABLE = False
Memory = None  # Default if import fails
try:
    # Try multiple paths to find HoloLoom
    possible_paths = [
        Path(__file__).resolve().parent.parent.parent,  # From file location (mythRL root)
        Path.cwd().parent.parent,  # From cwd (Promptly/promptly)
        Path.cwd().parent,  # From cwd (Promptly)
        Path.cwd(),  # From cwd (mythRL root)
    ]

    hololoom_root = None
    for path in possible_paths:
        hololoom_dir = path / "HoloLoom"
        if hololoom_dir.exists() and (hololoom_dir / "memory").exists():
            hololoom_root = path  # Add parent directory, not HoloLoom itself
            break

    if not hololoom_root:
        raise ImportError(f"HoloLoom not found in any of: {[str(p / 'HoloLoom') for p in possible_paths]}")

    if str(hololoom_root) not in sys.path:
        sys.path.insert(0, str(hololoom_root))

    hololoom_path = hololoom_root / "HoloLoom"

    from HoloLoom.memory.unified import UnifiedMemory, RecallStrategy
    from HoloLoom.memory.unified import Memory as HoloLoomMemory
    Memory = HoloLoomMemory
    HOLOLOOM_AVAILABLE = True
    print(f"[OK] HoloLoom found at: {hololoom_path}")
except ImportError as e:
    print(f"[WARN] HoloLoom not available: {e}")
    # Create stub Memory class for type hints
    from dataclasses import dataclass as dc
    @dc
    class Memory:
        content: str
        metadata: Dict[str, Any] = None

# Try to import Neo4j backend
NEO4J_AVAILABLE = False
try:
    from HoloLoom.memory.neo4j_graph import Neo4jGraphMemory
    NEO4J_AVAILABLE = True
    print("[OK] Neo4j backend available")
except ImportError:
    print("[WARN] Neo4j backend not available")


@dataclass
class UnifiedPrompt:
    """Prompt stored in both Promptly and HoloLoom"""
    prompt_id: str
    name: str
    content: str
    version: int
    created_at: str
    updated_at: str
    tags: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    usage_count: int = 0
    avg_quality: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_hololoom_memory(self) -> Memory:
        """Convert to HoloLoom Memory format"""
        content_parts = [
            f"# Prompt: {self.name}",
            "",
            f"**Version:** {self.version}",
            f"**Created:** {self.created_at}",
            "",
            "## Content",
            self.content,
            "",
            f"**Tags:** {', '.join(self.tags)}",
            f"**Related Concepts:** {', '.join(self.related_concepts)}",
            "",
            f"**Usage:** {self.usage_count} executions",
            f"**Quality:** {self.avg_quality:.2f}"
        ]

        return Memory(
            content="\n".join(content_parts),
            metadata={
                "type": "promptly_prompt",
                "prompt_id": self.prompt_id,
                "name": self.name,
                "version": self.version,
                "tags": self.tags,
                "related_concepts": self.related_concepts,
                "usage_count": self.usage_count,
                "avg_quality": self.avg_quality,
                **self.metadata
            }
        )


class HoloLoomUnifiedBridge:
    """
    Unified bridge connecting Promptly and HoloLoom.

    Enables:
    - Prompts stored in Neo4j knowledge graph
    - Semantic search across all prompts
    - Relationship mapping (prompt -> concept -> entity)
    - Multi-modal memory support
    - Unified analytics
    - Cross-system memory sharing
    """

    def __init__(
        self,
        user_id: str = "promptly_unified",
        enable_neo4j: bool = True,
        enable_mem0: bool = False,
        enable_qdrant: bool = False
    ):
        """
        Initialize unified bridge.

        Args:
            user_id: User identifier
            enable_neo4j: Use Neo4j graph database
            enable_mem0: Use Mem0 integration
            enable_qdrant: Use Qdrant vector DB
        """
        self.user_id = user_id
        self.memory = None
        self.enabled = HOLOLOOM_AVAILABLE

        if not HOLOLOOM_AVAILABLE:
            print("[WARN] HoloLoom not available - unified bridge disabled")
            return

        # Initialize HoloLoom memory with all features
        try:
            print(f"\n[INFO] Initializing HoloLoom unified memory...")
            print(f"  - User ID: {user_id}")

            # UnifiedMemory doesn't take these parameters - it auto-initializes backends
            self.memory = UnifiedMemory(user_id=user_id)

            print("[OK] HoloLoom unified bridge initialized")
            self.enabled = True

        except Exception as e:
            print(f"[ERROR] Failed to initialize HoloLoom: {e}")
            import traceback
            traceback.print_exc()
            self.enabled = False

    def store_prompt(self, prompt: UnifiedPrompt) -> Optional[str]:
        """
        Store prompt in HoloLoom knowledge graph.

        Args:
            prompt: UnifiedPrompt object

        Returns:
            Memory ID if stored successfully
        """
        if not self.enabled or not self.memory:
            return None

        try:
            # Convert prompt to text format for UnifiedMemory.store()
            content_parts = [
                f"# Prompt: {prompt.name}",
                f"Version: {prompt.version}",
                prompt.content,
                f"Tags: {', '.join(prompt.tags)}",
                f"Usage: {prompt.usage_count} | Quality: {prompt.avg_quality:.2f}"
            ]
            text = "\n".join(content_parts)

            # Store using UnifiedMemory API (returns string ID)
            memory_id = self.memory.store(
                text=text,
                context={
                    "type": "promptly_prompt",
                    "prompt_id": prompt.prompt_id,
                    "name": prompt.name,
                    "version": prompt.version,
                    "tags": prompt.tags,
                    "usage_count": prompt.usage_count,
                    "avg_quality": prompt.avg_quality
                },
                importance=min(prompt.avg_quality, 1.0) if prompt.avg_quality > 0 else 0.5
            )

            print(f"[OK] Stored prompt '{prompt.name}' in HoloLoom: {memory_id}")
            return memory_id

        except Exception as e:
            print(f"[ERROR] Failed to store prompt: {e}")
            import traceback
            traceback.print_exc()
            return None

    def search_prompts(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Semantic search for prompts.

        Args:
            query: Search query
            tags: Optional tag filters
            limit: Maximum results

        Returns:
            List of matching prompts with metadata
        """
        if not self.enabled or not self.memory:
            return []

        try:
            # Search in HoloLoom using recall API
            results = self.memory.recall(query, strategy=RecallStrategy.SIMILAR, limit=limit * 2)

            # Filter to prompts only
            prompt_results = [
                {
                    'id': mem.id,
                    'text': mem.text,
                    'relevance': mem.relevance,
                    'context': mem.context
                }
                for mem in results
                if mem.context.get('type') == 'promptly_prompt'
            ]

            # Filter by tags if specified
            if tags:
                prompt_results = [
                    r for r in prompt_results
                    if any(tag in r.get('context', {}).get('tags', []) for tag in tags)
                ]

            return prompt_results[:limit]

        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def link_prompt_to_concept(self, prompt_id: str, concept: str) -> bool:
        """
        Create relationship between prompt and concept in graph.

        Args:
            prompt_id: Prompt identifier
            concept: Concept name (e.g., "SQL optimization", "UI design")

        Returns:
            True if linked successfully
        """
        if not self.enabled or not self.memory:
            return False

        try:
            # Store concept as its own memory
            self.memory.store(
                text=f"Concept: {concept}",
                context={
                    "type": "concept",
                    "name": concept,
                    "linked_prompts": [prompt_id]
                },
                importance=0.6
            )

            print(f"[OK] Linked prompt {prompt_id} to concept '{concept}'")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to link: {e}")
            return False

    def get_related_prompts(self, prompt_id: str, limit: int = 5) -> List[Dict]:
        """
        Find prompts related to given prompt.

        Uses graph relationships and semantic similarity.

        Args:
            prompt_id: Source prompt ID
            limit: Maximum results

        Returns:
            List of related prompts
        """
        if not self.enabled or not self.memory:
            return []

        try:
            # Search for prompts to find the source
            all_results = self.memory.recall("prompt", strategy=RecallStrategy.SIMILAR, limit=100)
            source_prompt = None

            for mem in all_results:
                if mem.context.get('prompt_id') == prompt_id:
                    source_prompt = mem
                    break

            if not source_prompt:
                return []

            # Use similar_to() if available, otherwise search by text
            try:
                similar = self.memory.similar_to(source_prompt.id, limit=limit + 1)
                # Convert to our format and filter to prompts only
                related = [
                    {
                        'id': mem.id,
                        'text': mem.text,
                        'relevance': mem.relevance,
                        'context': mem.context
                    }
                    for mem in similar
                    if mem.context.get('type') == 'promptly_prompt'
                    and mem.context.get('prompt_id') != prompt_id
                ]
            except:
                # Fallback to text search
                results = self.search_prompts(source_prompt.text, limit=limit + 1)
                related = [r for r in results if r.get('context', {}).get('prompt_id') != prompt_id]

            return related[:limit]

        except Exception as e:
            print(f"[ERROR] Failed to find related: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_prompt_analytics(self) -> Dict[str, Any]:
        """
        Get unified analytics across Promptly and HoloLoom.

        Returns:
            Analytics dictionary
        """
        if not self.enabled or not self.memory:
            return {"enabled": False}

        try:
            # Use a broad recall query to get prompts
            # Since UnifiedMemory doesn't have list_all(), we'll search for "prompt"
            all_results = self.memory.recall("prompt", strategy=RecallStrategy.SIMILAR, limit=100)

            # Filter to promptly prompts only
            prompts = [
                m for m in all_results
                if m.context.get('type') == 'promptly_prompt'
            ]

            if not prompts:
                return {
                    "enabled": True,
                    "total_prompts": 0,
                    "message": "No prompts in HoloLoom yet"
                }

            # Calculate analytics
            total_prompts = len(prompts)
            total_usage = sum(
                m.context.get('usage_count', 0)
                for m in prompts
            )

            qualities = [
                m.context.get('avg_quality', 0)
                for m in prompts
                if m.context.get('avg_quality', 0) > 0
            ]
            avg_quality = sum(qualities) / len(qualities) if qualities else 0

            # Tag distribution
            tag_counts = {}
            for m in prompts:
                for tag in m.context.get('tags', []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Most used prompts
            most_used = sorted(
                prompts,
                key=lambda m: m.context.get('usage_count', 0),
                reverse=True
            )[:5]

            return {
                "enabled": True,
                "total_prompts": total_prompts,
                "total_usage": total_usage,
                "avg_quality": avg_quality,
                "tag_distribution": tag_counts,
                "most_used": [
                    {
                        "name": m.context.get('name'),
                        "usage": m.context.get('usage_count')
                    }
                    for m in most_used
                ]
            }

        except Exception as e:
            print(f"[ERROR] Analytics failed: {e}")
            import traceback
            traceback.print_exc()
            return {"enabled": True, "error": str(e)}

    def sync_from_promptly(self, promptly_instance) -> int:
        """
        Sync all prompts from Promptly to HoloLoom.

        Args:
            promptly_instance: Promptly() instance

        Returns:
            Number of prompts synced
        """
        if not self.enabled or not self.memory:
            return 0

        try:
            # Get all prompts from Promptly
            prompts = promptly_instance.list()
            synced = 0

            for prompt_data in prompts:
                # Create UnifiedPrompt
                unified = UnifiedPrompt(
                    prompt_id=f"promptly_{prompt_data['name']}_{prompt_data['version']}",
                    name=prompt_data['name'],
                    content=prompt_data['content'],
                    version=prompt_data['version'],
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    tags=prompt_data.get('metadata', {}).get('tags', []),
                    metadata=prompt_data.get('metadata', {})
                )

                # Store in HoloLoom
                if self.store_prompt(unified):
                    synced += 1

            print(f"[OK] Synced {synced} prompts from Promptly to HoloLoom")
            return synced

        except Exception as e:
            print(f"[ERROR] Sync failed: {e}")
            return 0


# Convenience function
def create_unified_bridge(
    user_id: str = "promptly_unified",
    enable_neo4j: bool = True
) -> HoloLoomUnifiedBridge:
    """Create unified bridge instance"""
    return HoloLoomUnifiedBridge(
        user_id=user_id,
        enable_neo4j=enable_neo4j
    )


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("HoloLoom + Promptly Unified Integration")
    print("=" * 60)

    if HOLOLOOM_AVAILABLE:
        print("\n[OK] HoloLoom available")

        # Create bridge
        bridge = create_unified_bridge(enable_neo4j=NEO4J_AVAILABLE)

        if bridge.enabled:
            print("\n[OK] Bridge enabled successfully")

            # Demo: Store a prompt
            demo_prompt = UnifiedPrompt(
                prompt_id="demo_001",
                name="SQL Optimizer",
                content="Optimize this SQL query for performance:\n{query}",
                version=1,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                tags=["sql", "optimization", "database"],
                related_concepts=["query optimization", "database performance"],
                usage_count=42,
                avg_quality=0.87
            )

            bridge.store_prompt(demo_prompt)

            # Get analytics
            analytics = bridge.get_prompt_analytics()
            print(f"\n[OK] Analytics:")
            for key, value in analytics.items():
                print(f"  {key}: {value}")

        else:
            print("\n[WARN] Bridge disabled")
    else:
        print("\n[WARN] HoloLoom not available")
        print("\nTo enable:")
        print("  1. Ensure HoloLoom is in parent directory")
        print("  2. Install dependencies: pip install -r HoloLoom/requirements.txt")
        print("  3. Optional: Setup Neo4j for graph features")
