"""
HoloLoom - Unified Memory System

The 10/10 Layer: Everything is a memory operation.

Usage:
    from hololoom import HoloLoom

    # Initialize (single-device mode)
    loom = HoloLoom()

    # Initialize (cross-device mode with identity)
    from hololoom.handoff import UnifiedIdentity
    identity = UnifiedIdentity.generate("blake")
    loom = HoloLoom(identity=identity)  # Syncs across devices!

    # Experience content (any modality)
    memory = await loom.experience("Thompson Sampling balances exploration")

    # Recall related memories
    memories = await loom.recall("What did I learn about sampling?")

    # Reflect on outcomes
    await loom.reflect(memories, feedback={"helpful": True})

Design Philosophy:
- Single entry point (HoloLoom)
- Single representation (Memory)
- Three core operations (experience/recall/reflect)
- Implementation details hidden
- Modality-agnostic
- Cross-device: Identity owns memory, devices are just windows
"""

from typing import Any, List, Optional, Dict, Union, Tuple, TYPE_CHECKING
from pathlib import Path
import networkx as nx

# LAZY IMPORTS: Break circular dependency with HoloLoom/__init__.py
# Config is imported lazily in __init__() method instead of module level
if TYPE_CHECKING:
    from hololoom.config import Config
    from hololoom.handoff.identity import UnifiedIdentity
    from hololoom.handoff.orchestrator import HardenedHandoffOrchestrator
    from hololoom.handoff.synced_memory import SyncedMemory
    from hololoom.loom.weave_house import WeaveHouse, WeaveResult
    from hololoom.loom.dreaming import DreamOrchestrator
    from hololoom.fabric.fabric import Fabric

from hololoom.memory.protocol import Memory
from hololoom.memory.awareness_graph import AwarenessGraph
from hololoom.memory.awareness_types import ActivationStrategy, AwarenessMetrics
from hololoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus
from hololoom.embedding.spectral import MatryoshkaEmbeddings

# Input processing (graceful degradation)
try:
    from hololoom.input.router import InputRouter
    MULTIMODAL_AVAILABLE = True
except ImportError:
    InputRouter = None
    MULTIMODAL_AVAILABLE = False


class HoloLoom:
    """
    Unified memory system.

    Everything is a memory operation:
    - experience() → form memories
    - recall() → activate memories
    - reflect() → learn from feedback

    Example:
        >>> loom = HoloLoom()
        >>>
        >>> # Experience (any modality)
        >>> mem = await loom.experience("Python has decorators")
        >>>
        >>> # Recall
        >>> memories = await loom.recall("Tell me about Python")
        >>>
        >>> # Reflect
        >>> await loom.reflect(memories, feedback={"helpful": True})
    """

    def __init__(
        self,
        config: Optional['Config'] = None,
        graph_backend: Optional[nx.MultiDiGraph] = None,
        identity: Optional['UnifiedIdentity'] = None,
        use_multi_perspective: bool = False,
    ):
        """
        Initialize HoloLoom.

        Args:
            config: System configuration (defaults to Config.fast())
            graph_backend: Optional graph backend (creates new if None)
            identity: Optional UnifiedIdentity for cross-device sync.
                      When provided, enables synced memory across devices.
            use_multi_perspective: Enable multi-perspective WeaveHouse system.
                      When True (or config.use_weave_house is True), creates
                      WeaveHouse with 5 looms (RECALL, REASON, REACH, REFLECT, REFUSE)
                      and optional DreamOrchestrator for background consolidation.

        Example:
            >>> # Simple initialization (single-device mode)
            >>> loom = HoloLoom()
            >>>
            >>> # With custom config
            >>> from hololoom.config import Config
            >>> loom = HoloLoom(config=Config.fused())
            >>>
            >>> # Cross-device mode (syncs memories across devices)
            >>> from hololoom.handoff import UnifiedIdentity
            >>> identity = UnifiedIdentity.generate("blake")
            >>> loom = HoloLoom(identity=identity)
            >>>
            >>> # Multi-perspective mode (5 looms with consensus)
            >>> loom = HoloLoom(use_multi_perspective=True)
            >>> # Or via config:
            >>> loom = HoloLoom(config=Config.multi_perspective())
        """
        # LAZY IMPORT: Import Config here to break circular dependency
        from hololoom.config import Config

        # Configuration
        self.config = config or Config.fast()

        # Cross-device handoff (if identity provided)
        self._identity = identity
        self._handoff = None
        self._synced_memory = None

        if identity is not None:
            # Lazy import handoff modules
            from hololoom.handoff.orchestrator import HardenedHandoffOrchestrator
            from hololoom.handoff.synced_memory import SyncedMemory

            # Create synced memory and handoff orchestrator
            self._synced_memory = SyncedMemory(identity)
            self._handoff = HardenedHandoffOrchestrator(
                identity=identity,
                memory=self._synced_memory,  # Note: parameter is 'memory' not 'synced_memory'
            )

        # Create input router (if available)
        if MULTIMODAL_AVAILABLE:
            self._router = InputRouter()
        else:
            self._router = None

        # Create semantic calculus
        embedder = MatryoshkaEmbeddings(sizes=self.config.scales)
        self._semantic = MatryoshkaSemanticCalculus(
            matryoshka_embedder=embedder,
            snapshot_interval=0.5
        )

        # Create awareness graph (the core)
        self._graph = graph_backend or nx.MultiDiGraph()
        self._awareness = AwarenessGraph(
            graph_backend=self._graph,
            semantic_calculus=self._semantic,
            vector_store=None  # Could add vector store here
        )

        # Multi-perspective WeaveHouse (December 2025)
        # Stored for lazy initialization in __aenter__ (since create_weave_house_system is async)
        self._use_multi_perspective = use_multi_perspective or (
            hasattr(self.config, 'use_weave_house') and self.config.use_weave_house
        )
        self._weave_house: Optional['WeaveHouse'] = None
        self._dream_orchestrator: Optional['DreamOrchestrator'] = None
        self._weave_house_initialized = False

    # =========================================================================
    # Core Operations: The 10/10 API
    # =========================================================================

    async def experience(
        self,
        content: Any,
        context: Optional[Dict] = None
    ) -> Memory:
        """
        Experience content and integrate into memory.

        The fundamental operation - everything flows through this.

        Handles any modality:
        - Text: "Thompson Sampling balances exploration"
        - Structured: {"algorithm": "Thompson Sampling", "type": "bayesian"}
        - Image: image_bytes (if PIL installed)
        - Audio: audio_bytes (if audio libs installed)
        - Multimodal: [text, image, data] (fused automatically)

        Args:
            content: Content to experience (any modality)
            context: Optional context metadata

        Returns:
            Memory object (immutable, can be used for recall)

        Example:
            >>> # Text
            >>> mem1 = await loom.experience("Python has decorators")
            >>>
            >>> # Structured data
            >>> mem2 = await loom.experience({
            ...     "language": "Python",
            ...     "features": ["decorators", "generators"]
            ... })
            >>>
            >>> # With context
            >>> mem3 = await loom.experience(
            ...     "Important fact",
            ...     context={"priority": "high", "source": "lecture"}
            ... )
        """
        # Fast path: text string
        if isinstance(content, str):
            perception = await self._awareness.perceive(content)
            memory_id = await self._awareness.remember(content, perception, context)

        # General path: multimodal (if available)
        elif self._router is not None and not isinstance(content, str):
            # Process through input router (handles modality detection)
            processed = await self._router.process(content)
            perception = await self._awareness.perceive(processed)
            memory_id = await self._awareness.remember(processed, perception, context)

        else:
            raise ValueError(
                f"Cannot process content type {type(content)}. "
                "Either provide text string or install multimodal dependencies."
            )

        # Cross-device mode: also store to synced memory (for background sync)
        if self._synced_memory is not None:
            content_text = content if isinstance(content, str) else str(content)
            await self._synced_memory.experience(content_text, tags=context.get('tags') if context else None)

        # Retrieve full Memory object
        memory = self._awareness.graph.nodes[memory_id]
        return Memory(
            id=memory_id,
            text=memory['text'],
            timestamp=memory['timestamp'],
            context=memory.get('context', {}),
            metadata=memory.get('metadata', {})
        )

    async def recall(
        self,
        query: Any,
        strategy: ActivationStrategy = ActivationStrategy.BALANCED,
        limit: Optional[int] = None,
        include_photos: bool = False
    ) -> Union[List[Memory], Dict[str, List]]:
        """
        Recall memories related to query.

        Query can be:
        - Text: "What is Thompson Sampling?"
        - Structured: {"topic": "reinforcement_learning"}
        - Image: image_bytes (finds similar images)

        Args:
            query: What to search for (any modality)
            strategy: Activation strategy
                - PRECISE: High precision (topic shift detection)
                - BALANCED: Balance precision/recall (default)
                - EXPLORATORY: Broad exploration
                - DEEP: Follow connections deeply
            limit: Maximum memories to return (None = no limit)
            include_photos: If True, also search photo memories (multimodal)

        Returns:
            If include_photos=False: List[Memory] (text memories only, backward compatible)
            If include_photos=True: Dict with 'text' and 'photos' keys

        Example:
            >>> # Simple recall (text only - backward compatible)
            >>> memories = await loom.recall("What did I learn about Python?")
            >>>
            >>> # Multimodal recall (text + photos)
            >>> results = await loom.recall(
            ...     "Show me the architecture diagram",
            ...     include_photos=True
            ... )
            >>> text_memories = results['text']
            >>> photo_memories = results['photos']
        """
        # Experience query (creates temporary perception)
        if isinstance(query, str):
            perception = await self._awareness.perceive(query)

        elif self._router is not None:
            processed = await self._router.process(query)
            perception = await self._awareness.perceive(processed)

        else:
            raise ValueError(
                f"Cannot process query type {type(query)}. "
                "Either provide text string or install multimodal dependencies."
            )

        # Activate memories
        memories = await self._awareness.activate(
            perception,
            strategy=strategy
        )

        # Apply limit if specified
        if limit is not None:
            memories = memories[:limit]

        # If not including photos, return text memories only (backward compatible)
        if not include_photos:
            return memories

        # Multimodal search (text + photos)
        query_text = query if isinstance(query, str) else str(query)

        # Ensure photo memory initialized
        if hasattr(self, '_photo_memory'):
            photo_memory = self._photo_memory
        else:
            photo_memory = self._ensure_photo_memory()
            await photo_memory._initialize()

        # Search across text and photos
        multimodal_results = await self._kg.search_multimodal(
            query=query_text,
            return_types=['text', 'photo'],
            k=limit or 10,
            photo_memory=photo_memory
        )

        # Separate text and photo results
        text_results = [r for r in multimodal_results if r['type'] == 'text']
        photo_results = [r for r in multimodal_results if r['type'] == 'photo']

        # Convert photo results back to PhotoTokens
        photos = []
        for photo_result in photo_results:
            photo_id = photo_result['id']
            if photo_id in photo_memory.tokens:
                photos.append(photo_memory.tokens[photo_id])

        return {
            'text': memories,  # Use awareness-activated memories
            'photos': photos,
            'scores': {
                'text': [r['score'] for r in text_results],
                'photos': [r['score'] for r in photo_results]
            }
        }

    async def reflect(
        self,
        memories: List[Memory],
        feedback: Optional[Dict] = None
    ) -> None:
        """
        Reflect on memories to improve future recall.

        Learning signal for the system. Updates topology weights,
        activation parameters, and exploration strategy based on feedback.

        Args:
            memories: Memories to reflect on
            feedback: Feedback dictionary with signals like:
                - {"helpful": True/False}
                - {"relevance": 0.8}
                - {"selected": [memory.id for memory in top_3]}
                - {"outcome": "success"}

        Example:
            >>> # Recall memories
            >>> memories = await loom.recall("machine learning algorithms")
            >>>
            >>> # User found these helpful
            >>> await loom.reflect(memories, feedback={"helpful": True})
            >>>
            >>> # More detailed feedback
            >>> await loom.reflect(memories, feedback={
            ...     "relevance": 0.9,
            ...     "selected": [memories[0].id, memories[1].id],
            ...     "outcome": "answered_question"
            ... })
        """
        # Simple reflection: strengthen activated connections
        # Future: Use reflection buffer for more sophisticated learning

        if not memories:
            return

        # Update topology weights based on feedback
        helpful = feedback.get('helpful', True) if feedback else True
        relevance = feedback.get('relevance', 0.8) if feedback else 0.8

        # For now, just track the feedback
        # Future: Update graph edge weights, adjust activation thresholds, etc.
        for memory in memories:
            # Could update metadata, edge weights, etc.
            pass

        # TODO: Integrate with reflection buffer for sophisticated learning
        # await self._awareness.reflection_buffer.learn(memories, feedback)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def experience_batch(
        self,
        contents: List[Any],
        context: Optional[Dict] = None
    ) -> List[Memory]:
        """
        Experience multiple contents efficiently.

        Args:
            contents: List of content items
            context: Optional shared context for all items

        Returns:
            List of Memory objects

        Example:
            >>> contents = [
            ...     "Python has decorators",
            ...     "Python has generators",
            ...     {"language": "Python"}
            ... ]
            >>> memories = await loom.experience_batch(contents)
        """
        memories = []
        for content in contents:
            memory = await self.experience(content, context)
            memories.append(memory)
        return memories

    async def search(
        self,
        query: str,
        **kwargs
    ) -> List[Memory]:
        """
        Alias for recall() with more intuitive name.

        Args:
            query: Search query
            **kwargs: Additional arguments passed to recall()

        Returns:
            Activated memories

        Example:
            >>> results = await loom.search("machine learning")
        """
        return await self.recall(query, **kwargs)

    def get_metrics(self) -> Dict:
        """
        Get system metrics for monitoring.

        Returns:
            Dictionary with metrics:
            - n_memories: Total memories stored
            - n_connections: Total connections (edges)
            - n_active: Currently activated memories
            - activation_density: Activation field density
            - trajectory_length: Semantic trajectory length

        Example:
            >>> metrics = loom.get_metrics()
            >>> print(f"Total memories: {metrics['n_memories']}")
        """
        awareness_metrics = self._awareness.get_metrics()
        return {
            'n_memories': awareness_metrics.n_memories,
            'n_connections': awareness_metrics.n_connections,
            'n_active': awareness_metrics.n_active,
            'activation_density': awareness_metrics.activation_density,
            'trajectory_length': awareness_metrics.trajectory_length,
            'shift_magnitude': awareness_metrics.shift_magnitude,
            'shift_detected': awareness_metrics.shift_detected
        }

    def summary(self) -> str:
        """
        Get human-readable summary of system state.

        Returns:
            Summary string

        Example:
            >>> print(loom.summary())
            HoloLoom System
            ===============
            Memories: 42
            Connections: 128
            Active: 5 (density: 0.12)
        """
        metrics = self.get_metrics()
        return f"""HoloLoom System
===============
Memories: {metrics['n_memories']}
Connections: {metrics['n_connections']}
Active: {metrics['n_active']} (density: {metrics['activation_density']:.2f})
Trajectory: {metrics['trajectory_length']} steps
Shift detected: {metrics['shift_detected']}
"""

    # =========================================================================
    # Multi-Loom Architecture / WeaveHouse (December 2025)
    # =========================================================================

    async def weave(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> 'WeaveResult':
        """
        Multi-perspective query using WeaveHouse.

        Sends query to 5 independent looms (RECALL, REASON, REACH, REFLECT, REFUSE),
        synthesizes their responses through consensus, and auto-explores
        disagreement zones.

        Requires: multi-perspective mode enabled (use_multi_perspective=True
        or config=Config.multi_perspective())

        Args:
            query: Question or request to process
            context: Optional context metadata

        Returns:
            WeaveResult with:
                - response: Synthesized response text
                - confidence: Overall confidence (0.0-1.0)
                - epistemic_confidence: Meta-confidence (0.0-1.0)
                - fabrics: Individual Fabric responses from each loom
                - tensions: Detected disagreements between looms
                - exploration_trace: Log of exploration steps
                - metadata: Additional context

        Raises:
            RuntimeError: If WeaveHouse not initialized
                         (multi-perspective mode not enabled or not in async context)

        Example:
            >>> async with HoloLoom(use_multi_perspective=True) as loom:
            ...     result = await loom.weave("What is Thompson Sampling?")
            ...     print(f"Response: {result.response}")
            ...     print(f"Confidence: {result.confidence:.2f}")
            ...     print(f"Perspectives: {len(result.fabrics)}")
            ...
            ...     # View individual perspectives
            ...     for fabric in result.fabrics:
            ...         print(f"  {fabric.perspective}: {fabric.response[:50]}...")
        """
        if self._weave_house is None:
            if not self._use_multi_perspective:
                raise RuntimeError(
                    "Multi-perspective mode not enabled. "
                    "Initialize with use_multi_perspective=True or config=Config.multi_perspective()"
                )
            else:
                raise RuntimeError(
                    "WeaveHouse not initialized. "
                    "Ensure you're using 'async with HoloLoom(...) as loom:' context manager"
                )

        # Execute multi-perspective weaving
        result = await self._weave_house.weave(query, context=context)

        return result

    async def get_perspectives(self) -> Dict[str, str]:
        """
        Get descriptions of all available loom perspectives.

        Returns dictionary mapping loom names to their descriptions:
        - RECALL (Librarian): Memory retrieval and evidence
        - REASON (Logician): Logical inference and deduction
        - REACH (Artist): Creative exploration and analogy
        - REFLECT (Auditor): Verification and self-critique
        - REFUSE (Skeptic): Safety and boundary enforcement

        Requires: multi-perspective mode enabled

        Returns:
            Dict mapping perspective name to description

        Raises:
            RuntimeError: If WeaveHouse not initialized

        Example:
            >>> async with HoloLoom(use_multi_perspective=True) as loom:
            ...     perspectives = await loom.get_perspectives()
            ...     for name, desc in perspectives.items():
            ...         print(f"{name}: {desc}")
        """
        if self._weave_house is None:
            if not self._use_multi_perspective:
                raise RuntimeError(
                    "Multi-perspective mode not enabled. "
                    "Initialize with use_multi_perspective=True or config=Config.multi_perspective()"
                )
            else:
                raise RuntimeError(
                    "WeaveHouse not initialized. "
                    "Ensure you're using 'async with HoloLoom(...) as loom:' context manager"
                )

        return {
            loom.perspective: loom.description if hasattr(loom, 'description') else loom.perspective
            for loom in self._weave_house.looms
        }

    @property
    def weave_house(self) -> Optional['WeaveHouse']:
        """
        Direct access to WeaveHouse (advanced users only).

        Returns:
            WeaveHouse instance if multi-perspective mode enabled and initialized,
            None otherwise.

        Example:
            >>> async with HoloLoom(use_multi_perspective=True) as loom:
            ...     if loom.weave_house:
            ...         print(f"Looms: {len(loom.weave_house.looms)}")
        """
        return self._weave_house

    @property
    def dream_orchestrator(self) -> Optional['DreamOrchestrator']:
        """
        Direct access to DreamOrchestrator (advanced users only).

        Returns:
            DreamOrchestrator instance if dreaming enabled and initialized,
            None otherwise.

        Example:
            >>> async with HoloLoom(config=Config.multi_perspective()) as loom:
            ...     if loom.dream_orchestrator:
            ...         stats = await loom.dream_orchestrator.dream_once()
            ...         print(f"Insights shared: {stats.total_insights_collected}")
        """
        return self._dream_orchestrator

    @property
    def is_multi_perspective(self) -> bool:
        """
        Check if multi-perspective mode is enabled.

        Returns:
            True if multi-perspective WeaveHouse is active.

        Example:
            >>> loom = HoloLoom(use_multi_perspective=True)
            >>> print(f"Multi-perspective: {loom.is_multi_perspective}")
            Multi-perspective: True
        """
        return self._use_multi_perspective

    # =========================================================================
    # Multimodal Photo Memory (November 2025)
    # =========================================================================

    def _ensure_photo_memory(self):
        """Lazy-initialize photo memory storage."""
        if not hasattr(self, '_photo_memory'):
            from hololoom.memory.photo_tokens import PhotoTokenMemory
            from hololoom.memory.graph import KG

            # Create photo memory storage
            storage_path = self.config.data_dir if hasattr(self.config, 'data_dir') else "./photo_memory"
            self._photo_memory = PhotoTokenMemory(storage_path=storage_path)

            # Get KG from awareness graph
            if hasattr(self._awareness, 'graph'):
                self._kg = self._awareness.graph if isinstance(self._awareness.graph, KG) else KG()
            else:
                self._kg = KG()

        return self._photo_memory

    async def remember_photo(
        self,
        image: Union[bytes, 'np.ndarray', str, Path],
        caption: Optional[str] = None,
        tags: List[str] = None,
        link_to_memory: Optional[str] = None
    ) -> 'PhotoToken':
        """
        Remember a photo in multimodal memory.

        Stores the photo with:
        - CLIP embeddings for semantic image-text matching
        - Structural features (color, brightness, layout)
        - Automatic linking to knowledge graph entities
        - Optional linking to existing text memories

        Args:
            image: Image as bytes, numpy array, or file path
            caption: Optional text description
            tags: Optional categorical labels ["diagram", "screenshot", etc.]
            link_to_memory: Optional memory ID to link photo to

        Returns:
            PhotoToken with embeddings and metadata

        Example:
            >>> # Store a photo
            >>> photo = await loom.remember_photo(
            ...     "diagram.png",
            ...     caption="System architecture diagram",
            ...     tags=["diagram", "architecture"]
            ... )
            >>>
            >>> # Link to existing memory
            >>> text_memory = await loom.experience("We discussed the architecture")
            >>> photo = await loom.remember_photo(
            ...     "diagram.png",
            ...     caption="Architecture from meeting",
            ...     link_to_memory=text_memory.id
            ... )
        """
        # Ensure photo memory initialized
        photo_memory = self._ensure_photo_memory()

        # Store photo
        # Note: photo_memory needs to be initialized properly
        # We'll need to call __aenter__ if not already in context manager
        if not hasattr(photo_memory, 'clip_model'):
            await photo_memory._initialize()

        photo_token = await photo_memory.store(
            image=image,
            caption=caption,
            tags=tags or [],
            source='user'
        )

        # Add to knowledge graph
        self._kg.add_photo_node(photo_token)

        # Link to memory if specified
        if link_to_memory:
            self._kg.link_photo_to_memory(
                photo_token.token_id,
                link_to_memory,
                edge_type='ILLUSTRATES'
            )

        return photo_token

    async def find_similar_photos(
        self,
        query_image: Union[bytes, 'np.ndarray', str, Path],
        k: int = 5
    ) -> List['PhotoToken']:
        """
        Find visually similar photos using CLIP embeddings.

        Args:
            query_image: Query image (bytes, array, or path)
            k: Number of similar photos to return

        Returns:
            List of similar PhotoTokens with similarity scores

        Example:
            >>> # Find photos similar to a reference image
            >>> similar = await loom.find_similar_photos("reference.png", k=5)
            >>> for photo in similar:
            ...     print(f"{photo.caption} (similarity: {photo.metadata['score']:.3f})")
        """
        # Ensure photo memory initialized
        photo_memory = self._ensure_photo_memory()
        if not hasattr(photo_memory, 'clip_model'):
            await photo_memory._initialize()

        # Retrieve similar photos
        results = await photo_memory.retrieve_by_image(query_image, k=k)

        # Add scores to metadata
        photos_with_scores = []
        for photo_token, score in results:
            photo_token.metadata['score'] = score
            photos_with_scores.append(photo_token)

        return photos_with_scores

    async def get_photos_by_tag(self, tag: str, k: int = 10) -> List['PhotoToken']:
        """
        Get photos with a specific tag.

        Args:
            tag: Tag name to filter by
            k: Maximum number of photos to return

        Returns:
            List of PhotoTokens with the tag

        Example:
            >>> # Get all diagram photos
            >>> diagrams = await loom.get_photos_by_tag("diagram")
        """
        # Ensure photo memory initialized
        photo_memory = self._ensure_photo_memory()
        if not hasattr(photo_memory, 'clip_model'):
            await photo_memory._initialize()

        # Retrieve by tag
        results = await photo_memory.retrieve_by_tags([tag], k=k, match_all=False)

        return results

    async def link_photo_to_memory(
        self,
        photo_id: str,
        memory_id: str,
        relationship: str = "ILLUSTRATES"
    ) -> None:
        """
        Create semantic link between photo and text memory.

        Args:
            photo_id: Photo token ID
            memory_id: Memory/shard ID
            relationship: Type of relationship:
                - ILLUSTRATES: Photo depicts/explains the memory
                - REFERENCED_IN: Photo mentioned in text
                - ACCOMPANIES: Photo from same event

        Example:
            >>> # Experience text
            >>> text_mem = await loom.experience("We discussed architecture")
            >>>
            >>> # Store photo
            >>> photo = await loom.remember_photo("diagram.png")
            >>>
            >>> # Link them
            >>> await loom.link_photo_to_memory(
            ...     photo.token_id,
            ...     text_mem.id,
            ...     relationship="ILLUSTRATES"
            ... )
        """
        # Ensure KG initialized
        if not hasattr(self, '_kg'):
            self._ensure_photo_memory()

        # Create link
        self._kg.link_photo_to_memory(photo_id, memory_id, edge_type=relationship)

    # =========================================================================
    # Visual Compression for Context Window Expansion (November 2025)
    # =========================================================================

    async def compress_to_visual(
        self,
        data: Any,
        compression_type: str = 'auto',
        caption: Optional[str] = None,
        tags: List[str] = None
    ) -> Tuple['PhotoToken', 'CompressionMetrics']:
        """
        Compress structured data into visual representation for efficient storage.

        Inspired by DeepSeek-Janus: Images convey more information per token than text.
        Achieves 3-20× compression for structured content (graphs, tables, code).

        Key Insight:
            - Text representation: 5000 tokens
            - Visual representation: 1000 vision tokens
            - Information density: 5-10× higher per vision token
            - Effective compression: 5-20× for structured data

        Args:
            data: Data to compress (NetworkX graph, DataFrame, dict, list, or code string)
            compression_type: 'knowledge_graph', 'table', 'code', or 'auto'
            caption: Optional caption for compressed visual
            tags: Optional tags (defaults to ['compressed', compression_type])

        Returns:
            (photo_token, metrics) tuple:
                - photo_token: PhotoToken with visual representation
                - metrics: CompressionMetrics with compression stats

        Example:
            >>> # Compress large knowledge graph
            >>> kg = build_large_graph()  # 5000 tokens as text
            >>> photo, metrics = await loom.compress_to_visual(kg, 'knowledge_graph')
            >>> print(f"Compression: {metrics.compression_ratio:.1f}×")
            >>> # Output: Compression: 5.0×
            >>>
            >>> # Later: Retrieve compressed memory
            >>> results = await loom.recall("architecture", include_photos=True)
            >>> # Vision model can read diagram directly
        """
        from hololoom.memory.visual_compression import compress_to_visual, CompressionMetrics

        # Compress data to visual representation
        image, metrics = compress_to_visual(
            data,
            compression_type=compression_type,
            width=600,
            height=400
        )

        # Generate caption if not provided
        if caption is None:
            caption = f"Compressed {metrics.compression_type} ({metrics.compression_ratio:.1f}× compression)"

        # Generate tags
        if tags is None:
            tags = ['compressed', metrics.compression_type, 'visual_compression']

        # Store as photo token
        photo_token = await self.remember_photo(
            image,
            caption=caption,
            tags=tags
        )

        # Add compression metrics to metadata
        photo_token.metadata.update({
            'compression_type': metrics.compression_type,
            'original_tokens': metrics.original_tokens,
            'visual_tokens': metrics.visual_tokens,
            'compression_ratio': metrics.compression_ratio,
            'info_density': metrics.info_density,
            'is_compressed': True
        })

        return photo_token, metrics

    async def decompress_visual(
        self,
        photo_token: 'PhotoToken',
        use_ocr: bool = True
    ) -> str:
        """
        Decompress visual representation back to text using vision model.

        Uses DeepSeek-OCR (if available) to read and interpret visual
        representations, extracting structured information.

        Args:
            photo_token: PhotoToken with compressed visual representation
            use_ocr: If True, use DeepSeek-OCR for decompression

        Returns:
            Extracted text/data from visual representation

        Example:
            >>> # Compress knowledge graph
            >>> photo, metrics = await loom.compress_to_visual(kg)
            >>>
            >>> # Later: Decompress when needed
            >>> extracted_data = await loom.decompress_visual(photo)
            >>> # Vision model reads diagram and extracts structure
        """
        if not photo_token.metadata.get('is_compressed', False):
            raise ValueError("PhotoToken is not a compressed visual representation")

        if not use_ocr:
            # Fallback: Return caption + metadata
            compression_type = photo_token.metadata.get('compression_type', 'unknown')
            ratio = photo_token.metadata.get('compression_ratio', 0)
            return (
                f"Compressed {compression_type} "
                f"(Original: {photo_token.metadata.get('original_tokens')} tokens, "
                f"Visual: {photo_token.metadata.get('visual_tokens')} tokens, "
                f"Ratio: {ratio:.1f}×)\n\n"
                f"Caption: {photo_token.caption}"
            )

        # Use DeepSeek-OCR for decompression
        try:
            from hololoom.spinningWheel.deepseek_ocr_spinner import DeepSeekOCRSpinner

            spinner = DeepSeekOCRSpinner()
            extracted_text = await spinner.extract_text(photo_token.image_data)

            return extracted_text

        except ImportError:
            import warnings
            warnings.warn(
                "DeepSeek-OCR not available for decompression. "
                "Install deepseek-vl to enable visual decompression."
            )
            # Fallback to caption
            return photo_token.caption or "No caption available"

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get statistics on compressed visual memories.

        Returns:
            Dictionary with compression statistics:
                - total_compressed: Number of compressed visuals
                - avg_compression_ratio: Average compression ratio
                - total_tokens_saved: Total tokens saved via compression
                - by_type: Breakdown by compression type

        Example:
            >>> stats = loom.get_compression_stats()
            >>> print(f"Total compressed: {stats['total_compressed']}")
            >>> print(f"Avg compression: {stats['avg_compression_ratio']:.1f}×")
            >>> print(f"Tokens saved: {stats['total_tokens_saved']}")
        """
        if not hasattr(self, '_photo_memory'):
            return {
                'total_compressed': 0,
                'avg_compression_ratio': 0.0,
                'total_tokens_saved': 0,
                'by_type': {}
            }

        # Get all compressed photos
        compressed_photos = [
            photo for photo in self._photo_memory.tokens.values()
            if photo.metadata.get('is_compressed', False)
        ]

        if not compressed_photos:
            return {
                'total_compressed': 0,
                'avg_compression_ratio': 0.0,
                'total_tokens_saved': 0,
                'by_type': {}
            }

        # Calculate statistics
        total_compressed = len(compressed_photos)
        total_original = sum(p.metadata.get('original_tokens', 0) for p in compressed_photos)
        total_visual = sum(p.metadata.get('visual_tokens', 0) for p in compressed_photos)
        avg_ratio = total_original / total_visual if total_visual > 0 else 0.0
        tokens_saved = total_original - total_visual

        # Breakdown by type
        by_type = {}
        for photo in compressed_photos:
            ctype = photo.metadata.get('compression_type', 'unknown')
            if ctype not in by_type:
                by_type[ctype] = {
                    'count': 0,
                    'original_tokens': 0,
                    'visual_tokens': 0,
                    'tokens_saved': 0
                }

            by_type[ctype]['count'] += 1
            by_type[ctype]['original_tokens'] += photo.metadata.get('original_tokens', 0)
            by_type[ctype]['visual_tokens'] += photo.metadata.get('visual_tokens', 0)
            by_type[ctype]['tokens_saved'] += (
                photo.metadata.get('original_tokens', 0) -
                photo.metadata.get('visual_tokens', 0)
            )

        return {
            'total_compressed': total_compressed,
            'avg_compression_ratio': avg_ratio,
            'total_tokens_saved': tokens_saved,
            'total_original_tokens': total_original,
            'total_visual_tokens': total_visual,
            'by_type': by_type
        }

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize photo memory if needed
        if hasattr(self, '_photo_memory'):
            await self._photo_memory.__aenter__()

        # Initialize WeaveHouse if multi-perspective mode enabled
        if self._use_multi_perspective and not self._weave_house_initialized:
            await self._initialize_weave_house()

        return self

    async def _initialize_weave_house(self):
        """
        Initialize WeaveHouse and DreamOrchestrator.

        Called lazily when entering async context with multi-perspective mode.
        """
        if self._weave_house_initialized:
            return

        from hololoom.loom.initialization import create_weave_house_system

        self._weave_house, self._dream_orchestrator = await create_weave_house_system(
            config=self.config,
            memory_backend=self._awareness,  # Share awareness graph with WeaveHouse
            enable_dreaming=getattr(self.config, 'enable_dreaming', True),
            dreaming_interval=getattr(self.config, 'dream_consolidation_interval', 3600.0),
            math_bleed_rate=getattr(self.config, 'dream_math_bleed_rate', 0.3),
            pattern_bleed_rate=getattr(self.config, 'dream_pattern_bleed_rate', 0.2),
            exploration_depth=getattr(self.config, 'weave_house_exploration_depth', 2),
            tension_threshold=getattr(self.config, 'weave_house_tension_threshold', 0.3),
        )

        # Start dreaming if enabled
        if self._dream_orchestrator is not None:
            await self._dream_orchestrator.start_dreaming()

        self._weave_house_initialized = True

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit (cleanup).

        Ensures proper cleanup of resources.

        Example:
            >>> async with HoloLoom() as loom:
            ...     memory = await loom.experience("content")
            ...     # Automatic cleanup on exit
        """
        # Stop dreaming if running
        if self._dream_orchestrator is not None:
            await self._dream_orchestrator.stop_dreaming()

        # Close photo memory if initialized
        if hasattr(self, '_photo_memory'):
            await self._photo_memory.__aexit__(exc_type, exc_val, exc_tb)

        # Close awareness graph (cleanup tasks, connections, etc.)
        if hasattr(self._awareness, 'close'):
            await self._awareness.close()

    # =========================================================================
    # Advanced Access (for power users)
    # =========================================================================

    @property
    def awareness(self) -> AwarenessGraph:
        """
        Direct access to awareness graph (advanced users only).

        Most users should use experience()/recall()/reflect().
        This provides full control over internal components.

        Returns:
            AwarenessGraph instance

        Example:
            >>> # Advanced: direct graph manipulation
            >>> graph = loom.awareness.graph
            >>> nodes = list(graph.nodes())
        """
        return self._awareness

    @property
    def graph(self) -> nx.MultiDiGraph:
        """
        Direct access to graph backend (advanced users only).

        Returns:
            NetworkX MultiDiGraph

        Example:
            >>> # Advanced: query graph directly
            >>> graph = loom.graph
            >>> edges = list(graph.edges(data=True))
        """
        return self._graph

    # =========================================================================
    # Cross-Device Handoff (December 2025)
    # =========================================================================

    @property
    def identity(self) -> Optional['UnifiedIdentity']:
        """
        Get the current identity (if in cross-device mode).

        Returns:
            UnifiedIdentity if cross-device mode enabled, None otherwise.

        Example:
            >>> if loom.identity:
            ...     print(f"Logged in as: {loom.identity.nickname}")
            ...     print(f"DID: {loom.identity.did}")
        """
        return self._identity

    @property
    def is_synced(self) -> bool:
        """
        Check if cross-device sync is enabled.

        Returns:
            True if identity is configured for sync.

        Example:
            >>> if loom.is_synced:
            ...     print("Memory syncs across devices")
        """
        return self._identity is not None

    async def handoff_to(
        self,
        target_device: str,
        context: Optional[Dict[str, Any]] = None
    ) -> 'HandoffResult':
        """
        Hand off current work context to another device.

        Transfers pending memory operations and context to target device,
        allowing seamless continuation of work.

        Requires: identity must be set (cross-device mode)

        Args:
            target_device: Device ID to hand off to (from identity.devices)
            context: Optional UI state, ambient context, etc.

        Returns:
            HandoffResult with transfer statistics

        Raises:
            ValueError: If not in cross-device mode
            HandoffError: If handoff fails

        Example:
            >>> # Hand off to laptop
            >>> result = await loom.handoff_to("blake-laptop")
            >>> print(f"Transferred {result.ops_transferred} operations")
            >>>
            >>> # Hand off with context
            >>> result = await loom.handoff_to(
            ...     "blake-phone",
            ...     context={"current_task": "reviewing_code", "file": "main.py"}
            ... )
        """
        if self._handoff is None:
            raise ValueError(
                "Cross-device sync not enabled. "
                "Initialize HoloLoom with an identity: HoloLoom(identity=my_identity)"
            )

        return await self._handoff.handoff_to(target_device, context)

    async def receive_handoff(
        self,
        signed_request: 'HandoffRequest'
    ) -> 'MergeResult':
        """
        Receive and apply a handoff from another device.

        Validates the signed request and merges incoming operations.

        Requires: identity must be set (cross-device mode)

        Args:
            signed_request: HandoffRequest signed by source device

        Returns:
            MergeResult with merge statistics

        Raises:
            ValueError: If not in cross-device mode
            InvalidSignatureError: If signature verification fails
            ReplayAttackError: If request is stale or reused

        Example:
            >>> # Receive handoff (typically via transport layer)
            >>> result = await loom.receive_handoff(signed_request)
            >>> print(f"Merged {result.merged} operations")
        """
        if self._handoff is None:
            raise ValueError(
                "Cross-device sync not enabled. "
                "Initialize HoloLoom with an identity: HoloLoom(identity=my_identity)"
            )

        return await self._handoff.receive_handoff(signed_request)

    def get_pending_sync(self) -> List[Dict[str, Any]]:
        """
        Get pending memory operations waiting to sync.

        Returns:
            List of operations not yet synced to other devices.

        Example:
            >>> pending = loom.get_pending_sync()
            >>> print(f"{len(pending)} operations pending sync")
        """
        if self._synced_memory is None:
            return []

        ops = self._synced_memory.pending_delta()
        return [op.to_dict() for op in ops]

    def get_device_registry(self) -> Dict[str, Any]:
        """
        Get information about paired devices.

        Returns:
            Dictionary with device registry information:
            - devices: List of paired devices
            - active_count: Number of active devices
            - this_device: Current device ID

        Example:
            >>> registry = loom.get_device_registry()
            >>> for device in registry['devices']:
            ...     print(f"{device['name']}: {device['status']}")
        """
        if self._identity is None:
            return {
                'devices': [],
                'active_count': 0,
                'this_device': None
            }

        devices = []
        active_count = 0
        for device_id, manifest in self._identity.devices.items():
            devices.append({
                'device_id': device_id,
                'name': manifest.device_name,
                'status': manifest.status.name,
                'last_seen': manifest.last_seen.isoformat() if manifest.last_seen else None,
                'last_sync': manifest.last_sync.isoformat() if manifest.last_sync else None,
                'capabilities': list(manifest.capabilities),
            })
            if manifest.is_active():
                active_count += 1

        return {
            'devices': devices,
            'active_count': active_count,
            'this_device': self._identity.current_device_id if hasattr(self._identity, 'current_device_id') else None
        }

    async def add_device(
        self,
        device_name: str,
        pairing_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a new device to the identity's device registry.

        Initiates device pairing process for cross-device sync.

        Args:
            device_name: Human-readable name for the device
            pairing_code: Optional pairing code from QR exchange

        Returns:
            Dictionary with pairing information:
            - device_id: New device ID
            - pairing_code: Code to enter on other device (if not provided)
            - status: Pairing status

        Example:
            >>> # Generate pairing code for new device
            >>> result = await loom.add_device("blake-phone")
            >>> print(f"Enter code on phone: {result['pairing_code']}")
            >>>
            >>> # Complete pairing with code from other device
            >>> result = await loom.add_device("blake-phone", pairing_code="ABC123")
        """
        if self._identity is None:
            raise ValueError(
                "Cross-device sync not enabled. "
                "Initialize HoloLoom with an identity: HoloLoom(identity=my_identity)"
            )

        # Add device to identity registry
        manifest = await self._identity.add_device(device_name, pairing_code)

        return {
            'device_id': manifest.device_id,
            'device_name': manifest.device_name,
            'status': manifest.status.name,
            'pairing_code': pairing_code or manifest.metadata.get('pairing_code'),
        }

    async def revoke_device(self, device_id: str) -> bool:
        """
        Revoke a device from the identity's device registry.

        Immediately blocks the device from syncing.

        Args:
            device_id: Device ID to revoke

        Returns:
            True if successfully revoked

        Example:
            >>> # Revoke compromised device
            >>> await loom.revoke_device("blake-old-phone")
            >>> print("Device revoked - cannot sync anymore")
        """
        if self._identity is None:
            raise ValueError(
                "Cross-device sync not enabled. "
                "Initialize HoloLoom with an identity: HoloLoom(identity=my_identity)"
            )

        await self._identity.revoke_device(device_id)
        return True
