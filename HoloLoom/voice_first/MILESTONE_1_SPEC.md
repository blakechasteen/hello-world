# Voice-First UX Layer - Milestone 1 Specification

**Feature**: Thread Branching, Merging, and Auto-Summarization
**Timeline**: 6 weeks (Week 1-6 post-MVP)
**Status**: 📋 Specification Phase
**Author**: Voice-First UX Team
**Date**: November 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Thread Branching](#thread-branching)
3. [Thread Merging](#thread-merging)
4. [Auto-Summarization](#auto-summarization)
5. [Integration Points](#integration-points)
6. [Testing Strategy](#testing-strategy)
7. [Implementation Timeline](#implementation-timeline)
8. [Voice Commands Reference](#voice-commands-reference)
9. [Performance Targets](#performance-targets)
10. [Dependencies](#dependencies)
11. [Success Criteria](#success-criteria)
12. [Future Enhancements](#future-enhancements)

---

## Executive Summary

### Milestone 1 Goals

**Problem**: Tab-switching and context-switching create cognitive overhead. Users lose focus when exploring tangential ideas.

**Solution**: Natural conversation flow with thread branching - like human dialogue where you can say "wait, that reminds me..." and come back later.

**Key Features**:
1. **Thread Branching** - Fork conversations to explore ideas without losing context
2. **Thread Merging** - Synthesize insights from parallel exploration
3. **Auto-Summarization** - LLM-generated summaries for quick context recovery

### Timeline

- **Week 1-2**: Thread Branching implementation
- **Week 3-4**: Thread Merging implementation
- **Week 5-6**: Auto-Summarization + integration + polish

### Success Criteria

✅ Users can fork threads by voice with <100ms latency
✅ Context inheritance works (last 30s + entities preserved)
✅ Merging combines insights from multiple threads
✅ Summaries are accurate and helpful (>80% user satisfaction)
✅ Zero data loss on fork/merge operations
✅ All 37 tests passing (15 branching + 12 merging + 10 summarization)

### Key Deliverables

**Code** (~600 lines):
- `thread/thread_branching.py` (220 lines)
- `thread/thread_merging.py` (200 lines)
- `thread/thread_summarizer.py` (180 lines)

**Tests** (~450 lines):
- `tests/test_thread_branching.py` (180 lines)
- `tests/test_thread_merging.py` (150 lines)
- `tests/test_thread_summarizer.py` (120 lines)

**Documentation**:
- This specification (~2,000 lines)
- Updated README.md
- Integration guide

---

## Thread Branching

### Overview

**Feature**: "fork this into a new idea about [topic]"

**Purpose**: Allow users to explore tangential ideas without losing the main conversation thread. Like saying "wait, that reminds me..." in a conversation.

**User Experience**:
```
[Main thread: Orchard Planning]
User: "how should I space apple trees?"
Elle: "12-15 feet for dwarf varieties, 18-25 for standard..."

User: "wait, this reminds me - fork this into biochar idea"
Elle: "Created 'biochar' branch from 'orchard planning'.
      Inherited context: apple spacing, soil health, nutrient cycles.
      You're now in 'biochar' thread."

[Biochar thread is now active - Main thread in background]
User: "how does biochar affect soil pH?"
Elle: [answers in biochar context]

User: "back to orchard"
Elle: "Switched to 'orchard planning'. You were discussing apple tree spacing."

[Orchard thread picks up exactly where it left off]
```

### Technical Specification

#### Data Model

```python
@dataclass
class ThreadBranch:
    """Represents a branched thread"""
    branch_id: str                    # Unique thread ID
    parent_id: str                    # Parent thread ID
    branch_name: str                  # User-provided name
    fork_timestamp: datetime          # When fork occurred
    inherited_context: List[Message]  # Messages from parent
    inherited_entities: List[str]     # Entities from parent
    metadata: Dict[str, Any]          # Additional metadata

    # YarnGraph integration
    graph_edge: KGEdge                # BRANCHED_FROM edge

@dataclass
class ContextWindow:
    """Context inherited from parent thread"""
    messages: List[Message]           # Last N messages
    entities: List[str]               # Entities mentioned
    topic: str                        # Thread topic
    window_seconds: int               # Time window (default: 30s)
    total_messages: int               # Total in parent
```

#### Core Algorithm

```python
# HoloLoom/voice_first/thread/thread_branching.py

class ThreadBrancher:
    """
    Handles thread branching with context inheritance.

    Features:
    - Inherit last N seconds of conversation
    - Extract and preserve entities
    - Create YarnGraph relationships
    - Preserve conversation style/persona
    """

    def __init__(
        self,
        thread_manager: ThreadManager,
        yarn_graph: KG,
        memory_window_seconds: int = 30
    ):
        """
        Initialize brancher.

        Args:
            thread_manager: Elle's thread manager
            yarn_graph: HoloLoom knowledge graph
            memory_window_seconds: Context window (default: 30s)
        """
        self.thread_manager = thread_manager
        self.graph = yarn_graph
        self.memory_window = memory_window_seconds

    async def fork_thread(
        self,
        parent_thread_id: str,
        branch_name: str,
        custom_context: Optional[List[Message]] = None
    ) -> ThreadBranch:
        """
        Fork a thread into new branch with context inheritance.

        Algorithm:
        1. Validate parent thread exists
        2. Extract recent context (last N seconds)
        3. Extract entities from context
        4. Create new thread with inherited data
        5. Add BRANCHED_FROM edge to YarnGraph
        6. Return branch metadata

        Args:
            parent_thread_id: ID of thread to fork
            branch_name: Name for new branch
            custom_context: Optional custom context (overrides time window)

        Returns:
            ThreadBranch with all metadata

        Raises:
            ThreadNotFoundError: If parent doesn't exist
            BranchingError: If fork fails
        """
        # 1. Get parent thread
        parent = self.thread_manager.get_thread(parent_thread_id)
        if parent is None:
            raise ThreadNotFoundError(f"Thread not found: {parent_thread_id}")

        # 2. Extract recent context
        if custom_context is not None:
            context_messages = custom_context
        else:
            context_messages = self._extract_recent_context(
                parent,
                window_seconds=self.memory_window
            )

        # 3. Extract entities
        entities = self._extract_entities(context_messages)

        # 4. Create new thread
        branch = self.thread_manager.create_thread(
            name=branch_name,
            topic=branch_name  # Could be more sophisticated
        )

        # 5. Add inherited messages
        for msg in context_messages:
            branch.messages.append(msg.copy())

        # 6. Add metadata
        branch.metadata['parent_id'] = parent_thread_id
        branch.metadata['fork_timestamp'] = datetime.now().isoformat()
        branch.metadata['inherited_entities'] = entities
        branch.metadata['context_window_seconds'] = self.memory_window
        branch.metadata['is_branch'] = True

        # 7. Create YarnGraph edge
        graph_edge = await self._create_graph_relationship(
            parent_id=parent_thread_id,
            branch_id=branch.id,
            entities=entities
        )

        # 8. Return branch metadata
        return ThreadBranch(
            branch_id=branch.id,
            parent_id=parent_thread_id,
            branch_name=branch_name,
            fork_timestamp=datetime.now(),
            inherited_context=context_messages,
            inherited_entities=entities,
            metadata=branch.metadata,
            graph_edge=graph_edge
        )

    def _extract_recent_context(
        self,
        thread: Thread,
        window_seconds: int
    ) -> List[Message]:
        """
        Extract messages from last N seconds.

        Args:
            thread: Source thread
            window_seconds: Time window

        Returns:
            List of messages within window
        """
        cutoff_time = datetime.now().timestamp() - window_seconds

        recent_messages = [
            msg for msg in thread.messages
            if msg.timestamp >= cutoff_time
        ]

        # Always include at least 1 message (the trigger)
        if len(recent_messages) == 0 and len(thread.messages) > 0:
            recent_messages = [thread.messages[-1]]

        return recent_messages

    def _extract_entities(self, messages: List[Message]) -> List[str]:
        """
        Extract entities from messages.

        Uses simple keyword extraction for MVP.
        Could be enhanced with NER in future.

        Args:
            messages: Messages to analyze

        Returns:
            List of entity strings
        """
        # Simple entity extraction (MVP)
        # Future: Use spaCy NER or LLM extraction

        text = " ".join([msg.content for msg in messages])

        # Extract nouns (simple approach)
        # For MVP, just extract capitalized words and common keywords
        import re

        # Capitalized words (potential entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)

        # Common domain keywords
        keywords = re.findall(
            r'\b(apple|biochar|compost|soil|tree|orchard|plant|garden)\b',
            text.lower()
        )

        entities = list(set(capitalized + keywords))
        return entities

    async def _create_graph_relationship(
        self,
        parent_id: str,
        branch_id: str,
        entities: List[str]
    ) -> KGEdge:
        """
        Create BRANCHED_FROM edge in YarnGraph.

        Args:
            parent_id: Parent thread ID
            branch_id: Branch thread ID
            entities: Shared entities

        Returns:
            Created KGEdge
        """
        edge = KGEdge(
            source=branch_id,
            target=parent_id,
            relation="BRANCHED_FROM",
            weight=1.0,
            metadata={
                'fork_timestamp': datetime.now().isoformat(),
                'shared_entities': entities,
                'relationship_type': 'thread_branch'
            }
        )

        await self.graph.add_edge(edge)
        return edge

    def get_branch_tree(self, root_thread_id: str) -> Dict:
        """
        Get complete branch tree for a thread.

        Args:
            root_thread_id: Root thread to analyze

        Returns:
            Tree structure showing all branches
        """
        # Query YarnGraph for BRANCHED_FROM edges
        branches = self.graph.get_edges_by_relation("BRANCHED_FROM")

        # Build tree
        tree = {
            'root': root_thread_id,
            'branches': []
        }

        for edge in branches:
            if edge.target == root_thread_id:  # This is a direct branch
                tree['branches'].append({
                    'id': edge.source,
                    'name': self.thread_manager.get_thread(edge.source).name,
                    'fork_time': edge.metadata.get('fork_timestamp'),
                    'entities': edge.metadata.get('shared_entities', [])
                })

        return tree
```

#### Voice Command Patterns

**Patterns to add to VoiceGrammar**:

```python
# Thread branching patterns (Milestone 1)
THREAD_BRANCHING_PATTERNS = [
    (re.compile(r'^(fork|branch|split) this into (?P<branch_name>.+)$', re.I), 0.95),
    (re.compile(r'^(this (is|seems) important|i have an idea)[.,;:]? (create|start) (a )?new (thread|branch)( for)?( (?P<branch_name>.+))?$', re.I), 0.85),
    (re.compile(r'^new branch:?\s*(?P<branch_name>.+)$', re.I), 0.90),
    (re.compile(r'^wait,? (that reminds me|i want to explore) - (?P<branch_name>.+)$', re.I), 0.85),
    (re.compile(r'^spin off (a )?branch (for|about) (?P<branch_name>.+)$', re.I), 0.90),
]

# Branch navigation patterns
BRANCH_NAVIGATION_PATTERNS = [
    (re.compile(r'^(show|list) (my )?branches?(\?)?$', re.I), 0.95),
    (re.compile(r'^what branches (do i have|exist)(\?)?$', re.I), 0.90),
    (re.compile(r'^show (me )?the thread tree$', re.I), 0.95),
]
```

#### Integration with VoiceRouter

```python
# In VoiceRouter._handle_thread_command()

async def _handle_thread_command(self, intent: CommandIntent, context: Dict) -> str:
    """Handle thread management commands."""

    # ... existing thread commands ...

    # NEW: Thread branching
    if intent.command_type == CommandType.THREAD_BRANCH:
        branch_name = intent.params.get('branch_name', 'new idea')

        # Get active thread
        active_thread_id = context.get('active_thread_id')
        if not active_thread_id:
            return "No active thread to branch from. Create a thread first."

        # Create branch
        brancher = ThreadBrancher(
            thread_manager=self.elle_handler.thread_manager,
            yarn_graph=self.hololoom_handler.orchestrator.kg_store,
            memory_window_seconds=30
        )

        branch = await brancher.fork_thread(
            parent_thread_id=active_thread_id,
            branch_name=branch_name
        )

        # Switch to branch
        self.elle_handler.thread_manager.set_active_thread(branch.branch_id)

        return (
            f"Created '{branch_name}' branch from current thread. "
            f"Inherited {len(branch.inherited_context)} messages and "
            f"{len(branch.inherited_entities)} entities. "
            f"You're now in '{branch_name}' thread."
        )
```

### Testing Strategy

**Test File**: `HoloLoom/voice_first/tests/test_thread_branching.py`

**Test Cases** (15 total):

```python
class TestThreadBranching:
    """Test thread branching functionality"""

    def test_fork_basic(self):
        """Test basic fork operation"""
        # Create parent thread with 5 messages
        # Fork to new branch
        # Verify branch created
        # Verify context inherited

    def test_fork_context_inheritance_time_window(self):
        """Test that only recent messages are inherited"""
        # Create thread with messages at t=0, t=20, t=40, t=60
        # Fork at t=60 with 30s window
        # Verify only t=40 and t=60 messages inherited

    def test_fork_entity_extraction(self):
        """Test entity extraction from context"""
        # Create thread discussing "apple trees" and "biochar"
        # Fork
        # Verify entities extracted: ["apple", "biochar", "trees"]

    def test_fork_graph_edge_creation(self):
        """Test BRANCHED_FROM edge in YarnGraph"""
        # Fork thread
        # Query graph for BRANCHED_FROM edge
        # Verify edge exists with correct metadata

    def test_fork_metadata_preservation(self):
        """Test metadata is preserved"""
        # Fork thread
        # Verify parent_id, fork_timestamp, entities in metadata

    def test_fork_from_nonexistent_thread(self):
        """Test error handling for invalid parent"""
        # Attempt to fork from non-existent thread
        # Verify ThreadNotFoundError raised

    def test_fork_with_custom_context(self):
        """Test custom context override"""
        # Fork with custom message list
        # Verify custom messages used instead of time window

    def test_fork_preserves_parent_thread(self):
        """Test parent thread unchanged"""
        # Create parent with 10 messages
        # Fork
        # Verify parent still has 10 messages

    def test_fork_multiple_branches(self):
        """Test multiple forks from same parent"""
        # Create parent
        # Fork to branch1
        # Fork to branch2
        # Verify both branches exist with different IDs

    def test_fork_nested_branches(self):
        """Test forking from a branch (nested)"""
        # Fork parent → branch1
        # Fork branch1 → branch2
        # Verify branch2 has branch1 as parent

    def test_get_branch_tree(self):
        """Test branch tree visualization"""
        # Create parent with 2 branches
        # Get branch tree
        # Verify tree structure correct

    def test_fork_voice_command_parsing(self):
        """Test voice command detection"""
        # Test "fork this into biochar"
        # Test "wait, that reminds me - composting"
        # Verify branch_name extracted correctly

    def test_fork_performance(self):
        """Test fork operation latency"""
        # Time fork operation
        # Verify <100ms for typical thread (10 messages)

    def test_fork_with_empty_parent(self):
        """Test forking empty thread"""
        # Create thread with 0 messages
        # Attempt fork
        # Verify graceful handling (at least 1 message inherited)

    def test_fork_integration_with_router(self):
        """Test end-to-end fork via VoiceRouter"""
        # Send "fork this into testing" via router
        # Verify branch created
        # Verify active thread switched
```

---

## Thread Merging

### Overview

**Feature**: "merge [thread1] and [thread2] into [target]"

**Purpose**: Combine insights from parallel exploration paths. Like synthesizing notes from multiple research sessions.

**User Experience**:
```
[User has 3 threads: main, biochar, cover_crops]

User: "list my threads"
Elle: "You have 3 threads:
       • main (Research - 10 messages)
       • biochar (3 messages)
       • cover_crops (5 messages)"

User: "merge biochar and cover_crops into main"
Elle: "Merging... synthesizing insights...

      Merged 'biochar' and 'cover_crops' into 'main'.

      Key insights:
      • Biochar increases carbon sequestration
      • Cover crops provide nitrogen fixing
      • Both improve soil structure

      8 messages added to 'main' thread."

[Main thread now contains synthesized insights]
```

### Technical Specification

#### Data Model

```python
class MergeStrategy(Enum):
    """Merge strategies"""
    APPEND = "append"                  # Simple concatenation
    SYNTHESIZE = "synthesize"          # LLM-generated synthesis
    PRESERVE_ALL = "preserve_all"      # Keep all messages, deduplicate

@dataclass
class MergeResult:
    """Result of merge operation"""
    target_thread_id: str              # Thread that received merge
    source_thread_ids: List[str]       # Threads that were merged in
    strategy: MergeStrategy            # Strategy used
    messages_added: int                # Count of messages added
    synthesis: Optional[str]           # LLM synthesis (if SYNTHESIZE)
    merged_timestamp: datetime         # When merge occurred
    conflicts: List[str]               # Any conflicts detected
    metadata: Dict[str, Any]           # Additional metadata
```

#### Core Algorithm

```python
# HoloLoom/voice_first/thread/thread_merging.py

class ThreadMerger:
    """
    Handles thread merging with multiple strategies.

    Strategies:
    - APPEND: Concatenate all messages chronologically
    - SYNTHESIZE: Use LLM to create synthesis summary
    - PRESERVE_ALL: Keep all messages, deduplicate by content
    """

    def __init__(
        self,
        thread_manager: ThreadManager,
        yarn_graph: KG,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize merger.

        Args:
            thread_manager: Elle's thread manager
            yarn_graph: HoloLoom knowledge graph
            llm_client: LLM for synthesis (optional, required for SYNTHESIZE)
        """
        self.thread_manager = thread_manager
        self.graph = yarn_graph
        self.llm_client = llm_client

    async def merge_threads(
        self,
        target_thread_id: str,
        source_thread_ids: List[str],
        strategy: MergeStrategy = MergeStrategy.SYNTHESIZE
    ) -> MergeResult:
        """
        Merge multiple threads into target thread.

        Args:
            target_thread_id: Thread to merge into
            source_thread_ids: Threads to merge from
            strategy: Merge strategy to use

        Returns:
            MergeResult with details

        Raises:
            ThreadNotFoundError: If any thread doesn't exist
            MergeError: If merge fails
        """
        # 1. Validate threads exist
        target = self.thread_manager.get_thread(target_thread_id)
        if target is None:
            raise ThreadNotFoundError(f"Target not found: {target_thread_id}")

        sources = []
        for sid in source_thread_ids:
            thread = self.thread_manager.get_thread(sid)
            if thread is None:
                raise ThreadNotFoundError(f"Source not found: {sid}")
            sources.append(thread)

        # 2. Collect all messages
        all_messages = []
        for source in sources:
            all_messages.extend(source.messages)

        # 3. Apply merge strategy
        if strategy == MergeStrategy.APPEND:
            result = self._merge_append(target, all_messages)
        elif strategy == MergeStrategy.SYNTHESIZE:
            result = await _merge_synthesize(target, sources, all_messages)
        elif strategy == MergeStrategy.PRESERVE_ALL:
            result = self._merge_preserve_all(target, all_messages)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 4. Create YarnGraph edges
        for source_id in source_thread_ids:
            await self._create_merge_edge(source_id, target_thread_id)

        # 5. Return result
        return MergeResult(
            target_thread_id=target_thread_id,
            source_thread_ids=source_thread_ids,
            strategy=strategy,
            messages_added=result['messages_added'],
            synthesis=result.get('synthesis'),
            merged_timestamp=datetime.now(),
            conflicts=result.get('conflicts', []),
            metadata=result.get('metadata', {})
        )

    def _merge_append(
        self,
        target: Thread,
        messages: List[Message]
    ) -> Dict:
        """
        Simple chronological concatenation.

        Args:
            target: Target thread
            messages: Messages to add

        Returns:
            Dict with messages_added count
        """
        # Sort by timestamp
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)

        # Append to target
        initial_count = len(target.messages)
        target.messages.extend(sorted_messages)

        return {
            'messages_added': len(sorted_messages),
            'strategy': 'append'
        }

    async def _merge_synthesize(
        self,
        target: Thread,
        sources: List[Thread],
        messages: List[Message]
    ) -> Dict:
        """
        LLM-generated synthesis of insights.

        Args:
            target: Target thread
            sources: Source threads
            messages: All messages to synthesize

        Returns:
            Dict with synthesis and messages_added
        """
        if self.llm_client is None:
            # Fallback to APPEND if no LLM
            return self._merge_append(target, messages)

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(target, sources, messages)

        # Get LLM synthesis
        synthesis = await self.llm_client.generate(prompt)

        # Add synthesis as single message
        synthesis_message = Message(
            role="assistant",
            content=f"**Synthesis of {len(sources)} threads:**\n\n{synthesis}",
            timestamp=datetime.now().timestamp(),
            metadata={
                'is_synthesis': True,
                'source_threads': [s.id for s in sources],
                'message_count': len(messages)
            }
        )

        target.messages.append(synthesis_message)

        return {
            'messages_added': 1,  # Just the synthesis
            'synthesis': synthesis,
            'strategy': 'synthesize',
            'metadata': {
                'source_message_count': len(messages),
                'synthesized': True
            }
        }

    def _build_synthesis_prompt(
        self,
        target: Thread,
        sources: List[Thread],
        messages: List[Message]
    ) -> str:
        """
        Build prompt for LLM synthesis.

        Uses metaprompt enhancement for quality.
        """
        from HoloLoom.prompting import create_metaprompt
        from HoloLoom.config import Config

        # Collect thread summaries
        thread_summaries = []
        for source in sources:
            summary = f"**{source.name}** ({len(source.messages)} messages):\n"
            for msg in source.messages[-5:]:  # Last 5 messages
                summary += f"- {msg.content[:100]}...\n"
            thread_summaries.append(summary)

        # Build casual request
        casual_request = f"""
        Synthesize insights from {len(sources)} conversation threads into a concise summary.

        Target thread: {target.name}

        Threads to merge:
        {chr(10).join(thread_summaries)}

        Create a synthesis that:
        1. Identifies key insights from each thread
        2. Finds connections and relationships
        3. Organizes insights clearly (bullet points preferred)
        4. Highlights any conflicts or contradictions
        5. Keeps it concise (3-5 bullet points max)
        """

        # Enhance with metaprompt
        config = Config.fast()
        config.llm_provider = "anthropic"

        enhanced_prompt = create_metaprompt(casual_request, config=config)

        return enhanced_prompt

    def _merge_preserve_all(
        self,
        target: Thread,
        messages: List[Message]
    ) -> Dict:
        """
        Keep all messages, deduplicate by content.

        Args:
            target: Target thread
            messages: Messages to add

        Returns:
            Dict with messages_added and duplicates_removed
        """
        # Get existing message contents for deduplication
        existing_contents = {msg.content for msg in target.messages}

        # Filter duplicates
        unique_messages = []
        duplicates = 0

        for msg in messages:
            if msg.content not in existing_contents:
                unique_messages.append(msg)
                existing_contents.add(msg.content)
            else:
                duplicates += 1

        # Sort and append
        unique_messages.sort(key=lambda m: m.timestamp)
        target.messages.extend(unique_messages)

        return {
            'messages_added': len(unique_messages),
            'strategy': 'preserve_all',
            'metadata': {
                'duplicates_removed': duplicates
            }
        }

    async def _create_merge_edge(
        self,
        source_id: str,
        target_id: str
    ) -> KGEdge:
        """Create MERGED_INTO edge in YarnGraph."""
        edge = KGEdge(
            source=source_id,
            target=target_id,
            relation="MERGED_INTO",
            weight=1.0,
            metadata={
                'merge_timestamp': datetime.now().isoformat(),
                'relationship_type': 'thread_merge'
            }
        )

        await self.graph.add_edge(edge)
        return edge
```

#### Voice Command Patterns

```python
# Thread merging patterns
THREAD_MERGING_PATTERNS = [
    (re.compile(r'^merge (?P<sources>.+) into (?P<target>\w+)$', re.I), 0.95),
    (re.compile(r'^combine (?P<sources>.+) (with|and) (?P<target>\w+)$', re.I), 0.90),
    (re.compile(r'^synthesize (?P<sources>.+) into (?P<target>\w+)$', re.I), 0.90),
]

# Example: "merge biochar and composting into main"
# Extracts: sources="biochar and composting", target="main"
```

### Testing Strategy

**Test File**: `HoloLoom/voice_first/tests/test_thread_merging.py`

**Test Cases** (12 total):

```python
class TestThreadMerging:
    """Test thread merging functionality"""

    def test_merge_append_basic(self):
        """Test basic APPEND merge"""
        # Create 2 source threads with 3 messages each
        # Merge into target
        # Verify 6 messages added chronologically

    def test_merge_synthesize_basic(self):
        """Test SYNTHESIZE merge with LLM"""
        # Create 2 source threads
        # Merge with SYNTHESIZE strategy
        # Verify synthesis message created
        # Verify synthesis contains key insights

    def test_merge_preserve_all_deduplication(self):
        """Test PRESERVE_ALL removes duplicates"""
        # Create threads with overlapping messages
        # Merge with PRESERVE_ALL
        # Verify duplicates not added

    def test_merge_graph_edge_creation(self):
        """Test MERGED_INTO edges created"""
        # Merge 2 threads
        # Query graph for MERGED_INTO edges
        # Verify edges exist

    def test_merge_nonexistent_thread(self):
        """Test error handling"""
        # Attempt merge with non-existent thread
        # Verify ThreadNotFoundError

    def test_merge_chronological_ordering(self):
        """Test messages sorted by timestamp"""
        # Create threads with messages at different times
        # Merge
        # Verify chronological order in target

    def test_merge_multiple_sources(self):
        """Test merging 3+ threads"""
        # Create 4 source threads
        # Merge all into target
        # Verify all messages present

    def test_merge_empty_thread(self):
        """Test merging empty thread"""
        # Create empty thread
        # Merge into target
        # Verify graceful handling (0 messages added)

    def test_merge_synthesis_prompt_quality(self):
        """Test LLM prompt includes key info"""
        # Build synthesis prompt
        # Verify thread names, message counts, key messages included

    def test_merge_voice_command_parsing(self):
        """Test command extraction"""
        # Test "merge biochar and composting into main"
        # Verify sources and target extracted correctly

    def test_merge_performance(self):
        """Test merge latency"""
        # Merge 2 threads (10 messages each)
        # APPEND: <500ms
        # SYNTHESIZE: <3s

    def test_merge_integration_with_router(self):
        """Test end-to-end merge via VoiceRouter"""
        # Send merge command via router
        # Verify merge executed
        # Verify response message accurate
```

---

## Auto-Summarization

### Overview

**Feature**: "summarize [thread]"

**Purpose**: Get quick context on a thread without reading all messages. Especially useful after returning to a background thread.

**User Experience**:
```
[User returns after working in other threads]

User: "summarize orchard planning thread"
Elle: "Orchard Planning (12 messages over 2 hours):

      • Discussed apple tree spacing (12-15ft dwarf, 18-25ft standard)
      • Explored soil preparation needs
      • Considered biochar amendment benefits
      • Branched to 'biochar' thread at 14:32

      Last message: 'how does biochar affect pH?' (30 minutes ago)"

[User now has full context without reading 12 messages]
```

### Technical Specification

#### Data Model

```python
class SummaryStyle(Enum):
    """Summary generation styles"""
    BULLET_POINTS = "bullet_points"    # • Key point 1 • Key point 2
    NARRATIVE = "narrative"            # Coherent paragraph
    DECISIONS = "decisions"            # What was decided
    QUESTIONS = "questions"            # Unanswered questions
    TIMELINE = "timeline"              # Chronological key events

@dataclass
class ThreadSummary:
    """Generated thread summary"""
    thread_id: str                     # Thread being summarized
    style: SummaryStyle                # Style used
    summary_text: str                  # Generated summary
    message_count: int                 # Messages in thread
    time_span: timedelta               # Thread duration
    generated_at: datetime             # When summary created
    key_entities: List[str]            # Entities mentioned
    branches: List[str]                # Branch thread IDs
    metadata: Dict[str, Any]           # Additional metadata
```

#### Core Algorithm

```python
# HoloLoom/voice_first/thread/thread_summarizer.py

class ThreadSummarizer:
    """
    LLM-powered thread summarization with multiple styles.

    Features:
    - Multiple summary styles (bullets, narrative, etc.)
    - Metaprompt-enhanced quality
    - Summary caching (TTL-based)
    - Entity extraction
    """

    def __init__(
        self,
        thread_manager: ThreadManager,
        llm_client: Any,
        cache_ttl_seconds: int = 300  # 5 minutes
    ):
        """
        Initialize summarizer.

        Args:
            thread_manager: Elle's thread manager
            llm_client: LLM for generation
            cache_ttl_seconds: Cache time-to-live
        """
        self.thread_manager = thread_manager
        self.llm_client = llm_client
        self.cache = {}  # {thread_id: (summary, timestamp)}
        self.cache_ttl = cache_ttl_seconds

    async def summarize_thread(
        self,
        thread_id: str,
        style: SummaryStyle = SummaryStyle.BULLET_POINTS,
        use_cache: bool = True
    ) -> ThreadSummary:
        """
        Generate thread summary.

        Args:
            thread_id: Thread to summarize
            style: Summary style
            use_cache: Use cached summary if available

        Returns:
            ThreadSummary object
        """
        # 1. Check cache
        if use_cache and thread_id in self.cache:
            cached_summary, cached_time = self.cache[thread_id]
            age = (datetime.now() - cached_time).total_seconds()

            if age < self.cache_ttl:
                # Cache hit
                return cached_summary

        # 2. Get thread
        thread = self.thread_manager.get_thread(thread_id)
        if thread is None:
            raise ThreadNotFoundError(f"Thread not found: {thread_id}")

        # 3. Extract metadata
        metadata = self._extract_metadata(thread)

        # 4. Build prompt based on style
        prompt = self._build_summary_prompt(thread, style, metadata)

        # 5. Generate summary
        summary_text = await self.llm_client.generate(prompt)

        # 6. Create summary object
        summary = ThreadSummary(
            thread_id=thread_id,
            style=style,
            summary_text=summary_text,
            message_count=len(thread.messages),
            time_span=self._calculate_time_span(thread),
            generated_at=datetime.now(),
            key_entities=metadata['entities'],
            branches=metadata['branches'],
            metadata=metadata
        )

        # 7. Cache summary
        self.cache[thread_id] = (summary, datetime.now())

        return summary

    def _extract_metadata(self, thread: Thread) -> Dict:
        """Extract metadata from thread for summary."""
        # Entities
        all_text = " ".join([msg.content for msg in thread.messages])
        entities = self._extract_entities(all_text)

        # Branches (if this thread has been forked)
        branches = thread.metadata.get('branches', [])

        # Topics discussed
        topics = self._extract_topics(thread.messages)

        return {
            'entities': entities,
            'branches': branches,
            'topics': topics,
            'is_branch': thread.metadata.get('is_branch', False),
            'parent_id': thread.metadata.get('parent_id')
        }

    def _build_summary_prompt(
        self,
        thread: Thread,
        style: SummaryStyle,
        metadata: Dict
    ) -> str:
        """
        Build LLM prompt for summary generation.

        Uses metaprompt enhancement for quality.
        """
        from HoloLoom.prompting import create_metaprompt
        from HoloLoom.config import Config

        # Collect recent messages (last 10 or all if fewer)
        recent_messages = thread.messages[-10:]
        messages_text = "\n".join([
            f"[{msg.role}]: {msg.content}"
            for msg in recent_messages
        ])

        # Build style-specific request
        if style == SummaryStyle.BULLET_POINTS:
            style_instruction = """
            Create a concise bullet-point summary (3-5 bullets max).
            Focus on key insights, decisions, and action items.
            """
        elif style == SummaryStyle.NARRATIVE:
            style_instruction = """
            Create a coherent paragraph summarizing the conversation.
            Tell the story of what was discussed and concluded.
            """
        elif style == SummaryStyle.DECISIONS:
            style_instruction = """
            Focus only on decisions that were made.
            List each decision clearly with context.
            If no decisions were made, say so.
            """
        elif style == SummaryStyle.QUESTIONS:
            style_instruction = """
            Identify unanswered questions from the conversation.
            List questions that need follow-up.
            If all questions were answered, say so.
            """
        elif style == SummaryStyle.TIMELINE:
            style_instruction = """
            Create a chronological timeline of key events.
            Format: [time] - what happened
            """
        else:
            style_instruction = "Summarize the conversation."

        # Build casual request
        casual_request = f"""
        Summarize this conversation thread.

        Thread: {thread.name}
        Messages: {len(thread.messages)}
        Entities: {', '.join(metadata['entities'][:5])}

        Recent messages:
        {messages_text}

        Style: {style_instruction}

        Requirements:
        - Be concise but informative
        - Highlight key insights
        - Note any branches created
        - Mention time context if relevant
        """

        # Enhance with metaprompt
        config = Config.fast()
        config.llm_provider = "anthropic"

        enhanced_prompt = create_metaprompt(casual_request, config=config)

        return enhanced_prompt

    def _calculate_time_span(self, thread: Thread) -> timedelta:
        """Calculate time span of thread."""
        if len(thread.messages) == 0:
            return timedelta(0)

        first = datetime.fromtimestamp(thread.messages[0].timestamp)
        last = datetime.fromtimestamp(thread.messages[-1].timestamp)

        return last - first

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction."""
        import re

        # Capitalized words
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)

        # Domain keywords
        keywords = re.findall(
            r'\b(apple|biochar|compost|soil|tree|orchard|plant|garden)\b',
            text.lower()
        )

        entities = list(set(capitalized + keywords))
        return entities[:10]  # Top 10

    def _extract_topics(self, messages: List[Message]) -> List[str]:
        """Extract main topics discussed."""
        # Simple approach: Most common nouns
        # Future: Use LLM or NLP for better topic extraction

        all_text = " ".join([msg.content for msg in messages])
        words = all_text.lower().split()

        # Count word frequency
        from collections import Counter
        word_counts = Counter(words)

        # Filter common words (simple stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        topics = [
            word for word, count in word_counts.most_common(10)
            if word not in stop_words and len(word) > 3
        ]

        return topics[:5]  # Top 5 topics

    def clear_cache(self, thread_id: Optional[str] = None):
        """Clear summary cache."""
        if thread_id is None:
            self.cache.clear()
        elif thread_id in self.cache:
            del self.cache[thread_id]
```

#### Voice Command Patterns

```python
# Summarization patterns
THREAD_SUMMARIZATION_PATTERNS = [
    (re.compile(r'^summarize (the |this )?(?P<thread_name>.+) (thread|conversation)$', re.I), 0.95),
    (re.compile(r'^(what|tell me what) (have we|did we) (discussed|talked about) (in |about )?(?P<thread_name>.+)(\?)?$', re.I), 0.90),
    (re.compile(r'^summary of (?P<thread_name>.+)$', re.I), 0.90),
    (re.compile(r'^summarize( this)?$', re.I), 0.85),  # Current thread
    (re.compile(r'^give me (a )?(quick )?summary$', re.I), 0.85),
]
```

### Testing Strategy

**Test File**: `HoloLoom/voice_first/tests/test_thread_summarizer.py`

**Test Cases** (10 total):

```python
class TestThreadSummarizer:
    """Test thread summarization"""

    def test_summarize_bullet_points(self):
        """Test BULLET_POINTS style"""
        # Create thread with 10 messages
        # Generate summary
        # Verify bullet point format
        # Verify key points captured

    def test_summarize_narrative(self):
        """Test NARRATIVE style"""
        # Generate narrative summary
        # Verify coherent paragraph

    def test_summarize_decisions(self):
        """Test DECISIONS style"""
        # Create thread with decisions made
        # Generate summary
        # Verify decisions listed

    def test_summarize_questions(self):
        """Test QUESTIONS style"""
        # Create thread with unanswered questions
        # Generate summary
        # Verify questions identified

    def test_summarize_caching(self):
        """Test cache functionality"""
        # Generate summary
        # Generate again
        # Verify cached version used (faster)
        # Verify cache TTL respected

    def test_summarize_entity_extraction(self):
        """Test entity extraction"""
        # Create thread mentioning entities
        # Generate summary
        # Verify entities in metadata

    def test_summarize_empty_thread(self):
        """Test empty thread handling"""
        # Create empty thread
        # Generate summary
        # Verify graceful handling

    def test_summarize_metadata_extraction(self):
        """Test metadata (topics, branches, etc.)"""
        # Create thread with branches
        # Generate summary
        # Verify branch info in metadata

    def test_summarize_voice_command_parsing(self):
        """Test command extraction"""
        # Test "summarize orchard planning thread"
        # Verify thread_name extracted

    def test_summarize_integration_with_router(self):
        """Test end-to-end via VoiceRouter"""
        # Send summary command
        # Verify summary generated and returned
```

---

## Integration Points

### 1. VoiceRouter Integration

**File**: `HoloLoom/voice_first/core/voice_router.py`

**Changes Needed**:
```python
# Add to _handle_thread_command() method

elif intent.command_type == CommandType.THREAD_BRANCH:
    # Fork current thread
    brancher = ThreadBrancher(...)
    branch = await brancher.fork_thread(...)
    return f"Created '{branch.branch_name}' branch..."

elif intent.command_type == CommandType.THREAD_MERGE:
    # Merge multiple threads
    merger = ThreadMerger(...)
    result = await merger.merge_threads(...)
    return f"Merged {len(result.source_thread_ids)} threads..."

elif intent.command_type == CommandType.THREAD_SUMMARIZE:
    # Summarize thread
    summarizer = ThreadSummarizer(...)
    summary = await summarizer.summarize_thread(...)
    return summary.summary_text
```

### 2. Elle.ThreadManager Integration

**File**: `elle/voice/threads.py`

**Extensions Needed**:
```python
class ThreadManager:
    # ... existing code ...

    # NEW: Track branches
    def add_branch(self, parent_id: str, branch_id: str):
        """Track branch relationship."""
        if parent_id not in self.threads:
            return

        parent = self.threads[parent_id]
        if 'branches' not in parent.metadata:
            parent.metadata['branches'] = []

        parent.metadata['branches'].append(branch_id)

    # NEW: Get all branches of a thread
    def get_branches(self, thread_id: str) -> List[Thread]:
        """Get all branches of a thread."""
        thread = self.get_thread(thread_id)
        if thread is None:
            return []

        branch_ids = thread.metadata.get('branches', [])
        return [self.get_thread(bid) for bid in branch_ids if bid in self.threads]
```

### 3. YarnGraph Integration

**File**: `HoloLoom/memory/graph.py`

**New Edge Types**:
```python
# Add to relation types
BRANCHED_FROM = "BRANCHED_FROM"  # Branch → Parent
MERGED_INTO = "MERGED_INTO"      # Source → Target
```

**Query Methods**:
```python
# In KG class

def get_branch_tree(self, root_id: str) -> Dict:
    """Get complete branch tree starting from root."""
    edges = self.get_edges_by_relation("BRANCHED_FROM")

    # Build tree recursively
    tree = {'id': root_id, 'children': []}

    for edge in edges:
        if edge.target == root_id:
            child_tree = self.get_branch_tree(edge.source)
            tree['children'].append(child_tree)

    return tree

def get_merge_history(self, thread_id: str) -> List[str]:
    """Get all threads that were merged into this one."""
    edges = self.get_edges_by_relation("MERGED_INTO")

    sources = [
        edge.source
        for edge in edges
        if edge.target == thread_id
    ]

    return sources
```

### 4. LLM Client Integration

**File**: `HoloLoom/llm/unified_client.py`

**Usage**:
```python
from HoloLoom.llm import UnifiedLLMClient

# For synthesis and summarization
llm_client = UnifiedLLMClient(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

# Generate synthesis
synthesis = await llm_client.generate(prompt)
```

### 5. Metaprompt Integration

**File**: `HoloLoom/prompting/metaprompt.py`

**Usage**:
```python
from HoloLoom.prompting import create_metaprompt
from HoloLoom.config import Config

# Enhance summarization prompt
config = Config.fast()
config.llm_provider = "anthropic"

enhanced = create_metaprompt(casual_request, config=config)
```

---

## Testing Strategy

### Test Organization

```
HoloLoom/voice_first/tests/
├── test_thread_branching.py      # 15 tests
├── test_thread_merging.py        # 12 tests
├── test_thread_summarizer.py     # 10 tests
├── test_milestone1_integration.py # 8 integration tests
└── test_milestone1_e2e.py        # 5 end-to-end tests
```

**Total**: 50 tests

### Unit Tests (37 tests)

**Per-component testing** (see detailed specs above for each):
- Thread Branching: 15 tests
- Thread Merging: 12 tests
- Thread Summarizer: 10 tests

### Integration Tests (8 tests)

**File**: `test_milestone1_integration.py`

```python
class TestMilestone1Integration:
    """Integration tests across components"""

    def test_fork_and_summarize(self):
        """Fork thread, then summarize both"""
        # Create parent thread
        # Fork to branch
        # Summarize parent
        # Summarize branch
        # Verify summaries reflect relationship

    def test_fork_merge_cycle(self):
        """Fork, modify, merge back"""
        # Fork thread
        # Add messages to branch
        # Merge branch back to parent
        # Verify synthesis includes branch insights

    def test_multiple_forks_and_merge(self):
        """Complex branch tree"""
        # Create main thread
        # Fork to branch1, branch2
        # Add content to both branches
        # Merge both back to main
        # Verify synthesis combines all insights

    def test_nested_branches(self):
        """Branch from a branch"""
        # Fork main → branch1
        # Fork branch1 → branch2
        # Verify branch2 has correct parent
        # Summarize branch2 (should mention ancestry)

    def test_branch_with_graph_navigation(self):
        """YarnGraph relationships"""
        # Fork multiple times
        # Query graph for BRANCHED_FROM edges
        # Get branch tree
        # Verify tree structure correct

    def test_merge_with_conflicts(self):
        """Merging threads with overlapping content"""
        # Create threads with duplicate messages
        # Merge with PRESERVE_ALL
        # Verify deduplication

    def test_summarization_cache_invalidation(self):
        """Cache invalidates on thread change"""
        # Summarize thread
        # Add messages
        # Summarize again
        # Verify new summary (cache miss)

    def test_voice_commands_end_to_end(self):
        """Complete voice pipeline"""
        # Send "fork this into biochar"
        # Send "add some messages"
        # Send "summarize biochar"
        # Send "merge biochar into main"
        # Verify all operations work via voice
```

### End-to-End Tests (5 tests)

**File**: `test_milestone1_e2e.py`

```python
class TestMilestone1E2E:
    """Complete user journeys"""

    @pytest.mark.asyncio
    async def test_research_workflow(self):
        """Simulates multi-threaded research"""
        agent = UnifiedVoiceAgent()
        await agent.initialize()

        # Main research thread
        await agent.process("start thread regenerative agriculture")
        await agent.process("what is biochar?")

        # Branch 1: Biochar deep dive
        await agent.process("fork this into biochar production")
        await agent.process("how is biochar made?")
        await agent.process("what are the benefits?")

        # Back to main
        await agent.process("back to regenerative agriculture")
        await agent.process("what about cover crops?")

        # Branch 2: Cover crops
        await agent.process("fork this into cover crops")
        await agent.process("which cover crops fix nitrogen?")

        # Merge insights
        await agent.process("back to regenerative agriculture")
        await agent.process("merge biochar production and cover crops into regenerative agriculture")

        # Verify synthesis
        summary = await agent.process("summarize regenerative agriculture")
        assert "biochar" in summary.lower()
        assert "cover crops" in summary.lower()

    @pytest.mark.asyncio
    async def test_orchard_planning_workflow(self):
        """Orchard planning example from spec"""
        # (See example in Thread Branching section)

    @pytest.mark.asyncio
    async def test_quick_context_recovery(self):
        """Return to thread after time away"""
        agent = UnifiedVoiceAgent()
        await agent.initialize()

        # Create thread
        await agent.process("start thread orchard")
        await agent.process("how far apart should apple trees be?")

        # Switch away
        await agent.process("start thread composting")
        # ... do other work ...

        # Return and get context
        summary = await agent.process("summarize orchard")
        # Should quickly tell user what happened
        assert "apple" in summary.lower()
        assert "spacing" in summary.lower()

    @pytest.mark.asyncio
    async def test_parallel_exploration(self):
        """Work on multiple threads in parallel"""
        # Create 3 threads
        # Fork each thread
        # Merge all branches
        # Verify synthesis

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Graceful error handling"""
        # Try to fork non-existent thread
        # Try to merge with invalid thread
        # Verify helpful error messages
```

### Performance Benchmarks

**File**: `test_milestone1_performance.py`

```python
class TestMilestone1Performance:
    """Performance benchmarks"""

    def test_fork_latency(self):
        """Fork should be <100ms"""
        # Time fork operation
        # Average over 10 runs
        # Assert <100ms

    def test_merge_append_latency(self):
        """APPEND merge should be <500ms"""
        # Time merge with 20 messages
        # Assert <500ms

    def test_merge_synthesize_latency(self):
        """SYNTHESIZE merge should be <3s"""
        # Time LLM-based merge
        # Assert <3s

    def test_summarize_latency(self):
        """Summary generation <2s"""
        # Time summarization
        # Assert <2s

    def test_cache_speedup(self):
        """Cache should be 10x+ faster"""
        # Time uncached summary
        # Time cached summary
        # Assert cached is 10x+ faster
```

---

## Implementation Timeline

### Week 1-2: Thread Branching

**Days 1-3**: Core Implementation
- Create `thread/thread_branching.py`
- Implement `ThreadBrancher` class
- Implement `fork_thread()` method
- Implement context extraction logic
- Implement entity extraction

**Days 4-5**: Integration
- Add `BRANCHED_FROM` edge to YarnGraph
- Update `VoiceGrammar` with branching patterns
- Update `VoiceRouter` to handle `THREAD_BRANCH` commands
- Integrate with `Elle.ThreadManager`

**Days 6-8**: Testing
- Write 15 unit tests
- Fix bugs
- Performance optimization
- User testing with 3 test users

**Days 9-10**: Documentation
- Update README.md
- Write integration guide
- Create examples

**Deliverable**: Working thread branching by voice

---

### Week 3-4: Thread Merging

**Days 11-13**: Core Implementation
- Create `thread/thread_merging.py`
- Implement `ThreadMerger` class
- Implement merge strategies (APPEND, SYNTHESIZE, PRESERVE_ALL)
- Implement LLM synthesis prompt building

**Days 14-15**: Integration
- Add `MERGED_INTO` edge to YarnGraph
- Update `VoiceGrammar` with merging patterns
- Update `VoiceRouter` to handle `THREAD_MERGE` commands
- Integrate with metaprompt system

**Days 16-18**: Testing
- Write 12 unit tests
- Test merge strategies
- Test LLM synthesis quality
- User testing

**Days 19-20**: Polish
- Optimize merge performance
- Improve synthesis quality
- Update documentation

**Deliverable**: Working thread merging with synthesis

---

### Week 5-6: Auto-Summarization

**Days 21-23**: Core Implementation
- Create `thread/thread_summarizer.py`
- Implement `ThreadSummarizer` class
- Implement summary styles (BULLET_POINTS, NARRATIVE, etc.)
- Implement caching logic

**Days 24-25**: Integration
- Update `VoiceGrammar` with summarization patterns
- Update `VoiceRouter` to handle `THREAD_SUMMARIZE` commands
- Integrate with metaprompt system
- Test LLM quality

**Days 26-28**: Testing
- Write 10 unit tests
- Test all summary styles
- Test cache effectiveness
- Integration tests (8 tests)
- End-to-end tests (5 tests)

**Days 29-30**: Final Polish
- Performance optimization
- User testing with 5+ users
- Bug fixes
- Complete documentation
- Prepare for Milestone 2

**Deliverable**: Production-ready Milestone 1

---

## Voice Commands Reference

### Complete Grammar Patterns

```python
# Thread Branching (5 patterns)
"fork this into [topic]"
"wait, that reminds me - [topic]"
"this is important, create a new branch for [topic]"
"new branch: [topic]"
"spin off a branch about [topic]"

# Thread Merging (3 patterns)
"merge [thread1] and [thread2] into [target]"
"combine [thread1] and [thread2]"
"synthesize [threads] into [target]"

# Thread Summarization (5 patterns)
"summarize [thread]"
"summary of [thread]"
"what have we discussed in [thread]?"
"give me a quick summary"
"tell me what we talked about"

# Branch Navigation (3 patterns)
"list my branches"
"show thread tree"
"what branches do i have?"
```

### Example Conversations

**Example 1: Research Workflow**
```
User: "start thread regenerative agriculture"
Elle: "Created 'regenerative agriculture' thread."

User: "what is biochar?"
Elle: "Biochar is a carbon-rich material..."

User: "fork this into biochar deep dive"
Elle: "Created 'biochar deep dive' branch. Inherited context about biochar basics."

User: "how is it produced?"
Elle: [answers in biochar context]

User: "back to regenerative agriculture"
Elle: "Switched to 'regenerative agriculture'."

User: "what about cover crops?"
Elle: [answers]

User: "fork into cover crops"
Elle: "Created 'cover crops' branch."

[... work in cover crops ...]

User: "back to regenerative agriculture"
Elle: "Switched back."

User: "merge biochar deep dive and cover crops into regenerative agriculture"
Elle: "Merging... synthesizing insights...

      Merged insights:
      • Biochar: pyrolysis process, carbon sequestration
      • Cover crops: nitrogen fixing, soil structure
      • Both improve soil health and fertility

      5 messages added to thread."

User: "summarize regenerative agriculture"
Elle: "Regenerative Agriculture (15 messages, 45 minutes):

      • Explored biochar production and benefits
      • Studied cover crop nitrogen fixing
      • Discovered synergies between both approaches
      • Created 2 branches for deep dives

      Key insight: Combining biochar + cover crops maximizes soil health."
```

**Example 2: Context Recovery**
```
[User returns after 3 hours]

User: "what threads do i have?"
Elle: "You have 4 threads:
       • orchard planning (12 messages, 3 hours ago)
       • biochar (5 messages, 2 hours ago)
       • composting (8 messages, 1 hour ago)
       • main research (20 messages, active)"

User: "summarize orchard planning"
Elle: "Orchard Planning (12 messages, started 3 hours ago):

      • Discussed apple tree spacing (dwarf: 12-15ft, standard: 18-25ft)
      • Considered soil preparation with biochar
      • Branched to 'biochar' thread for deep dive
      • Last topic: pollination requirements

      Status: Paused, waiting for biochar insights"

User: "switch to orchard planning"
Elle: "Switched to 'orchard planning'. Last message: 'how many pollinators do i need?'"

[User picks up exactly where they left off]
```

---

## Performance Targets

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| **Fork thread** | <100ms | Context extraction + graph edge |
| **Merge (APPEND)** | <500ms | Chronological sort + append |
| **Merge (SYNTHESIZE)** | <3s | LLM call for synthesis |
| **Merge (PRESERVE_ALL)** | <500ms | Deduplication + append |
| **Summarize (cached)** | <10ms | Cache hit |
| **Summarize (uncached)** | <2s | LLM generation |
| **Cache speedup** | 10x+ | Cached vs uncached |
| **Context extraction** | <50ms | Last N seconds of messages |
| **Entity extraction** | <20ms | Simple regex/NLP |
| **Graph edge creation** | <30ms | YarnGraph write |

**Overall UX Target**: All voice operations feel instant (<500ms perceived latency)

---

## Dependencies

### Required

1. **Elle.ThreadManager** - Thread creation, management, switching
   - File: `elle/voice/threads.py`
   - Status: ✅ Exists (Milestone 2 complete)

2. **HoloLoom.memory.graph.KG** - YarnGraph for relationships
   - File: `HoloLoom/memory/graph.py`
   - Status: ✅ Exists (production ready)

3. **LLM Client** - For synthesis and summarization
   - File: `HoloLoom/llm/unified_client.py`
   - Status: ✅ Exists (supports Anthropic, OpenAI, Ollama)

4. **Metaprompt System** - For quality enhancement
   - File: `HoloLoom/prompting/metaprompt.py`
   - Status: ✅ Exists (November 2025 implementation)

### Optional

1. **Summary Cache** - For faster repeated summaries
   - Implementation: In-memory dict with TTL
   - Status: ⏳ To be implemented (simple)

2. **Advanced NER** - Better entity extraction
   - Library: spaCy or LLM-based
   - Status: ⏳ Future enhancement (MVP uses regex)

3. **Conflict Resolution** - Smart merge conflict handling
   - Status: ⏳ Future enhancement (Milestone 2)

---

## Success Criteria

### Functional Requirements

✅ **Thread Branching**:
- Users can fork threads by voice
- Context inheritance works (last 30s)
- Entities preserved across fork
- YarnGraph edges created
- Zero data loss

✅ **Thread Merging**:
- Multiple threads can be merged
- SYNTHESIZE strategy produces quality summaries
- Deduplication works (PRESERVE_ALL)
- Chronological ordering maintained

✅ **Auto-Summarization**:
- Summaries are accurate and helpful
- Multiple styles work (BULLET_POINTS, NARRATIVE, etc.)
- Caching speeds up repeated requests
- Metadata extraction works

### Non-Functional Requirements

✅ **Performance**:
- All operations meet latency targets
- Cache provides 10x+ speedup
- No performance degradation with thread growth

✅ **Quality**:
- >80% user satisfaction with summaries
- >90% accuracy in entity extraction
- LLM synthesis is coherent and insightful

✅ **Reliability**:
- Zero data loss in fork/merge operations
- Graceful error handling
- Robust cache invalidation

### Test Coverage

✅ **Tests**: 50 total tests passing
- Unit: 37 tests
- Integration: 8 tests
- End-to-end: 5 tests

✅ **Code Quality**:
- All functions documented
- Type hints complete
- Logging comprehensive

---

## Future Enhancements

### Milestone 2 Additions

**Smart Conflict Resolution**:
- Detect contradictions in merged threads
- Offer resolution strategies
- Highlight conflicts for user review

**Advanced Entity Linking**:
- Use LLM or spaCy for NER
- Link entities across threads
- Build entity graph

**Summary Personalization**:
- Learn user's preferred summary style
- Adapt verbosity to context
- Personalized summaries per user

### Milestone 3 Additions

**Temporal Navigation**:
- "take me back to yesterday's discussion"
- Time-based thread filtering
- Temporal patterns in summaries

**Auto-Branching**:
- Detect topic shifts
- Offer to auto-fork
- Smart branch naming

**Thread Archival**:
- Auto-archive inactive threads
- Compressed summaries for old threads
- Restore from archive

---

## Conclusion

Milestone 1 transforms voice-first UX from basic thread management to **natural thought flow orchestration**:

- **Branching** eliminates tab-switching cognitive overhead
- **Merging** enables parallel exploration with synthesis
- **Summarization** provides instant context recovery

**Timeline**: 6 weeks
**Effort**: ~1,200 lines of code + 800 lines of tests
**Impact**: Eliminates context-switching, enables natural conversation flow

This specification provides complete guidance for implementation. Each component is detailed with algorithms, integration points, testing strategy, and success criteria.

**Ready to build!** 🚀
