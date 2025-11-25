"""
Search Task - Example Task with Knowledge Base Integration

Demonstrates:
- Integration with VoiceSOPEditor (knowledge base)
- Async generator progress updates
- Real-world task implementation
- Search result processing

Created: 2025-11-22
Part of: Voice UX Milestone 2 Phase 3
"""

import asyncio
from typing import AsyncIterator, List, Dict, Any
from pathlib import Path

# Import task system components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from voice.task_system import (
    TaskProgress, TaskState, BaseTask, register_task
)


@register_task("search")
class SearchTask(BaseTask):
    """
    Example task: Search knowledge base

    Demonstrates multi-step search with progress tracking:
    1. Parse query
    2. Search SOPs
    3. Rank results
    4. Extract snippets
    5. Format response

    Usage:
        task = SearchTask(query="How to create a thread?")
        async for progress in task.execute(context):
            print(f"{progress.current_step}: {progress.progress_pct:.0%}")
    """

    def __init__(self, query: str, max_results: int = 5):
        """
        Initialize search task

        Args:
            query: Search query
            max_results: Maximum number of results to return
        """
        super().__init__(name=f"Search: {query[:30]}...")
        self.query = query
        self.max_results = max_results
        self.total_steps = 5
        self.results: List[Dict[str, Any]] = []

    async def execute(self, context: dict) -> AsyncIterator[TaskProgress]:
        """
        Execute search with progress updates

        Args:
            context: Execution context (sop_dir, editor instance, etc.)

        Yields:
            TaskProgress objects with current state
        """
        self._state = TaskState.RUNNING

        # Get SOP directory from context
        sop_dir = context.get("sop_dir", "elle/sops")

        # Step 1: Parse query (extract keywords, intent)
        await self._check_pause()
        if self._check_cancelled():
            yield self._cancelled_progress("Parsing query", 0.0, 0)
            return

        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.RUNNING,
            progress_pct=0.2,
            current_step="Parsing query",
            total_steps=self.total_steps,
            completed_steps=0,
            status_message=f"Parsing: {self.query}"
        )

        # Simulate query parsing
        keywords = self.query.lower().split()
        await asyncio.sleep(0.3)

        # Step 2: Search SOPs
        await self._check_pause()
        if self._check_cancelled():
            yield self._cancelled_progress("Searching SOPs", 0.2, 1)
            return

        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.RUNNING,
            progress_pct=0.4,
            current_step="Searching SOPs",
            total_steps=self.total_steps,
            completed_steps=1,
            status_message=f"Searching for: {', '.join(keywords[:3])}"
        )

        # Simulate SOP search (in real implementation, use VoiceSOPEditor)
        self.results = await self._search_sops(sop_dir, keywords)
        await asyncio.sleep(0.5)

        # Step 3: Rank results
        await self._check_pause()
        if self._check_cancelled():
            yield self._cancelled_progress("Ranking results", 0.4, 2)
            return

        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.RUNNING,
            progress_pct=0.6,
            current_step="Ranking results",
            total_steps=self.total_steps,
            completed_steps=2,
            status_message=f"Found {len(self.results)} matches"
        )

        # Simulate ranking
        self.results = await self._rank_results(self.results, keywords)
        await asyncio.sleep(0.3)

        # Step 4: Extract snippets
        await self._check_pause()
        if self._check_cancelled():
            yield self._cancelled_progress("Extracting snippets", 0.6, 3)
            return

        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.RUNNING,
            progress_pct=0.8,
            current_step="Extracting snippets",
            total_steps=self.total_steps,
            completed_steps=3,
            status_message=f"Processing top {min(len(self.results), self.max_results)} results"
        )

        # Extract relevant snippets
        for result in self.results[:self.max_results]:
            result["snippet"] = self._extract_snippet(result["content"], keywords)
        await asyncio.sleep(0.3)

        # Step 5: Format response
        await self._check_pause()
        if self._check_cancelled():
            yield self._cancelled_progress("Formatting response", 0.8, 4)
            return

        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.RUNNING,
            progress_pct=0.9,
            current_step="Formatting response",
            total_steps=self.total_steps,
            completed_steps=4,
            status_message="Preparing final results"
        )

        response = self._format_results()
        await asyncio.sleep(0.2)

        # Completed
        self._state = TaskState.COMPLETED
        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            progress_pct=1.0,
            current_step="Complete",
            total_steps=self.total_steps,
            completed_steps=self.total_steps,
            status_message=f"Found {len(self.results[:self.max_results])} results for: {self.query}"
        )

    async def _search_sops(self, sop_dir: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search SOP files (simulated)

        In real implementation, use VoiceSOPEditor.search()
        """
        # Simulated search results
        mock_sops = [
            {
                "file": "threading.md",
                "title": "Thread Management",
                "content": "To create a new thread, use the + command followed by the thread name. "
                          "For example: '+ baking' creates a thread about baking.",
                "relevance": 0.95
            },
            {
                "file": "navigation.md",
                "title": "Navigation Commands",
                "content": "Navigate between threads using t<number> (e.g., t3 for thread 3). "
                          "Use 'back' to return to previous thread, 'home' for thread 1.",
                "relevance": 0.75
            },
            {
                "file": "commands.md",
                "title": "Voice Commands",
                "content": "Available commands: threads (list all), + <name> (create), "
                          "t<number> (switch), back, home, stop, pause, resume.",
                "relevance": 0.65
            }
        ]

        # Filter by keyword relevance
        results = []
        for sop in mock_sops:
            if any(kw in sop["content"].lower() for kw in keywords):
                results.append(sop)

        return results

    async def _rank_results(
        self,
        results: List[Dict[str, Any]],
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """Rank results by relevance"""
        # Simple ranking: count keyword matches
        for result in results:
            content_lower = result["content"].lower()
            match_count = sum(1 for kw in keywords if kw in content_lower)
            result["match_count"] = match_count
            result["relevance"] = result.get("relevance", 0.5) * (1 + match_count * 0.1)

        # Sort by relevance
        return sorted(results, key=lambda r: r["relevance"], reverse=True)

    def _extract_snippet(self, content: str, keywords: List[str], max_length: int = 100) -> str:
        """Extract relevant snippet containing keywords"""
        # Find first sentence with keyword
        sentences = content.split(". ")
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in keywords):
                # Truncate if too long
                if len(sentence) > max_length:
                    return sentence[:max_length] + "..."
                return sentence

        # Fallback: first sentence
        return sentences[0][:max_length] + "..." if len(sentences[0]) > max_length else sentences[0]

    def _format_results(self) -> str:
        """Format search results as human-readable text"""
        if not self.results:
            return "No results found"

        lines = [f"Found {len(self.results[:self.max_results])} results:\n"]

        for idx, result in enumerate(self.results[:self.max_results], 1):
            lines.append(f"{idx}. {result['title']} ({result['file']})")
            lines.append(f"   {result.get('snippet', result['content'][:80])}")
            lines.append(f"   Relevance: {result['relevance']:.0%}\n")

        return "\n".join(lines)

    def _cancelled_progress(self, step: str, pct: float, completed: int) -> TaskProgress:
        """Helper to create cancelled progress"""
        return TaskProgress(
            task_id=self.task_id,
            state=TaskState.CANCELLED,
            progress_pct=pct,
            current_step=step,
            total_steps=self.total_steps,
            completed_steps=completed,
            status_message=f"Search cancelled at {step}"
        )


@register_task("search_knowledge")
class SearchKnowledgeTask(BaseTask):
    """
    Advanced search task with knowledge graph integration

    Demonstrates:
    - Multi-source search (SOPs + knowledge graph)
    - Entity extraction
    - Relationship traversal
    """

    def __init__(self, query: str, include_related: bool = True):
        super().__init__(name=f"Knowledge search: {query[:30]}...")
        self.query = query
        self.include_related = include_related
        self.total_steps = 4

    async def execute(self, context: dict) -> AsyncIterator[TaskProgress]:
        """Execute knowledge graph search"""
        self._state = TaskState.RUNNING

        steps = [
            ("Extracting entities", 0.25, 0.5),
            ("Searching knowledge graph", 0.5, 1.0),
            ("Finding related concepts", 0.75, 0.8),
            ("Synthesizing results", 1.0, 0.4),
        ]

        for idx, (step_name, progress_pct, duration) in enumerate(steps):
            if self._check_cancelled():
                yield TaskProgress(
                    task_id=self.task_id,
                    state=TaskState.CANCELLED,
                    progress_pct=progress_pct,
                    current_step=step_name,
                    total_steps=self.total_steps,
                    completed_steps=idx,
                    status_message="Knowledge search cancelled"
                )
                return

            await self._check_pause()

            yield TaskProgress(
                task_id=self.task_id,
                state=TaskState.RUNNING,
                progress_pct=progress_pct,
                current_step=step_name,
                total_steps=self.total_steps,
                completed_steps=idx,
                status_message=f"Knowledge search: {step_name}"
            )

            await asyncio.sleep(duration)

        # Completed
        self._state = TaskState.COMPLETED
        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            progress_pct=1.0,
            current_step="Complete",
            total_steps=self.total_steps,
            completed_steps=self.total_steps,
            status_message=f"Knowledge search complete: {self.query}"
        )


# Example usage
async def demo_search_task():
    """Demo SearchTask execution"""
    print("\n" + "="*60)
    print("SEARCH TASK DEMO")
    print("="*60)

    # Create task
    task = SearchTask(query="How to create a thread?", max_results=3)

    print(f"\n[Task Created] {task.name} (ID: {task.task_id})")
    print(f"Query: {task.query}")
    print(f"Max results: {task.max_results}")

    # Execute
    print("\n[Execution Started]")
    context = {"sop_dir": "elle/sops"}

    async for progress in task.execute(context):
        bar_length = 30
        filled = int(bar_length * progress.progress_pct)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"  [{bar}] {progress.progress_pct:>5.0%} | {progress.current_step}")

    # Show results
    print(f"\n[Results]")
    for idx, result in enumerate(task.results[:task.max_results], 1):
        print(f"\n{idx}. {result['title']} ({result['relevance']:.0%} relevance)")
        print(f"   {result.get('snippet', '')}")

    print(f"\n[Task Complete] Final state: {task._state.value}")


async def demo_search_with_pause():
    """Demo search with pause/resume"""
    print("\n" + "="*60)
    print("SEARCH WITH PAUSE DEMO")
    print("="*60)

    task = SearchTask(query="navigation commands")
    print(f"\n[Task Created] {task.name}")

    context = {"sop_dir": "elle/sops"}
    progress_gen = task.execute(context)

    # Process first 2 steps
    for i in range(2):
        progress = await progress_gen.__anext__()
        print(f"  Step {i+1}: {progress.current_step} ({progress.progress_pct:.0%})")

    # Pause
    print("\n[Pausing search...]")
    await task.pause()

    # Resume after delay
    async def resume_after_delay():
        await asyncio.sleep(0.5)
        print("\n[Resuming search...]")
        await task.resume()

    resume_task = asyncio.create_task(resume_after_delay())
    progress = await progress_gen.__anext__()
    await resume_task

    # Complete remaining steps
    async for progress in progress_gen:
        print(f"  Step: {progress.current_step} ({progress.progress_pct:.0%})")

    print(f"\n[Search Complete] Found {len(task.results)} results")


if __name__ == "__main__":
    print("SearchTask Example Implementation")
    print("Demonstrates knowledge base integration\n")

    # Run demos
    asyncio.run(demo_search_task())
    asyncio.run(demo_search_with_pause())

    print("\n" + "="*60)
    print("ALL DEMOS COMPLETE")
    print("="*60)
