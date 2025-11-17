"""
Paper Memory System
===================

Integrates research papers with HoloLoom's persistent memory.

**Philosophy**: Each paper section becomes a memory shard with rich metadata,
enabling cross-paper retrieval and synthesis.

Author: HoloLoom Team
Date: 2025-11-17
"""

from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

try:
    from HoloLoom import HoloLoom
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    import warnings
    warnings.warn("HoloLoom not available. Install from repository.")

from HoloLoom.research_assistant.pdf_ingestion import ResearchPaper, PaperSection


# ============================================================================
# Paper Memory System
# ============================================================================

class PaperMemorySystem:
    """
    Stores and retrieves research papers using HoloLoom's memory system.

    **Features**:
    - Section-based memory shards (one per paper section)
    - Rich metadata (paper title, authors, section, citations)
    - Cross-paper retrieval
    - Knowledge graph construction from entities
    - Paper tracking and statistics

    Example:
        >>> memory = PaperMemorySystem()
        >>> await memory.initialize()
        >>>
        >>> # Ingest paper
        >>> paper = quick_ingest("attention.pdf")
        >>> await memory.ingest_paper(paper)
        >>>
        >>> # Query across papers
        >>> results = await memory.query_papers("What is self-attention?")
        >>> for result in results:
        >>>     print(f"{result['paper']}: {result['content'][:100]}")
    """

    def __init__(self):
        """Initialize paper memory system."""
        if not HOLOLOOM_AVAILABLE:
            raise RuntimeError("HoloLoom not available. Cannot create PaperMemorySystem.")

        self.loom: Optional[HoloLoom] = None
        self.papers: List[ResearchPaper] = []
        self.paper_index: Dict[str, ResearchPaper] = {}  # title -> paper

    async def initialize(self):
        """Initialize HoloLoom instance."""
        self.loom = HoloLoom()
        # HoloLoom uses async context manager, but we'll manage lifecycle manually
        # In production, use: async with HoloLoom() as loom

    async def ingest_paper(self, paper: ResearchPaper) -> Dict[str, Any]:
        """
        Store research paper in HoloLoom memory.

        Creates memory shards for:
        - Paper metadata (title, authors, abstract)
        - Each section (Introduction, Methods, Results, etc.)
        - Citations (if available)

        Args:
            paper: ResearchPaper object from PDF ingestion

        Returns:
            Dict with ingestion statistics

        Example:
            >>> paper = quick_ingest("bert.pdf")
            >>> stats = await memory.ingest_paper(paper)
            >>> print(f"Stored {stats['sections_stored']} sections")
        """
        if not self.loom:
            await self.initialize()

        # Track paper
        self.papers.append(paper)
        self.paper_index[paper.title] = paper

        sections_stored = 0
        memories_created = 0

        # 1. Store metadata memory
        metadata_content = self._create_metadata_content(paper)
        await self.loom.experience(
            content=metadata_content,
            metadata={
                'type': 'paper_metadata',
                'paper_title': paper.title,
                'authors': paper.authors,
                'num_sections': len(paper.sections),
                'num_citations': len(paper.citations),
                'ingestion_time': datetime.now().isoformat()
            }
        )
        memories_created += 1

        # 2. Store abstract memory
        if paper.abstract and len(paper.abstract) > 20:
            await self.loom.experience(
                content=f"Abstract from '{paper.title}': {paper.abstract}",
                metadata={
                    'type': 'paper_abstract',
                    'paper_title': paper.title,
                    'section': 'Abstract'
                }
            )
            memories_created += 1

        # 3. Store section memories
        for section in paper.sections:
            if len(section.content.strip()) > 50:  # Skip tiny sections
                section_content = self._create_section_content(paper, section)

                await self.loom.experience(
                    content=section_content,
                    metadata={
                        'type': 'paper_section',
                        'paper_title': paper.title,
                        'section_title': section.title,
                        'authors': paper.authors
                    }
                )

                sections_stored += 1
                memories_created += 1

        # 4. Store citations memory (if available)
        if paper.citations:
            citations_content = self._create_citations_content(paper)
            await self.loom.experience(
                content=citations_content,
                metadata={
                    'type': 'paper_citations',
                    'paper_title': paper.title,
                    'num_citations': len(paper.citations)
                }
            )
            memories_created += 1

        return {
            'paper_title': paper.title,
            'sections_stored': sections_stored,
            'memories_created': memories_created,
            'total_papers': len(self.papers)
        }

    async def query_papers(
        self,
        query: str,
        max_results: int = 10,
        paper_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query across all ingested papers.

        Args:
            query: Natural language query
            max_results: Maximum results to return
            paper_filter: Optional paper title to filter results

        Returns:
            List of results with content and metadata

        Example:
            >>> results = await memory.query_papers(
            ...     "What is the Transformer architecture?",
            ...     max_results=5
            ... )
            >>> for r in results:
            ...     print(f"{r['paper']}, {r['section']}: {r['snippet']}")
        """
        if not self.loom:
            await self.initialize()

        # Retrieve memories from HoloLoom
        memories = await self.loom.recall(query, k=max_results)

        # Format results
        results = []
        for memory in memories:
            metadata = memory.get('metadata', {})

            # Filter by paper if specified
            if paper_filter and metadata.get('paper_title') != paper_filter:
                continue

            results.append({
                'content': memory.get('content', ''),
                'paper': metadata.get('paper_title', 'Unknown'),
                'section': metadata.get('section_title', metadata.get('section', 'Unknown')),
                'type': metadata.get('type', 'unknown'),
                'snippet': memory.get('content', '')[:200] + '...'
            })

        return results

    async def get_paper_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all ingested papers with metadata.

        Returns:
            List of paper summaries

        Example:
            >>> papers = await memory.get_paper_list()
            >>> for p in papers:
            ...     print(f"{p['title']} by {', '.join(p['authors'])}")
        """
        return [
            {
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract[:200] + '...' if len(paper.abstract) > 200 else paper.abstract,
                'num_sections': len(paper.sections),
                'num_citations': len(paper.citations),
                'num_figures': len(paper.figures)
            }
            for paper in self.papers
        ]

    def get_knowledge_graph(self) -> Dict[str, Any]:
        """
        Construct knowledge graph from ingested papers.

        Extracts:
        - Paper nodes
        - Author nodes
        - Citation edges
        - Section relationships

        Returns:
            Dict with nodes, edges, and metadata for gap identification

        Example:
            >>> kg = memory.get_knowledge_graph()
            >>> print(f"Papers: {len(kg['nodes'])}")
            >>> print(f"Citations: {len(kg['edges'])}")
        """
        nodes = []
        edges = []
        node_counts = {}

        # Add paper nodes
        for paper in self.papers:
            paper_node = paper.title
            nodes.append(paper_node)
            node_counts[paper_node] = len(paper.sections)

            # Add author nodes and edges
            for author in paper.authors:
                if author not in nodes:
                    nodes.append(author)
                    node_counts[author] = 0

                # Author -> Paper edge
                edges.append((author, paper_node, 'AUTHORED'))
                node_counts[author] += 1

            # Add citation edges (if papers cite each other)
            for citation in paper.citations:
                # Check if citation matches any loaded paper
                for other_paper in self.papers:
                    if other_paper.title in citation:
                        edges.append((paper_node, other_paper.title, 'CITES'))
                        break

        return {
            'nodes': nodes,
            'edges': edges,
            'node_counts': node_counts
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dict with statistics

        Example:
            >>> stats = memory.get_statistics()
            >>> print(f"Total papers: {stats['total_papers']}")
        """
        total_sections = sum(len(p.sections) for p in self.papers)
        total_citations = sum(len(p.citations) for p in self.papers)
        total_figures = sum(len(p.figures) for p in self.papers)

        return {
            'total_papers': len(self.papers),
            'total_sections': total_sections,
            'total_citations': total_citations,
            'total_figures': total_figures,
            'avg_sections_per_paper': total_sections / len(self.papers) if self.papers else 0,
            'avg_citations_per_paper': total_citations / len(self.papers) if self.papers else 0
        }

    # Private helper methods

    def _create_metadata_content(self, paper: ResearchPaper) -> str:
        """Create content string for metadata memory."""
        authors_str = ', '.join(paper.authors)
        return (
            f"Research Paper: '{paper.title}' by {authors_str}. "
            f"Contains {len(paper.sections)} sections "
            f"and {len(paper.citations)} citations."
        )

    def _create_section_content(self, paper: ResearchPaper, section: PaperSection) -> str:
        """Create content string for section memory."""
        # Truncate very long sections (>2000 chars)
        content = section.content
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"

        return (
            f"From paper '{paper.title}', section '{section.title}':\n\n"
            f"{content}"
        )

    def _create_citations_content(self, paper: ResearchPaper) -> str:
        """Create content string for citations memory."""
        citations_preview = paper.citations[:5]  # First 5 citations
        citations_str = '\n'.join(f"  {i+1}. {c[:100]}..." for i, c in enumerate(citations_preview))

        return (
            f"References from '{paper.title}':\n\n"
            f"{citations_str}\n\n"
            f"(Total: {len(paper.citations)} citations)"
        )


# ============================================================================
# Utility Functions
# ============================================================================

async def ingest_and_query_example():
    """
    Example workflow: ingest papers and query.

    Example:
        >>> await ingest_and_query_example()
    """
    from HoloLoom.research_assistant.pdf_ingestion import quick_ingest

    # Create memory system
    memory = PaperMemorySystem()
    await memory.initialize()

    # Ingest papers (example)
    # paper1 = quick_ingest("attention.pdf")
    # stats1 = await memory.ingest_paper(paper1)
    # print(f"Ingested: {stats1}")

    # Query papers
    # results = await memory.query_papers("What is self-attention?")
    # for r in results:
    #     print(f"{r['paper']}: {r['snippet']}")

    print("Example workflow ready. Uncomment to use with actual PDFs.")
