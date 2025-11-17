"""
Research Assistant Demo
=======================

Demonstrates the Personal Research Assistant capabilities without Streamlit.

This script shows how to:
1. Ingest PDF papers programmatically
2. Query across papers
3. Run synthesis to discover patterns/contradictions/gaps
4. Generate responses with insights

**Usage**:
    PYTHONPATH=. python demos/demo_research_assistant.py

Author: HoloLoom Team
Date: 2025-11-17
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any

from HoloLoom.research_assistant.pdf_ingestion import PDFIngestionPipeline
from HoloLoom.research_assistant.paper_memory import PaperMemorySystem
from HoloLoom.research_assistant.synthesis_engine import ResearchSynthesisEngine


# ============================================================================
# Demo Configuration
# ============================================================================

# Sample paper metadata (for demo without actual PDFs)
SAMPLE_PAPERS = [
    {
        'title': 'Attention Is All You Need',
        'authors': ['Vaswani et al.'],
        'abstract': 'We propose the Transformer, a novel architecture based entirely on attention mechanisms...',
        'sections': [
            {
                'title': 'Introduction',
                'content': 'The Transformer uses self-attention to compute representations without recurrence or convolutions. This enables parallel processing and long-range dependencies.'
            },
            {
                'title': 'Model Architecture',
                'content': 'The Transformer uses multi-head self-attention and position-wise feed-forward networks. The attention mechanism computes queries, keys, and values.'
            },
            {
                'title': 'Results',
                'content': 'The Transformer achieves state-of-the-art results on machine translation tasks, outperforming previous recurrent and convolutional models.'
            }
        ],
        'citations': [
            'Neural Machine Translation by Jointly Learning to Align and Translate',
            'Sequence to Sequence Learning with Neural Networks'
        ]
    },
    {
        'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
        'authors': ['Devlin et al.'],
        'abstract': 'We introduce BERT, which uses bidirectional Transformers for language understanding...',
        'sections': [
            {
                'title': 'Introduction',
                'content': 'BERT applies bidirectional training to the Transformer architecture, enabling it to learn deep contextualized representations from unlabeled text.'
            },
            {
                'title': 'Method',
                'content': 'BERT uses masked language modeling and next sentence prediction for pre-training. The bidirectional attention allows each token to attend to all other tokens.'
            },
            {
                'title': 'Experiments',
                'content': 'BERT achieves state-of-the-art results on 11 NLP tasks including question answering and natural language inference.'
            }
        ],
        'citations': [
            'Attention Is All You Need',
            'Deep Contextualized Word Representations'
        ]
    },
    {
        'title': 'Language Models are Few-Shot Learners',
        'authors': ['Brown et al.'],
        'abstract': 'We train GPT-3, a 175 billion parameter language model, and evaluate its few-shot learning capabilities...',
        'sections': [
            {
                'title': 'Introduction',
                'content': 'GPT-3 demonstrates that scaling up language models enables few-shot learning without task-specific fine-tuning.'
            },
            {
                'title': 'Approach',
                'content': 'GPT-3 uses causal (unidirectional) attention in the Transformer architecture, trained on a diverse corpus of text.'
            },
            {
                'title': 'Results',
                'content': 'GPT-3 achieves strong few-shot performance on many NLP tasks, sometimes matching or exceeding fine-tuned models.'
            }
        ],
        'citations': [
            'Attention Is All You Need',
            'Language Models are Unsupervised Multitask Learners'
        ]
    }
]


# ============================================================================
# Demo Functions
# ============================================================================

def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_section(title: str):
    """Print formatted section."""
    print(f"\n--- {title} ---\n")


async def create_mock_paper(paper_data: Dict[str, Any]):
    """Create a mock ResearchPaper object from sample data."""
    from HoloLoom.research_assistant.pdf_ingestion import ResearchPaper, PaperSection

    sections = [
        PaperSection(
            title=s['title'],
            content=s['content'],
            page_start=0,
            page_end=0
        )
        for s in paper_data['sections']
    ]

    return ResearchPaper(
        title=paper_data['title'],
        authors=paper_data['authors'],
        abstract=paper_data['abstract'],
        sections=sections,
        citations=paper_data['citations'],
        figures=[],
        metadata={'source': 'demo'},
        pdf_path=None
    )


async def demo_ingestion():
    """Demonstrate paper ingestion."""
    print_header("Demo 1: PDF Ingestion")

    memory = PaperMemorySystem()
    await memory.initialize()

    print("Ingesting 3 research papers...")
    print("(Using mock papers for demo - in production, use PDFs)\n")

    for i, paper_data in enumerate(SAMPLE_PAPERS, 1):
        paper = await create_mock_paper(paper_data)

        print(f"{i}. Ingesting: {paper.title}")
        stats = await memory.ingest_paper(paper)

        print(f"   ✓ Stored {stats['sections_stored']} sections")
        print(f"   ✓ Created {stats['memories_created']} memories")
        print(f"   ✓ Total papers: {stats['total_papers']}\n")

    # Show statistics
    stats = memory.get_statistics()
    print("\n📊 Collection Statistics:")
    print(f"   Total papers: {stats['total_papers']}")
    print(f"   Total sections: {stats['total_sections']}")
    print(f"   Total citations: {stats['total_citations']}")
    print(f"   Avg sections/paper: {stats['avg_sections_per_paper']:.1f}")

    return memory


async def demo_querying(memory: PaperMemorySystem):
    """Demonstrate cross-paper querying."""
    print_header("Demo 2: Cross-Paper Querying")

    queries = [
        "What is the Transformer architecture?",
        "How does BERT differ from GPT-3?",
        "What is self-attention?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")

        results = await memory.query_papers(query, max_results=3)

        print(f"\n   Found {len(results)} relevant passages:\n")

        for j, result in enumerate(results[:2], 1):
            print(f"   {j}. [{result['paper']}] {result['section']}")
            print(f"      {result['snippet'][:150]}...\n")


async def demo_synthesis(memory: PaperMemorySystem):
    """Demonstrate synthesis insights."""
    print_header("Demo 3: Cross-Paper Synthesis")

    # Create synthesis engine
    synthesis = ResearchSynthesisEngine()
    await synthesis.start()

    # Register papers for synthesis
    print("Registering papers for synthesis...")
    for paper_data in SAMPLE_PAPERS:
        paper = await create_mock_paper(paper_data)
        for section in paper.sections:
            synthesis.register_paper_content(paper.title, section.content)

    # Register some sample queries (for pattern detection)
    sample_queries = [
        {
            'text': 'What is the Transformer architecture?',
            'confidence': 0.95,
            'timestamp': datetime.now(),
            'entities': ['Transformer'],
            'motifs': ['architecture']
        },
        {
            'text': 'How does attention work?',
            'confidence': 0.92,
            'timestamp': datetime.now(),
            'entities': ['attention'],
            'motifs': ['mechanism']
        },
        {
            'text': 'What is self-attention?',
            'confidence': 0.94,
            'timestamp': datetime.now(),
            'entities': ['self-attention'],
            'motifs': ['mechanism']
        }
    ]

    for query in sample_queries:
        synthesis.register_paper_query(query)

    # Run synthesis
    print("Running synthesis cycle...\n")
    kg = memory.get_knowledge_graph()
    results = await synthesis.synthesize(kg)

    # Display results
    print_section("Discovered Patterns")
    patterns = synthesis.get_patterns()
    if patterns:
        for i, pattern in enumerate(patterns, 1):
            print(f"{i}. {pattern.content}")
            print(f"   Sources: {len(pattern.source_memories)} queries")
            print(f"   Confidence: {pattern.confidence:.2%}\n")
    else:
        print("No patterns discovered (need more queries)")

    print_section("Contradictions Detected")
    contradictions = synthesis.get_contradictions()
    if contradictions:
        for i, c in enumerate(contradictions, 1):
            print(f"{i}. Type: {c.type.value}")
            print(f"   {c.explanation}")
            print(f"   Score: {c.score:.2f}\n")
    else:
        print("No contradictions detected")

    print_section("Knowledge Gaps Identified")
    gaps = synthesis.get_gaps()
    if gaps:
        for i, gap in enumerate(gaps[:3], 1):
            print(f"{i}. {gap.bridge_concept}")
            print(f"   Missing: {', '.join(gap.missing_concepts[:3])}")
            print(f"   Importance: {gap.importance:.2%}")
            if gap.suggested_queries:
                print(f"   Suggested: {gap.suggested_queries[0]}\n")
    else:
        print("No knowledge gaps identified")

    # Summary
    print_section("Synthesis Summary")
    summary = synthesis.get_synthesis_summary()
    print(f"Total cycles: {summary['total_cycles']}")
    print(f"Patterns: {summary['num_patterns']}")
    print(f"Contradictions: {summary['num_contradictions']}")
    print(f"Gaps: {summary['num_gaps']}")
    print(f"Duration: {summary['cycle_duration_ms']:.1f}ms")

    await synthesis.stop()

    return synthesis


async def demo_insights(synthesis: ResearchSynthesisEngine):
    """Demonstrate query-aware insights."""
    print_header("Demo 4: Query-Aware Insights")

    query = "attention mechanism"
    print(f"Getting insights for: '{query}'\n")

    insights = synthesis.get_insights_for_query(query)

    if insights['patterns']:
        print("Relevant Patterns:")
        for pattern in insights['patterns'][:2]:
            print(f"  • {pattern}")
        print()

    if insights['contradictions']:
        print("Relevant Contradictions:")
        for contradiction in insights['contradictions'][:2]:
            print(f"  ⚠ {contradiction}")
        print()

    if insights['gaps']:
        print("Relevant Gaps:")
        for gap in insights['gaps'][:2]:
            print(f"  🕳 {gap}")
        print()


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  Personal Research Assistant Demo")
    print("  'Your research papers remember each other'")
    print("=" * 60)

    # Demo 1: Ingestion
    memory = await demo_ingestion()

    # Demo 2: Querying
    await demo_querying(memory)

    # Demo 3: Synthesis
    synthesis = await demo_synthesis(memory)

    # Demo 4: Insights
    await demo_insights(synthesis)

    # Conclusion
    print_header("Demo Complete!")
    print("Next steps:")
    print("  1. Run the Streamlit chatbot:")
    print("     streamlit run HoloLoom/research_assistant/chatbot.py")
    print()
    print("  2. Upload real PDF papers")
    print()
    print("  3. Explore patterns, contradictions, and gaps!")
    print()
    print("✨ Your research papers remember each other ✨\n")


if __name__ == "__main__":
    asyncio.run(main())
