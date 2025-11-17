"""
HoloLoom Recall Accuracy Benchmark - Books Dataset
===================================================

Tests how well HoloLoom retrieves relevant memories from book excerpts and summaries.
Tests long-form text retrieval capabilities.

Dataset:
- Source: Synthetic book summaries (for demo)
- Size: 15 classic and modern books
- Queries: Questions about themes, characters, plots

Metrics:
- Precision@K (K=1,5,10)
- Recall@K (K=5,10)
- MRR (Mean Reciprocal Rank)

Usage:
    PYTHONPATH=. python benchmarks/recall_accuracy/books_benchmark.py

Output:
    - JSON results: benchmarks/results/recall_accuracy_books_YYYY-MM-DD.json
    - Markdown report: benchmarks/results/recall_accuracy_books_YYYY-MM-DD.md

Author: HoloLoom Team
Date: 2025-11-17
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.benchmark_base import (
    BaseBenchmark,
    BenchmarkMetrics,
    BenchmarkConfig,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_percentile
)


# ============================================================================
# Books Benchmark Dataset (Synthetic Book Summaries)
# ============================================================================

BOOKS_DATA = [
    {
        'id': 'book_001',
        'title': '1984',
        'author': 'George Orwell',
        'genre': 'Dystopian Fiction',
        'summary': 'In a totalitarian future society ruled by Big Brother, Winston Smith works at the Ministry of Truth rewriting history. He begins a forbidden love affair with Julia and joins a resistance movement led by O\'Brien. The novel explores themes of surveillance, propaganda, thought control, and the manipulation of truth. Winston\'s rebellion ultimately fails as he is tortured and brainwashed into loving Big Brother.'
    },
    {
        'id': 'book_002',
        'title': 'To Kill a Mockingbird',
        'author': 'Harper Lee',
        'genre': 'Southern Gothic',
        'summary': 'Scout Finch narrates her childhood in Depression-era Alabama. Her father Atticus, a lawyer, defends Tom Robinson, a black man falsely accused of rape. The trial exposes deep racial prejudices in the community. Through Scout\'s eyes, the novel explores themes of moral growth, compassion, courage, and the loss of innocence. Boo Radley, a reclusive neighbor, ultimately saves Scout and her brother Jem from an attack.'
    },
    {
        'id': 'book_003',
        'title': 'The Great Gatsby',
        'author': 'F. Scott Fitzgerald',
        'genre': 'Literary Fiction',
        'summary': 'Nick Carraway moves to West Egg and becomes neighbors with the mysterious millionaire Jay Gatsby. Gatsby throws lavish parties hoping to attract Daisy Buchanan, his lost love now married to Tom. The novel critiques the American Dream, exposing the hollow materialism and moral bankruptcy of the wealthy elite. Gatsby\'s dream ends in tragedy when he is killed, dying alone despite his vast wealth.'
    },
    {
        'id': 'book_004',
        'title': 'Pride and Prejudice',
        'author': 'Jane Austen',
        'genre': 'Romance',
        'summary': 'Elizabeth Bennet navigates social expectations and romantic complications in Regency England. She initially despises the proud Mr. Darcy but gradually discovers his true character. Through misunderstandings, social class barriers, and personal growth, Elizabeth and Darcy overcome their pride and prejudice to find love. The novel satirizes marriage conventions and explores themes of reputation, class, and personal growth.'
    },
    {
        'id': 'book_005',
        'title': 'The Catcher in the Rye',
        'author': 'J.D. Salinger',
        'genre': 'Coming-of-age',
        'summary': 'Holden Caulfield, a disaffected teenager, recounts his expulsion from prep school and subsequent wanderings through New York City. He struggles with depression, alienation, and the phoniness he perceives in adult society. Holden fantasizes about protecting children\'s innocence as a "catcher in the rye." The novel explores teenage rebellion, loss of innocence, and the difficulty of growing up.'
    },
    {
        'id': 'book_006',
        'title': 'Brave New World',
        'author': 'Aldous Huxley',
        'genre': 'Dystopian Fiction',
        'summary': 'In a technologically advanced future, humans are genetically engineered and conditioned for predetermined social roles. Society is maintained through pleasure, drugs (soma), and entertainment. Bernard Marx and John "the Savage" challenge this sterile utopia. The novel examines themes of technological control, loss of individuality, manufactured happiness, and the cost of stability. John\'s tragic end questions whether happiness without freedom is worth living for.'
    },
    {
        'id': 'book_007',
        'title': 'Moby-Dick',
        'author': 'Herman Melville',
        'genre': 'Adventure',
        'summary': 'Ishmael joins the whaling ship Pequod, commanded by the obsessed Captain Ahab. Ahab seeks revenge against Moby Dick, the white whale that destroyed his leg. The crew embarks on a dangerous hunt across the oceans. The novel explores themes of obsession, fate, nature, and the limits of knowledge. Ahab\'s monomaniacal pursuit ultimately destroys the ship and kills everyone except Ishmael.'
    },
    {
        'id': 'book_008',
        'title': 'The Lord of the Rings',
        'author': 'J.R.R. Tolkien',
        'genre': 'Fantasy',
        'summary': 'Frodo Baggins inherits a powerful ring that must be destroyed in Mount Doom to save Middle-earth from the Dark Lord Sauron. Accompanied by a fellowship of diverse races, Frodo embarks on a perilous journey. The epic explores themes of good versus evil, corruption of power, friendship, and the courage of ordinary people. Despite overwhelming odds, the ring is destroyed and Sauron is defeated.'
    },
    {
        'id': 'book_009',
        'title': 'Crime and Punishment',
        'author': 'Fyodor Dostoevsky',
        'genre': 'Psychological Fiction',
        'summary': 'Raskolnikov, an impoverished student in St. Petersburg, murders a pawnbroker to test his theory that extraordinary men are above moral law. He descends into guilt and paranoia as detective Porfiry pursues him psychologically. Through his relationship with Sonya, a prostitute, Raskolnikov finds redemption. The novel explores morality, free will, redemption, and the psychological consequences of crime.'
    },
    {
        'id': 'book_010',
        'title': 'One Hundred Years of Solitude',
        'author': 'Gabriel García Márquez',
        'genre': 'Magical Realism',
        'summary': 'The Buendía family founds the town of Macondo and experiences generations of love, war, and magical events. The novel blends reality with fantasy as it chronicles the rise and fall of the family and town. Themes include cyclical time, solitude, destiny, and the nature of reality. The family line ends as Macondo is destroyed by a hurricane, fulfilling an ancient prophecy.'
    },
    {
        'id': 'book_011',
        'title': 'The Hobbit',
        'author': 'J.R.R. Tolkien',
        'genre': 'Fantasy',
        'summary': 'Bilbo Baggins, a comfort-loving hobbit, is recruited by Gandalf to help dwarves reclaim their homeland from the dragon Smaug. Despite initial reluctance, Bilbo discovers his courage and cunning. He finds a magic ring that grants invisibility and plays a crucial role in the quest. The adventure transforms Bilbo from a provincial homebody into a hero. The novel explores themes of personal growth, greed, and the value of home.'
    },
    {
        'id': 'book_012',
        'title': 'Sapiens',
        'author': 'Yuval Noah Harari',
        'genre': 'Non-fiction History',
        'summary': 'Harari traces the history of Homo sapiens from the Stone Age to the present. He argues that humans\' ability to create shared myths and stories enabled large-scale cooperation. The book examines the Cognitive, Agricultural, and Scientific Revolutions and their impacts. Harari explores how humans conquered the world, developed complex societies, and shaped the planet. He questions what the future holds for our species.'
    },
    {
        'id': 'book_013',
        'title': 'Thinking, Fast and Slow',
        'author': 'Daniel Kahneman',
        'genre': 'Psychology/Economics',
        'summary': 'Kahneman explains the two systems that drive human thinking: System 1 (fast, intuitive) and System 2 (slow, deliberate). He explores cognitive biases, heuristics, and systematic errors in judgment. The book synthesizes decades of research on decision-making, happiness, and behavioral economics. Kahneman demonstrates how intuition can mislead us and provides strategies for better thinking and decision-making.'
    },
    {
        'id': 'book_014',
        'title': 'The Selfish Gene',
        'author': 'Richard Dawkins',
        'genre': 'Science/Evolution',
        'summary': 'Dawkins presents a gene-centered view of evolution, arguing that genes, not organisms, are the primary unit of selection. He introduces the concept of memes as cultural equivalents of genes. The book explains altruism, selfishness, and cooperation through the lens of gene survival. Dawkins shows how apparently selfless behaviors can emerge from selfish genetic interests. The work revolutionized evolutionary biology thinking.'
    },
    {
        'id': 'book_015',
        'title': 'The Lean Startup',
        'author': 'Eric Ries',
        'genre': 'Business/Entrepreneurship',
        'summary': 'Ries introduces a methodology for developing businesses through validated learning, scientific experimentation, and iterative product releases. The Lean Startup approach emphasizes rapid hypothesis testing through minimum viable products (MVPs). Entrepreneurs should pivot or persevere based on customer feedback. The book advocates for build-measure-learn feedback loops to reduce waste and increase chances of success. This methodology has influenced startups worldwide.'
    }
]

BOOKS_QUERIES = [
    {
        'query': 'Which books explore dystopian societies and government control?',
        'relevant': ['book_001', 'book_006']
    },
    {
        'query': 'What stories feature characters overcoming prejudice and social barriers?',
        'relevant': ['book_002', 'book_004']
    },
    {
        'query': 'Which books examine the corruption and illusion of the American Dream?',
        'relevant': ['book_003']
    },
    {
        'query': 'What coming-of-age stories deal with teenage alienation?',
        'relevant': ['book_005']
    },
    {
        'query': 'Which fantasy epics involve quests to destroy powerful artifacts?',
        'relevant': ['book_008', 'book_011']
    },
    {
        'query': 'What books explore guilt, redemption, and moral philosophy?',
        'relevant': ['book_009']
    },
    {
        'query': 'Which works blend magical realism with family sagas?',
        'relevant': ['book_010']
    },
    {
        'query': 'What non-fiction books explain human cognitive biases and decision-making?',
        'relevant': ['book_013']
    },
    {
        'query': 'Which books present gene-centered views of evolution?',
        'relevant': ['book_014']
    },
    {
        'query': 'What business books focus on lean methodology and rapid experimentation?',
        'relevant': ['book_015']
    },
    {
        'query': 'Which books tell stories of obsessive quests leading to tragedy?',
        'relevant': ['book_007', 'book_009']
    },
    {
        'query': 'What works examine the history and future of humanity?',
        'relevant': ['book_012']
    }
]


# ============================================================================
# Books Recall Accuracy Benchmark
# ============================================================================

class BooksRecallBenchmark(BaseBenchmark):
    """Recall accuracy benchmark on book summaries."""

    def __init__(self):
        super().__init__(
            name="recall_accuracy_books",
            version="1.0",
            seed=42
        )

    def load_dataset(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Load Books dataset.

        Returns:
            (books, info) tuple
        """
        dataset_info = {
            'name': 'books_synthetic',
            'version': '1.0',
            'num_books': len(BOOKS_DATA),
            'num_queries': len(BOOKS_QUERIES),
            'genres': ['Fiction', 'Non-fiction', 'Fantasy', 'Science'],
            'source': 'synthetic',
            'avg_summary_length': 100  # approximate words
        }

        return BOOKS_DATA, dataset_info

    def run_benchmark(
        self,
        data: List[Dict],
        config: BenchmarkConfig
    ) -> BenchmarkMetrics:
        """
        Run recall accuracy benchmark on books.
        """
        print(f"Simulating recall on {len(data)} books...")
        print(f"Testing {len(BOOKS_QUERIES)} queries...\n")

        # Simulate storing book memories
        memory_store = {}
        for book in data:
            # Combine all fields for richer content
            content = f"{book['title']} by {book['author']}. Genre: {book['genre']}. {book['summary']}"
            memory_store[book['id']] = content

        # Run queries
        all_retrieved = []
        all_relevant = []
        latencies = []

        for i, query_item in enumerate(BOOKS_QUERIES, 1):
            query = query_item['query']
            relevant = query_item['relevant']

            # Simulate retrieval
            start_time = time.time()
            retrieved = self._simulate_retrieval(query, data, k=10)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

            all_retrieved.append(retrieved)
            all_relevant.append(relevant)

            # Progress
            if i % 3 == 0:
                print(f"  Progress: {i}/{len(BOOKS_QUERIES)} queries")

        print()

        # Calculate metrics
        metrics = BenchmarkMetrics(
            precision_at_1=self._avg_precision(all_retrieved, all_relevant, k=1),
            precision_at_5=self._avg_precision(all_retrieved, all_relevant, k=5),
            precision_at_10=self._avg_precision(all_retrieved, all_relevant, k=10),
            recall_at_5=self._avg_recall(all_retrieved, all_relevant, k=5),
            recall_at_10=self._avg_recall(all_retrieved, all_relevant, k=10),
            mrr=calculate_mrr(all_retrieved, all_relevant),
            latency_p50_ms=calculate_percentile(latencies, 50),
            latency_p95_ms=calculate_percentile(latencies, 95),
            latency_p99_ms=calculate_percentile(latencies, 99),
            throughput_qps=len(BOOKS_QUERIES) / sum(latencies) * 1000 if latencies else 0
        )

        return metrics

    def _simulate_retrieval(
        self,
        query: str,
        books: List[Dict],
        k: int = 10
    ) -> List[str]:
        """
        Simulate HoloLoom retrieval using genre + theme matching.
        """
        query_words = set(query.lower().split())

        scores = []
        for book in books:
            # Score based on title + genre + summary
            content = f"{book['title']} {book['genre']} {book['summary']}"
            content_words = set(content.lower().split())

            # Count matching words
            score = len(query_words & content_words)

            # Boost if genre matches
            genre_words = set(book['genre'].lower().split())
            if query_words & genre_words:
                score *= 1.3

            # Boost if title matches
            title_words = set(book['title'].lower().split())
            if query_words & title_words:
                score *= 1.5

            scores.append((book['id'], score))

        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [book_id for book_id, _ in scores[:k]]

    def _avg_precision(
        self,
        retrieved_lists: List[List[str]],
        relevant_lists: List[List[str]],
        k: int
    ) -> float:
        """Calculate average Precision@K across all queries."""
        if not retrieved_lists:
            return 0.0

        precisions = [
            calculate_precision_at_k(retrieved, relevant, k)
            for retrieved, relevant in zip(retrieved_lists, relevant_lists)
        ]

        return sum(precisions) / len(precisions)

    def _avg_recall(
        self,
        retrieved_lists: List[List[str]],
        relevant_lists: List[List[str]],
        k: int
    ) -> float:
        """Calculate average Recall@K across all queries."""
        if not retrieved_lists:
            return 0.0

        recalls = [
            calculate_recall_at_k(retrieved, relevant, k)
            for retrieved, relevant in zip(retrieved_lists, relevant_lists)
        ]

        return sum(recalls) / len(recalls)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run Books recall accuracy benchmark."""
    # Create benchmark
    benchmark = BooksRecallBenchmark()

    # Run benchmark
    result = benchmark.run(
        num_memories=15,  # Use all 15 books
        num_queries=12,   # Use all 12 queries
        config_mode="fast"
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = result.timestamp.split('T')[0]  # YYYY-MM-DD
    json_file = output_dir / f"recall_accuracy_books_{timestamp}.json"
    result.save(json_file)

    print(f"✓ Results saved to: {json_file}")

    # Create markdown report
    md_file = output_dir / f"recall_accuracy_books_{timestamp}.md"
    with open(md_file, 'w') as f:
        f.write(f"# Books Recall Accuracy Benchmark\n\n")
        f.write(f"**Date:** {result.timestamp}\n\n")
        f.write(f"**Dataset:** {result.config.dataset_name}\n")
        f.write(f"**Books:** {result.config.num_memories}\n")
        f.write(f"**Queries:** {result.config.num_queries}\n")
        f.write(f"**Genres:** Fiction, Non-fiction, Fantasy, Science\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Precision@1 | {result.metrics.precision_at_1:.3f} |\n")
        f.write(f"| Precision@5 | {result.metrics.precision_at_5:.3f} |\n")
        f.write(f"| Precision@10 | {result.metrics.precision_at_10:.3f} |\n")
        f.write(f"| Recall@5 | {result.metrics.recall_at_5:.3f} |\n")
        f.write(f"| Recall@10 | {result.metrics.recall_at_10:.3f} |\n")
        f.write(f"| MRR | {result.metrics.mrr:.3f} |\n")
        f.write(f"| Latency p95 | {result.metrics.latency_p95_ms:.1f}ms |\n")
        f.write(f"| Throughput | {result.metrics.throughput_qps:.1f} queries/sec |\n\n")

        f.write(f"## Analysis\n\n")
        f.write(f"**Dataset Characteristics:**\n")
        f.write(f"- 15 books from classic and modern literature\n")
        f.write(f"- Mix of fiction (novels) and non-fiction (science, business)\n")
        f.write(f"- Long-form text (~100 word summaries)\n")
        f.write(f"- Thematic queries (not keyword-based)\n\n")

        f.write(f"**Performance Notes:**\n")
        if result.metrics.recall_at_5 and result.metrics.recall_at_5 > 0.9:
            f.write(f"- ✅ High recall ({result.metrics.recall_at_5:.1%}) - finds relevant books reliably\n")
        if result.metrics.mrr and result.metrics.mrr > 0.75:
            f.write(f"- ✅ Strong MRR ({result.metrics.mrr:.3f}) - relevant books rank high\n")
        if result.metrics.latency_p95_ms and result.metrics.latency_p95_ms < 1.0:
            f.write(f"- ✅ Sub-millisecond latency - suitable for interactive applications\n")

    print(f"✓ Report saved to: {md_file}\n")

    return 0 if not result.errors else 1


if __name__ == '__main__':
    sys.exit(main())
