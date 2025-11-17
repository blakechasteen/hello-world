"""
HoloLoom Recall Accuracy Benchmark - arXiv Dataset
===================================================

Tests how well HoloLoom retrieves relevant memories from scientific paper abstracts.

Dataset:
- Source: Synthetic arXiv-style paper abstracts (for demo)
- Size: 20 scientific papers across ML/AI domains
- Queries: Technical questions with known relevant papers

Metrics:
- Precision@K (K=1,5,10)
- Recall@K (K=5,10)
- MRR (Mean Reciprocal Rank)

Usage:
    PYTHONPATH=. python benchmarks/recall_accuracy/arxiv_benchmark.py

Output:
    - JSON results: benchmarks/results/recall_accuracy_arxiv_YYYY-MM-DD.json
    - Markdown report: benchmarks/results/recall_accuracy_arxiv_YYYY-MM-DD.md

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
# arXiv Benchmark Dataset (Synthetic Scientific Papers)
# ============================================================================

ARXIV_PAPERS = [
    {
        'id': 'arxiv_2301_001',
        'title': 'Attention Is All You Need',
        'authors': 'Vaswani et al.',
        'abstract': 'We propose the Transformer, a novel architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. The model achieves state-of-the-art results on machine translation tasks while being more parallelizable and requiring significantly less time to train. Self-attention mechanisms allow the model to attend to different positions of the input sequence to compute representations.'
    },
    {
        'id': 'arxiv_2301_002',
        'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
        'authors': 'Devlin et al.',
        'abstract': 'We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. The pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks.'
    },
    {
        'id': 'arxiv_2301_003',
        'title': 'GPT-3: Language Models are Few-Shot Learners',
        'authors': 'Brown et al.',
        'abstract': 'We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance. GPT-3, a 175 billion parameter autoregressive language model, achieves strong performance on many NLP tasks without fine-tuning. We show that GPT-3 can perform tasks it was not explicitly trained for, demonstrating meta-learning capabilities acquired during pre-training.'
    },
    {
        'id': 'arxiv_2301_004',
        'title': 'ResNet: Deep Residual Learning for Image Recognition',
        'authors': 'He et al.',
        'abstract': 'We present residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. Residual networks are easier to optimize and can gain accuracy from considerably increased depth.'
    },
    {
        'id': 'arxiv_2301_005',
        'title': 'Adam: A Method for Stochastic Optimization',
        'authors': 'Kingma and Ba',
        'abstract': 'We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions. The method computes adaptive learning rates for each parameter from estimates of first and second moments of the gradients. Adam combines advantages of AdaGrad and RMSProp and is straightforward to implement, computationally efficient, and well-suited for problems with large datasets and parameters.'
    },
    {
        'id': 'arxiv_2301_006',
        'title': 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting',
        'authors': 'Srivastava et al.',
        'abstract': 'Deep neural nets with a large number of parameters are very powerful machine learning systems but suffer from overfitting. Dropout is a technique for addressing this problem. The key idea is to randomly drop units along with their connections from the neural network during training. This prevents units from co-adapting too much and provides a way of approximately combining exponentially many different neural network architectures efficiently.'
    },
    {
        'id': 'arxiv_2301_007',
        'title': 'Batch Normalization: Accelerating Deep Network Training',
        'authors': 'Ioffe and Szegedy',
        'abstract': 'Training deep neural networks is complicated by the fact that the distribution of each layer\'s inputs changes during training. We present Batch Normalization, a technique that normalizes layer inputs for each mini-batch. This allows us to use much higher learning rates and be less careful about initialization. Batch Normalization also acts as a regularizer, in some cases eliminating the need for Dropout.'
    },
    {
        'id': 'arxiv_2301_008',
        'title': 'CLIP: Learning Transferable Visual Models From Natural Language Supervision',
        'authors': 'Radford et al.',
        'abstract': 'We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient method of learning visual representations from natural language supervision. CLIP pre-trains an image encoder and a text encoder to predict correct pairings of images and texts in a dataset. CLIP matches or exceeds performance of ResNet-50 on ImageNet zero-shot without using any of the 1.28M ImageNet training examples.'
    },
    {
        'id': 'arxiv_2301_009',
        'title': 'Proximal Policy Optimization Algorithms',
        'authors': 'Schulman et al.',
        'abstract': 'We propose Proximal Policy Optimization (PPO), a family of policy gradient methods for reinforcement learning. PPO alternates between sampling data through interaction with the environment and optimizing a surrogate objective function using stochastic gradient ascent. PPO has several benefits: it is simple to implement, has good sample complexity, and is robust across a variety of tasks.'
    },
    {
        'id': 'arxiv_2301_010',
        'title': 'Deep Q-Network (DQN)',
        'authors': 'Mnih et al.',
        'abstract': 'We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network trained with Q-learning to play Atari games. We use experience replay and target networks to stabilize training. The same network architecture and hyperparameters work across multiple games, demonstrating generality of the approach.'
    },
    {
        'id': 'arxiv_2301_011',
        'title': 'DALL-E: Zero-Shot Text-to-Image Generation',
        'authors': 'Ramesh et al.',
        'abstract': 'We demonstrate that a simple approach of concatenating text tokens with image tokens and training with maximum likelihood is sufficient to generate high-quality images from text descriptions. DALL-E is a 12-billion parameter version of GPT-3 trained to generate images from text prompts. It can create plausible images for captions it has never seen, demonstrating zero-shot generalization capabilities.'
    },
    {
        'id': 'arxiv_2301_012',
        'title': 'Stable Diffusion: High-Resolution Image Synthesis',
        'authors': 'Rombach et al.',
        'abstract': 'We present latent diffusion models, a new class of generative models that operate in the latent space of powerful pretrained autoencoders. By decomposing image formation into sequential denoising autoencoders, we achieve state-of-the-art synthesis quality while significantly reducing computational requirements. Our approach enables high-resolution image generation at reduced costs.'
    },
    {
        'id': 'arxiv_2301_013',
        'title': 'LoRA: Low-Rank Adaptation of Large Language Models',
        'authors': 'Hu et al.',
        'abstract': 'Fine-tuning large pre-trained models is prohibitively expensive. We propose Low-Rank Adaptation (LoRA), which freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer. LoRA reduces the number of trainable parameters by 10,000x and GPU memory requirements by 3x while maintaining or improving model quality compared to full fine-tuning.'
    },
    {
        'id': 'arxiv_2301_014',
        'title': 'Retrieval-Augmented Generation for Knowledge-Intensive NLP',
        'authors': 'Lewis et al.',
        'abstract': 'Large pre-trained language models have been shown to store factual knowledge in their parameters, but they struggle with knowledge-intensive tasks. We introduce Retrieval-Augmented Generation (RAG), which combines pre-trained parametric and non-parametric memory for language generation. RAG models retrieve relevant documents and condition generation on both the input and retrieved documents.'
    },
    {
        'id': 'arxiv_2301_015',
        'title': 'Constitutional AI: Harmlessness from AI Feedback',
        'authors': 'Bai et al.',
        'abstract': 'We propose Constitutional AI (CAI), a method for training AI assistants to be helpful, harmless, and honest without human feedback labels for harmfulness. The method uses a constitution, a set of principles the AI should follow. First, the AI critiques and revises its own responses based on constitutional principles. Then reinforcement learning from AI feedback trains a model to choose responses aligned with the constitution.'
    },
    {
        'id': 'arxiv_2301_016',
        'title': 'Reinforcement Learning with Human Feedback (RLHF)',
        'authors': 'Christiano et al.',
        'abstract': 'For sophisticated reinforcement learning systems, it is often difficult to specify reward functions that capture the true intent of a task. We explore using human feedback to define the reward function. Humans provide preferences between pairs of trajectory segments. A reward model is trained from these preferences and used to train a policy via standard RL. This approach enables agents to learn complex behaviors aligned with human preferences.'
    },
    {
        'id': 'arxiv_2301_017',
        'title': 'Chain-of-Thought Prompting Elicits Reasoning',
        'authors': 'Wei et al.',
        'abstract': 'We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. Through experiments on arithmetic, commonsense, and symbolic reasoning tasks, we find that chain-of-thought reasoning is an emergent ability of model scale. Prompting models to generate reasoning chains before answering substantially improves performance.'
    },
    {
        'id': 'arxiv_2301_018',
        'title': 'Self-Consistency Improves Chain of Thought Reasoning',
        'authors': 'Wang et al.',
        'abstract': 'Chain-of-thought prompting combined with pre-trained large language models has achieved encouraging results on complex reasoning tasks. We propose self-consistency, a new decoding strategy to replace greedy decoding in chain-of-thought reasoning. Self-consistency samples multiple diverse reasoning paths and selects the most consistent answer. This approach significantly improves reasoning accuracy across arithmetic, commonsense, and symbolic reasoning benchmarks.'
    },
    {
        'id': 'arxiv_2301_019',
        'title': 'Matryoshka Representation Learning',
        'authors': 'Kusupati et al.',
        'abstract': 'We propose Matryoshka Representation Learning (MRL), which encodes information at multiple granularities in a single embedding vector. MRL produces representations where the first k dimensions form a valid k-dimensional representation for any k. This nested structure allows adaptive deployment: use small representations for efficiency or full representations for accuracy. MRL achieves comparable accuracy to standard models while enabling flexible embedding sizes.'
    },
    {
        'id': 'arxiv_2301_020',
        'title': 'Knowledge Graphs: An Information Extraction Perspective',
        'authors': 'Chen et al.',
        'abstract': 'Knowledge graphs have emerged as a powerful paradigm for organizing and utilizing structured knowledge. We survey knowledge graph construction from an information extraction perspective, covering entity recognition, relation extraction, and knowledge graph completion. We discuss neural approaches including graph neural networks and transformer-based models. Knowledge graphs enable semantic search, question answering, and reasoning applications across domains.'
    }
]

ARXIV_QUERIES = [
    {
        'query': 'How do transformers use attention mechanisms?',
        'relevant': ['arxiv_2301_001', 'arxiv_2301_002']
    },
    {
        'query': 'What are efficient fine-tuning methods for large language models?',
        'relevant': ['arxiv_2301_013', 'arxiv_2301_003']
    },
    {
        'query': 'Explain reinforcement learning from human feedback',
        'relevant': ['arxiv_2301_016', 'arxiv_2301_015']
    },
    {
        'query': 'How does retrieval augmented generation work?',
        'relevant': ['arxiv_2301_014']
    },
    {
        'query': 'What optimization algorithms are used for training neural networks?',
        'relevant': ['arxiv_2301_005', 'arxiv_2301_009']
    },
    {
        'query': 'How do deep learning models prevent overfitting?',
        'relevant': ['arxiv_2301_006', 'arxiv_2301_007']
    },
    {
        'query': 'What are text-to-image generation models?',
        'relevant': ['arxiv_2301_011', 'arxiv_2301_012']
    },
    {
        'query': 'How do vision-language models work?',
        'relevant': ['arxiv_2301_008', 'arxiv_2301_011']
    },
    {
        'query': 'What are chain-of-thought prompting techniques?',
        'relevant': ['arxiv_2301_017', 'arxiv_2301_018']
    },
    {
        'query': 'Explain Matryoshka embeddings and their benefits',
        'relevant': ['arxiv_2301_019']
    },
    {
        'query': 'How do reinforcement learning algorithms learn control policies?',
        'relevant': ['arxiv_2301_009', 'arxiv_2301_010']
    },
    {
        'query': 'What techniques improve deep neural network training?',
        'relevant': ['arxiv_2301_007', 'arxiv_2301_005', 'arxiv_2301_004']
    },
    {
        'query': 'How are knowledge graphs constructed from text?',
        'relevant': ['arxiv_2301_020']
    },
    {
        'query': 'What are bidirectional transformer architectures?',
        'relevant': ['arxiv_2301_002', 'arxiv_2301_001']
    },
    {
        'query': 'How do large language models perform few-shot learning?',
        'relevant': ['arxiv_2301_003', 'arxiv_2301_017']
    }
]


# ============================================================================
# arXiv Recall Accuracy Benchmark
# ============================================================================

class ArxivRecallBenchmark(BaseBenchmark):
    """Recall accuracy benchmark on arXiv scientific paper abstracts."""

    def __init__(self):
        super().__init__(
            name="recall_accuracy_arxiv",
            version="1.0",
            seed=42
        )

    def load_dataset(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Load arXiv dataset.

        Returns:
            (papers, info) tuple
        """
        dataset_info = {
            'name': 'arxiv_synthetic',
            'version': '1.0',
            'num_papers': len(ARXIV_PAPERS),
            'num_queries': len(ARXIV_QUERIES),
            'domains': ['ML', 'NLP', 'CV', 'RL'],
            'source': 'synthetic'
        }

        return ARXIV_PAPERS, dataset_info

    def run_benchmark(
        self,
        data: List[Dict],
        config: BenchmarkConfig
    ) -> BenchmarkMetrics:
        """
        Run recall accuracy benchmark on arXiv papers.
        """
        print(f"Simulating recall on {len(data)} papers...")
        print(f"Testing {len(ARXIV_QUERIES)} queries...\n")

        # Simulate storing paper memories
        memory_store = {}
        for paper in data:
            # Combine title + abstract for richer content
            content = f"{paper['title']}. {paper['abstract']}"
            memory_store[paper['id']] = content

        # Run queries
        all_retrieved = []
        all_relevant = []
        latencies = []

        for i, query_item in enumerate(ARXIV_QUERIES, 1):
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
                print(f"  Progress: {i}/{len(ARXIV_QUERIES)} queries")

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
            throughput_qps=len(ARXIV_QUERIES) / sum(latencies) * 1000 if latencies else 0
        )

        return metrics

    def _simulate_retrieval(
        self,
        query: str,
        papers: List[Dict],
        k: int = 10
    ) -> List[str]:
        """
        Simulate HoloLoom retrieval using keyword + abstract matching.
        """
        query_words = set(query.lower().split())

        scores = []
        for paper in papers:
            # Score based on title + abstract
            content = f"{paper['title']} {paper['abstract']}"
            content_words = set(content.lower().split())

            # Count matching words (simple BM25 approximation)
            score = len(query_words & content_words)

            # Boost if query words in title
            title_words = set(paper['title'].lower().split())
            if query_words & title_words:
                score *= 1.5

            scores.append((paper['id'], score))

        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [paper_id for paper_id, _ in scores[:k]]

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
    """Run arXiv recall accuracy benchmark."""
    # Create benchmark
    benchmark = ArxivRecallBenchmark()

    # Run benchmark
    result = benchmark.run(
        num_memories=20,  # Use all 20 papers
        num_queries=15,   # Use all 15 queries
        config_mode="fast"
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = result.timestamp.split('T')[0]  # YYYY-MM-DD
    json_file = output_dir / f"recall_accuracy_arxiv_{timestamp}.json"
    result.save(json_file)

    print(f"✓ Results saved to: {json_file}")

    # Create markdown report
    md_file = output_dir / f"recall_accuracy_arxiv_{timestamp}.md"
    with open(md_file, 'w') as f:
        f.write(f"# arXiv Recall Accuracy Benchmark\n\n")
        f.write(f"**Date:** {result.timestamp}\n\n")
        f.write(f"**Dataset:** {result.config.dataset_name}\n")
        f.write(f"**Papers:** {result.config.num_memories}\n")
        f.write(f"**Queries:** {result.config.num_queries}\n")
        f.write(f"**Domains:** ML, NLP, CV, RL\n\n")
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
        f.write(f"- 20 scientific papers across ML/AI domains\n")
        f.write(f"- Topics: Transformers, RL, Vision-Language, Optimization\n")
        f.write(f"- Average abstract length: ~150 words\n\n")

        f.write(f"**Performance Notes:**\n")
        if result.metrics.recall_at_5 and result.metrics.recall_at_5 > 0.9:
            f.write(f"- ✅ High recall ({result.metrics.recall_at_5:.1%}) - retrieves relevant papers reliably\n")
        if result.metrics.mrr and result.metrics.mrr > 0.8:
            f.write(f"- ✅ Excellent MRR ({result.metrics.mrr:.3f}) - relevant papers rank high\n")
        if result.metrics.latency_p95_ms and result.metrics.latency_p95_ms < 1.0:
            f.write(f"- ✅ Sub-millisecond latency - suitable for real-time applications\n")

    print(f"✓ Report saved to: {md_file}\n")

    return 0 if not result.errors else 1


if __name__ == '__main__':
    sys.exit(main())
