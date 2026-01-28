"""
Shared test datasets for benchmarks.
"""

from typing import List, Dict, Any

# ============================================================================
# CORPUS: Documents to search over
# ============================================================================

EVALUATION_CORPUS: List[Dict[str, str]] = [
    # Machine Learning - Bandits
    {"id": "ml_1", "text": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. It balances exploration and exploitation by sampling from posterior distributions of each arm's reward probability."},
    {"id": "ml_2", "text": "The multi-armed bandit problem is a classic reinforcement learning problem where an agent must choose between multiple options (arms) with uncertain rewards to maximize cumulative reward."},
    {"id": "ml_3", "text": "UCB (Upper Confidence Bound) is another algorithm for the multi-armed bandit problem that uses confidence intervals to balance exploration and exploitation deterministically."},
    {"id": "ml_4", "text": "Epsilon-greedy is a simple exploration strategy that explores randomly with probability epsilon and exploits the best known option with probability 1-epsilon."},
    {"id": "ml_5", "text": "The exploration-exploitation tradeoff is fundamental to decision making under uncertainty. Too much exploration wastes resources, too little misses better options."},

    # Machine Learning - Bayesian
    {"id": "ml_6", "text": "Bayesian inference updates prior beliefs with observed data using Bayes' theorem to form posterior distributions. It is fundamental to probabilistic machine learning."},
    {"id": "ml_7", "text": "Posterior distributions in Bayesian statistics represent updated beliefs after observing data, combining prior knowledge with likelihood of observed evidence."},
    {"id": "ml_8", "text": "Prior distributions encode our beliefs before seeing data. Choosing appropriate priors is crucial in Bayesian modeling and can significantly affect results."},

    # Neural Networks
    {"id": "nn_1", "text": "Neural networks are computational models inspired by biological neurons. They learn representations through backpropagation of errors and gradient descent optimization."},
    {"id": "nn_2", "text": "Transformers use self-attention mechanisms to process sequences in parallel. They are the foundation of modern language models like GPT, BERT, and Claude."},
    {"id": "nn_3", "text": "Attention mechanisms allow neural networks to focus on relevant parts of the input when producing output, enabling better handling of long-range dependencies."},
    {"id": "nn_4", "text": "Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to network weights."},

    # Information Retrieval
    {"id": "ir_1", "text": "BM25 is a probabilistic ranking function used in information retrieval based on term frequency, inverse document frequency, and document length normalization."},
    {"id": "ir_2", "text": "Cosine similarity measures the angle between two vectors, commonly used to compare text embeddings. Values range from -1 (opposite) to 1 (identical)."},
    {"id": "ir_3", "text": "TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic reflecting how important a word is to a document in a corpus."},
    {"id": "ir_4", "text": "Vector embeddings represent text as dense numerical vectors in a high-dimensional space where semantically similar texts are close together."},

    # RAG and Knowledge Systems
    {"id": "rag_1", "text": "RAG (Retrieval-Augmented Generation) combines retrieval systems with language models to ground responses in external knowledge, reducing hallucinations."},
    {"id": "rag_2", "text": "Knowledge graphs represent entities and relationships as nodes and edges, enabling structured reasoning over facts and multi-hop queries."},
    {"id": "rag_3", "text": "Hybrid retrieval combines dense vector search with sparse keyword matching (like BM25) using techniques like Reciprocal Rank Fusion."},
    {"id": "rag_4", "text": "Semantic search uses embeddings to find documents by meaning rather than exact keyword matches, enabling more natural query understanding."},

    # HoloLoom Specific
    {"id": "hl_1", "text": "Matryoshka embeddings encode information at multiple scales (like nested dolls), allowing flexible precision vs computation tradeoffs by truncating dimensions."},
    {"id": "hl_2", "text": "Hot pattern feedback tracks which knowledge elements are frequently accessed and boosts their retrieval weight, adapting to usage patterns."},
    {"id": "hl_3", "text": "Thompson Sampling in policy selection samples from Beta distributions of each tool's success rate, naturally balancing exploration of new tools with exploitation of known good ones."},
    {"id": "hl_4", "text": "Multipass retrieval performs multiple rounds of search with increasing importance thresholds, progressively filtering to the most relevant results."},

    # General CS
    {"id": "cs_1", "text": "Reinforcement learning trains agents to make sequential decisions by maximizing cumulative rewards through trial and error in an environment."},
    {"id": "cs_2", "text": "Caching stores frequently accessed data in fast memory to reduce latency. Cache hit rates measure how often requested data is found in cache."},
    {"id": "cs_3", "text": "Latency is the time delay between a request and response. P95 latency means 95% of requests complete within that time."},

    # Optimization & Training (expanded January 2026)
    {"id": "opt_1", "text": "Gradient descent is an optimization algorithm that iteratively adjusts parameters in the direction of steepest descent of the loss function. Learning rate controls step size."},
    {"id": "opt_2", "text": "Adam optimizer combines momentum and adaptive learning rates. It maintains exponentially decaying averages of past gradients and squared gradients."},
    {"id": "opt_3", "text": "Overfitting occurs when a model memorizes training data rather than learning generalizable patterns. Regularization techniques like dropout and weight decay help prevent it."},

    # Evaluation Metrics
    {"id": "eval_1", "text": "Precision measures the fraction of retrieved documents that are relevant. Recall measures the fraction of relevant documents that are retrieved. The F1 score is their harmonic mean."},
    {"id": "eval_2", "text": "NDCG (Normalized Discounted Cumulative Gain) evaluates ranking quality by giving higher weight to relevant documents appearing earlier in the results list."},
    {"id": "eval_3", "text": "Mean Reciprocal Rank (MRR) measures the average position of the first relevant result across queries. Higher MRR indicates better top-result placement."},

    # Graph & Structured Knowledge
    {"id": "gk_1", "text": "Graph neural networks operate on graph-structured data by aggregating information from neighboring nodes. They enable learning on social networks, molecular structures, and knowledge graphs."},
    {"id": "gk_2", "text": "Entity resolution identifies when different records refer to the same real-world entity. It is crucial for knowledge graph construction and data integration."},
    {"id": "gk_3", "text": "Multi-hop reasoning traverses multiple edges in a knowledge graph to answer complex queries that require connecting multiple facts together."},

    # Embeddings & Representations
    {"id": "emb_1", "text": "Word2Vec learns word embeddings by predicting surrounding words (CBOW) or predicting a word from context (Skip-gram). Similar words have similar vector representations."},
    {"id": "emb_2", "text": "Sentence embeddings capture the meaning of entire sentences as fixed-length vectors. Models like Sentence-BERT fine-tune transformers for semantic similarity tasks."},
    {"id": "emb_3", "text": "Dimensionality reduction techniques like PCA and t-SNE project high-dimensional embeddings into lower dimensions for visualization while preserving important structure."},

    # Systems & Scalability
    {"id": "sys_1", "text": "Distributed systems split computation across multiple machines for scalability. Consensus protocols like Raft and Paxos ensure consistency despite failures."},
    {"id": "sys_2", "text": "Vector databases like Qdrant and Pinecone are optimized for approximate nearest neighbor search on high-dimensional embeddings using algorithms like HNSW."},
    {"id": "sys_3", "text": "Batch processing handles large datasets efficiently by processing records in groups rather than individually. Stream processing handles data in real-time as it arrives."},
]

# ============================================================================
# QUERIES: Test queries with ground truth relevance
# ============================================================================

EVALUATION_QUERIES: List[Dict[str, Any]] = [
    # Easy - Direct factual
    {
        "id": "q1",
        "query": "What is Thompson Sampling?",
        "relevant_ids": ["ml_1", "ml_2", "hl_3"],
        "difficulty": "easy",
        "type": "factual",
    },
    {
        "id": "q2",
        "query": "How do transformers work?",
        "relevant_ids": ["nn_2", "nn_3"],
        "difficulty": "easy",
        "type": "factual",
    },
    {
        "id": "q3",
        "query": "What is BM25?",
        "relevant_ids": ["ir_1", "ir_3"],
        "difficulty": "easy",
        "type": "factual",
    },

    # Medium - Requires understanding relationships
    {
        "id": "q4",
        "query": "How do bandit algorithms balance exploration and exploitation?",
        "relevant_ids": ["ml_1", "ml_3", "ml_4", "ml_5", "hl_3"],
        "difficulty": "medium",
        "type": "analytical",
    },
    {
        "id": "q5",
        "query": "Explain Bayesian approaches to machine learning",
        "relevant_ids": ["ml_1", "ml_6", "ml_7", "ml_8"],
        "difficulty": "medium",
        "type": "conceptual",
    },
    {
        "id": "q6",
        "query": "What retrieval methods are used in RAG systems?",
        "relevant_ids": ["rag_1", "rag_3", "rag_4", "ir_1", "ir_2", "ir_4"],
        "difficulty": "medium",
        "type": "factual",
    },
    {
        "id": "q7",
        "query": "How do knowledge graphs help with reasoning?",
        "relevant_ids": ["rag_2", "rag_1"],
        "difficulty": "medium",
        "type": "conceptual",
    },

    # Hard - Requires synthesis across topics
    {
        "id": "q8",
        "query": "Compare different exploration strategies in reinforcement learning",
        "relevant_ids": ["ml_1", "ml_3", "ml_4", "ml_5", "cs_1"],
        "difficulty": "hard",
        "type": "comparative",
    },
    {
        "id": "q9",
        "query": "How can hybrid retrieval improve search quality over vector-only approaches?",
        "relevant_ids": ["rag_3", "ir_1", "ir_2", "ir_4", "rag_4"],
        "difficulty": "hard",
        "type": "analytical",
    },
    {
        "id": "q10",
        "query": "What techniques does HoloLoom use to improve retrieval over time?",
        "relevant_ids": ["hl_1", "hl_2", "hl_3", "hl_4"],
        "difficulty": "hard",
        "type": "factual",
    },

    # Edge cases
    {
        "id": "q11",
        "query": "latency optimization caching performance",  # Keyword-heavy
        "relevant_ids": ["cs_2", "cs_3"],
        "difficulty": "easy",
        "type": "keyword",
    },
    {
        "id": "q12",
        "query": "How do neural networks learn?",  # Broad
        "relevant_ids": ["nn_1", "nn_4", "nn_3"],
        "difficulty": "medium",
        "type": "conceptual",
    },
]

# ============================================================================
# SUBSETS for different evaluation modes
# ============================================================================

def get_queries_for_mode(mode: str) -> List[Dict[str, Any]]:
    """Get appropriate query subset for evaluation mode."""
    if mode == "quick":
        # Just easy queries
        return [q for q in EVALUATION_QUERIES if q["difficulty"] == "easy"]
    elif mode == "standard":
        # Easy + medium
        return [q for q in EVALUATION_QUERIES if q["difficulty"] in ["easy", "medium"]]
    else:
        # Full - all queries
        return EVALUATION_QUERIES


def get_corpus() -> List[Dict[str, str]]:
    """Get the evaluation corpus."""
    return EVALUATION_CORPUS
