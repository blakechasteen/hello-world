"""
CRM Embedding Service

Production-ready embedding generation using HoloLoom's multi-scale embeddings.
Integrates with MatryoshkaEmbeddings for semantic similarity and search.

Features:
- Multi-scale embeddings (96, 192, 384 dimensions)
- Automatic caching for performance
- Fallback to simple embedder if sentence-transformers unavailable
- Entity-specific embedding generation
"""

from typing import Optional, List, Dict, Any
import numpy as np
import warnings

from crm_app.models import Contact, Company, Deal, Activity

# HoloLoom imports
try:
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    _HAVE_MATRYOSHKA = True
except ImportError:
    MatryoshkaEmbeddings = None
    _HAVE_MATRYOSHKA = False
    warnings.warn("MatryoshkaEmbeddings not available, using simple embedder")

try:
    from HoloLoom.input.simple_embedder import SimpleEmbedder
    _HAVE_SIMPLE = True
except ImportError:
    SimpleEmbedder = None
    _HAVE_SIMPLE = False


class CRMEmbeddingService:
    """
    Embedding service for CRM entities using HoloLoom's multi-scale embeddings

    This service generates semantic embeddings for contacts, companies, deals,
    and activities to enable similarity search and natural language queries.

    Architecture:
    - Uses HoloLoom's MatryoshkaEmbeddings (384d base with 96d, 192d scales)
    - Falls back to SimpleEmbedder if sentence-transformers unavailable
    - Caches embeddings for performance
    - Provides entity-specific text generation for best semantic quality
    """

    def __init__(
        self,
        scales: List[int] = None,
        use_matryoshka: bool = True,
        cache_enabled: bool = True
    ):
        """
        Initialize embedding service

        Args:
            scales: Embedding scales to use (default: [96, 192, 384])
            use_matryoshka: Use multi-scale embeddings if available
            cache_enabled: Enable embedding caching
        """
        self.scales = scales or [96, 192, 384]
        self.cache_enabled = cache_enabled
        self.embedding_dim = max(self.scales)  # Use largest scale by default

        # Initialize embedder
        if use_matryoshka and _HAVE_MATRYOSHKA:
            try:
                self.embedder = MatryoshkaEmbeddings(sizes=self.scales)
                self.backend = "matryoshka"
                print(f"[CRM Embedding] Using MatryoshkaEmbeddings with scales {self.scales}")
            except Exception as e:
                warnings.warn(f"Failed to initialize MatryoshkaEmbeddings: {e}")
                self.embedder = self._create_fallback_embedder()
                self.backend = "simple"
        else:
            self.embedder = self._create_fallback_embedder()
            self.backend = "simple"

    def _create_fallback_embedder(self):
        """Create fallback simple embedder"""
        if _HAVE_SIMPLE:
            print("[CRM Embedding] Using SimpleEmbedder (fallback)")
            return SimpleEmbedder(dimension=self.embedding_dim)
        else:
            # Ultimate fallback: basic hash-based embedder
            print("[CRM Embedding] Using basic hash embedder (fallback)")
            return self._BasicHashEmbedder(dimension=self.embedding_dim)

    # ========================================================================
    # Core Embedding Methods
    # ========================================================================

    def embed_text(self, text: str, scale: Optional[int] = None) -> np.ndarray:
        """
        Generate embedding for text

        Args:
            text: Text to embed
            scale: Specific scale to use (default: largest scale)

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            return np.zeros(scale or self.embedding_dim, dtype=np.float32)

        try:
            if self.backend == "matryoshka":
                # MatryoshkaEmbeddings
                embeddings = self.embedder.encode([text], scales=self.scales)

                # Get requested scale
                if scale and scale in self.scales:
                    scale_idx = self.scales.index(scale)
                    return embeddings[0][scale_idx]
                else:
                    # Return largest scale by default
                    return embeddings[0][-1]

            else:
                # SimpleEmbedder or fallback
                embedding = self.embedder.encode(text)

                # If scale requested and different from default, project
                if scale and scale != self.embedding_dim:
                    embedding = self._project_to_scale(embedding, scale)

                return embedding

        except Exception as e:
            warnings.warn(f"Embedding generation failed: {e}")
            return np.zeros(scale or self.embedding_dim, dtype=np.float32)

    def embed_batch(self, texts: List[str], scale: Optional[int] = None) -> np.ndarray:
        """
        Generate embeddings for batch of texts (more efficient)

        Args:
            texts: List of texts to embed
            scale: Specific scale to use (default: largest scale)

        Returns:
            Matrix of embeddings (n_texts × embedding_dim)
        """
        if not texts:
            return np.zeros((0, scale or self.embedding_dim), dtype=np.float32)

        try:
            if self.backend == "matryoshka":
                embeddings = self.embedder.encode(texts, scales=self.scales)

                if scale and scale in self.scales:
                    scale_idx = self.scales.index(scale)
                    return np.array([emb[scale_idx] for emb in embeddings])
                else:
                    return np.array([emb[-1] for emb in embeddings])

            else:
                embeddings = [self.embedder.encode(text) for text in texts]
                embeddings = np.array(embeddings)

                if scale and scale != self.embedding_dim:
                    embeddings = np.array([
                        self._project_to_scale(emb, scale)
                        for emb in embeddings
                    ])

                return embeddings

        except Exception as e:
            warnings.warn(f"Batch embedding failed: {e}")
            return np.zeros((len(texts), scale or self.embedding_dim), dtype=np.float32)

    # ========================================================================
    # Entity-Specific Methods
    # ========================================================================

    def embed_contact(
        self,
        contact: Contact,
        company: Optional[Company] = None,
        scale: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embedding for contact with optional company context

        Args:
            contact: Contact entity
            company: Optional company for richer context
            scale: Specific embedding scale

        Returns:
            Contact embedding vector
        """
        text = self._contact_to_text(contact, company)
        return self.embed_text(text, scale=scale)

    def embed_company(self, company: Company, scale: Optional[int] = None) -> np.ndarray:
        """Generate embedding for company"""
        text = self._company_to_text(company)
        return self.embed_text(text, scale=scale)

    def embed_deal(
        self,
        deal: Deal,
        contact: Optional[Contact] = None,
        company: Optional[Company] = None,
        scale: Optional[int] = None
    ) -> np.ndarray:
        """Generate embedding for deal with optional context"""
        text = self._deal_to_text(deal, contact, company)
        return self.embed_text(text, scale=scale)

    def embed_activity(
        self,
        activity: Activity,
        contact: Optional[Contact] = None,
        scale: Optional[int] = None
    ) -> np.ndarray:
        """Generate embedding for activity"""
        text = self._activity_to_text(activity, contact)
        return self.embed_text(text, scale=scale)

    # ========================================================================
    # Text Generation (Entity → Embedding-Ready Text)
    # ========================================================================

    def _contact_to_text(self, contact: Contact, company: Optional[Company] = None) -> str:
        """Convert contact to embedding-ready text"""
        parts = []

        # Name and title
        parts.append(f"Contact: {contact.name}")
        if contact.title:
            parts.append(f"Title: {contact.title}")

        # Company context
        if company:
            parts.append(f"Company: {company.name}")
            parts.append(f"Industry: {company.industry}")
            parts.append(f"Company Size: {company.size.value}")
        elif contact.company_id:
            parts.append(f"Has company affiliation")

        # Engagement
        if contact.engagement_level:
            parts.append(f"Engagement: {contact.engagement_level}")

        # Tags (keywords for semantic matching)
        if contact.tags:
            parts.append(f"Tags: {', '.join(contact.tags)}")

        # Notes (rich semantic content)
        if contact.notes:
            parts.append(f"Notes: {contact.notes}")

        return ". ".join(parts)

    def _company_to_text(self, company: Company) -> str:
        """Convert company to embedding-ready text"""
        parts = [
            f"Company: {company.name}",
            f"Industry: {company.industry}",
            f"Size: {company.size.value} employees"
        ]

        if company.website:
            parts.append(f"Website: {company.website}")

        if company.tags:
            parts.append(f"Tags: {', '.join(company.tags)}")

        if company.notes:
            parts.append(f"Notes: {company.notes}")

        return ". ".join(parts)

    def _deal_to_text(
        self,
        deal: Deal,
        contact: Optional[Contact] = None,
        company: Optional[Company] = None
    ) -> str:
        """Convert deal to embedding-ready text"""
        parts = [
            f"Deal: {deal.title}",
            f"Stage: {deal.stage.value}",
            f"Value: {deal.currency} {deal.value:,.0f}",
            f"Probability: {deal.probability * 100:.0f}%"
        ]

        if contact:
            parts.append(f"Contact: {contact.name}")

        if company:
            parts.append(f"Company: {company.name} ({company.industry})")

        if deal.notes:
            parts.append(f"Notes: {deal.notes}")

        return ". ".join(parts)

    def _activity_to_text(self, activity: Activity, contact: Optional[Contact] = None) -> str:
        """Convert activity to embedding-ready text"""
        parts = [
            f"Activity Type: {activity.type.value}",
            f"Subject: {activity.subject}" if activity.subject else ""
        ]

        if contact:
            parts.append(f"Contact: {contact.name}")

        if activity.content:
            parts.append(f"Content: {activity.content}")

        if activity.outcome:
            parts.append(f"Outcome: {activity.outcome.value}")

        return ". ".join(p for p in parts if p)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _project_to_scale(self, embedding: np.ndarray, target_scale: int) -> np.ndarray:
        """Project embedding to smaller scale (simple truncation)"""
        if len(embedding) <= target_scale:
            # Pad if too small
            padded = np.zeros(target_scale, dtype=np.float32)
            padded[:len(embedding)] = embedding
            return padded
        else:
            # Truncate and renormalize
            projected = embedding[:target_scale].copy()
            norm = np.linalg.norm(projected)
            if norm > 0:
                projected /= norm
            return projected

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    # ========================================================================
    # Fallback Embedder (Basic Hash-Based)
    # ========================================================================

    class _BasicHashEmbedder:
        """Basic hash-based embedder for ultimate fallback"""

        def __init__(self, dimension: int = 384):
            self.dimension = dimension

        def encode(self, text: str) -> np.ndarray:
            """Generate hash-based embedding"""
            import hashlib

            tokens = text.lower().split()
            embedding = np.zeros(self.dimension, dtype=np.float32)

            for token in tokens:
                if len(token) < 3:
                    continue

                hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
                idx = hash_val % self.dimension
                embedding[idx] += 1.0

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

            return embedding


# ============================================================================
# Convenience Functions
# ============================================================================

def create_embedding_service(
    scales: List[int] = None,
    backend: str = "auto"
) -> CRMEmbeddingService:
    """
    Factory function to create embedding service

    Args:
        scales: Embedding scales (default: [96, 192, 384])
        backend: 'auto', 'matryoshka', or 'simple'

    Returns:
        Configured CRMEmbeddingService
    """
    use_matryoshka = backend in ("auto", "matryoshka")
    return CRMEmbeddingService(scales=scales, use_matryoshka=use_matryoshka)
