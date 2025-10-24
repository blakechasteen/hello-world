"""
Smart Deduplication System
===========================

Handles multiple types of duplicate detection:

1. **Exact Duplicates**: Same content hash
2. **URL Variants**: Same content, different URLs (tracking params, etc.)
3. **Near Duplicates**: Similar content (fuzzy matching)
4. **Canonical Resolution**: Multiple URLs → single canonical version
5. **Version Tracking**: Same content updated over time

Uses multiple strategies:
- Content hashing (exact)
- MinHash LSH (near-duplicate detection)
- Simhash (fuzzy fingerprinting)
- URL normalization
- Temporal tracking
"""

import hashlib
import re
from typing import List, Optional, Set, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from collections import defaultdict

try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_AVAILABLE = True
except ImportError:
    MINHASH_AVAILABLE = False
    MinHash = None
    MinHashLSH = None


@dataclass
class ContentSignature:
    """Signature for duplicate detection."""
    content_hash: str           # SHA256 of full content
    partial_hash: str           # Hash of first 1KB (quick check)
    simhash: Optional[int]      # Simhash fingerprint (fuzzy)
    minhash: Optional[object]   # MinHash signature (near-duplicate)
    word_count: int             # Word count
    char_count: int             # Character count
    normalized_url: str         # Canonical URL


@dataclass
class DuplicateGroup:
    """Group of duplicate/near-duplicate content."""
    canonical_id: str                    # Primary memory ID
    canonical_url: str                   # Primary URL
    duplicate_ids: List[str] = field(default_factory=list)
    duplicate_urls: List[str] = field(default_factory=list)
    similarity_scores: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class URLNormalizer:
    """Normalize URLs to detect duplicates."""

    # Query parameters to remove (tracking, analytics)
    REMOVE_PARAMS = {
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
        'fbclid', 'gclid', 'msclkid',
        'ref', 'source', 'campaign',
        '_ga', '_gid',
        'mc_cid', 'mc_eid',
    }

    # Protocol normalization
    PROTOCOL_ALIASES = {
        'http': 'https',  # Prefer HTTPS
    }

    @classmethod
    def normalize(cls, url: str) -> str:
        """
        Normalize URL to canonical form.

        Handles:
        - Protocol normalization (http → https)
        - Remove tracking parameters
        - Sort query parameters
        - Remove fragments
        - Lowercase domain
        - Remove trailing slashes
        - www normalization
        """
        parsed = urlparse(url)

        # Normalize protocol
        scheme = cls.PROTOCOL_ALIASES.get(parsed.scheme, parsed.scheme)

        # Normalize domain (lowercase, remove www)
        netloc = parsed.netloc.lower()
        if netloc.startswith('www.'):
            netloc = netloc[4:]

        # Filter query parameters
        query_params = parse_qs(parsed.query)
        filtered_params = {
            k: v for k, v in query_params.items()
            if k not in cls.REMOVE_PARAMS
        }

        # Sort parameters for consistency
        sorted_query = urlencode(sorted(filtered_params.items()), doseq=True)

        # Normalize path (remove trailing slash, except root)
        path = parsed.path
        if path != '/' and path.endswith('/'):
            path = path[:-1]

        # Reconstruct URL (no fragment)
        normalized = urlunparse((
            scheme,
            netloc,
            path,
            parsed.params,
            sorted_query,
            ''  # No fragment
        ))

        return normalized


class SimhashCalculator:
    """Calculate Simhash fingerprint for fuzzy matching."""

    @staticmethod
    def calculate(text: str, width: int = 64) -> int:
        """
        Calculate Simhash fingerprint.

        Simhash creates a fingerprint where similar documents
        have fingerprints with small Hamming distance.

        Args:
            text: Text to fingerprint
            width: Bit width (default 64)

        Returns:
            Integer fingerprint
        """
        # Tokenize
        tokens = text.lower().split()

        # Create feature hashes
        v = [0] * width

        for token in tokens:
            # Hash token
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)

            # Update vector
            for i in range(width):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1

        # Create fingerprint
        fingerprint = 0
        for i in range(width):
            if v[i] > 0:
                fingerprint |= (1 << i)

        return fingerprint

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Calculate Hamming distance between two fingerprints."""
        return bin(hash1 ^ hash2).count('1')

    @staticmethod
    def similarity(hash1: int, hash2: int, width: int = 64) -> float:
        """
        Calculate similarity (0-1) between two fingerprints.

        Returns:
            1.0 = identical, 0.0 = completely different
        """
        distance = SimhashCalculator.hamming_distance(hash1, hash2)
        return 1.0 - (distance / width)


class DeduplicationEngine:
    """Main deduplication engine."""

    def __init__(
        self,
        exact_threshold: float = 1.0,      # Exact match
        near_threshold: float = 0.85,       # Near-duplicate threshold
        fuzzy_threshold: float = 0.75,      # Fuzzy match threshold
        use_minhash: bool = True            # Use MinHash LSH
    ):
        self.exact_threshold = exact_threshold
        self.near_threshold = near_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.use_minhash = use_minhash and MINHASH_AVAILABLE

        # Storage
        self.content_hashes: Dict[str, str] = {}  # hash → memory_id
        self.url_to_id: Dict[str, str] = {}       # normalized_url → memory_id
        self.simhashes: Dict[str, int] = {}       # memory_id → simhash
        self.duplicate_groups: Dict[str, DuplicateGroup] = {}

        # MinHash LSH index
        if self.use_minhash:
            self.lsh = MinHashLSH(threshold=near_threshold, num_perm=128)
            self.minhashes: Dict[str, MinHash] = {}

    def create_signature(
        self,
        content: str,
        url: str,
        memory_id: str
    ) -> ContentSignature:
        """Create content signature for duplicate detection."""

        # Full content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Partial hash (first 1KB for quick check)
        partial_hash = hashlib.sha256(content[:1024].encode()).hexdigest()

        # Word/char counts
        words = content.split()
        word_count = len(words)
        char_count = len(content)

        # Simhash
        simhash = SimhashCalculator.calculate(content)

        # MinHash (if available)
        minhash = None
        if self.use_minhash:
            minhash = MinHash(num_perm=128)
            for word in words:
                minhash.update(word.encode())

        # Normalized URL
        normalized_url = URLNormalizer.normalize(url)

        return ContentSignature(
            content_hash=content_hash,
            partial_hash=partial_hash,
            simhash=simhash,
            minhash=minhash,
            word_count=word_count,
            char_count=char_count,
            normalized_url=normalized_url
        )

    def check_duplicate(
        self,
        signature: ContentSignature,
        memory_id: str
    ) -> Optional[Tuple[str, float, str]]:
        """
        Check if content is duplicate.

        Returns:
            Tuple of (existing_memory_id, similarity, duplicate_type) or None
        """

        # 1. Exact content match
        if signature.content_hash in self.content_hashes:
            existing_id = self.content_hashes[signature.content_hash]
            return (existing_id, 1.0, 'exact_content')

        # 2. Exact URL match (normalized)
        if signature.normalized_url in self.url_to_id:
            existing_id = self.url_to_id[signature.normalized_url]
            return (existing_id, 1.0, 'exact_url')

        # 3. Near-duplicate via MinHash LSH
        if self.use_minhash and signature.minhash:
            similar = self.lsh.query(signature.minhash)
            if similar:
                # Get most similar
                best_match = similar[0]
                similarity = self._calculate_minhash_similarity(
                    signature.minhash,
                    self.minhashes[best_match]
                )
                if similarity >= self.near_threshold:
                    return (best_match, similarity, 'near_duplicate')

        # 4. Fuzzy match via Simhash
        for existing_id, existing_simhash in self.simhashes.items():
            similarity = SimhashCalculator.similarity(
                signature.simhash,
                existing_simhash
            )
            if similarity >= self.fuzzy_threshold:
                return (existing_id, similarity, 'fuzzy_match')

        return None

    def add_content(
        self,
        signature: ContentSignature,
        memory_id: str
    ) -> Optional[str]:
        """
        Add content to dedup index.

        Returns:
            Canonical memory_id if duplicate found, else None
        """

        # Check for duplicates
        duplicate = self.check_duplicate(signature, memory_id)

        if duplicate:
            canonical_id, similarity, dup_type = duplicate

            # Add to duplicate group
            if canonical_id not in self.duplicate_groups:
                self.duplicate_groups[canonical_id] = DuplicateGroup(
                    canonical_id=canonical_id,
                    canonical_url=self.url_to_id.get(canonical_id, '')
                )

            group = self.duplicate_groups[canonical_id]
            group.duplicate_ids.append(memory_id)
            group.duplicate_urls.append(signature.normalized_url)
            group.similarity_scores.append(similarity)
            group.updated_at = datetime.now()

            return canonical_id

        # Not a duplicate - add to index
        self.content_hashes[signature.content_hash] = memory_id
        self.url_to_id[signature.normalized_url] = memory_id
        self.simhashes[memory_id] = signature.simhash

        if self.use_minhash and signature.minhash:
            self.lsh.insert(memory_id, signature.minhash)
            self.minhashes[memory_id] = signature.minhash

        return None

    def _calculate_minhash_similarity(self, mh1: MinHash, mh2: MinHash) -> float:
        """Calculate Jaccard similarity between MinHash signatures."""
        if not MINHASH_AVAILABLE:
            return 0.0
        return mh1.jaccard(mh2)

    def get_duplicates(self, memory_id: str) -> Optional[DuplicateGroup]:
        """Get all duplicates of a memory."""
        return self.duplicate_groups.get(memory_id)

    def get_canonical(self, memory_id: str) -> str:
        """
        Get canonical ID for a memory.

        If memory is a duplicate, returns canonical ID.
        Otherwise returns the memory_id itself.
        """
        # Check if this is a duplicate in any group
        for canonical_id, group in self.duplicate_groups.items():
            if memory_id in group.duplicate_ids:
                return canonical_id

        # Not a duplicate, it's canonical
        return memory_id

    def merge_duplicates(
        self,
        memory_id1: str,
        memory_id2: str
    ) -> str:
        """
        Manually mark two memories as duplicates.

        Returns canonical_id
        """
        # Determine canonical (use earlier ID)
        canonical_id = min(memory_id1, memory_id2)
        duplicate_id = max(memory_id1, memory_id2)

        # Create or update group
        if canonical_id not in self.duplicate_groups:
            self.duplicate_groups[canonical_id] = DuplicateGroup(
                canonical_id=canonical_id,
                canonical_url=''
            )

        group = self.duplicate_groups[canonical_id]
        if duplicate_id not in group.duplicate_ids:
            group.duplicate_ids.append(duplicate_id)
            group.similarity_scores.append(1.0)  # Manual merge = 100% similar
            group.updated_at = datetime.now()

        return canonical_id

    def stats(self) -> Dict:
        """Get deduplication statistics."""
        total_duplicates = sum(
            len(group.duplicate_ids)
            for group in self.duplicate_groups.values()
        )

        return {
            'total_content_hashes': len(self.content_hashes),
            'total_urls': len(self.url_to_id),
            'total_simhashes': len(self.simhashes),
            'duplicate_groups': len(self.duplicate_groups),
            'total_duplicates': total_duplicates,
            'dedup_rate': total_duplicates / max(len(self.content_hashes), 1),
            'using_minhash': self.use_minhash
        }


# Convenience functions
def normalize_url(url: str) -> str:
    """Normalize URL to canonical form."""
    return URLNormalizer.normalize(url)


def calculate_content_hash(content: str) -> str:
    """Calculate SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using Simhash.

    Returns:
        Similarity score 0-1 (1.0 = identical)
    """
    hash1 = SimhashCalculator.calculate(text1)
    hash2 = SimhashCalculator.calculate(text2)
    return SimhashCalculator.similarity(hash1, hash2)
