"""
HoloLoom Mathematical Modules - Hofstadter Sequences
=====================================================
Self-referential memory indexing inspired by Gödel, Escher, Bach.

Uses Hofstadter sequences (G, H, Q) to create recursive memory indices
that encode temporal and structural relationships.

Philosophy:
"Strange loops" in memory - memories that reference themselves through
mathematical recursion, creating emergent hierarchical patterns.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


# ============================================================================
# Hofstadter Sequences
# ============================================================================

class HofstadterSequences:
    """
    Implementation of self-referential Hofstadter sequences.
    
    Sequences:
    - G(n) = n - G(G(n-1))  [Forward temporal jumps]
    - H(n) = n - H(H(H(n-1)))  [Backward temporal resonance]
    - Q(n) = Q(n - Q(n-1)) + Q(n - Q(n-2))  [Chaotic associations]
    - R(n) = R(n-1) + R(n-R(n-1))  [Recursive growth]
    
    These sequences create non-linear memory addressing patterns
    that mirror how human memory associates concepts.
    """
    
    def __init__(self, max_n: int = 10000):
        """
        Args:
            max_n: Maximum sequence index to compute
        """
        self.max_n = max_n
        
        # Initialize caches with base cases
        self._g_cache = {0: 0}
        self._h_cache = {0: 0, 1: 1}
        self._q_cache = {1: 1, 2: 1}
        self._r_cache = {1: 1, 2: 1}
    
    def G(self, n: int) -> int:
        """
        Hofstadter G sequence: G(n) = n - G(G(n-1))
        
        Properties:
        - Encodes forward temporal jumps
        - Grows roughly as n/phi (golden ratio)
        - Creates self-similar patterns
        
        Use case: Find the "next" memory to explore
        """
        if n <= 0:
            return 0
        
        if n in self._g_cache:
            return self._g_cache[n]
        
        # Prevent infinite recursion
        if n > self.max_n:
            return n // 2
        
        result = n - self.G(self.G(n - 1))
        self._g_cache[n] = result
        return result
    
    def H(self, n: int) -> int:
        """
        Hofstadter H sequence: H(n) = n - H(H(H(n-1)))
        
        Properties:
        - Triple recursion creates deeper resonance
        - Slower growth than G
        - Encodes backward temporal connections
        
        Use case: Find memories that led to current state
        """
        if n <= 1:
            return n
        
        if n in self._h_cache:
            return self._h_cache[n]
        
        if n > self.max_n:
            return n // 3
        
        result = n - self.H(self.H(self.H(n - 1)))
        self._h_cache[n] = result
        return result
    
    def Q(self, n: int) -> int:
        """
        Hofstadter Q sequence: Q(n) = Q(n - Q(n-1)) + Q(n - Q(n-2))
        
        Properties:
        - Chaotic, unpredictable behavior
        - Creates non-linear jumps
        - Fibonacci-like structure with Q feedback
        
        Use case: Associative memory jumps (creative connections)
        """
        if n <= 2:
            return 1
        
        if n in self._q_cache:
            return self._q_cache[n]
        
        if n > self.max_n:
            return max(1, n // 4)
        
        result = self.Q(n - self.Q(n - 1)) + self.Q(n - self.Q(n - 2))
        self._q_cache[n] = result
        return result
    
    def R(self, n: int) -> int:
        """
        Hofstadter R sequence: R(n) = R(n-1) + R(n-R(n-1))
        
        Properties:
        - Recursive growth pattern
        - Similar to Fibonacci but self-referential
        - Creates exponential-like growth with plateaus
        
        Use case: Memory importance/salience growth
        """
        if n <= 2:
            return 1
        
        if n in self._r_cache:
            return self._r_cache[n]
        
        if n > self.max_n:
            return n
        
        result = self.R(n - 1) + self.R(n - self.R(n - 1))
        self._r_cache[n] = result
        return result
    
    def generate_sequence(self, sequence_type: str, length: int) -> List[int]:
        """
        Generate a sequence of specified type.
        
        Args:
            sequence_type: 'G', 'H', 'Q', or 'R'
            length: Number of terms to generate
        
        Returns:
            List of sequence values
        """
        sequence_map = {
            'G': self.G,
            'H': self.H,
            'Q': self.Q,
            'R': self.R
        }
        
        if sequence_type not in sequence_map:
            raise ValueError(f"Unknown sequence type: {sequence_type}")
        
        func = sequence_map[sequence_type]
        return [func(i) for i in range(1, length + 1)]


# ============================================================================
# Memory Indexing
# ============================================================================

@dataclass
class MemoryIndex:
    """Index for a single memory using Hofstadter sequences."""
    memory_id: int
    forward: int  # G-sequence (next memory)
    backward: int  # H-sequence (prior memory)
    associate: int  # Q-sequence (creative jump)
    salience: int  # R-sequence (importance)
    temporal_phase: int  # Time bucket


class HofstadterMemoryIndex:
    """
    Memory indexing system using Hofstadter sequences.
    
    Creates a web of self-referential memory connections where:
    - Each memory has multiple "resonance indices"
    - Indices create non-linear traversal patterns
    - Emergent patterns arise from sequence properties
    
    Usage:
        indexer = HofstadterMemoryIndex()
        idx = indexer.index_memory(42, timestamp=time.time())
        next_memory = idx.forward
        related_memories = indexer.find_resonance([42, 43, 44])
    """
    
    def __init__(self, max_n: int = 10000):
        """
        Args:
            max_n: Maximum sequence index
        """
        self.sequences = HofstadterSequences(max_n=max_n)
        self.max_n = max_n
    
    def index_memory(
        self,
        memory_id: int,
        timestamp: float,
        salience_base: float = 0.5
    ) -> MemoryIndex:
        """
        Generate Hofstadter indices for a memory.
        
        Args:
            memory_id: Unique memory identifier
            timestamp: Unix timestamp
            salience_base: Base salience value [0, 1]
        
        Returns:
            MemoryIndex with all computed indices
        """
        # Map memory_id to sequence index space
        n = memory_id % self.max_n
        if n == 0:
            n = 1
        
        # Generate indices
        forward = self.sequences.G(n)
        backward = self.sequences.H(n)
        associate = self.sequences.Q(n)
        
        # Salience grows with R-sequence
        salience_factor = self.sequences.R(min(n, 100))  # Cap for stability
        salience = int(salience_base * salience_factor)
        
        # Temporal phase (bucketed time)
        temporal_phase = int(timestamp) % self.max_n
        
        return MemoryIndex(
            memory_id=memory_id,
            forward=forward,
            backward=backward,
            associate=associate,
            salience=salience,
            temporal_phase=temporal_phase
        )
    
    def find_resonance(
        self,
        memory_ids: List[int],
        depth: int = 3,
        min_score: float = 0.5
    ) -> List[Tuple[int, int, float]]:
        """
        Find memories that resonate through Hofstadter indices.
        
        Two memories resonate if their sequences intersect
        within 'depth' steps.
        
        Args:
            memory_ids: List of memory IDs to check
            depth: How many steps to look for intersections
            min_score: Minimum resonance score
        
        Returns:
            List of (mem_a, mem_b, resonance_score) tuples
        """
        resonances = []
        
        # Get indices for all memories
        indices = {mid: self.index_memory(mid, 0.0) for mid in memory_ids}
        
        # Check all pairs
        for i, id_a in enumerate(memory_ids):
            idx_a = indices[id_a]
            
            for id_b in memory_ids[i+1:]:
                idx_b = indices[id_b]
                
                score = self._compute_resonance_score(idx_a, idx_b, depth)
                
                if score >= min_score:
                    resonances.append((id_a, id_b, score))
        
        return sorted(resonances, key=lambda x: x[2], reverse=True)
    
    def _compute_resonance_score(
        self,
        idx_a: MemoryIndex,
        idx_b: MemoryIndex,
        depth: int
    ) -> float:
        """
        Compute resonance score between two memory indices.
        
        Considers:
        - Forward sequence proximity
        - Backward sequence proximity
        - Associative sequence proximity
        - Temporal alignment
        """
        score = 0.0
        
        # Forward resonance (G-sequence)
        if abs(idx_a.forward - idx_b.forward) < depth:
            score += 0.3
        
        # Backward resonance (H-sequence)
        if abs(idx_a.backward - idx_b.backward) < depth:
            score += 0.25
        
        # Associative resonance (Q-sequence)
        # Q is chaotic, so allow larger depth
        if abs(idx_a.associate - idx_b.associate) < depth * 2:
            score += 0.25
        
        # Temporal alignment
        temporal_diff = abs(idx_a.temporal_phase - idx_b.temporal_phase)
        temporal_similarity = 1.0 - min(temporal_diff / self.max_n, 1.0)
        score += 0.2 * temporal_similarity
        
        return min(score, 1.0)
    
    def traverse_sequence(
        self,
        start_id: int,
        sequence_type: str = 'G',
        steps: int = 10
    ) -> List[int]:
        """
        Traverse memory space following a Hofstadter sequence.
        
        Args:
            start_id: Starting memory ID
            sequence_type: 'forward' (G), 'backward' (H), or 'associate' (Q)
            steps: Number of steps to traverse
        
        Returns:
            List of memory IDs in traversal order
        """
        path = [start_id]
        current = start_id
        
        for _ in range(steps):
            idx = self.index_memory(current, 0.0)
            
            if sequence_type == 'forward':
                next_id = idx.forward
            elif sequence_type == 'backward':
                next_id = idx.backward
            elif sequence_type == 'associate':
                next_id = idx.associate
            else:
                raise ValueError(f"Unknown sequence type: {sequence_type}")
            
            # Avoid cycles
            if next_id in path:
                break
            
            path.append(next_id)
            current = next_id
        
        return path
    
    def sequence_statistics(self, memory_ids: List[int]) -> Dict[str, float]:
        """
        Compute statistical properties of memory indices.
        
        Returns:
            - avg_forward: Average forward jump
            - avg_backward: Average backward jump
            - associate_entropy: Entropy of associative jumps
            - resonance_density: Fraction of pairs that resonate
        """
        if not memory_ids:
            return {}
        
        indices = [self.index_memory(mid, 0.0) for mid in memory_ids]
        
        # Forward/backward averages
        avg_forward = np.mean([idx.forward for idx in indices])
        avg_backward = np.mean([idx.backward for idx in indices])
        
        # Associative entropy
        associates = [idx.associate for idx in indices]
        unique_associates = len(set(associates))
        associate_entropy = unique_associates / len(associates) if associates else 0.0
        
        # Resonance density
        resonances = self.find_resonance(memory_ids, min_score=0.5)
        max_pairs = len(memory_ids) * (len(memory_ids) - 1) / 2
        resonance_density = len(resonances) / max_pairs if max_pairs > 0 else 0.0
        
        return {
            'avg_forward': float(avg_forward),
            'avg_backward': float(avg_backward),
            'associate_entropy': float(associate_entropy),
            'resonance_density': float(resonance_density),
            'num_memories': len(memory_ids)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== Hofstadter Memory Indexing Demo ===\n")
    
    # Create indexer
    indexer = HofstadterMemoryIndex(max_n=1000)
    
    # Index some memories
    memory_ids = [10, 25, 42, 73, 100]
    
    print("Memory Indices:")
    for mid in memory_ids:
        idx = indexer.index_memory(mid, timestamp=1700000000.0)
        print(f"  Memory {mid:3d}: "
              f"forward={idx.forward:3d}, "
              f"backward={idx.backward:3d}, "
              f"associate={idx.associate:3d}, "
              f"salience={idx.salience:3d}")
    
    print("\nResonance Analysis:")
    resonances = indexer.find_resonance(memory_ids, depth=5)
    for mem_a, mem_b, score in resonances[:5]:
        print(f"  {mem_a} ⟷ {mem_b}: resonance={score:.3f}")
    
    print("\nSequence Traversal (forward from 42):")
    path = indexer.traverse_sequence(42, sequence_type='forward', steps=8)
    print(f"  Path: {' → '.join(map(str, path))}")
    
    print("\nStatistics:")
    stats = indexer.sequence_statistics(memory_ids)
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n✓ Demo complete!")
