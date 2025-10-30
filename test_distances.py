import sys
sys.path.insert(0, '.')
import numpy as np
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus

# Create embedder and semantic calculator
embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)

# Generate some example positions
texts = [
    "Thompson Sampling balances exploration",
    "Machine learning uses algorithms",
    "Python is a programming language"
]

positions = []
for text in texts:
    async def process():
        async def word_stream():
            for word in text.split():
                yield word
        
        snapshot = None
        async for s in semantic.stream_analyze(word_stream()):
            snapshot = s
        return snapshot
    
    import asyncio
    snapshot = asyncio.run(process())
    position = snapshot.matryoshka_position
    positions.append(position)
    print(f"Position shape: {position.shape}, norm: {np.linalg.norm(position):.3f}")

# Calculate distances
for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[j])
        print(f"Distance between text {i} and {j}: {dist:.3f}")
