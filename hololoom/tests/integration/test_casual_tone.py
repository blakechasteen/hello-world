import pytest

pytest.importorskip("gradio")

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.consciousness_ui_simple import SimpleDualStream

async def test():
    gen = SimpleDualStream()
    i, e, t = await gen.generate(
        'What are quantum computing applications?', 
        'Quantum computing has applications in cryptography, drug discovery, optimization problems, and machine learning.'
    )
    print('\n=== NEW CASUAL TONE ===\n')
    print(e)
    print('\n======================\n')

asyncio.run(test())
