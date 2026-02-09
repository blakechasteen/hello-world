"""
Memory Bus demo entry point — 2026-02-09

Usage:
    python -m memory_bus "What treatment should I apply to Aurora next?"

Requires: docker compose -f docker-compose.memory.yml up -d
"""

import asyncio
import sys

from .demo import run_demo


def main() -> None:
    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What treatment should I apply to Aurora next?"
    asyncio.run(run_demo(query_text))


if __name__ == "__main__":
    main()
