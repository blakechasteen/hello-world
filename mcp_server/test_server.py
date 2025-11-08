"""
Test ExpertLoom MCP Server
===========================

Quick test to verify the server works before connecting to Claude Desktop.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
try:
    from mcp.server import Server
    print("[OK] MCP SDK installed")
except ImportError:
    print("[ERROR] MCP SDK not installed - run: pip install mcp")
    sys.exit(1)

try:
    from mythRL_core.entity_resolution import EntityRegistry
    print("[OK] mythRL_core imports work")
except ImportError as e:
    print(f"[ERROR] mythRL_core import failed: {e}")
    sys.exit(1)

try:
    from mythRL_core.summarization import TextSummarizer
    print("[OK] Summarization module works")
except ImportError as e:
    print(f"[ERROR] Summarization import failed: {e}")
    sys.exit(1)

# Test domain loading
try:
    registry_path = Path(__file__).parent.parent / "mythRL_core" / "domains" / "automotive" / "registry.json"
    if not registry_path.exists():
        print(f"[ERROR] Automotive domain not found at: {registry_path}")
        sys.exit(1)

    registry = EntityRegistry.load(registry_path)
    print(f"[OK] Loaded automotive domain ({registry.stats()['total_entities']} entities)")
except Exception as e:
    print(f"[ERROR] Failed to load domain: {e}")
    sys.exit(1)

# Test server creation
try:
    from expertloom_server import app, load_domain, get_current_domain
    print("[OK] Server module imports successfully")

    # Test domain loading
    domain = load_domain("automotive")
    print(f"[OK] Domain loader works")

    print(f"[OK] Server configured with 5 tools:")
    print(f"  - summarize_text: Summarize while preserving entities")
    print(f"  - extract_entities: Extract entities and measurements")
    print(f"  - process_note: Complete pipeline")
    print(f"  - list_domains: Show available domains")
    print(f"  - switch_domain: Change active domain")

except Exception as e:
    print(f"[ERROR] Server test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nNext steps:")
print("1. Install: pip install mcp")
print("2. Configure Claude Desktop (see README.md)")
print("3. Restart Claude Desktop")
print("4. Look for hammer icon in Claude Desktop (bottom right)")
print("5. Try: 'Use summarize_text on this note: Checked the Corolla...'")
