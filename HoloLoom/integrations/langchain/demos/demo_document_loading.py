"""
Demo: Document Loading with LangChain Integration
===================================================

Shows how to load 100+ document formats using UniversalDocumentLoader.

Author: Claude Code
Date: 2025-11-20
"""

import asyncio
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from HoloLoom.integrations.langchain import UniversalDocumentLoader, load_documents


def demo_basic_loading():
    """Demo basic document loading."""
    print("\n" + "="*60)
    print("Demo 1: Basic Document Loading")
    print("="*60 + "\n")

    loader = UniversalDocumentLoader()

    # Example 1: Text file (if exists)
    print("1️⃣  Text File Loading")
    print("   Status: Would load .txt, .md files")
    print("   Usage: loader.load('document.txt')\n")

    # Example 2: PDF (if exists)
    print("2️⃣  PDF Loading")
    print("   Status: Would load .pdf files with PyPDF")
    print("   Usage: loader.load('research.pdf')\n")

    # Example 3: Web URL
    print("3️⃣  Web Loading")
    print("   Status: Would load from URLs")
    print("   Usage: loader.load('https://example.com')\n")


def demo_directory_loading():
    """Demo directory loading."""
    print("\n" + "="*60)
    print("Demo 2: Directory Loading")
    print("="*60 + "\n")

    print("📁 Directory Loading Example")
    print("   loader.load_directory('docs/', glob_pattern='**/*.md')")
    print("   → Recursively loads all Markdown files\n")

    print("📁 With Exclusions")
    print("   loader.load_directory('src/', exclude=['**/test_*.py'])")
    print("   → Loads all Python files except tests\n")


def demo_specialized_loaders():
    """Demo specialized loaders."""
    print("\n" + "="*60)
    print("Demo 3: Specialized Loaders")
    print("="*60 + "\n")

    print("🐙 GitHub Repository")
    print("   from HoloLoom.integrations.langchain import load_github_repo")
    print("   shards = load_github_repo('https://github.com/user/repo', branch='main')")
    print("   → Loads all files from repository\n")

    print("💬 Slack Workspace")
    print("   from HoloLoom.integrations.langchain import load_slack_workspace")
    print("   shards = load_slack_workspace('slack_export/')")
    print("   → Loads Slack export directory\n")

    print("📓 Notion Database")
    print("   from HoloLoom.integrations.langchain import load_notion_database")
    print("   shards = load_notion_database(notion_token, database_id)")
    print("   → Loads Notion database\n")


def demo_supported_formats():
    """Show supported formats."""
    print("\n" + "="*60)
    print("Demo 4: Supported Formats (100+)")
    print("="*60 + "\n")

    from HoloLoom.integrations.langchain import supported_document_types

    try:
        formats = supported_document_types()

        print("📄 Document Formats:")
        for ext, loader in list(formats.items())[:10]:
            print(f"   {ext:12} → {loader}")
        print(f"   ... and {len(formats) - 10} more formats\n")

    except Exception as e:
        print(f"   ⚠️  {e}\n")
        print("   Example formats:")
        print("   .pdf        → PDFs")
        print("   .docx       → Word documents")
        print("   .pptx       → PowerPoint")
        print("   .xlsx       → Excel")
        print("   .ipynb      → Jupyter notebooks")
        print("   .html       → Web pages")
        print("   .json       → JSON data")
        print("   .csv        → CSV data\n")


def demo_integration_with_hololoom():
    """Demo HoloLoom integration."""
    print("\n" + "="*60)
    print("Demo 5: Integration with HoloLoom")
    print("="*60 + "\n")

    print("📚 Complete RAG Pipeline:")
    print("""
    from HoloLoom import HoloLoom
    from HoloLoom.integrations.langchain import load_documents

    # Load documents
    shards = load_documents('docs/')

    # Ingest to HoloLoom
    async with HoloLoom() as loom:
        for shard in shards:
            await loom.experience(shard.content)

        # Query
        memories = await loom.recall('What did I learn?')
    """)


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print(" "*15 + "LangChain Document Loading Demo")
    print("="*70)

    demo_basic_loading()
    demo_directory_loading()
    demo_specialized_loaders()
    demo_supported_formats()
    demo_integration_with_hololoom()

    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70 + "\n")

    print("Next Steps:")
    print("1. Install dependencies: pip install langchain langchain-community")
    print("2. Try loading your own documents")
    print("3. See README.md for complete API reference\n")


if __name__ == "__main__":
    main()
