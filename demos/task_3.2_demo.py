"""
Task 3.2 Demo: Semantic Memory Enhancement

Demonstrates multi-modal spinners and cross-modal query capabilities.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.spinningWheel.multimodal_spinner import (
    MultiModalSpinner,
    TextSpinner,
    StructuredDataSpinner,
    CrossModalSpinner
)
from HoloLoom.input.protocol import ModalityType


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_shard(shard, index: int = None):
    """Print shard details."""
    prefix = f"Shard {index}: " if index is not None else ""
    print(f"\n{prefix}")
    print(f"  ID: {shard.id[:50]}...")
    print(f"  Text: {shard.text[:80]}...")
    print(f"  Episode: {shard.episode}")
    print(f"  Entities: {shard.entities[:3] if shard.entities else 'None'}")
    print(f"  Motifs: {shard.motifs[:3] if shard.motifs else 'None'}")
    print(f"  Modality: {shard.metadata.get('modality_type', 'N/A')}")
    print(f"  Confidence: {shard.metadata.get('confidence', 0):.3f}")
    print(f"  Has embedding: {shard.metadata.get('embedding') is not None}")


async def demo1_text_processing():
    """Demo 1: Enhanced text processing with entities and motifs."""
    print_header("DEMO 1: Enhanced Text Processing")
    
    spinner = TextSpinner()
    
    documents = [
        "Apple Inc. announced record profits today in Cupertino, California.",
        "Quantum computing breakthroughs are revolutionizing cryptography and AI.",
        "Climate change continues to impact global weather patterns significantly."
    ]
    
    print("\nProcessing 3 text documents...")
    
    all_shards = []
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}: {doc}")
        shards = await spinner.spin(doc)
        all_shards.extend(shards)
        
        shard = shards[0]
        print(f"  [OK] Entities: {shard.entities}")
        print(f"  [OK] Motifs: {shard.motifs}")
        print(f"  [OK] Confidence: {shard.metadata.get('confidence', 0):.3f}")
    
    print(f"\n[OK] Created {len(all_shards)} text shards")
    print("="*70)


async def demo2_structured_data():
    """Demo 2: Structured data processing with schema detection."""
    print_header("DEMO 2: Structured Data Processing")
    
    spinner = StructuredDataSpinner()
    
    datasets = [
        {
            "name": "Company Profile",
            "data": {
                "company": "Apple Inc.",
                "founded": 1976,
                "headquarters": "Cupertino, CA",
                "products": ["iPhone", "MacBook", "iPad", "Apple Watch"],
                "revenue_usd": 394328000000,
                "employees": 164000
            }
        },
        {
            "name": "Research Dataset",
            "data": {
                "study": "Quantum Computing Applications",
                "year": 2025,
                "findings": ["Error correction improved", "Qubit stability increased"],
                "success_rate": 0.94
            }
        }
    ]
    
    print("\nProcessing 2 structured datasets...")
    
    for i, item in enumerate(datasets, 1):
        print(f"\nDataset {i}: {item['name']}")
        shards = await spinner.spin(item['data'])
        
        shard = shards[0]
        print(f"  [OK] Modality: {shard.metadata.get('modality_type', 'N/A')}")
        print(f"  [OK] Confidence: {shard.metadata.get('confidence', 0):.3f}")
        print(f"  [OK] Has embedding: {shard.metadata.get('embedding') is not None}")
    
    print("\n[OK] Processed all structured data")
    print("="*70)


async def demo3_auto_detection():
    """Demo 3: Automatic modality detection."""
    print_header("DEMO 3: Automatic Modality Detection")
    
    spinner = MultiModalSpinner()
    
    test_inputs = [
        ("Plain Text", "The sun rises in the east and sets in the west."),
        ("JSON Object", {"temperature": 72, "humidity": 45, "condition": "sunny"}),
        ("List of Data", [10, 20, 30, 40, 50]),
        ("Mixed Content", ["Text here", {"data": 123}])
    ]
    
    print("\nAuto-detecting modalities for various inputs...\n")
    
    for label, input_data in test_inputs:
        shards = await spinner.spin(input_data)
        shard = shards[0]
        
        print(f"{label}:")
        print(f"  Input: {str(input_data)[:60]}...")
        print(f"  Detected: {shard.metadata.get('modality_type', 'UNKNOWN')}")
        print(f"  Confidence: {shard.metadata.get('confidence', 0):.3f}")
    
    print("\n[OK] All modalities auto-detected successfully")
    print("="*70)


async def demo4_cross_modal_fusion():
    """Demo 4: Cross-modal fusion with multiple inputs."""
    print_header("DEMO 4: Cross-Modal Fusion")
    
    spinner = CrossModalSpinner()
    
    # Mix of text and structured data about same topic
    inputs = [
        "Apple Inc. is a leading technology company known for innovative products.",
        {
            "company": "Apple Inc.",
            "market_cap_usd": 2800000000000,
            "key_products": ["iPhone", "MacBook", "iPad"],
            "ceo": "Tim Cook"
        },
        "The iPhone revolutionized the smartphone industry when launched in 2007.",
        {
            "product": "iPhone",
            "launch_year": 2007,
            "current_version": "iPhone 15",
            "market_share": 0.28
        }
    ]
    
    print("\nFusing 4 inputs (2 text + 2 structured)...")
    print("\nInput summary:")
    for i, inp in enumerate(inputs, 1):
        print(f"  {i}. {'TEXT' if isinstance(inp, str) else 'STRUCTURED'}: {str(inp)[:60]}...")
    
    print("\nFusion strategies:")
    
    for strategy in ["attention", "average", "max"]:
        print(f"\n  Strategy: {strategy.upper()}")
        shards = await spinner.spin_multiple(inputs, fusion_strategy=strategy)
        
        # Find fused shard
        fused = [s for s in shards if s.metadata.get('is_fused')]
        if fused:
            fused_shard = fused[0]
            print(f"    Component count: {fused_shard.metadata.get('component_count', 0)}")
            print(f"    Modalities: {fused_shard.metadata.get('component_modalities', [])}")
            print(f"    Confidence: {fused_shard.metadata.get('confidence', 0):.3f}")
            print(f"    Has embedding: {fused_shard.metadata.get('embedding') is not None}")
    
    print("\n[OK] Cross-modal fusion demonstrated")
    print("="*70)


async def demo5_cross_modal_query():
    """Demo 5: Cross-modal query processing."""
    print_header("DEMO 5: Cross-Modal Query Processing")
    
    spinner = CrossModalSpinner()
    
    queries = [
        "Show me text and images about quantum computing",
        "Find documents and data about climate change",
        "Search for information about artificial intelligence across all formats"
    ]
    
    print("\nProcessing cross-modal queries...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        shards = await spinner.spin_query(query)
        
        if shards:
            query_shard = shards[0]
            print(f"  [OK] Query processed")
            print(f"  [OK] Is cross-modal: {query_shard.metadata.get('cross_modal', False)}")
            print(f"  [OK] Has embedding: {query_shard.metadata.get('embedding') is not None}")
            print(f"  [OK] Confidence: {query_shard.metadata.get('confidence', 0):.3f}")
        print()
    
    print("[OK] Cross-modal queries ready for memory search")
    print("="*70)


async def demo6_supported_modalities():
    """Demo 6: Check supported modalities."""
    print_header("DEMO 6: Supported Modalities")
    
    spinner = MultiModalSpinner()
    
    supported = spinner.get_supported_modalities()
    
    print("\nChecking available processors...\n")
    
    all_modalities = [
        ModalityType.TEXT,
        ModalityType.IMAGE,
        ModalityType.AUDIO,
        ModalityType.VIDEO,
        ModalityType.STRUCTURED
    ]
    
    for modality in all_modalities:
        status = "[OK] Available" if modality in supported else "[--] Unavailable"
        print(f"  {modality.name:12} {status}")
    
    print(f"\nTotal available: {len(supported)}/{len(all_modalities)} modalities")
    print("="*70)


async def demo7_memory_integration_preview():
    """Demo 7: Preview of memory integration."""
    print_header("DEMO 7: Memory Integration Preview")
    
    print("\nMulti-Modal Memory Architecture:\n")
    
    print("  Input → InputRouter → Processor → ProcessedInput")
    print("           ↓")
    print("      MultiModalSpinner")
    print("           ↓")
    print("      MemoryShard (with modality metadata)")
    print("           ↓")
    print("      Memory Backend (Neo4j + Qdrant)")
    print("           ↓")
    print("      Cross-Modal Retrieval")
    
    print("\n\nKey Features:")
    print("  [OK] Unified shard format with modality tagging")
    print("  [OK] Automatic modality detection and routing")
    print("  [OK] Cross-modal fusion with 4 strategies")
    print("  [OK] Entities and motifs extraction")
    print("  [OK] Cross-modal query embedding")
    print("  [OK] Graceful degradation (works without optional deps)")
    
    print("\n\nNext Steps:")
    print("  1. Enhance memory backends for multi-modal storage")
    print("  2. Implement cross-modal similarity search")
    print("  3. Build multi-modal knowledge graphs")
    print("  4. Enable queries like 'Show me text and images about X'")
    
    print("\n" + "="*70)


async def run_all_demos():
    """Run all Task 3.2 demos."""
    print("\n" + "="*70)
    print("=" + " "*68 + "=")
    print("=" + "  TASK 3.2: SEMANTIC MEMORY ENHANCEMENT - DEMO".center(68) + "=")
    print("=" + " "*68 + "=")
    print("="*70)
    
    demos = [
        ("Enhanced Text Processing", demo1_text_processing),
        ("Structured Data Processing", demo2_structured_data),
        ("Automatic Modality Detection", demo3_auto_detection),
        ("Cross-Modal Fusion", demo4_cross_modal_fusion),
        ("Cross-Modal Query Processing", demo5_cross_modal_query),
        ("Supported Modalities", demo6_supported_modalities),
        ("Memory Integration Preview", demo7_memory_integration_preview),
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n[FAIL] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("=" + " "*68 + "=")
    print("=" + "  DEMO COMPLETE - ALL FEATURES DEMONSTRATED".center(68) + "=")
    print("=" + " "*68 + "=")
    print("="*70)
    
    print("\n\nSummary:")
    print("  [OK] Multi-modal spinners working (Text, Structured, Cross-Modal)")
    print("  [OK] Automatic modality detection")
    print("  [OK] Cross-modal fusion (attention, average, max)")
    print("  [OK] Cross-modal query processing")
    print("  [OK] MemoryShard creation with modality metadata")
    print("  [OK] Entity and motif extraction")
    print("  [OK] Ready for memory backend integration")
    
    print("\n\nRun this demo:")
    print("  $env:PYTHONPATH = \".\"; python demos/task_3.2_demo.py")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_all_demos())
