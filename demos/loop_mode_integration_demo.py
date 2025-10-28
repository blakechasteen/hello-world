#!/usr/bin/env python3
"""
🔁 Loop Mode Integration Demo
==============================
Demonstrates continuous narrative processing with priority queues,
rate limiting, and learning mode across all domains.

This showcases the revolutionary loop engine:
- 24/7 continuous processing
- Priority queue management (LOW, NORMAL, HIGH, URGENT)
- Rate limiting for sustainable processing
- Checkpoint/resume capability
- Learning mode for domain pattern recognition
- Real-time statistics tracking
"""

import asyncio
import sys
from pathlib import Path

# Add mythRL to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom_narrative.loop_engine import NarrativeLoopEngine, LoopMode, Priority
from HoloLoom.cross_domain_adapter import CrossDomainAdapter
import time


async def demonstrate_batch_mode():
    """Demo 1: Batch processing - process queue then stop."""
    print("\n" + "="*70)
    print("🎯 DEMO 1: BATCH MODE - Process Queue Then Stop")
    print("="*70)
    
    engine = NarrativeLoopEngine(
        mode=LoopMode.BATCH,
        rate_limit=5,  # 5 tasks/sec
        learning_mode=True
    )
    
    # Add diverse tasks
    tasks = [
        ("Odysseus faced the Cyclops and overcame his pride.", "mythology", Priority.NORMAL),
        ("Sarah pivoted her startup three times before finding product-market fit.", "business", Priority.HIGH),
        ("Dr. Chen's experiment contradicted 50 years of theory.", "science", Priority.URGENT),
        ("In therapy, I finally faced what I'd avoided for years.", "personal", Priority.NORMAL),
        ("User interviews revealed we'd solved the wrong problem.", "product", Priority.LOW),
    ]
    
    print(f"\n📥 Adding {len(tasks)} tasks to queue...")
    for text, domain, priority in tasks:
        await engine.add_task(text, domain, priority)
        print(f"   • {priority.name:7} | {domain:10} | {text[:50]}...")
    
    print(f"\n▶️  Starting batch processing...")
    start = time.perf_counter()
    
    # Track results
    results = []
    def on_result(task, result):
        results.append((task, result))
        print(f"   ✓ Processed {task.domain:10} (Priority: {task.priority.name})")
    
    engine.on_result = on_result
    
    # Run until queue empty
    await engine.run()
    
    elapsed = time.perf_counter() - start
    
    print(f"\n📊 Batch Processing Complete!")
    print(f"   • Processed: {len(results)} tasks")
    print(f"   • Time: {elapsed:.2f}s")
    print(f"   • Rate: {len(results)/elapsed:.1f} tasks/sec")
    print(f"   • Avg Time: {engine.stats.average_time_ms:.1f}ms per task")
    
    # Show domain distribution
    print(f"\n🎭 Domain Distribution:")
    domain_counts = {}
    for task, _ in results:
        domain_counts[task.domain] = domain_counts.get(task.domain, 0) + 1
    
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"   • {domain:10}: {count} tasks")


async def demonstrate_priority_queue():
    """Demo 2: Priority queue - urgent tasks processed first."""
    print("\n" + "="*70)
    print("⚡ DEMO 2: PRIORITY QUEUE - Urgent Tasks First")
    print("="*70)
    
    engine = NarrativeLoopEngine(
        mode=LoopMode.BATCH,
        rate_limit=10,  # Faster processing
        learning_mode=False
    )
    
    # Add tasks with different priorities (intentionally out of order)
    tasks = [
        ("Low priority: Historical event analysis.", None, Priority.LOW),
        ("Normal priority: Business case study.", None, Priority.NORMAL),
        ("URGENT: Security incident narrative.", None, Priority.URGENT),
        ("Low priority: Product feedback summary.", None, Priority.LOW),
        ("HIGH priority: Customer crisis story.", None, Priority.HIGH),
        ("Normal priority: Team retrospective.", None, Priority.NORMAL),
        ("URGENT: System failure investigation.", None, Priority.URGENT),
    ]
    
    print(f"\n📥 Adding tasks in random priority order:")
    for text, domain, priority in tasks:
        await engine.add_task(text, domain, priority)
        print(f"   {priority.name:7}: {text}")
    
    print(f"\n▶️  Processing with priority queue (urgent → high → normal → low)...")
    
    results = []
    def on_result(task, result):
        results.append(task)
        priority_emoji = {
            Priority.URGENT: "🔴",
            Priority.HIGH: "🟠", 
            Priority.NORMAL: "🟡",
            Priority.LOW: "🟢"
        }
        print(f"   {priority_emoji[task.priority]} {task.priority.name:7} | {task.text[:60]}")
    
    engine.on_result = on_result
    
    await engine.run()
    
    print(f"\n✅ Priority Queue Demo Complete!")
    print(f"\n📈 Processing Order Verification:")
    print(f"   • URGENT tasks: {sum(1 for t in results[:2] if t.priority == Priority.URGENT)} (expected at top)")
    print(f"   • HIGH tasks: {sum(1 for t in results if t.priority == Priority.HIGH)}")
    print(f"   • NORMAL tasks: {sum(1 for t in results if t.priority == Priority.NORMAL)}")
    print(f"   • LOW tasks: {sum(1 for t in results if t.priority == Priority.LOW)}")


async def demonstrate_continuous_mode():
    """Demo 3: Continuous mode - runs forever (we'll stop it manually)."""
    print("\n" + "="*70)
    print("🔄 DEMO 3: CONTINUOUS MODE - 24/7 Processing")
    print("="*70)
    
    engine = NarrativeLoopEngine(
        mode=LoopMode.CONTINUOUS,
        rate_limit=3,  # Conservative rate for 24/7
        learning_mode=True,
        checkpoint_interval=5  # Save state every 5 tasks
    )
    
    # Add initial tasks
    initial_tasks = [
        ("Perseus received divine gifts before facing Medusa.", "mythology", Priority.NORMAL),
        ("The company pivoted from B2C to B2B after user research.", "business", Priority.NORMAL),
    ]
    
    print(f"\n📥 Starting with {len(initial_tasks)} tasks...")
    for text, domain, priority in initial_tasks:
        await engine.add_task(text, domain, priority)
    
    print(f"\n▶️  Starting continuous mode (will run for 5 seconds)...")
    print(f"   💡 In production, this would run 24/7 processing incoming tasks")
    
    results = []
    def on_result(task, result):
        results.append(task)
        print(f"   ✓ [{len(results)}] {task.domain:10} | {task.text[:50]}...")
    
    def on_checkpoint(checkpoint_path):
        print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    engine.on_result = on_result
    engine.on_checkpoint = on_checkpoint
    
    # Start loop in background
    loop_task = asyncio.create_task(engine.run())
    
    # Add tasks dynamically while running
    await asyncio.sleep(1)
    print(f"\n➕ Adding more tasks while loop is running...")
    
    dynamic_tasks = [
        ("The experiment failed three times before the breakthrough.", "science", Priority.HIGH),
        ("After hitting rock bottom, I finally asked for help.", "personal", Priority.URGENT),
        ("Beta testers found the hidden use case we never imagined.", "product", Priority.NORMAL),
    ]
    
    for text, domain, priority in dynamic_tasks:
        await engine.add_task(text, domain, priority)
        print(f"   ➕ Added: {text[:50]}...")
        await asyncio.sleep(0.5)
    
    # Let it run a bit more
    await asyncio.sleep(3)
    
    # Stop gracefully
    print(f"\n🛑 Stopping continuous mode...")
    engine.stop()
    await loop_task
    
    print(f"\n📊 Continuous Mode Statistics:")
    print(f"   • Tasks Processed: {engine.stats.tasks_processed}")
    print(f"   • Total Time: {engine.stats.total_time_seconds:.1f}s")
    print(f"   • Rate: {engine.stats.tasks_per_second:.2f} tasks/sec")
    print(f"   • Checkpoints Created: {engine.stats.checkpoints_created}")
    
    if engine.stats.domain_patterns:
        print(f"\n🧠 Learned Domain Patterns:")
        for domain, count in sorted(engine.stats.domain_patterns.items(), key=lambda x: -x[1]):
            print(f"   • {domain:10}: {count} samples")


async def demonstrate_learning_mode():
    """Demo 4: Learning mode - auto-detect domains and improve over time."""
    print("\n" + "="*70)
    print("🧠 DEMO 4: LEARNING MODE - Auto-Detect & Improve")
    print("="*70)
    
    engine = NarrativeLoopEngine(
        mode=LoopMode.BATCH,
        rate_limit=5,
        learning_mode=True
    )
    
    # Add tasks WITHOUT domain specified
    tasks = [
        "Prometheus stole fire from the gods to help humanity.",
        "Our competitor launched a feature we'd been planning for months.",
        "The telescope revealed galaxies beyond our wildest imagination.",
        "My father's Alzheimer's diagnosis forced me to become the parent.",
        "Users were hacking our product to solve problems we never intended.",
        "The revolution began with a single protestor refusing to move.",
    ]
    
    print(f"\n📥 Adding {len(tasks)} tasks WITHOUT domain labels...")
    print(f"   💡 Learning mode will auto-detect and improve over time\n")
    
    for text in tasks:
        await engine.add_task(text, domain=None, priority=Priority.NORMAL)
        print(f"   • {text}")
    
    print(f"\n▶️  Processing with auto-detection...")
    
    detected_domains = []
    def on_result(task, result):
        detected = result.get('domain', 'unknown')
        detected_domains.append((task.text[:50], detected))
        confidence = result.get('insights', {}).get('domain_confidence', 0)
        print(f"   🔍 Detected: {detected:10} (conf: {confidence:.2f}) | {task.text[:40]}...")
    
    engine.on_result = on_result
    
    await engine.run()
    
    print(f"\n📊 Learning Mode Results:")
    print(f"   • Tasks Processed: {len(detected_domains)}")
    print(f"\n🎯 Domain Detection Accuracy:")
    
    expected = {
        "Prometheus": "mythology",
        "competitor": "business",
        "telescope": "science",
        "Alzheimer": "personal",
        "hacking our product": "product",
        "revolution": "history"
    }
    
    for text, detected in detected_domains:
        key = [k for k in expected.keys() if k.lower() in text.lower()]
        expected_domain = expected[key[0]] if key else "unknown"
        match = "✓" if detected == expected_domain else "✗"
        print(f"   {match} {text:50} → {detected:10} (expected: {expected_domain})")
    
    if engine.stats.domain_patterns:
        print(f"\n🧠 Learned Patterns (for future predictions):")
        for domain, count in sorted(engine.stats.domain_patterns.items(), key=lambda x: -x[1]):
            print(f"   • {domain:10}: {count} training samples")


async def demonstrate_rate_limiting():
    """Demo 5: Rate limiting - prevent overload."""
    print("\n" + "="*70)
    print("🚦 DEMO 5: RATE LIMITING - Sustainable Processing")
    print("="*70)
    
    # Test different rate limits
    rates = [2, 5, 10]
    
    for rate_limit in rates:
        print(f"\n📊 Testing rate limit: {rate_limit} tasks/sec")
        
        engine = NarrativeLoopEngine(
            mode=LoopMode.BATCH,
            rate_limit=rate_limit,
            learning_mode=False
        )
        
        # Add 10 quick tasks
        for i in range(10):
            await engine.add_task(
                f"Test narrative #{i+1} for rate limiting demo.",
                domain="mythology",
                priority=Priority.NORMAL
            )
        
        start = time.perf_counter()
        await engine.run()
        elapsed = time.perf_counter() - start
        
        actual_rate = 10 / elapsed
        print(f"   • Completed 10 tasks in {elapsed:.2f}s")
        print(f"   • Actual rate: {actual_rate:.2f} tasks/sec")
        print(f"   • Target rate: {rate_limit} tasks/sec")
        print(f"   • Compliance: {'✓' if actual_rate <= rate_limit + 0.5 else '✗'}")


async def main():
    """Run all loop mode demonstrations."""
    print("\n" + "="*70)
    print("🔁 mythRL LOOP MODE INTEGRATION DEMO")
    print("="*70)
    print("\n🚀 Revolutionary Continuous Processing Engine:")
    print("   • Batch, Continuous, and Scheduled modes")
    print("   • Priority queue (URGENT → HIGH → NORMAL → LOW)")
    print("   • Rate limiting for sustainable 24/7 operation")
    print("   • Checkpoint/resume for fault tolerance")
    print("   • Learning mode for domain auto-detection")
    print("   • Real-time statistics and monitoring")
    
    try:
        await demonstrate_batch_mode()
        await demonstrate_priority_queue()
        await demonstrate_continuous_mode()
        await demonstrate_learning_mode()
        await demonstrate_rate_limiting()
        
        print("\n" + "="*70)
        print("✨ ALL DEMOS COMPLETE!")
        print("="*70)
        print("\n🎯 Key Takeaways:")
        print("   1. BATCH mode: Perfect for processing queued items then stopping")
        print("   2. PRIORITY queue: Urgent tasks jump to front of line")
        print("   3. CONTINUOUS mode: 24/7 processing with dynamic task addition")
        print("   4. LEARNING mode: Auto-detect domains and improve accuracy")
        print("   5. RATE LIMITING: Sustainable processing prevents overload")
        print("\n💡 Integration Ready:")
        print("   • React Dashboard: Real-time loop controls and statistics")
        print("   • FastAPI Backend: RESTful API for loop management")
        print("   • WebSocket Stream: Live event updates")
        print("   • Production Deploy: Ready for 24/7 narrative intelligence")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
