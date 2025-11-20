"""
Full System Demonstration - Elle Core

Shows complete workflow:
1. Create SOPs for products
2. Run tasks with time/profit tracking
3. Get AI-powered recommendations
4. Use voice interface
5. Query knowledge base

Run: python -m elle.demo_full_system
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from elle.sop_schema import SOP, StepType, UnitType, create_bread_sop_example
from elle.tracker import TaskTracker
from elle.voice_interface import VoiceSOPEditor
from elle.mirrorcore import DecisionEngine, ElleKnowledge


async def demo():
    print("\n" + "="*70)
    print(" ELLE CORE - Full System Demonstration")
    print(" Farm & Kitchen Cooperative Intelligence")
    print("="*70 + "\n")

    # ========================================================================
    # STEP 1: Create SOPs
    # ========================================================================
    print("📝 STEP 1: Creating SOPs\n")

    # Create bread SOP
    bread_sop = create_bread_sop_example()
    print(f"✓ Created SOP: {bread_sop.name}")
    print(f"  Version: {bread_sop.version}")
    print(f"  Batch Size: {bread_sop.batch_size} {bread_sop.batch_unit}")
    print(f"  Time: {bread_sop.total_time_minutes} minutes")
    print(f"  Cost: ${bread_sop.total_cost:.2f}")
    print(f"  Profit: ${bread_sop.profit_per_batch:.2f} ({bread_sop.profit_margin:.0f}% margin)")
    print(f"  Hourly ROI: ${bread_sop.hourly_roi:.2f}/hour")

    # Create The GOAT SOP
    goat_sop = SOP(
        sop_id="GOAT_001",
        name="The GOAT - Oatmeal Drink Production",
        category="Kitchen",
        description="Ready-to-drink oat-based breakfast blend with honey and fruit.",
        batch_size=24,
        batch_unit="bottles (12oz)",
        total_time_minutes=90,
        selling_price=5.00,
        target_margin=0.65,
        tags=["goat", "drink", "breakfast", "quick"]
    )

    # Add ingredients
    goat_sop.add_ingredient("Rolled Oats", 5, "lbs", UnitType.WEIGHT, 0.60, "Costco")
    goat_sop.add_ingredient("Honey", 2, "lbs", UnitType.WEIGHT, 1.50, "Farm")
    goat_sop.add_ingredient("Fruit Puree", 3, "lbs", UnitType.WEIGHT, 2.00, "Local")
    goat_sop.add_ingredient("Water", 2, "gallons", UnitType.VOLUME, 0.00)

    # Add steps
    goat_sop.add_step(
        StepType.PREP,
        "Blend Base",
        "Blend oats with water until smooth (about 3 minutes)",
        duration_minutes=10,
        tools_required=["High-speed blender"]
    )

    goat_sop.add_step(
        StepType.PROCESS,
        "Add Honey and Fruit",
        "Add honey and fruit puree. Blend until fully incorporated.",
        duration_minutes=5
    )

    goat_sop.add_step(
        StepType.QC,
        "Taste Test",
        "Check sweetness and consistency. Adjust as needed.",
        duration_minutes=5,
        quality_checkpoints=["Smooth texture", "Balanced sweetness", "No lumps"]
    )

    goat_sop.add_step(
        StepType.PROCESS,
        "Bottle",
        "Fill 12oz bottles, leaving 1/2 inch headspace. Cap immediately.",
        duration_minutes=30,
        tools_required=["Funnel", "Bottles", "Caps"],
        safety_notes="Ensure bottles are sanitized"
    )

    goat_sop.add_step(
        StepType.QC,
        "Label and Date",
        "Apply labels with production date and ingredients.",
        duration_minutes=15
    )

    goat_sop.add_step(
        StepType.CLEANUP,
        "Refrigerate",
        "Move to refrigerator. Product good for 7-10 days.",
        duration_minutes=5,
        temperature=38,
        temperature_unit="F"
    )

    print(f"\n✓ Created SOP: {goat_sop.name}")
    print(f"  Cost: ${goat_sop.total_cost:.2f}")
    print(f"  Profit: ${goat_sop.profit_per_batch:.2f} ({goat_sop.profit_margin:.0f}% margin)")
    print(f"  Hourly ROI: ${goat_sop.hourly_roi:.2f}/hour")

    # Save SOPs
    sop_dir = Path("elle/sops")
    sop_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(sop_dir / f"{bread_sop.sop_id}.json", "w") as f:
        json.dump(bread_sop.to_dict(), f, indent=2)

    with open(sop_dir / f"{goat_sop.sop_id}.json", "w") as f:
        json.dump(goat_sop.to_dict(), f, indent=2)

    # Also export to markdown
    examples_dir = Path("elle/examples")
    examples_dir.mkdir(parents=True, exist_ok=True)

    with open(examples_dir / "BREAD_SOP.md", "w") as f:
        f.write(bread_sop.to_markdown())

    with open(examples_dir / "GOAT_SOP.md", "w") as f:
        f.write(goat_sop.to_markdown())

    print(f"\n✓ Saved SOPs to {sop_dir}/")
    print(f"✓ Saved markdown to {examples_dir}/")

    input("\n[Press Enter to continue to Step 2...]")

    # ========================================================================
    # STEP 2: Task Tracking
    # ========================================================================
    print("\n" + "="*70)
    print("⏱️  STEP 2: Real-Time Task Tracking\n")

    tracker = TaskTracker()

    print("Starting bread production task...\n")

    # Start task
    task_id = await tracker.start(
        task_name="Bake sourdough bread batch 12",
        sop_id="BREAD_001",
        category="Bakery",
        batch_number=12
    )

    # Simulate work
    print("... working (simulated 2.5 hours) ...")
    await asyncio.sleep(2)  # In real use, this would be actual work time

    print("\nTaking a break...\n")
    await tracker.pause(task_id)
    await asyncio.sleep(1)  # 15 minute break

    print("Resuming work...\n")
    await tracker.resume(task_id)
    await asyncio.sleep(1)

    # Complete task
    print("Finishing up...\n")
    result = await tracker.end(
        task_id=task_id,
        units=24,
        revenue=144.00,  # 24 loaves × $6
        material_cost=bread_sop.material_cost,
        quality_score=9.2,
        notes="Perfect rise, customers loved the texture"
    )

    input("\n[Press Enter to continue to Step 3...]")

    # ========================================================================
    # STEP 3: Decision Support
    # ========================================================================
    print("\n" + "="*70)
    print("🤖 STEP 3: AI-Powered Decision Support\n")

    engine = DecisionEngine()

    print("Getting daily recommendations based on ROI and seasonal focus...\n")

    recommendations = await engine.get_daily_recommendations(
        available_hours=8.0,
        available_cash=500.0
    )

    print("📋 TOP RECOMMENDATIONS:\n")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. [{rec.priority}] {rec.action}")
        print(f"   💡 {rec.reasoning}")
        print(f"   💰 Estimated Profit: ${rec.estimated_profit:.2f}")
        print(f"   ⏱️  Time Required: {rec.time_required:.1f} hours")
        print()

    print("\n📊 OPTIMIZED RESOURCE ALLOCATION:\n")
    allocation = await engine.allocate_resources(
        available_hours=8.0,
        available_cash=500.0
    )

    print(f"Hours Allocated: {allocation.total_hours:.1f} / 8.0 ({allocation.utilization:.0f}% utilization)")
    print(f"Expected Revenue: ${allocation.expected_revenue:.2f}")
    print(f"Expected Profit: ${allocation.expected_profit:.2f}")
    print(f"\nTask Sequence:")
    for i, task in enumerate(allocation.tasks, 1):
        print(f"  {i}. [{task['priority']}] {task['task']}")
        print(f"     {task['time_hours']:.1f} hours → ${task['profit']:.2f} profit")

    input("\n[Press Enter to continue to Step 4...]")

    # ========================================================================
    # STEP 4: Voice Interface
    # ========================================================================
    print("\n" + "="*70)
    print("🎤 STEP 4: Voice Command Interface\n")

    editor = VoiceSOPEditor(use_hololoom=False)  # Skip HoloLoom for this demo

    print("Testing voice commands (text simulation)...\n")

    # Test commands
    test_commands = [
        "show bread sop",
        "what's the profit for bread?",
        "update bread sop: increase proofing time to 50 minutes",
    ]

    for cmd in test_commands:
        print(f"👤 User: \"{cmd}\"")
        response = await editor.process_voice_command(cmd)
        print(f"🤖 Elle: {response}\n")

    input("\n[Press Enter to continue to Step 5...]")

    # ========================================================================
    # STEP 5: Analytics Summary
    # ========================================================================
    print("\n" + "="*70)
    print("📈 STEP 5: Analytics Summary\n")

    # Get analytics
    analytics = await tracker.get_analytics(days=30)

    print("📊 PERFORMANCE METRICS (Last 30 Days):\n")
    print(f"  Total Tasks: {analytics['total_tasks']}")
    print(f"  Total Revenue: ${analytics['total_revenue']:.2f}")
    print(f"  Total Profit: ${analytics['total_profit']:.2f}")
    print(f"  Average Hourly ROI: ${analytics['avg_hourly_roi']:.2f}/hour")
    print(f"  Average Margin: {analytics['avg_margin']:.1f}%")
    print(f"  Total Hours Worked: {analytics['total_hours']:.1f} hours")

    if analytics['best_product']:
        print(f"  🏆 Best Product: {analytics['best_product']}")

    # Cash flow prediction
    print("\n💵 CASH FLOW PREDICTION (Next 4 Weeks):\n")
    predictions = await engine.predict_cash_flow(weeks_ahead=4)

    for pred in predictions:
        print(f"  Week {pred['week']}: ${pred['predicted_revenue']:.2f} revenue, ${pred['predicted_profit']:.2f} profit")

    # Bottleneck analysis
    print("\n⚠️  BOTTLENECK ANALYSIS:\n")
    bottlenecks = await engine.detect_bottlenecks(days=30)

    if bottlenecks:
        for bottleneck in bottlenecks:
            print(f"  • {bottleneck['description']}")
            print(f"    → {bottleneck['suggestion']}\n")
    else:
        print("  ✓ No bottlenecks detected. Operations running smoothly!\n")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("✨ DEMONSTRATION COMPLETE")
    print("="*70 + "\n")

    print("What Elle Core provides:")
    print("  ✓ Voice-editable SOPs with automatic cost tracking")
    print("  ✓ Real-time time/profit analysis")
    print("  ✓ AI-powered product prioritization")
    print("  ✓ Resource optimization and scheduling")
    print("  ✓ Predictive cash flow analytics")
    print("  ✓ Bottleneck detection and recommendations")
    print("  ✓ Complete integration with HoloLoom/MirrorCore")
    print("\nNext steps:")
    print("  1. Run: python -m elle.voice_interface (for interactive voice session)")
    print("  2. Check: elle/examples/ for SOP markdown files")
    print("  3. Review: elle/data/tasks.db for task history")
    print("  4. Integrate: Connect with HoloLoom RAG for knowledge queries")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(demo())
