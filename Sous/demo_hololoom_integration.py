#!/usr/bin/env python3
"""
Demo: SOUS + HoloLoom Integration

Demonstrates:
1. Memory storage for cooking experiences
2. RAG-powered recipe suggestions
3. Agentic meal planning
4. Voice-controlled cooking assistance
"""

import asyncio
import sys
from pathlib import Path

# Add sous to path
sous_path = Path(__file__).parent / "Sous"
sys.path.insert(0, str(sous_path))

from sous.services.hololoom_bridge import SousHoloLoomBridge, HOLOLOOM_AVAILABLE
from sous.services.voice_assistant import SousVoiceAssistant


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"🤖 {title}")
    print("=" * 80 + "\n")


async def demo_memory_storage():
    """Demo 1: Memory storage for cooking knowledge"""
    print_section("DEMO 1: Memory Storage")

    if not HOLOLOOM_AVAILABLE:
        print("⚠️  HoloLoom not available - skipping memory demo")
        return

    print("Storing cooking experiences in HoloLoom's memory system...\n")

    async with SousHoloLoomBridge() as bridge:
        # Remember a recipe
        print("📖 Remembering a recipe...")
        result = await bridge.remember_recipe({
            "name": "Classic Spaghetti Carbonara",
            "description": "Traditional Italian pasta with eggs, cheese, and guanciale",
            "prep_time": 15,
            "ingredients": ["spaghetti", "eggs", "pecorino cheese", "guanciale", "black pepper"],
            "tags": ["italian", "pasta", "quick", "dinner"]
        })
        print(f"   ✓ Stored: {result.get('memory_id', 'N/A')}")
        print(f"   Text: {result.get('text', 'N/A')}\n")

        # Remember a cooking session
        print("👨‍🍳 Remembering a cooking session...")
        result = await bridge.remember_cooking_session(
            recipes_made=["Carbonara", "Garlic Bread"],
            success=True,
            notes="Carbonara turned out perfectly creamy. Garlic bread was a bit too crispy."
        )
        print(f"   ✓ Session recorded: {result.get('status')}")
        print(f"   Learned from experience: {result.get('learned')}\n")

        # Remember a substitution
        print("🔄 Remembering an ingredient substitution...")
        result = await bridge.remember_ingredient_substitution(
            original="guanciale",
            substitute="pancetta",
            recipe="Carbonara",
            worked_well=True,
            notes="Pancetta worked great, slightly less fatty but still delicious"
        )
        print(f"   ✓ Substitution: {result.get('substitution')}")
        print(f"   Success: {result.get('success')}\n")

        # Get memory metrics
        print("📊 Memory system metrics:")
        metrics = await bridge.get_memory_metrics()
        if metrics.get('status') == 'active':
            print(f"   Activation nodes: {metrics.get('activation', {}).get('active_nodes', 0)}")
            print(f"   Coherence: {metrics.get('coherence', {}).get('global_coherence', 0):.2f}")


async def demo_rag_suggestions():
    """Demo 2: RAG-powered recipe suggestions"""
    print_section("DEMO 2: RAG-Powered Recipe Suggestions")

    if not HOLOLOOM_AVAILABLE:
        print("⚠️  HoloLoom not available - skipping RAG demo")
        return

    print("Using HoloLoom's RAG system for intelligent recipe suggestions...\n")

    async with SousHoloLoomBridge(enable_rag=True) as bridge:
        # Suggest recipes based on ingredients
        print("🥗 What can I make with these ingredients?")
        ingredients = ["chicken breast", "broccoli", "garlic", "soy sauce", "rice"]
        print(f"   Ingredients: {', '.join(ingredients)}\n")

        result = await bridge.suggest_recipes_for_ingredients(
            ingredients=ingredients,
            dietary_restrictions=["gluten-free"]
        )

        if result['status'] == 'success':
            print(f"   AI Response: {result['response']}\n")
            print(f"   Confidence: {result['confidence']:.2%}")
        else:
            print(f"   Status: {result['status']}\n")

        # Find recipe by description
        print("\n🔍 Finding recipes by description...")
        result = await bridge.find_recipe_by_description(
            "healthy quick dinner for busy weeknight"
        )

        if result['status'] == 'success':
            print(f"   Description: {result['description']}")
            print(f"   Response: {result['response'][:200]}...")
            print(f"   Confidence: {result['confidence']:.2%}\n")

        # Suggest substitutions
        print("🔄 What can I substitute for buttermilk?")
        result = await bridge.suggest_substitutions(
            missing_ingredient="buttermilk",
            recipe_context="pancakes"
        )

        if result['status'] == 'success':
            print(f"   Response: {result['response']}")
            print(f"   Confidence: {result['confidence']:.2%}")


async def demo_agentic_planning():
    """Demo 3: Agentic meal planning"""
    print_section("DEMO 3: Agentic Meal Planning")

    if not HOLOLOOM_AVAILABLE:
        print("⚠️  HoloLoom not available - skipping agentic demo")
        return

    print("Using HoloLoom's agentic reasoning for intelligent meal planning...\n")

    async with SousHoloLoomBridge(enable_agentic=True) as bridge:
        # Plan meals for the week
        print("📅 Planning a week of meals...")
        result = await bridge.plan_meals_for_week(
            dietary_preferences=["vegetarian", "high-protein"],
            budget=150.0,
            time_constraints={
                "monday": 30,
                "wednesday": 45,
                "friday": 60
            }
        )

        if result['status'] == 'success':
            print(f"   Query: {result['query']}\n")
            print(f"   Plan: {result['plan'][:300]}...\n")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Reasoning steps: {result['reasoning_steps']}")
            if result.get('considerations'):
                print(f"   Key considerations: {', '.join(result['considerations'][:3])}")
        else:
            print(f"   Status: {result['status']}\n")

        # Optimize shopping list
        print("\n🛒 Optimizing shopping list...")
        result = await bridge.optimize_shopping_list(
            shopping_list=["tomatoes", "basil", "mozzarella", "pasta", "olive oil", "garlic"],
            pantry_items=["pasta", "olive oil", "salt", "pepper"],
            local_stores=["Whole Foods", "Trader Joe's", "Local Farmers Market"]
        )

        if result['status'] == 'success':
            print(f"   Original items: {result['original_count']}")
            print(f"   Strategy: {result['strategy'][:200]}...")
            print(f"   Confidence: {result['confidence']:.2%}\n")

        # Solve cooking problem
        print("🤔 Solving a cooking problem...")
        result = await bridge.solve_cooking_problem(
            problem="My sauce is too watery, how can I thicken it?",
            context="Making tomato sauce for pasta"
        )

        if result['status'] == 'success':
            print(f"   Problem: {result['problem']}")
            print(f"   Solution: {result['solution'][:200]}...")
            print(f"   Reasoning steps: {', '.join(result['steps'])}")
            print(f"   Confidence: {result['confidence']:.2%}")


async def demo_voice_assistant():
    """Demo 4: Voice-controlled cooking assistance"""
    print_section("DEMO 4: Voice-Controlled Cooking")

    print("Demonstrating hands-free voice assistance...\n")

    async with SousVoiceAssistant(voice_enabled=True, use_hololoom=True) as assistant:
        # Load a recipe
        print("📖 Loading recipe for voice guidance...")
        recipe = {
            "name": "Chocolate Chip Cookies",
            "steps": [
                {"description": "Preheat oven to 375°F"},
                {"description": "Mix butter and sugars until fluffy"},
                {"description": "Add eggs and vanilla, mix well"},
                {"description": "Combine dry ingredients in separate bowl"},
                {"description": "Gradually add dry ingredients to wet"},
                {"description": "Fold in chocolate chips"},
                {"description": "Drop spoonfuls onto baking sheet"},
                {"description": "Bake for 10-12 minutes until golden"}
            ]
        }

        result = assistant.load_recipe(recipe)
        print(f"   ✓ {result['message']}\n")

        # Simulate voice commands
        commands = [
            "start recipe",
            "set timer for 10 minutes",
            "what's in this step?",
            "next step",
            "check timer",
            "how much is 375 fahrenheit in celsius?",
            "next step",
            "what can I substitute for butter?",
        ]

        print("🎤 Simulating voice commands:\n")

        for i, command in enumerate(commands, 1):
            print(f"   {i}. You: \"{command}\"")

            result = await assistant.process_voice_command(command)

            if result['understood']:
                print(f"      🤖 Assistant: {result['response']}\n")
            else:
                print(f"      ⚠️  Not understood (confidence: {result.get('confidence', 0):.0%})\n")

            # Add delay for realism
            await asyncio.sleep(0.5)

        # Show status
        print("\n📊 Voice assistant status:")
        status = assistant.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")


async def demo_complete_workflow():
    """Demo 5: Complete workflow from planning to cooking"""
    print_section("DEMO 5: Complete Workflow")

    if not HOLOLOOM_AVAILABLE:
        print("⚠️  HoloLoom not available - showing simplified workflow")

    print("Complete workflow: Plan → Shop → Cook → Learn\n")

    async with SousHoloLoomBridge(enable_rag=True, enable_agentic=True) as bridge:
        # Step 1: Plan meals
        print("1️⃣  PLANNING: What should I cook this week?")
        if bridge.agent:
            result = await bridge.plan_meals_for_week(
                dietary_preferences=["balanced", "family-friendly"],
                budget=200.0
            )
            if result['status'] == 'success':
                print(f"   AI Plan: {result['plan'][:150]}...\n")
        else:
            print("   (Manual planning)\n")

        # Step 2: Optimize shopping
        print("2️⃣  SHOPPING: Optimize my grocery list")
        if bridge.agent:
            result = await bridge.optimize_shopping_list(
                shopping_list=["chicken", "rice", "vegetables", "pasta"],
                pantry_items=["olive oil", "garlic"],
                local_stores=["Kroger", "Whole Foods"]
            )
            if result['status'] == 'success':
                print(f"   Strategy: {result['strategy'][:150]}...\n")
        else:
            print("   (Manual shopping)\n")

        # Step 3: Cook with voice guidance
        print("3️⃣  COOKING: Voice-guided cooking")
        async with SousVoiceAssistant(use_hololoom=True) as assistant:
            recipe = {
                "name": "Chicken Stir-Fry",
                "steps": [
                    {"description": "Cut chicken into bite-sized pieces"},
                    {"description": "Heat oil in wok over high heat"},
                    {"description": "Cook chicken until golden, 5-7 minutes"}
                ]
            }
            assistant.load_recipe(recipe)
            response = await assistant.process_voice_command("start recipe")
            print(f"   Voice: {response['response']}\n")

        # Step 4: Remember and learn
        print("4️⃣  LEARNING: Remember this experience")
        if bridge.loom:
            result = await bridge.remember_cooking_session(
                recipes_made=["Chicken Stir-Fry"],
                success=True,
                notes="Great weeknight dinner, very quick"
            )
            print(f"   ✓ Experience stored in memory\n")
        else:
            print("   (Experience noted)\n")

    print("✅ Complete workflow demonstrated!")


async def main():
    """Run all demos"""
    print("=" * 80)
    print("🤖 SOUS + HoloLoom Integration Demo")
    print("=" * 80)
    print("\nIntegrating kitchen management with AI intelligence:\n")
    print("  1. Memory storage for cooking experiences")
    print("  2. RAG-powered recipe suggestions")
    print("  3. Agentic meal planning")
    print("  4. Voice-controlled cooking assistance")
    print("  5. Complete workflow")

    if not HOLOLOOM_AVAILABLE:
        print("\n⚠️  Warning: HoloLoom not available")
        print("   Some features will be limited or skipped")
        print("   Make sure HoloLoom is installed: pip install -e HoloLoom")

    print("\n" + "=" * 80)

    # Run demos
    await demo_memory_storage()
    await demo_rag_suggestions()
    await demo_agentic_planning()
    await demo_voice_assistant()
    await demo_complete_workflow()

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 Demo Complete!")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Memory system: Stores and recalls cooking experiences")
    print("  ✓ RAG system: Intelligent recipe suggestions and Q&A")
    print("  ✓ Agentic reasoning: Complex meal planning and optimization")
    print("  ✓ Voice assistant: Hands-free cooking guidance")
    print("  ✓ Complete integration: Seamless AI-powered kitchen management")

    print("\n💡 Try it yourself:")
    print("   python demo_hololoom_integration.py")
    print("   Or integrate into your SOUS app.py\n")


if __name__ == "__main__":
    asyncio.run(main())
