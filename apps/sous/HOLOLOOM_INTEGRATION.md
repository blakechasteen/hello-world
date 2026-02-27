# SOUS + HoloLoom Integration

Complete AI-powered kitchen management system integrating SOUS with HoloLoom's advanced neural decision-making capabilities.

## Overview

This integration brings enterprise-grade AI intelligence to home cooking:

- **Memory System**: Stores and recalls cooking experiences, recipes, and substitutions
- **RAG (Retrieval-Augmented Generation)**: Intelligent recipe suggestions and Q&A
- **Agentic Reasoning**: Complex meal planning and shopping optimization
- **Voice Assistant**: Hands-free cooking guidance

## Architecture

```
SOUS Kitchen Management
    ↓
HoloLoom Bridge (sous/services/hololoom_bridge.py)
    ├── Memory Integration → hololoom.experience()
    ├── RAG System → SimpleRAG.query()
    ├── Agentic Reasoning → AgenticOrchestrator.reason()
    └── Voice Assistant (sous/services/voice_assistant.py)
            ├── Recipe Navigation
            ├── Timer Management
            ├── Ingredient Queries
            └── Substitution Suggestions
```

## Features

### 1. Memory Integration

Store and recall cooking experiences in HoloLoom's knowledge graph:

```python
from sous.services.hololoom_bridge import SousHoloLoomBridge

async with SousHoloLoomBridge() as bridge:
    # Remember a recipe
    await bridge.remember_recipe({
        "name": "Spaghetti Carbonara",
        "ingredients": ["spaghetti", "eggs", "cheese"],
        "tags": ["italian", "pasta"]
    })

    # Remember a cooking session
    await bridge.remember_cooking_session(
        recipes_made=["Carbonara"],
        success=True,
        notes="Turned out perfectly!"
    )

    # Remember a substitution
    await bridge.remember_ingredient_substitution(
        original="guanciale",
        substitute="pancetta",
        recipe="Carbonara",
        worked_well=True
    )
```

**What it does**: Builds a persistent knowledge graph of your cooking experiences that gets smarter over time.

### 2. RAG-Powered Recipe Suggestions

Ask natural language questions and get intelligent suggestions:

```python
async with SousHoloLoomBridge(enable_rag=True) as bridge:
    # Suggest recipes based on ingredients
    result = await bridge.suggest_recipes_for_ingredients(
        ingredients=["chicken", "broccoli", "garlic"],
        dietary_restrictions=["gluten-free"]
    )
    print(result['response'])  # AI-generated suggestions

    # Find recipes by description
    result = await bridge.find_recipe_by_description(
        "healthy quick dinner for busy weeknight"
    )

    # Get substitution suggestions
    result = await bridge.suggest_substitutions(
        missing_ingredient="buttermilk",
        recipe_context="pancakes"
    )
```

**What it does**: Uses HoloLoom's Level 4 RAG system (semantic search + graph traversal + agentic reasoning) to answer cooking questions.

### 3. Agentic Meal Planning

Complex multi-step reasoning for meal planning and optimization:

```python
async with SousHoloLoomBridge(enable_agentic=True) as bridge:
    # Plan a week of meals
    result = await bridge.plan_meals_for_week(
        dietary_preferences=["vegetarian", "high-protein"],
        budget=150.0,
        time_constraints={"monday": 30, "friday": 60}
    )

    # Optimize shopping list
    result = await bridge.optimize_shopping_list(
        shopping_list=["tomatoes", "basil", "pasta"],
        pantry_items=["pasta", "olive oil"],
        local_stores=["Whole Foods", "Trader Joe's"]
    )

    # Solve cooking problems
    result = await bridge.solve_cooking_problem(
        problem="My sauce is too watery",
        context="Making tomato sauce"
    )
```

**What it does**: Uses HoloLoom's agentic orchestrator with Thompson Sampling exploration to generate comprehensive meal plans and solve complex cooking problems.

### 4. Voice-Controlled Cooking

Hands-free cooking assistance with natural language understanding:

```python
from sous.services.voice_assistant import SousVoiceAssistant

async with SousVoiceAssistant(use_hololoom=True) as assistant:
    # Load a recipe
    assistant.load_recipe(recipe_dict)

    # Process voice commands
    await assistant.process_voice_command("start recipe")
    await assistant.process_voice_command("set timer for 10 minutes")
    await assistant.process_voice_command("what's in this step?")
    await assistant.process_voice_command("next step")
    await assistant.process_voice_command("how much butter?")
    await assistant.process_voice_command("what can I substitute for eggs?")
```

**Supported commands**:
- **Navigation**: "next step", "previous step", "start recipe"
- **Timers**: "set timer for 10 minutes", "check timer", "cancel timer"
- **Ingredients**: "how much sugar?", "what's in this step?"
- **Substitutions**: "what can I use instead of butter?"
- **Measurements**: "convert 2 cups to tablespoons"
- **Temperature**: "350 fahrenheit to celsius"
- **General**: "help", or ask any cooking question

## Installation

### Prerequisites

1. **HoloLoom**: Install the HoloLoom package
   ```bash
   pip install -e HoloLoom
   ```

2. **Optional dependencies** (for full features):
   ```bash
   pip install torch sentence-transformers networkx ollama
   ```

### Quick Start

1. **Basic usage** (no AI):
   ```bash
   python Sous/sous/app.py
   ```

2. **With AI integration**:
   ```bash
   python Sous/sous/app.py --ai
   ```

3. **With voice assistant**:
   ```bash
   python Sous/sous/app.py --ai --voice
   ```

4. **Run full demo**:
   ```bash
   python Sous/demo_hololoom_integration.py
   ```

## Demo

The `demo_hololoom_integration.py` script showcases all integration features:

```bash
python Sous/demo_hololoom_integration.py
```

**Demo includes**:
1. Memory storage for recipes and experiences
2. RAG-powered recipe suggestions
3. Agentic meal planning
4. Voice-controlled cooking
5. Complete workflow (plan → shop → cook → learn)

## Usage Examples

### Example 1: Store Cooking Experience

```python
import asyncio
from sous.services.hololoom_bridge import remember_cooking_experience

async def main():
    result = await remember_cooking_experience(
        description="Made chocolate chip cookies. Used brown butter technique - amazing!",
        success=True
    )
    print(f"Stored memory: {result['memory_id']}")

asyncio.run(main())
```

### Example 2: Ask Cooking Question

```python
import asyncio
from sous.services.hololoom_bridge import ask_cooking_question

async def main():
    result = await ask_cooking_question(
        "What's the best way to cook a steak medium-rare?"
    )
    print(result['answer'])
    print(f"Confidence: {result['confidence']:.0%}")

asyncio.run(main())
```

### Example 3: Plan a Meal

```python
import asyncio
from sous.services.hololoom_bridge import plan_meal_intelligently

async def main():
    result = await plan_meal_intelligently(
        requirements="Dinner for 4 people, vegetarian, Italian-inspired",
        constraints={"time": 45, "budget": 40}
    )
    print(result['plan'])

asyncio.run(main())
```

### Example 4: Voice-Guided Cooking

```python
import asyncio
from sous.services.voice_assistant import SousVoiceAssistant

async def main():
    async with SousVoiceAssistant(use_hololoom=True) as assistant:
        # Load recipe
        recipe = {
            "name": "Pasta Carbonara",
            "steps": [
                {"description": "Boil water and cook pasta"},
                {"description": "Fry guanciale until crispy"},
                {"description": "Mix eggs and cheese"},
                {"description": "Combine pasta with egg mixture"}
            ]
        }
        assistant.load_recipe(recipe)

        # Simulate cooking session
        commands = [
            "start recipe",
            "set timer for 10 minutes",
            "next step",
            "what can I substitute for guanciale?",
            "check timer",
            "next step"
        ]

        for command in commands:
            response = await assistant.process_voice_command(command)
            print(f"You: {command}")
            print(f"Assistant: {response['response']}\n")

asyncio.run(main())
```

## Integration with SOUS Features

The HoloLoom integration works seamlessly with all existing SOUS features:

### Garden Harvests
```python
# Store garden harvest experiences in memory
await bridge.remember_cooking_session(
    recipes_made=["Tomato Salad"],
    success=True,
    notes="Used fresh tomatoes from backyard garden - incredible flavor!"
)
```

### Seasonal Calendar
```python
# Ask about seasonal cooking
result = await bridge.rag.query(
    "What vegetables are in season right now and what should I cook?"
)
```

### Local Farms
```python
# Optimize shopping with local farm context
result = await bridge.optimize_shopping_list(
    shopping_list=["tomatoes", "peppers", "eggs"],
    pantry_items=[],
    local_stores=["Smith Family Farm", "Downtown Farmers Market"]
)
```

### Waste Tracking
```python
# Learn from waste patterns
await bridge.loom.experience(
    "Spinach went bad after 5 days. Need to use leafy greens faster or buy smaller quantities."
)
```

## Performance

### Latency
- **Memory operations**: <50ms (store/recall)
- **RAG queries**: ~150-600ms (depending on mode)
- **Agentic planning**: ~750-900ms (multi-step reasoning)
- **Voice processing**: <100ms (command parsing)

### Memory Usage
- **HoloLoom base**: ~200MB
- **With embeddings**: ~400MB
- **Full system**: ~600MB

## Configuration

### HoloLoom Modes

The bridge uses `Config.fast()` by default (good balance of speed and quality):

```python
from hololoom.config import Config

# Fast mode (default) - 150ms queries
config = Config.fast()

# Fused mode - 300ms queries, best quality
config = Config.fused()

# Bare mode - 50ms queries, basic features
config = Config.bare()
```

### Enable/Disable Features

```python
# Memory only (fastest)
bridge = SousHoloLoomBridge(enable_rag=False, enable_agentic=False)

# RAG only (recipe suggestions)
bridge = SousHoloLoomBridge(enable_rag=True, enable_agentic=False)

# Full features
bridge = SousHoloLoomBridge(enable_rag=True, enable_agentic=True)
```

### Voice Assistant Options

```python
# Basic voice (no AI)
assistant = SousVoiceAssistant(voice_enabled=True, use_hololoom=False)

# Full AI voice
assistant = SousVoiceAssistant(voice_enabled=True, use_hololoom=True)
```

## Troubleshooting

### HoloLoom not available
```
⚠️  HoloLoom not available - running in basic mode
```

**Solution**: Install HoloLoom:
```bash
pip install -e HoloLoom
```

### Import errors
```python
# Check HoloLoom availability
from sous.services.hololoom_bridge import HOLOLOOM_AVAILABLE
print(f"HoloLoom available: {HOLOLOOM_AVAILABLE}")
```

### Performance issues

1. **Use faster mode**:
   ```python
   config = Config.bare()  # Fastest
   ```

2. **Disable features you don't need**:
   ```python
   bridge = SousHoloLoomBridge(
       enable_rag=True,      # Keep this
       enable_agentic=False  # Disable this if not needed
   )
   ```

3. **Check memory usage**:
   ```python
   metrics = await bridge.get_memory_metrics()
   print(metrics)
   ```

## Architecture Details

### HoloLoom Bridge (`sous/services/hololoom_bridge.py`)

**Purpose**: Main integration layer connecting SOUS to HoloLoom

**Key components**:
- `SousHoloLoomBridge`: Main bridge class
- Memory integration methods (`remember_*`)
- RAG integration methods (`suggest_*`, `find_*`)
- Agentic methods (`plan_*`, `optimize_*`, `solve_*`)

**Total**: 470 lines

### Voice Assistant (`sous/services/voice_assistant.py`)

**Purpose**: Hands-free cooking with natural language understanding

**Key components**:
- `SousVoiceAssistant`: Main voice controller
- `VoiceCommand`: Parsed command structure
- Command handlers for 7 command types
- Timer management
- Recipe navigation

**Total**: 580 lines

## API Reference

See inline documentation in:
- `sous/services/hololoom_bridge.py`
- `sous/services/voice_assistant.py`

All methods include docstrings with:
- Purpose
- Parameters
- Return values
- Usage examples

## Future Enhancements

Planned features:

1. **Multi-language support** for voice commands
2. **Image recognition** for ingredient identification
3. **Nutritional analysis** using HoloLoom's analytical capabilities
4. **Social features** (share recipes through HoloLoom memory)
5. **Personalization** (learn your preferences over time)
6. **Integration with smart devices** (Alexa, Google Home)

## License

Same as SOUS and HoloLoom parent projects.

## Credits

- **SOUS**: Kitchen management system
- **HoloLoom**: Neural decision-making platform
- **Integration**: Blake (November 2025)

## Support

For issues or questions:
1. Check this README
2. Run the demo: `python Sous/demo_hololoom_integration.py`
3. Check HoloLoom documentation: `hololoom/CLAUDE.md`
