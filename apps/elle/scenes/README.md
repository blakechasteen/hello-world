"""Test scenes for Elle simulation.

These are canonical scenarios used for:
- Testing Elle's decision-making
- Regression testing when prompts/models change
- Demonstrating Elle's behavior

Each scene is a JSON file containing:
- scene_id: Unique identifier
- timestamp: When scene was captured
- location: Where in the world
- weather, time_of_day: Ambient conditions
- summary: High-level description
- tags: Quick classification
- objects: List of things in the scene
- relations: How objects relate spatially

Scenes represent different situations:
- shed_cluttered.json: Organizational challenge, medium priority
- bed_cucumber_healthy.json: Healthy scene, minimal action needed
- fence_broken_urgent.json: Structural issue, high priority

To add new scenes:
1. Create a new .json file following the schema
2. Give it a descriptive name
3. Include tags that match expected decision patterns
4. Test with CLI: `python -m elle.adapters.cli_adapter.cli simulate --scene scenes/your_scene.json`
"""
