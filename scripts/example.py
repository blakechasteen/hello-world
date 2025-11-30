#!/usr/bin/env python3
"""
Golden Path Example: Complete flow from scene to action.

This demonstrates Elle's full decision loop:
1. Load a test scene
2. Create request with user intent
3. Engine loads memory
4. Policy decides what to do
5. Action returned

Run with:
    python example.py --scene elle/scenes/shed_cluttered.json
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add elle to path
sys.path.insert(0, str(Path(__file__).parent))

from elle.domain import (
    SceneSnapshot,
    ObjectInScene,
    SpatialRelation,
    UserIntent,
    UserState,
    InteractionMode,
    ScanType,
)
from elle.engine import ElleEngine, create_request
from elle.core import EllePolicy, PromptBuilder, create_llm_client, LLMProvider
from elle.memory import InMemoryMemoryStore
from elle.tools import create_default_registry
from elle.infra import ElleConfig


def load_scene_from_json(path: Path) -> SceneSnapshot:
    """Load scene from JSON file."""
    
    with open(path) as f:
        data = json.load(f)
    
    # Parse objects
    objects = []
    for obj_data in data.get('objects', []):
        objects.append(ObjectInScene(
            id=obj_data['id'],
            name=obj_data['name'],
            category=obj_data.get('category', 'unknown'),
            location=obj_data.get('location', 'unknown'),
            condition=obj_data.get('condition'),
            quantity=obj_data.get('quantity'),
        ))
    
    # Parse relations
    relations = []
    for rel_data in data.get('relations', []):
        relations.append(SpatialRelation(
            subject=rel_data['subject'],
            predicate=rel_data['predicate'],
            object=rel_data['object'],
        ))
    
    return SceneSnapshot(
        scene_id=data.get('scene_id', 'example_scene'),
        timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
        location=data['location'],
        objects=objects,
        relations=relations,
        weather=data.get('weather'),
        time_of_day=data.get('time_of_day'),
        summary=data.get('summary'),
        tags=data.get('tags', []),
    )


def main():
    """Run golden path example."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Elle Golden Path Example")
    logger.info("=" * 60)
    
    # 1. Load configuration
    logger.info("\n1. Loading configuration...")
    config = ElleConfig.load_profile('dev')
    
    # Check for API key if using real LLM
    if config.llm.provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set. Using stub mode.")
            logger.info("Set ANTHROPIC_API_KEY to use real LLM.")
            # Create a stub that returns mock JSON
            from elle.core.llm_client import LLMClient
            
            class StubLLMClient:
                def complete(self, prompt, **kwargs):
                    return '''```json
{
  "mode": "consulting",
  "utterance": "That left shelf looks approachable. Ten minutes?",
  "visuals": {
    "style": "soft_highlight",
    "target_object_ids": ["obj_rake"],
    "color": "amber",
    "position": "beside_user"
  },
  "suggested_task_id": "task_001",
  "suggested_task_summary": "Clear space on left shelf",
  "task_priority": "medium",
  "task_estimate_minutes": 10,
  "followups": [],
  "reasoning": "User slow-scanned cluttered shed. Left shelf is most accessible. Suggesting small win."
}
```'''
            
            llm_client = StubLLMClient()
        else:
            config.llm.api_key = api_key
            llm_client = create_llm_client(
                LLMProvider.ANTHROPIC,
                {
                    'api_key': config.llm.api_key,
                    'model': config.llm.model,
                    'timeout': config.llm.timeout,
                }
            )
    else:
        logger.info(f"Using {config.llm.provider} provider")
        llm_client = create_llm_client(
            LLMProvider(config.llm.provider),
            {
                'api_key': config.llm.api_key,
                'model': config.llm.model,
                'timeout': config.llm.timeout,
            }
        )
    
    logger.info(f"  Profile: {config.profile}")
    logger.info(f"  LLM: {config.llm.provider} / {config.llm.model}")
    logger.info(f"  Memory: {config.memory.backend}")
    
    # 2. Initialize components
    logger.info("\n2. Initializing components...")
    
    memory_store = InMemoryMemoryStore()
    tool_registry = create_default_registry()
    prompt_builder = PromptBuilder()
    
    policy = EllePolicy(
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )
    
    engine = ElleEngine(
        policy=policy,
        memory=memory_store,
        tools=tool_registry,
    )
    
    logger.info("  ✓ Memory store")
    logger.info("  ✓ Tool registry")
    logger.info("  ✓ Policy")
    logger.info("  ✓ Engine")
    
    # 3. Load test scene
    logger.info("\n3. Loading test scene...")
    
    scene_path = Path(__file__).parent / 'elle' / 'scenes' / 'shed_cluttered.json'
    scene = load_scene_from_json(scene_path)
    
    logger.info(f"  Location: {scene.location}")
    logger.info(f"  Objects: {scene.object_count}")
    logger.info(f"  Tags: {', '.join(scene.tags)}")
    logger.info(f"  Summary: {scene.summary}")
    
    # 4. Create user intent
    logger.info("\n4. Creating user intent...")
    
    intent = UserIntent(
        mode=InteractionMode.SEEKING_GUIDANCE,
        scan_type=ScanType.SLOW_SCAN,
        is_tired=True,
        is_rushed=False,
    )
    
    user = UserState(
        user_id='blake',
        name='Blake',
        current_energy_level='medium',
    )
    
    logger.info(f"  Mode: {intent.mode.value}")
    logger.info(f"  Scan: {intent.scan_type.value}")
    logger.info(f"  Tired: {intent.is_tired}")
    
    # 5. Create request
    logger.info("\n5. Creating request...")
    
    request = create_request(
        scene=scene,
        intent=intent,
        user=user,
        source_adapter='example',
    )
    
    logger.info(f"  Request ID: {request.request_id}")
    
    # 6. Process through engine
    logger.info("\n6. Processing through engine...")
    logger.info("  (This calls the LLM - may take a moment)")
    
    action = engine.handle(request)
    
    # 7. Display results
    logger.info("\n7. Elle's Response:")
    logger.info("=" * 60)
    logger.info(f"  Mode: {action.mode.value}")
    
    if action.utterance:
        logger.info(f"  Says: \"{action.utterance}\"")
    
    if action.visuals:
        logger.info(f"  Visual: {action.visuals.style.value}")
        if action.visuals.target_object_ids:
            logger.info(f"  Highlighting: {', '.join(action.visuals.target_object_ids)}")
        if action.visuals.color:
            logger.info(f"  Color: {action.visuals.color}")
    
    if action.has_task:
        logger.info(f"  Task: {action.suggested_task_summary}")
        logger.info(f"  Priority: {action.task_priority.value if action.task_priority else 'none'}")
        logger.info(f"  Estimate: {action.task_estimate_minutes} minutes")
    
    if action.reasoning:
        logger.info(f"\n  Reasoning: {action.reasoning}")
    
    logger.info("=" * 60)
    logger.info("\n✓ Golden path complete!")
    
    # 8. Show what was recorded in memory
    logger.info("\n8. Memory recorded:")
    memory_snapshot = memory_store.load_memory('blake')
    logger.info(f"  Total interactions: {memory_snapshot.total_interactions}")
    logger.info(f"  Recent locations: {', '.join(memory_snapshot.recent_locations)}")


if __name__ == '__main__':
    main()
