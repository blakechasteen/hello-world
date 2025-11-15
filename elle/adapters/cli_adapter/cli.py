"""CLI adapter for testing and simulation."""

import click
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from ...engine import ElleEngine, create_request
from ...domain import (
    SceneSnapshot,
    UserIntent,
    UserState,
    ScanType,
    InteractionMode,
)


@click.group()
def cli():
    """Elle CLI: testing and simulation interface."""
    pass


@cli.command()
@click.option('--scene', type=click.Path(exists=True), required=True, help='Path to scene JSON')
@click.option('--intent', type=str, default='seeking_guidance', help='Interaction mode')
@click.option('--scan', type=str, default='slow_scan', help='Scan type')
@click.option('--user-id', type=str, default='test_user', help='User ID')
@click.option('--tired', is_flag=True, help='User is tired')
@click.option('--rushed', is_flag=True, help='User is rushed')
def simulate(
    scene: str,
    intent: str,
    scan: str,
    user_id: str,
    tired: bool,
    rushed: bool,
):
    """Simulate Elle's response to a scene."""
    
    import os
    from ...core import EllePolicy, PromptBuilder, create_llm_client, LLMProvider
    from ...memory import InMemoryMemoryStore
    from ...tools import create_default_registry
    from ...infra import ElleConfig
    
    click.echo("=" * 60)
    click.echo("Elle CLI Simulation")
    click.echo("=" * 60)
    
    # Load scene from JSON
    scene_data = json.loads(Path(scene).read_text())
    scene_snapshot = _json_to_scene(scene_data)
    
    click.echo(f"\nScene: {scene_snapshot.location}")
    click.echo(f"Objects: {scene_snapshot.object_count}")
    click.echo(f"Tags: {', '.join(scene_snapshot.tags)}")
    
    # Build intent
    user_intent = UserIntent(
        mode=InteractionMode(intent),
        scan_type=ScanType(scan),
        is_tired=tired,
        is_rushed=rushed,
    )
    
    # Build user state (minimal for testing)
    user_state = UserState(
        user_id=user_id,
        name="Test User",
    )
    
    click.echo(f"\nUser: {user_state.name}")
    click.echo(f"Intent: {intent}, Scan: {scan}")
    click.echo(f"Tired: {tired}, Rushed: {rushed}")
    
    # Initialize Elle
    click.echo("\nInitializing Elle...")
    
    config = ElleConfig.load_profile('dev')
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        click.echo("⚠️  ANTHROPIC_API_KEY not set - using stub LLM")
        
        class StubLLMClient:
            def complete(self, prompt, **kwargs):
                return '''```json
{
  "mode": "consulting",
  "utterance": "CLI stub response",
  "visuals": {"style": "invisible"},
  "reasoning": "Stub LLM used (no API key)"
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
    
    # Create and process request
    request = create_request(
        scene=scene_snapshot,
        intent=user_intent,
        user=user_state,
        source_adapter="cli",
    )
    
    click.echo("\nProcessing... (calling LLM)")
    action = engine.handle(request)
    
    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("Elle's Response")
    click.echo("=" * 60)
    click.echo(f"Mode: {action.mode.value}")
    
    if action.utterance:
        click.echo(f"Says: \"{action.utterance}\"")
    
    if action.visuals and action.visuals.style.value != "invisible":
        click.echo(f"Visual: {action.visuals.style.value}")
        if action.visuals.target_object_ids:
            click.echo(f"Highlighting: {', '.join(action.visuals.target_object_ids)}")
    
    if action.has_task:
        click.echo(f"\nSuggested Task: {action.suggested_task_summary}")
        click.echo(f"Priority: {action.task_priority.value if action.task_priority else 'none'}")
        click.echo(f"Estimate: {action.task_estimate_minutes} min")
    
    if action.reasoning:
        click.echo(f"\nReasoning: {action.reasoning}")
    
    click.echo("=" * 60)


@cli.command()
@click.argument('scenes_dir', type=click.Path(exists=True))
def batch(scenes_dir: str):
    """Run batch simulation on all scenes in directory."""
    
    scenes_path = Path(scenes_dir)
    scene_files = list(scenes_path.glob("*.json"))
    
    click.echo(f"Found {len(scene_files)} scenes")
    
    for scene_file in scene_files:
        click.echo(f"\nProcessing: {scene_file.name}")
        # TODO: Process each scene
        

@cli.command()
def interactive():
    """Start interactive REPL for Elle."""
    
    click.echo("Elle Interactive Mode")
    click.echo("Type 'help' for commands, 'exit' to quit")
    
    while True:
        try:
            user_input = click.prompt("\n>", type=str)
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            if user_input.lower() == 'help':
                click.echo("Commands:")
                click.echo("  scene <path>  - Load scene from JSON")
                click.echo("  intent <mode> - Set interaction mode")
                click.echo("  go            - Process current context")
                click.echo("  exit          - Quit")
                continue
            
            # TODO: Process commands
            click.echo("Command not yet implemented")
            
        except (KeyboardInterrupt, EOFError):
            break
    
    click.echo("\nGoodbye!")


def _json_to_scene(data: dict) -> SceneSnapshot:
    """Convert JSON dict to SceneSnapshot."""
    
    from ...domain import ObjectInScene, SpatialRelation
    
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
    
    # Parse relations (if any)
    relations = []
    for rel_data in data.get('relations', []):
        relations.append(SpatialRelation(
            subject=rel_data['subject'],
            predicate=rel_data['predicate'],
            object=rel_data['object'],
        ))
    
    return SceneSnapshot(
        scene_id=data.get('scene_id', 'cli_scene'),
        timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
        location=data['location'],
        objects=objects,
        relations=relations,
        weather=data.get('weather'),
        time_of_day=data.get('time_of_day'),
        summary=data.get('summary'),
        tags=data.get('tags', []),
    )


if __name__ == '__main__':
    cli()
