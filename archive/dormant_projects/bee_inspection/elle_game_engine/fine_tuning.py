# fine_tuning.py
# Fine-Tuning Support for BigPlay Engine
# Version: 1.0.0
#
# Export conversation data, create fine-tuning datasets, and manage model versions.

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path


class DatasetQuality(Enum):
    """Quality tiers for fine-tuning data"""
    HIGH = "high"          # Perfect examples (5-star quality)
    MEDIUM = "medium"      # Good examples (3-4 star)
    LOW = "low"            # Acceptable (1-2 star)
    EXCLUDED = "excluded"  # Do not use for training


@dataclass
class ConversationExample:
    """Single conversation example for fine-tuning"""
    messages: List[Dict[str, str]]  # OpenAI format: [{"role": "...", "content": "..."}]
    quality: DatasetQuality = DatasetQuality.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_jsonl(self) -> str:
        """Convert to JSONL format for fine-tuning"""
        return json.dumps({
            "messages": self.messages,
            "metadata": self.metadata
        })

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic format"""
        # Anthropic uses different format
        return {
            "prompt": "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in self.messages[:-1]  # All but last
            ]),
            "completion": self.messages[-1]["content"]
        }


@dataclass
class FineTuningDataset:
    """Complete fine-tuning dataset"""
    examples: List[ConversationExample] = field(default_factory=list)
    name: str = "bigplay_dataset"
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)

    def add_example(self, example: ConversationExample):
        """Add example to dataset"""
        self.examples.append(example)

    def filter_by_quality(self, min_quality: DatasetQuality) -> "FineTuningDataset":
        """Filter dataset by minimum quality"""
        quality_order = {
            DatasetQuality.HIGH: 3,
            DatasetQuality.MEDIUM: 2,
            DatasetQuality.LOW: 1,
            DatasetQuality.EXCLUDED: 0
        }

        min_level = quality_order[min_quality]
        filtered = FineTuningDataset(name=f"{self.name}_filtered", version=self.version)

        for example in self.examples:
            if quality_order[example.quality] >= min_level:
                filtered.add_example(example)

        return filtered

    def export_openai_jsonl(self, output_path: str):
        """Export to OpenAI fine-tuning format (JSONL)"""
        with open(output_path, 'w') as f:
            for example in self.examples:
                f.write(example.to_jsonl() + '\n')

    def export_anthropic_jsonl(self, output_path: str):
        """Export to Anthropic format"""
        with open(output_path, 'w') as f:
            for example in self.examples:
                f.write(json.dumps(example.to_anthropic_format()) + '\n')

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        quality_counts = {}
        for example in self.examples:
            quality = example.quality.value
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        return {
            "total_examples": len(self.examples),
            "quality_distribution": quality_counts,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat()
        }


class FineTuningExporter:
    """
    Export BigPlay conversations to fine-tuning datasets.

    Features:
    - Convert game conversations to training format
    - Quality filtering
    - Dataset versioning
    - OpenAI and Anthropic support
    """

    def __init__(self, output_dir: str = "./fine_tuning_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_conversation(
        self,
        game_state: Dict[str, Any],
        player_message: str,
        npc_response: str,
        npc_id: str,
        quality: DatasetQuality = DatasetQuality.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationExample:
        """
        Export single conversation to training format

        Args:
            game_state: Game state context
            player_message: What player said
            npc_response: NPC's response
            npc_id: NPC identifier
            quality: Quality rating
            metadata: Optional metadata

        Returns:
            ConversationExample ready for fine-tuning
        """
        # Build system prompt with game context
        system_content = self._build_system_prompt(game_state, npc_id)

        # Build messages in OpenAI format
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": player_message},
            {"role": "assistant", "content": npc_response}
        ]

        # Add metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "npc_id": npc_id,
            "scene_id": game_state.get("scene_id", "unknown"),
            "exported_at": datetime.now().isoformat()
        })

        return ConversationExample(
            messages=messages,
            quality=quality,
            metadata=metadata
        )

    def export_multi_turn_conversation(
        self,
        conversation_turns: List[Dict[str, str]],
        game_state: Dict[str, Any],
        quality: DatasetQuality = DatasetQuality.MEDIUM
    ) -> ConversationExample:
        """
        Export multi-turn conversation

        Args:
            conversation_turns: List of {"speaker": "...", "message": "..."}
            game_state: Game state context
            quality: Quality rating

        Returns:
            ConversationExample
        """
        system_content = self._build_system_prompt(game_state)

        messages = [{"role": "system", "content": system_content}]

        for turn in conversation_turns:
            role = "user" if turn["speaker"] == "player" else "assistant"
            messages.append({
                "role": role,
                "content": turn["message"]
            })

        return ConversationExample(
            messages=messages,
            quality=quality,
            metadata={
                "scene_id": game_state.get("scene_id"),
                "turn_count": len(conversation_turns)
            }
        )

    def create_dataset_from_logs(
        self,
        log_file: str,
        min_quality: Optional[DatasetQuality] = None
    ) -> FineTuningDataset:
        """
        Create dataset from BigPlay logs

        Args:
            log_file: Path to JSONL log file
            min_quality: Minimum quality to include

        Returns:
            FineTuningDataset
        """
        dataset = FineTuningDataset(
            name=Path(log_file).stem,
            version="1.0"
        )

        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)

                    # Extract conversation data
                    if "player_message" in log_entry and "npc_response" in log_entry:
                        example = self.export_conversation(
                            game_state=log_entry.get("game_state", {}),
                            player_message=log_entry["player_message"],
                            npc_response=log_entry["npc_response"],
                            npc_id=log_entry.get("npc_id", "unknown"),
                            quality=DatasetQuality(log_entry.get("quality", "medium"))
                        )
                        dataset.add_example(example)

                except (json.JSONDecodeError, KeyError) as e:
                    # Skip malformed entries
                    continue

        # Filter if requested
        if min_quality:
            dataset = dataset.filter_by_quality(min_quality)

        return dataset

    def _build_system_prompt(
        self,
        game_state: Dict[str, Any],
        npc_id: Optional[str] = None
    ) -> str:
        """Build system prompt with game context"""
        parts = [
            "You are an NPC in a video game.",
            "Respond naturally to the player based on the game context."
        ]

        if npc_id:
            parts.append(f"You are playing the role of {npc_id}.")

        # Add scene context
        if "scene_id" in game_state:
            parts.append(f"Current location: {game_state['scene_id']}")

        # Add NPC info
        if "npcs" in game_state and npc_id:
            for npc in game_state["npcs"]:
                if npc.get("id") == npc_id:
                    parts.append(f"Role: {npc.get('role', 'NPC')}")
                    parts.append(f"Mood: {npc.get('mood', 'neutral')}")
                    break

        # Add world info
        if "world" in game_state:
            world = game_state["world"]
            parts.append(f"Time: {world.get('time_of_day', 'unknown')}")
            if "weather" in world:
                parts.append(f"Weather: {world['weather']}")

        return " ".join(parts)


@dataclass
class ModelVersion:
    """Fine-tuned model version"""
    model_id: str
    version: str
    provider: str  # "openai" or "anthropic"
    base_model: str  # e.g., "gpt-4o-mini"
    training_dataset: str
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "provider": self.provider,
            "base_model": self.base_model,
            "training_dataset": self.training_dataset,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "metrics": self.metrics
        }


class ModelVersionManager:
    """
    Manage fine-tuned model versions and A/B testing.

    Features:
    - Version tracking
    - A/B testing support
    - Performance metrics
    - Automatic fallback
    """

    def __init__(self, config_file: str = "./model_versions.json"):
        self.config_file = Path(config_file)
        self.versions: Dict[str, ModelVersion] = {}
        self._load_versions()

    def add_version(self, version: ModelVersion):
        """Add new model version"""
        self.versions[version.model_id] = version
        self._save_versions()

    def set_active(self, model_id: str):
        """Set model as active (for production use)"""
        # Deactivate all
        for version in self.versions.values():
            version.is_active = False

        # Activate target
        if model_id in self.versions:
            self.versions[model_id].is_active = True
            self._save_versions()

    def get_active_model(self) -> Optional[ModelVersion]:
        """Get currently active model"""
        for version in self.versions.values():
            if version.is_active:
                return version
        return None

    def get_model_for_ab_test(
        self,
        player_id: str,
        split_ratio: float = 0.5
    ) -> Optional[ModelVersion]:
        """
        Get model for A/B testing

        Args:
            player_id: Player identifier (for consistent assignment)
            split_ratio: Percentage to use new model (0.0-1.0)

        Returns:
            ModelVersion to use
        """
        # Hash player ID to get consistent assignment
        hash_value = hash(player_id)
        use_new = (hash_value % 100) / 100.0 < split_ratio

        active = self.get_active_model()
        versions_list = list(self.versions.values())

        if use_new and len(versions_list) > 1:
            # Use newest non-active model
            newest = max(
                [v for v in versions_list if not v.is_active],
                key=lambda v: v.created_at,
                default=None
            )
            return newest or active

        return active

    def update_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ):
        """Update model performance metrics"""
        if model_id in self.versions:
            self.versions[model_id].metrics.update(metrics)
            self._save_versions()

    def compare_models(
        self,
        model_id_a: str,
        model_id_b: str
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        version_a = self.versions.get(model_id_a)
        version_b = self.versions.get(model_id_b)

        if not version_a or not version_b:
            raise ValueError("Invalid model IDs")

        return {
            "model_a": {
                "id": model_id_a,
                "metrics": version_a.metrics
            },
            "model_b": {
                "id": model_id_b,
                "metrics": version_b.metrics
            },
            "delta": {
                key: version_b.metrics.get(key, 0) - version_a.metrics.get(key, 0)
                for key in set(list(version_a.metrics.keys()) + list(version_b.metrics.keys()))
            }
        }

    def _load_versions(self):
        """Load versions from config file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                for model_id, version_data in data.items():
                    self.versions[model_id] = ModelVersion(
                        model_id=version_data["model_id"],
                        version=version_data["version"],
                        provider=version_data["provider"],
                        base_model=version_data["base_model"],
                        training_dataset=version_data["training_dataset"],
                        created_at=datetime.fromisoformat(version_data["created_at"]),
                        is_active=version_data["is_active"],
                        metrics=version_data.get("metrics", {})
                    )

    def _save_versions(self):
        """Save versions to config file"""
        data = {
            model_id: version.to_dict()
            for model_id, version in self.versions.items()
        }

        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)


# Helper functions for common workflows

def export_session_to_dataset(
    session_data: List[Dict[str, Any]],
    output_file: str,
    provider: str = "openai"
):
    """
    Export game session to fine-tuning dataset

    Args:
        session_data: List of conversation turns
        output_file: Output JSONL file
        provider: "openai" or "anthropic"
    """
    exporter = FineTuningExporter()
    dataset = FineTuningDataset(name=Path(output_file).stem)

    for turn in session_data:
        if "player_message" in turn and "npc_response" in turn:
            example = exporter.export_conversation(
                game_state=turn.get("game_state", {}),
                player_message=turn["player_message"],
                npc_response=turn["npc_response"],
                npc_id=turn.get("npc_id", "unknown")
            )
            dataset.add_example(example)

    # Export
    if provider == "openai":
        dataset.export_openai_jsonl(output_file)
    else:
        dataset.export_anthropic_jsonl(output_file)


def create_ab_test(
    active_model_id: str,
    new_model_id: str,
    split_ratio: float = 0.1
) -> ModelVersionManager:
    """
    Set up A/B test between models

    Args:
        active_model_id: Current production model
        new_model_id: New model to test
        split_ratio: Percentage of traffic for new model (default: 10%)

    Returns:
        ModelVersionManager configured for A/B testing
    """
    manager = ModelVersionManager()
    manager.set_active(active_model_id)
    return manager
