"""
Elle Game Engine - Active Development
======================================

New features for the Elle game engine.
Core engine is in archive/dormant_projects/bee_inspection/elle_game_engine.

New Features (2025-12):
- Branching Dialogue System
- Output Validation Pipeline
- Voice Integration with Caching

Version: 0.3.0
"""

__version__ = "0.3.0"

from .branching_dialogue import (
    DialogueNode,
    DialogueChoice,
    DialogueTree,
    DialogueCondition,
    DialogueEffect,
    BranchingDialogueEngine,
)

from .output_validation import (
    ValidationSeverity,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationContext,
    ValidationRule,
    BaseValidationRule,
    ConsistencyCheckRule,
    SafetyFilterRule,
    CanonAlignmentRule,
    ToneMatchRule,
    CharacterVoiceRule,
    ValidationPipeline,
    create_validation_pipeline,
    create_strict_pipeline,
    create_permissive_pipeline,
)

from .voice_integration import (
    VoiceProvider,
    VoiceProviderType,
    VoiceProfile,
    VoiceCacheEntry,
    VoiceCache,
    VoiceManager,
    MockVoiceProvider,
    create_voice_manager,
    create_mock_voice_manager,
)

__all__ = [
    # Branching Dialogue
    'DialogueNode',
    'DialogueChoice',
    'DialogueTree',
    'DialogueCondition',
    'DialogueEffect',
    'BranchingDialogueEngine',
    # Output Validation
    'ValidationSeverity',
    'ValidationCategory',
    'ValidationIssue',
    'ValidationResult',
    'ValidationContext',
    'ValidationRule',
    'BaseValidationRule',
    'ConsistencyCheckRule',
    'SafetyFilterRule',
    'CanonAlignmentRule',
    'ToneMatchRule',
    'CharacterVoiceRule',
    'ValidationPipeline',
    'create_validation_pipeline',
    'create_strict_pipeline',
    'create_permissive_pipeline',
    # Voice Integration
    'VoiceProvider',
    'VoiceProviderType',
    'VoiceProfile',
    'VoiceCacheEntry',
    'VoiceCache',
    'VoiceManager',
    'MockVoiceProvider',
    'create_voice_manager',
    'create_mock_voice_manager',
]
