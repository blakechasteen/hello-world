"""
Unit Tests: ChatHistorySpinner
================================

Tests for chat history ingestion into HoloLoom memory system.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from hololoom.apps.workflow_builder.conversation_manager import ConversationManager
from hololoom.spinningWheel.chat_history import (
    ChatHistorySpinner,
    ImportanceScore
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
        db_path = f.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def conversation_manager(temp_db):
    """Create ConversationManager with test database."""
    return ConversationManager(temp_db)


@pytest.fixture
def sample_conversation(conversation_manager):
    """Create a sample conversation for testing."""
    conv = conversation_manager.create_conversation("Test Conversation")

    conversation_manager.add_message(
        conv.id,
        'user',
        "What is Thompson Sampling?"
    )

    conversation_manager.add_message(
        conv.id,
        'assistant',
        "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
        metadata={'confidence': 0.9, 'tool': 'answer'}
    )

    conversation_manager.add_message(
        conv.id,
        'user',
        "How does it work?"
    )

    conversation_manager.add_message(
        conv.id,
        'assistant',
        "It samples from Beta distributions to balance exploration and exploitation.",
        metadata={'confidence': 0.85}
    )

    return conv


class TestChatHistorySpinner:
    """Tests for ChatHistorySpinner."""

    @pytest.mark.asyncio
    async def test_spin_conversation(self, conversation_manager, sample_conversation):
        """Test spinning a single conversation into shards."""
        spinner = ChatHistorySpinner(conversation_manager, importance_threshold=0.0)

        shards = await spinner.spin_conversation(sample_conversation.id)

        assert len(shards) > 0, "Should create at least one shard"
        assert all(shard.id.startswith('chat_') for shard in shards), \
            "Shard IDs should have chat_ prefix"
        assert all(shard.episode == sample_conversation.title for shard in shards), \
            "All shards should reference conversation title"

    @pytest.mark.asyncio
    async def test_importance_filtering(self, conversation_manager):
        """Test that importance threshold filters low-importance messages."""
        # Create conversation with varying importance
        conv = conversation_manager.create_conversation("Mixed Importance")

        # High importance - technical question
        conversation_manager.add_message(
            conv.id, 'user',
            "Explain the difference between Thompson Sampling and UCB algorithms?"
        )
        conversation_manager.add_message(
            conv.id, 'assistant',
            "Thompson Sampling uses probability matching while UCB uses optimism.",
            metadata={'confidence': 0.92}
        )

        # Low importance - greeting
        conversation_manager.add_message(conv.id, 'user', "Thanks!")
        conversation_manager.add_message(conv.id, 'assistant', "You're welcome!")

        spinner = ChatHistorySpinner(conversation_manager, importance_threshold=0.5)
        shards = await spinner.spin_conversation(conv.id)

        # Should filter out low-importance greeting
        assert len(shards) < 2, "Low importance messages should be filtered"

    @pytest.mark.asyncio
    async def test_spin_all(self, conversation_manager, sample_conversation):
        """Test spinning all conversations."""
        # Create second conversation
        conv2 = conversation_manager.create_conversation("Another Conversation")
        conversation_manager.add_message(conv2.id, 'user', "What is HoloLoom?")
        conversation_manager.add_message(
            conv2.id, 'assistant',
            "HoloLoom is a neural decision-making system.",
            metadata={'confidence': 0.88}
        )

        spinner = ChatHistorySpinner(conversation_manager, importance_threshold=0.0)
        shards = await spinner.spin_all()

        assert len(shards) > 0, "Should create shards from all conversations"

        # Check that shards from both conversations are present
        conv_ids = {shard.metadata['conversation_id'] for shard in shards}
        assert sample_conversation.id in conv_ids
        assert conv2.id in conv_ids

    @pytest.mark.asyncio
    async def test_entity_extraction(self, conversation_manager):
        """Test entity extraction from conversations."""
        conv = conversation_manager.create_conversation("Entity Test")

        conversation_manager.add_message(
            conv.id, 'user',
            "Tell me about Thompson Sampling and Bayesian inference"
        )
        conversation_manager.add_message(
            conv.id, 'assistant',
            "Thompson Sampling uses Bayesian methods for exploration.",
            metadata={'confidence': 0.9}
        )

        spinner = ChatHistorySpinner(conversation_manager, extract_entities=True)
        shards = await spinner.spin_conversation(conv.id, min_importance=0.0)

        assert len(shards) > 0
        shard = shards[0]

        # Should extract capitalized terms as entities
        assert len(shard.entities) > 0, "Should extract entities"
        # "Thompson" and "Sampling" might be extracted
        entity_text = ' '.join(shard.entities).lower()
        assert any(term in entity_text for term in ['thompson', 'bayesian']), \
            "Should extract key entities"

    @pytest.mark.asyncio
    async def test_motif_extraction(self, conversation_manager):
        """Test motif extraction from conversations."""
        conv = conversation_manager.create_conversation("Motif Test")

        conversation_manager.add_message(
            conv.id, 'user',
            "What is reinforcement learning? How does it work?"
        )
        conversation_manager.add_message(
            conv.id, 'assistant',
            "Reinforcement learning is a type of machine learning.",
            metadata={'confidence': 0.9}
        )

        spinner = ChatHistorySpinner(conversation_manager, extract_motifs=True)
        shards = await spinner.spin_conversation(conv.id, min_importance=0.0)

        assert len(shards) > 0
        shard = shards[0]

        # Should extract questions as motifs
        assert len(shard.motifs) > 0, "Should extract motifs from questions"

    @pytest.mark.asyncio
    async def test_metadata_preservation(self, conversation_manager, sample_conversation):
        """Test that conversation metadata is preserved in shards."""
        spinner = ChatHistorySpinner(conversation_manager)
        shards = await spinner.spin_conversation(sample_conversation.id, min_importance=0.0)

        assert len(shards) > 0
        shard = shards[0]

        # Check metadata
        assert 'conversation_id' in shard.metadata
        assert 'conversation_title' in shard.metadata
        assert 'turn_index' in shard.metadata
        assert 'timestamp' in shard.metadata
        assert 'importance_score' in shard.metadata
        assert 'spinner' in shard.metadata

        assert shard.metadata['conversation_id'] == sample_conversation.id
        assert shard.metadata['conversation_title'] == sample_conversation.title
        assert shard.metadata['spinner'] == 'ChatHistorySpinner'

    @pytest.mark.asyncio
    async def test_importance_scoring_signals(self, conversation_manager):
        """Test importance scoring signals."""
        conv = conversation_manager.create_conversation("Scoring Test")

        # Technical question with high confidence
        conversation_manager.add_message(
            conv.id, 'user',
            "Explain how Thompson Sampling updates its Beta distributions?"
        )
        conversation_manager.add_message(
            conv.id, 'assistant',
            "Thompson Sampling maintains Beta(α,β) distributions. "
            "On success, α increases. On failure, β increases.",
            metadata={'confidence': 0.95, 'reasoning_steps': ['analyze', 'synthesize']}
        )

        spinner = ChatHistorySpinner(conversation_manager, importance_threshold=0.0)
        shards = await spinner.spin_conversation(conv.id, min_importance=0.0)

        assert len(shards) > 0
        shard = shards[0]

        # Check importance signals
        assert 'importance_score' in shard.metadata
        assert 'importance_signals' in shard.metadata
        assert 'importance_reason' in shard.metadata

        score = shard.metadata['importance_score']
        signals = shard.metadata['importance_signals']

        # Should have high importance due to technical content
        assert score > 0.5, "Technical conversation should have high importance"

        # Should have positive signals for technical terms, questions, confidence
        assert 'technical' in signals
        assert signals.get('technical', 0) > 0, "Should detect technical terms"

    @pytest.mark.asyncio
    async def test_tag_filtering(self, conversation_manager):
        """Test filtering conversations by tag."""
        # Create tagged conversation
        conv1 = conversation_manager.create_conversation("Tagged Conversation")
        conversation_manager.update_tags(conv1.id, ['bandits', 'exploration'])
        conversation_manager.add_message(conv1.id, 'user', "What are bandits?")
        conversation_manager.add_message(
            conv1.id, 'assistant', "Bandits are decision algorithms."
        )

        # Create untagged conversation
        conv2 = conversation_manager.create_conversation("Untagged")
        conversation_manager.add_message(conv2.id, 'user', "Hello")
        conversation_manager.add_message(conv2.id, 'assistant', "Hi there!")

        spinner = ChatHistorySpinner(conversation_manager, importance_threshold=0.0)
        shards = await spinner.spin_by_tag('bandits')

        # Should only get shards from tagged conversation
        conv_ids = {shard.metadata['conversation_id'] for shard in shards}
        assert conv1.id in conv_ids
        assert conv2.id not in conv_ids

    @pytest.mark.asyncio
    async def test_project_filtering(self, conversation_manager):
        """Test filtering conversations by project."""
        # Create project
        project = conversation_manager.create_project("ML Research", color="#667eea")

        # Create conversation in project
        conv1 = conversation_manager.create_conversation("In Project")
        conversation_manager.move_to_project(conv1.id, project.id)
        conversation_manager.add_message(conv1.id, 'user', "Project question")
        conversation_manager.add_message(conv1.id, 'assistant', "Project answer")

        # Create conversation outside project
        conv2 = conversation_manager.create_conversation("Outside Project")
        conversation_manager.add_message(conv2.id, 'user', "Other question")
        conversation_manager.add_message(conv2.id, 'assistant', "Other answer")

        spinner = ChatHistorySpinner(conversation_manager, importance_threshold=0.0)
        shards = await spinner.spin_by_project(project.id)

        # Should only get shards from project conversation
        conv_ids = {shard.metadata['conversation_id'] for shard in shards}
        assert conv1.id in conv_ids
        assert conv2.id not in conv_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])