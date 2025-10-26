#!/usr/bin/env python3
"""
Team Learning System

Mines high-quality interactions to:
- Extract successful conversation patterns
- Build training datasets from best examples
- Identify expert knowledge and best practices
- Create few-shot learning examples
- Generate documentation from conversations

Usage:
    from HoloLoom.chatops.team_learning import TeamLearningSystem

    system = TeamLearningSystem()

    # Mine conversations
    insights = await system.mine_conversations(room_id, date_range)

    # Extract training data
    training_data = system.get_training_examples(min_quality=0.9)

    # Generate documentation
    docs = await system.generate_documentation(topic="incident_response")

    # Find expert knowledge
    expertise = system.extract_expertise(user_id)
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import re


@dataclass
class Interaction:
    """A query-response interaction"""
    interaction_id: str
    timestamp: datetime
    user_id: str
    query: str
    response: str
    context: Dict[str, Any]

    # Quality metrics
    quality_score: float
    user_feedback: Optional[str] = None
    follow_up_questions: int = 0
    resolution_time: Optional[float] = None

    # Classification
    query_type: str = "general"
    complexity: str = "medium"  # simple, medium, complex
    topics: List[str] = field(default_factory=list)


@dataclass
class TrainingExample:
    """Training example extracted from interaction"""
    example_id: str
    query: str
    response: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Quality indicators
    quality_score: float = 0.0
    expert_validated: bool = False
    usage_count: int = 0


@dataclass
class BestPractice:
    """Best practice extracted from conversations"""
    practice_id: str
    title: str
    description: str
    category: str
    examples: List[str] = field(default_factory=list)
    evidence_count: int = 0
    confidence: float = 0.0
    expert_sources: List[str] = field(default_factory=list)


@dataclass
class ExpertKnowledge:
    """Expert knowledge profile"""
    user_id: str
    expertise_areas: Dict[str, float]  # topic -> confidence score
    best_responses: List[str] = field(default_factory=list)
    teaching_patterns: List[str] = field(default_factory=list)
    contribution_count: int = 0
    avg_quality_score: float = 0.0


class TeamLearningSystem:
    """
    System for learning from team interactions.

    Extracts:
    - High-quality training examples
    - Best practices and patterns
    - Expert knowledge profiles
    - Documentation from conversations
    """

    def __init__(
        self,
        min_quality_threshold: float = 0.8,
        storage_path: Optional[Path] = None
    ):
        self.min_quality_threshold = min_quality_threshold
        self.storage_path = storage_path or Path("./chatops_data/team_learning")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Storage
        self.interactions: Dict[str, Interaction] = {}
        self.training_examples: Dict[str, TrainingExample] = {}
        self.best_practices: Dict[str, BestPractice] = {}
        self.expert_profiles: Dict[str, ExpertKnowledge] = {}

        # Statistics
        self.stats = {
            "total_interactions": 0,
            "high_quality_interactions": 0,
            "training_examples_created": 0,
            "best_practices_identified": 0,
            "experts_profiled": 0
        }

        # Load existing data
        self._load_data()

        logging.info("TeamLearningSystem initialized")

    async def mine_conversations(
        self,
        room_id: str,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        min_quality: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Mine conversations for learning opportunities.

        Args:
            room_id: Matrix room ID
            date_range: Optional date range to mine
            min_quality: Minimum quality threshold

        Returns:
            Mining results with insights
        """

        min_quality = min_quality or self.min_quality_threshold

        # Fetch interactions (would integrate with conversation memory)
        interactions = await self._fetch_interactions(room_id, date_range)

        # Filter by quality
        high_quality = [
            interaction for interaction in interactions
            if interaction.quality_score >= min_quality
        ]

        # Extract training examples
        new_examples = self._extract_training_examples(high_quality)

        # Identify best practices
        new_practices = self._identify_best_practices(high_quality)

        # Update expert profiles
        self._update_expert_profiles(high_quality)

        # Update statistics
        self.stats["total_interactions"] += len(interactions)
        self.stats["high_quality_interactions"] += len(high_quality)
        self.stats["training_examples_created"] += len(new_examples)
        self.stats["best_practices_identified"] += len(new_practices)

        # Save data
        self._save_data()

        return {
            "interactions_analyzed": len(interactions),
            "high_quality_found": len(high_quality),
            "training_examples": len(new_examples),
            "best_practices": len(new_practices),
            "experts_updated": len(set(i.user_id for i in high_quality))
        }

    def _extract_training_examples(
        self,
        interactions: List[Interaction]
    ) -> List[TrainingExample]:
        """Extract training examples from interactions"""

        examples = []

        for interaction in interactions:
            # Skip low quality
            if interaction.quality_score < self.min_quality_threshold:
                continue

            # Create example
            example = TrainingExample(
                example_id=f"train_{interaction.interaction_id}",
                query=interaction.query,
                response=interaction.response,
                context=self._format_context(interaction.context),
                metadata={
                    "query_type": interaction.query_type,
                    "complexity": interaction.complexity,
                    "topics": interaction.topics,
                    "timestamp": interaction.timestamp.isoformat(),
                    "user_id": interaction.user_id
                },
                quality_score=interaction.quality_score
            )

            # Store
            self.training_examples[example.example_id] = example
            examples.append(example)

        return examples

    def _identify_best_practices(
        self,
        interactions: List[Interaction]
    ) -> List[BestPractice]:
        """Identify best practices from interactions"""

        practices = []

        # Group by query type
        by_type = defaultdict(list)
        for interaction in interactions:
            by_type[interaction.query_type].append(interaction)

        # Extract patterns from each type
        for query_type, type_interactions in by_type.items():
            # Find common successful approaches
            patterns = self._find_common_patterns(type_interactions)

            for pattern in patterns:
                practice_id = f"bp_{query_type}_{len(self.best_practices)}"

                practice = BestPractice(
                    practice_id=practice_id,
                    title=pattern["title"],
                    description=pattern["description"],
                    category=query_type,
                    examples=pattern["examples"],
                    evidence_count=pattern["count"],
                    confidence=pattern["confidence"],
                    expert_sources=pattern["experts"]
                )

                self.best_practices[practice_id] = practice
                practices.append(practice)

        return practices

    def _find_common_patterns(
        self,
        interactions: List[Interaction]
    ) -> List[Dict[str, Any]]:
        """Find common patterns in interactions"""

        patterns = []

        # Pattern 1: Successful incident response structure
        incident_patterns = self._extract_incident_patterns(interactions)
        patterns.extend(incident_patterns)

        # Pattern 2: Effective code review approach
        review_patterns = self._extract_review_patterns(interactions)
        patterns.extend(review_patterns)

        # Pattern 3: Clear explanation structure
        explanation_patterns = self._extract_explanation_patterns(interactions)
        patterns.extend(explanation_patterns)

        return patterns

    def _extract_incident_patterns(
        self,
        interactions: List[Interaction]
    ) -> List[Dict[str, Any]]:
        """Extract incident response patterns"""

        patterns = []

        # Pattern: "Severity assessment first"
        severity_first = [
            i for i in interactions
            if i.query_type == "incident" and
            re.search(r"(severity|priority|critical|high|low).*?(first|upfront|immediately)", i.response, re.IGNORECASE)
        ]

        if len(severity_first) >= 3:
            patterns.append({
                "title": "Assess Severity First",
                "description": "Always start incident responses with severity assessment",
                "examples": [i.response[:200] for i in severity_first[:3]],
                "count": len(severity_first),
                "confidence": min(len(severity_first) / 10, 1.0),
                "experts": list(set(i.user_id for i in severity_first))
            })

        # Pattern: "Root cause before remediation"
        root_cause_first = [
            i for i in interactions
            if i.query_type == "incident" and
            re.search(r"root cause.*?(before|prior to|first)", i.response, re.IGNORECASE)
        ]

        if len(root_cause_first) >= 3:
            patterns.append({
                "title": "Identify Root Cause Before Remediation",
                "description": "Investigate root cause before proposing fixes",
                "examples": [i.response[:200] for i in root_cause_first[:3]],
                "count": len(root_cause_first),
                "confidence": min(len(root_cause_first) / 10, 1.0),
                "experts": list(set(i.user_id for i in root_cause_first))
            })

        return patterns

    def _extract_review_patterns(
        self,
        interactions: List[Interaction]
    ) -> List[Dict[str, Any]]:
        """Extract code review patterns"""

        patterns = []

        # Pattern: "Security concerns highlighted"
        security_focus = [
            i for i in interactions
            if i.query_type == "code_review" and
            re.search(r"(security|vulnerability|exploit|injection)", i.response, re.IGNORECASE)
        ]

        if len(security_focus) >= 3:
            patterns.append({
                "title": "Highlight Security Concerns",
                "description": "Always check and highlight security issues in code reviews",
                "examples": [i.response[:200] for i in security_focus[:3]],
                "count": len(security_focus),
                "confidence": min(len(security_focus) / 10, 1.0),
                "experts": list(set(i.user_id for i in security_focus))
            })

        return patterns

    def _extract_explanation_patterns(
        self,
        interactions: List[Interaction]
    ) -> List[Dict[str, Any]]:
        """Extract explanation patterns"""

        patterns = []

        # Pattern: "Use examples"
        with_examples = [
            i for i in interactions
            if re.search(r"(for example|e\.g\.|such as|like)", i.response, re.IGNORECASE)
        ]

        if len(with_examples) >= 5:
            patterns.append({
                "title": "Use Concrete Examples",
                "description": "Include examples to illustrate abstract concepts",
                "examples": [i.response[:200] for i in with_examples[:3]],
                "count": len(with_examples),
                "confidence": min(len(with_examples) / 20, 1.0),
                "experts": list(set(i.user_id for i in with_examples))
            })

        return patterns

    def _update_expert_profiles(self, interactions: List[Interaction]):
        """Update expert knowledge profiles"""

        # Group by user
        by_user = defaultdict(list)
        for interaction in interactions:
            by_user[interaction.user_id].append(interaction)

        # Update profiles
        for user_id, user_interactions in by_user.items():
            profile = self.expert_profiles.get(user_id, ExpertKnowledge(
                user_id=user_id,
                expertise_areas={}
            ))

            # Extract topics and quality
            for interaction in user_interactions:
                for topic in interaction.topics:
                    current_score = profile.expertise_areas.get(topic, 0.0)
                    # Weight by quality and recency
                    new_score = current_score * 0.9 + interaction.quality_score * 0.1
                    profile.expertise_areas[topic] = new_score

            # Update best responses
            best = sorted(user_interactions, key=lambda x: x.quality_score, reverse=True)[:5]
            profile.best_responses = [i.interaction_id for i in best]

            # Update stats
            profile.contribution_count += len(user_interactions)
            profile.avg_quality_score = sum(i.quality_score for i in user_interactions) / len(user_interactions)

            self.expert_profiles[user_id] = profile

        self.stats["experts_profiled"] = len(self.expert_profiles)

    def get_training_examples(
        self,
        min_quality: Optional[float] = None,
        query_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[TrainingExample]:
        """
        Get training examples for fine-tuning or few-shot learning.

        Args:
            min_quality: Minimum quality score
            query_type: Filter by query type
            limit: Maximum number of examples

        Returns:
            List of training examples
        """

        min_quality = min_quality or self.min_quality_threshold

        # Filter examples
        examples = [
            ex for ex in self.training_examples.values()
            if ex.quality_score >= min_quality
        ]

        if query_type:
            examples = [
                ex for ex in examples
                if ex.metadata.get("query_type") == query_type
            ]

        # Sort by quality
        examples.sort(key=lambda x: x.quality_score, reverse=True)

        if limit:
            examples = examples[:limit]

        return examples

    def get_few_shot_examples(
        self,
        query: str,
        n: int = 3
    ) -> List[TrainingExample]:
        """
        Get few-shot examples similar to query.

        Args:
            query: Query to match
            n: Number of examples

        Returns:
            Most relevant training examples
        """

        # Classify query
        query_type = self._classify_query(query)

        # Get examples of same type
        examples = self.get_training_examples(query_type=query_type)

        # Simple relevance scoring (would use embeddings in production)
        scored = []
        query_words = set(query.lower().split())

        for example in examples:
            example_words = set(example.query.lower().split())
            overlap = len(query_words & example_words)
            scored.append((overlap, example))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [ex for _, ex in scored[:n]]

    async def generate_documentation(
        self,
        topic: str,
        min_examples: int = 5
    ) -> Optional[str]:
        """
        Generate documentation from conversations.

        Args:
            topic: Topic to document
            min_examples: Minimum number of examples needed

        Returns:
            Generated documentation
        """

        # Find relevant interactions
        relevant = [
            interaction for interaction in self.interactions.values()
            if topic.lower() in " ".join(interaction.topics).lower() and
            interaction.quality_score >= self.min_quality_threshold
        ]

        if len(relevant) < min_examples:
            return None

        # Extract best practices
        practices = [
            bp for bp in self.best_practices.values()
            if topic.lower() in bp.category.lower()
        ]

        # Find experts
        experts = self.find_experts(topic, limit=3)

        # Generate documentation
        doc = f"# {topic.replace('_', ' ').title()}\n\n"
        doc += f"*Auto-generated from {len(relevant)} high-quality team interactions*\n\n"

        # Overview
        doc += "## Overview\n\n"
        doc += f"This guide is based on {len(relevant)} successful {topic} interactions from our team.\n\n"

        # Best Practices
        if practices:
            doc += "## Best Practices\n\n"
            for practice in practices:
                doc += f"### {practice.title}\n\n"
                doc += f"{practice.description}\n\n"
                doc += f"**Evidence**: {practice.evidence_count} examples (confidence: {practice.confidence:.0%})\n\n"

                if practice.examples:
                    doc += "**Example:**\n```\n"
                    doc += practice.examples[0][:200] + "...\n"
                    doc += "```\n\n"

        # Common Patterns
        doc += "## Common Patterns\n\n"
        patterns = self._extract_common_sequences(relevant)
        for i, pattern in enumerate(patterns[:5], 1):
            doc += f"{i}. {pattern}\n"
        doc += "\n"

        # Expert Contributors
        if experts:
            doc += "## Expert Contributors\n\n"
            for expert in experts:
                profile = self.expert_profiles[expert]
                doc += f"- **{expert}**: {profile.contribution_count} contributions, "
                doc += f"avg quality {profile.avg_quality_score:.2f}\n"
            doc += "\n"

        # Recent Examples
        doc += "## Recent Examples\n\n"
        recent = sorted(relevant, key=lambda x: x.timestamp, reverse=True)[:3]
        for example in recent:
            doc += f"### Example from {example.timestamp.strftime('%Y-%m-%d')}\n\n"
            doc += f"**Query**: {example.query}\n\n"
            doc += f"**Approach**: {example.response[:300]}...\n\n"

        return doc

    def find_experts(
        self,
        topic: str,
        limit: int = 5
    ) -> List[str]:
        """
        Find experts on a topic.

        Args:
            topic: Topic to find experts for
            limit: Maximum number of experts

        Returns:
            List of user IDs sorted by expertise
        """

        scored_experts = []

        for user_id, profile in self.expert_profiles.items():
            # Check if user has expertise in topic
            expertise_score = profile.expertise_areas.get(topic, 0.0)

            # Weight by overall quality and contribution count
            final_score = (
                expertise_score * 0.6 +
                profile.avg_quality_score * 0.3 +
                min(profile.contribution_count / 50, 1.0) * 0.1
            )

            if final_score > 0:
                scored_experts.append((final_score, user_id))

        scored_experts.sort(reverse=True)

        return [user_id for _, user_id in scored_experts[:limit]]

    def get_expert_profile(self, user_id: str) -> Optional[ExpertKnowledge]:
        """Get expert profile for user"""
        return self.expert_profiles.get(user_id)

    def export_training_dataset(
        self,
        output_path: Path,
        format: str = "jsonl",
        min_quality: float = 0.9
    ):
        """
        Export training dataset.

        Args:
            output_path: Output file path
            format: Format (jsonl, json, csv)
            min_quality: Minimum quality threshold
        """

        examples = self.get_training_examples(min_quality=min_quality)

        if format == "jsonl":
            with open(output_path, 'w') as f:
                for example in examples:
                    entry = {
                        "messages": [
                            {"role": "user", "content": example.query},
                            {"role": "assistant", "content": example.response}
                        ],
                        "metadata": example.metadata
                    }
                    f.write(json.dumps(entry) + "\n")

        elif format == "json":
            data = [
                {
                    "query": ex.query,
                    "response": ex.response,
                    "quality_score": ex.quality_score,
                    "metadata": ex.metadata
                }
                for ex in examples
            ]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        logging.info(f"Exported {len(examples)} training examples to {output_path}")

    def _extract_common_sequences(self, interactions: List[Interaction]) -> List[str]:
        """Extract common response sequences"""

        sequences = []

        # Look for common structures
        for interaction in interactions:
            # Extract sentence structure
            sentences = interaction.response.split('. ')
            if len(sentences) >= 3:
                # First three sentence pattern
                pattern = " → ".join(s[:30] + "..." for s in sentences[:3])
                sequences.append(pattern)

        # Find most common
        counter = Counter(sequences)
        return [seq for seq, _ in counter.most_common(10)]

    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["incident", "error", "down"]):
            return "incident"
        elif any(word in query_lower for word in ["review", "pr", "code"]):
            return "code_review"
        elif any(word in query_lower for word in ["deploy", "workflow"]):
            return "workflow"
        else:
            return "general"

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for training example"""
        if not context:
            return ""

        parts = []
        for key, value in context.items():
            if isinstance(value, (str, int, float)):
                parts.append(f"{key}: {value}")

        return " | ".join(parts)

    async def _fetch_interactions(
        self,
        room_id: str,
        date_range: Optional[Tuple[datetime, datetime]]
    ) -> List[Interaction]:
        """Fetch interactions from storage (stub)"""

        # This would integrate with conversation memory
        # For now, return empty list
        return []

    def _save_data(self):
        """Save learning data to disk"""

        # Save training examples
        examples_file = self.storage_path / "training_examples.json"
        examples_data = {
            ex_id: {
                "query": ex.query,
                "response": ex.response,
                "quality_score": ex.quality_score,
                "metadata": ex.metadata
            }
            for ex_id, ex in self.training_examples.items()
        }
        with open(examples_file, 'w') as f:
            json.dump(examples_data, f, indent=2)

        # Save best practices
        practices_file = self.storage_path / "best_practices.json"
        practices_data = {
            bp_id: {
                "title": bp.title,
                "description": bp.description,
                "category": bp.category,
                "confidence": bp.confidence,
                "evidence_count": bp.evidence_count
            }
            for bp_id, bp in self.best_practices.items()
        }
        with open(practices_file, 'w') as f:
            json.dump(practices_data, f, indent=2)

        # Save expert profiles
        experts_file = self.storage_path / "expert_profiles.json"
        experts_data = {
            user_id: {
                "expertise_areas": profile.expertise_areas,
                "contribution_count": profile.contribution_count,
                "avg_quality_score": profile.avg_quality_score
            }
            for user_id, profile in self.expert_profiles.items()
        }
        with open(experts_file, 'w') as f:
            json.dump(experts_data, f, indent=2)

    def _load_data(self):
        """Load learning data from disk"""

        try:
            # Load training examples
            examples_file = self.storage_path / "training_examples.json"
            if examples_file.exists():
                with open(examples_file, 'r') as f:
                    examples_data = json.load(f)
                    self.stats["training_examples_created"] = len(examples_data)

            # Load best practices
            practices_file = self.storage_path / "best_practices.json"
            if practices_file.exists():
                with open(practices_file, 'r') as f:
                    practices_data = json.load(f)
                    self.stats["best_practices_identified"] = len(practices_data)

            # Load expert profiles
            experts_file = self.storage_path / "expert_profiles.json"
            if experts_file.exists():
                with open(experts_file, 'r') as f:
                    experts_data = json.load(f)
                    self.stats["experts_profiled"] = len(experts_data)

            logging.info(f"Loaded learning data: {len(self.training_examples)} examples, {len(self.best_practices)} practices")

        except Exception as e:
            logging.error(f"Failed to load data: {e}")


# Demo
async def demo_team_learning():
    """Demonstrate team learning system"""

    print("📚 Team Learning System Demo\n")

    system = TeamLearningSystem()

    # Simulate some high-quality interactions
    print("Creating sample interactions...\n")

    # TODO: Add sample interactions

    # Get training examples
    examples = system.get_training_examples(min_quality=0.8, limit=5)
    print(f"Training Examples: {len(examples)}")

    # Find experts
    experts = system.find_experts("incident_response", limit=3)
    print(f"\nExperts on incident_response: {experts}")

    # Generate documentation
    docs = await system.generate_documentation("incident_response")
    if docs:
        print(f"\nGenerated Documentation:\n{docs[:500]}...")

    # Export dataset
    output_path = Path("./training_dataset.jsonl")
    system.export_training_dataset(output_path, format="jsonl")
    print(f"\nExported training dataset to {output_path}")


if __name__ == "__main__":
    asyncio.run(demo_team_learning())
