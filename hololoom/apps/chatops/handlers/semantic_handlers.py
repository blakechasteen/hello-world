#!/usr/bin/env python3
"""
Semantic State Matrix Bot Handlers
===================================
Phase 8 integration: Semantic State -> Policy -> Matrix Chat

Commands:
- !semantic <query> - Analyze query semantic state (244D -> 8D)
- !semantic shift - Show topic shift status in conversation
- !semantic tools - Show semantic-aware tool suggestions
- !semantic bandit - Show Thompson Sampling with semantic adjustments
- !semantic compare <q1> | <q2> - Compare two queries semantically
- !semantic replay [limit] - Replay conversation semantic trajectory
- !semantic suggest - Suggest next action based on context
- !semantic help - Show semantic commands help

Created: 2025-12-23
Updated: 2025-12-25 - Added compare, replay, suggest commands
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field

try:
    from nio import MatrixRoom, RoomMessageText
except ImportError:
    pass

import numpy as np

# Import semantic calculus components
try:
    from hololoom.semantic_calculus import (
        PolicySemanticState as SemanticState,
        SemanticToolSelector,
        SemanticAwareBandit,
        SEMANTIC_CATEGORIES,
        TOPIC_SHIFT_THRESHOLD,
        SIGNIFICANT_SHIFT_THRESHOLD,
    )
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    SemanticState = None
    SemanticToolSelector = None
    SemanticAwareBandit = None
    SEMANTIC_CATEGORIES = {}

# Handler registry
from hololoom.apps.chatops.handlers.handler_registry import (
    HandlerRegistry, HandlerCategory, chatops_handler
)

# Multi-agent integration (Phase 7)
try:
    from hololoom.apps.chatops.handlers.multi_agent_semantic import (
        notify_semantic_update,
        get_semantic_registry
    )
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False
    notify_semantic_update = None
    get_semantic_registry = None

logger = logging.getLogger(__name__)


def _format_error_with_recovery(error_type: str, details: str, recovery_steps: List[str]) -> str:
    """Format error message with actionable recovery steps."""
    steps_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(recovery_steps))
    return f"**{error_type}**\n\n{details}\n\n**Recovery Steps:**\n{steps_text}"


@dataclass
class ConversationSemanticContext:
    """Tracks semantic state across conversation turns."""
    room_id: str
    states: List[Any] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    topic_shifts: List[Dict[str, Any]] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)


class SemanticMatrixHandlers:
    """
    Matrix bot handlers for Phase 8 Semantic State integration.

    Provides visibility into:
    - 244D semantic space analysis
    - 8D policy feature extraction
    - Topic shift detection
    - Semantic-aware tool selection
    - Thompson Sampling with semantic priors
    """

    def __init__(self, bot, embedder=None):
        """
        Initialize semantic handlers.

        Args:
            bot: MatrixBot instance
            embedder: Optional embedding function (query -> 244D vector)
        """
        self.bot = bot
        self.embedder = embedder

        # Track conversation semantic context per room
        self._room_contexts: Dict[str, ConversationSemanticContext] = {}

        # Initialize semantic components
        if SEMANTIC_AVAILABLE:
            self.tool_selector = SemanticToolSelector()
            self.bandit = SemanticAwareBandit(
                tools=["answer", "search", "notion_write", "calc"],
                semantic_weight=0.3
            )
            logger.info("Semantic handlers initialized (Phase 8)")
        else:
            self.tool_selector = None
            self.bandit = None
            logger.warning("Semantic calculus not available - handlers will show mock data")

    def _get_room_context(self, room_id: str) -> ConversationSemanticContext:
        """Get or create conversation context for a room."""
        if room_id not in self._room_contexts:
            self._room_contexts[room_id] = ConversationSemanticContext(room_id=room_id)
        return self._room_contexts[room_id]

    def _create_mock_embedding(self, text: str, dim: int = 244) -> np.ndarray:
        """Create a deterministic mock embedding from text."""
        # Use text hash for reproducible but varied embeddings
        seed = hash(text) % (2**31)
        np.random.seed(seed)
        embedding = np.random.randn(dim) * 0.5

        # Bias based on keywords in text
        text_lower = text.lower()

        # Detect category hints in text
        category_keywords = {
            "Creative": ["write", "poem", "story", "creative", "imagine", "create"],
            "Cognitive": ["analyze", "explain", "understand", "think", "reason"],
            "Emotional": ["feel", "emotion", "happy", "sad", "love", "hate"],
            "Temporal": ["when", "time", "history", "before", "after", "future"],
            "Narrative": ["tell", "story", "narrative", "plot", "character"],
            "Core": ["what", "define", "is", "are", "basic", "fundamental"],
        }

        detected_cat = "Core"  # Default
        for cat, keywords in category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected_cat = cat
                break

        # Boost detected category
        if detected_cat in SEMANTIC_CATEGORIES:
            indices = SEMANTIC_CATEGORIES[detected_cat]
            for idx in indices:
                if idx < dim:
                    embedding[idx] += 2.0

        return embedding

    @chatops_handler(
        command="semantic",
        description="Analyze query semantic state (Phase 8: 244D -> 8D policy features)",
        usage="!semantic <query> | shift | tools | bandit | help",
        category=HandlerCategory.LEARNING,
        aliases=["sem", "semantics"]
    )
    async def handle_semantic(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Main semantic analysis command.

        Usage:
            !semantic <query>     - Analyze semantic state of query
            !semantic shift       - Show topic shift detection
            !semantic tools       - Show semantic tool suggestions
            !semantic bandit      - Show Thompson Sampling adjustments
            !semantic help        - Show this help
        """
        if not args:
            await self._show_semantic_help(room)
            return

        parts = args.split(maxsplit=1)
        subcommand = parts[0].lower()
        subargs = parts[1] if len(parts) > 1 else ""

        if subcommand == "help":
            await self._show_semantic_help(room)
        elif subcommand == "shift":
            await self._show_topic_shifts(room, event)
        elif subcommand == "tools":
            await self._show_tool_suggestions(room, event, subargs)
        elif subcommand == "bandit":
            await self._show_bandit_analysis(room, event, subargs)
        elif subcommand == "compare":
            await self._compare_queries(room, event, subargs)
        elif subcommand == "replay":
            await self._replay_conversation(room, event, subargs)
        elif subcommand == "suggest":
            await self._suggest_action(room, event)
        else:
            # Treat entire args as a query to analyze
            await self._analyze_query(room, event, args)

    async def _show_semantic_help(self, room: MatrixRoom):
        """Show semantic command help."""
        help_text = """
**🧠 Semantic Analysis Commands (Phase 8)**

`!semantic <query>` - Analyze semantic state
  → Shows 244D → 8D projection, category scores, momentum

`!semantic shift` - Topic shift detection
  → Shows conversation topic shifts in this room

`!semantic tools [query]` - Tool suggestions
  → Shows which tools are suggested based on semantic context

`!semantic bandit [query]` - Thompson Sampling
  → Shows how semantic state adjusts bandit priors

`!semantic compare <q1> | <q2>` - Compare queries
  → Compares two queries semantically with cosine similarity

`!semantic replay [limit]` - Conversation replay
  → Shows semantic trajectory of last N queries (default 10)

`!semantic suggest` - Action suggestions
  → Suggests next action based on semantic context

**Architecture:**
```
Query → 244D embedding → SemanticState → 8D features → NeuralCore
                              ↓
                      SemanticAwareBandit
                      (adjusts priors by category)
```

**16 Semantic Categories:**
Core, Narrative, Emotional, Relational, Archetypal, Philosophical,
Transformation, Ethical, Creative, Cognitive, Temporal, Spatial,
Character, Plot, Theme, Style
"""
        await self.bot.send_message(room.room_id, help_text, markdown=True)

    async def _analyze_query(self, room: MatrixRoom, event: RoomMessageText, query: str):
        """Analyze semantic state of a query."""
        if not SEMANTIC_AVAILABLE:
            await self.bot.send_message(
                room.room_id,
                "⚠️ Semantic calculus not available. Install dependencies.",
                markdown=True
            )
            return

        try:
            # Create embedding (use real embedder if available, else mock)
            if self.embedder:
                embedding = self.embedder(query)
            else:
                embedding = self._create_mock_embedding(query)

            # Get room context for previous state
            context = self._get_room_context(room.room_id)
            prev_state = context.states[-1] if context.states else None

            # Create semantic state
            state = SemanticState.from_embedding(embedding, prev_state)

            # Store in context
            context.states.append(state)
            context.queries.append(query)
            context.last_update = datetime.now()

            # Notify multi-agent registry (Phase 7)
            if MULTI_AGENT_AVAILABLE and notify_semantic_update:
                try:
                    await notify_semantic_update(
                        room_id=room.room_id,
                        semantic_state=state,
                        query=query,
                        topic=state.dominant_category
                    )
                except Exception as e:
                    logger.debug(f"Multi-agent notification skipped: {e}")

            # Check for topic shift
            if state.topic_shift_detected:
                shift_type = "MAJOR" if state.shift_magnitude > SIGNIFICANT_SHIFT_THRESHOLD else "minor"
                context.topic_shifts.append({
                    "query": query,
                    "magnitude": state.shift_magnitude,
                    "type": shift_type,
                    "from_category": prev_state.dominant_category if prev_state else None,
                    "to_category": state.dominant_category,
                    "timestamp": datetime.now().isoformat()
                })

            # Get 8D feature vector
            features = state.to_feature_vector()

            # Get top categories
            sorted_cats = sorted(state.category_scores.items(), key=lambda x: x[1], reverse=True)[:5]

            # Format response
            shift_indicator = ""
            if state.topic_shift_detected:
                shift_indicator = f"\n⚡ **Topic Shift Detected!** (magnitude: {state.shift_magnitude:.3f})"

            response = f"""
**🧠 Semantic Analysis**

**Query:** `{query[:100]}`

**Semantic State:**
• Dominant Category: **{state.dominant_category}**
• Momentum: {state.momentum:.3f}
• Complexity: {state.complexity:.3f}
• Shift Magnitude: {state.shift_magnitude:.3f}
{shift_indicator}

**8D Policy Features:**
```
[momentum, complexity, top5_dims[0-4], shift_mag]
[{', '.join(f'{x:.2f}' for x in features)}]
```

**Top 5 Categories:**
"""
            for cat, score in sorted_cats:
                bar = "█" * int(score * 20)
                response += f"• {cat}: {bar} {score:.2f}\n"

            response += f"\n*Conversation turn: {len(context.queries)} | Shifts detected: {len(context.topic_shifts)}*"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Semantic analysis error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ Error: {str(e)}")

    async def _show_topic_shifts(self, room: MatrixRoom, event: RoomMessageText):
        """Show topic shift history for the room."""
        context = self._get_room_context(room.room_id)

        if not context.topic_shifts:
            await self.bot.send_message(
                room.room_id,
                f"""
**📊 Topic Shift Detection**

No topic shifts detected in this room yet.

**Thresholds:**
• Minor shift: velocity > {TOPIC_SHIFT_THRESHOLD}
• Major shift: velocity > {SIGNIFICANT_SHIFT_THRESHOLD}

Use `!semantic <query>` to analyze queries and detect shifts.
""",
                markdown=True
            )
            return

        response = f"""
**📊 Topic Shifts in This Room**

**Thresholds:** minor > {TOPIC_SHIFT_THRESHOLD}, major > {SIGNIFICANT_SHIFT_THRESHOLD}

**Detected Shifts:**
"""

        for i, shift in enumerate(context.topic_shifts[-10:], 1):  # Last 10
            icon = "⚡" if shift["type"] == "MAJOR" else "→"
            from_cat = shift.get("from_category", "?")
            to_cat = shift.get("to_category", "?")
            response += f"{i}. {icon} `{from_cat}` → `{to_cat}` (mag: {shift['magnitude']:.2f})\n"
            response += f"   Query: `{shift['query'][:50]}...`\n"

        response += f"\n**Total:** {len(context.topic_shifts)} shifts in {len(context.queries)} queries"

        await self.bot.send_message(room.room_id, response, markdown=True)

    async def _show_tool_suggestions(self, room: MatrixRoom, event: RoomMessageText, query: str):
        """Show semantic-aware tool suggestions."""
        if not SEMANTIC_AVAILABLE or not self.tool_selector:
            await self.bot.send_message(
                room.room_id,
                "⚠️ SemanticToolSelector not available.",
                markdown=True
            )
            return

        try:
            # Get or create semantic state
            if query:
                embedding = self._create_mock_embedding(query)
                state = SemanticState.from_embedding(embedding)
            else:
                # Use last state from context
                context = self._get_room_context(room.room_id)
                if not context.states:
                    await self.bot.send_message(
                        room.room_id,
                        "No semantic state available. Use `!semantic tools <query>` with a query.",
                        markdown=True
                    )
                    return
                state = context.states[-1]
                query = context.queries[-1] if context.queries else "(last query)"

            # Get suggestions
            suggested_tool = self.tool_selector.suggest_tool(state)
            should_branch = self.tool_selector.should_branch_thread(state)

            response = f"""
**🔧 Semantic Tool Selection**

**Query:** `{query[:80]}`

**Semantic Context:**
• Category: **{state.dominant_category}**
• Momentum: {state.momentum:.3f}
• Complexity: {state.complexity:.3f}
• Shift: {state.shift_magnitude:.3f}

**Recommendations:**
• **Suggested Tool:** `{suggested_tool}`
• **Branch Thread?** {'✅ Yes (topic shifted)' if should_branch else '❌ No (continue thread)'}

**Logic:**
"""

            if state.topic_shift_detected:
                response += "→ Topic shift detected → suggest branching\n"
            if state.momentum < 0.25:
                response += "→ Low momentum → may need clarification\n"
            if state.complexity > 0.8:
                response += "→ High complexity → may need decomposition\n"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Tool suggestion error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ Error: {str(e)}")

    async def _show_bandit_analysis(self, room: MatrixRoom, event: RoomMessageText, query: str):
        """Show Thompson Sampling with semantic adjustments."""
        if not SEMANTIC_AVAILABLE or not self.bandit:
            await self.bot.send_message(
                room.room_id,
                "⚠️ SemanticAwareBandit not available.",
                markdown=True
            )
            return

        try:
            # Get or create semantic state
            if query:
                embedding = self._create_mock_embedding(query)
                state = SemanticState.from_embedding(embedding)
            else:
                context = self._get_room_context(room.room_id)
                if not context.states:
                    await self.bot.send_message(
                        room.room_id,
                        "No semantic state available. Use `!semantic bandit <query>`.",
                        markdown=True
                    )
                    return
                state = context.states[-1]
                query = context.queries[-1] if context.queries else "(last query)"

            # Get adjusted priors
            adjusted = self.bandit.adjust_priors(state)
            expected = self.bandit.get_expected_rewards(state)

            # Sample a tool (for demo)
            sampled_tool = self.bandit.sample(semantic_state=state)

            # Get category affinities
            affinities = SemanticAwareBandit.CATEGORY_TOOL_AFFINITIES.get(
                state.dominant_category, {}
            )

            response = f"""
**🎰 SemanticAwareBandit Analysis**

**Query:** `{query[:80]}`
**Dominant Category:** `{state.dominant_category}`
**Semantic Weight:** {self.bandit.semantic_weight}

**Category → Tool Affinities:**
"""
            for tool, boost in sorted(affinities.items(), key=lambda x: x[1], reverse=True):
                response += f"• {tool}: {boost:.1f}x\n"

            response += f"\n**Adjusted Priors (α, β):**\n"
            for tool, (alpha, beta) in adjusted.items():
                exp = expected[tool]
                bar = "█" * int(exp * 20)
                response += f"• `{tool}`: α={alpha:.2f}, β={beta:.2f} → E[X]={exp:.3f} {bar}\n"

            response += f"\n**Sampled Tool:** `{sampled_tool}` ⭐"

            # Show bandit stats
            stats = self.bandit.get_stats()
            total_updates = sum(stats['alphas'][t] + stats['betas'][t] - 2.0 for t in self.bandit.tools)
            response += f"\n\n*Bandit has learned from ~{int(total_updates)} observations*"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Bandit analysis error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ Error: {str(e)}")

    async def _compare_queries(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """Compare two queries semantically using | separator."""
        if not SEMANTIC_AVAILABLE:
            error_msg = _format_error_with_recovery(
                "⚠️ Semantic Unavailable",
                "Semantic calculus components not loaded.",
                ["Ensure HoloLoom.semantic_calculus is installed", "Check import errors in logs"]
            )
            await self.bot.send_message(room.room_id, error_msg, markdown=True)
            return

        if "|" not in args:
            await self.bot.send_message(
                room.room_id,
                "**Usage:** `!semantic compare <query1> | <query2>`\n\n"
                "**Example:** `!semantic compare What is ML? | Explain machine learning`",
                markdown=True
            )
            return

        try:
            parts = args.split("|", 1)
            query1 = parts[0].strip()
            query2 = parts[1].strip()

            if not query1 or not query2:
                await self.bot.send_message(
                    room.room_id, "Both queries must be non-empty.", markdown=True
                )
                return

            # Create embeddings for both
            emb1 = self._create_mock_embedding(query1)
            emb2 = self._create_mock_embedding(query2)

            # Calculate cosine similarity
            cos_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

            # Create semantic states
            state1 = SemanticState.from_embedding(emb1)
            state2 = SemanticState.from_embedding(emb2)

            # Get top 3 categories for each
            cats1 = sorted(state1.category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            cats2 = sorted(state2.category_scores.items(), key=lambda x: x[1], reverse=True)[:3]

            response = f"""
**🔬 Semantic Comparison**

**Query 1:** `{query1[:60]}{'...' if len(query1) > 60 else ''}`
**Query 2:** `{query2[:60]}{'...' if len(query2) > 60 else ''}`

**Cosine Similarity:** {cos_sim:.3f} {'🟢 Similar' if cos_sim > 0.7 else '🟡 Moderate' if cos_sim > 0.4 else '🔴 Different'}

| Metric | Query 1 | Query 2 |
|--------|---------|---------|
| Dominant | {state1.dominant_category} | {state2.dominant_category} |
| Momentum | {state1.momentum:.3f} | {state2.momentum:.3f} |
| Complexity | {state1.complexity:.3f} | {state2.complexity:.3f} |

**Top Categories - Query 1:**
"""
            for cat, score in cats1:
                bar = "█" * int(score * 15)
                response += f"• {cat}: {bar} {score:.2f}\n"

            response += "\n**Top Categories - Query 2:**\n"
            for cat, score in cats2:
                bar = "█" * int(score * 15)
                response += f"• {cat}: {bar} {score:.2f}\n"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Compare error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ Error: {str(e)}")

    async def _replay_conversation(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """Replay conversation semantic analysis trajectory."""
        context = self._get_room_context(room.room_id)

        if not context.queries:
            await self.bot.send_message(
                room.room_id,
                "**📜 Semantic Replay**\n\n"
                "No conversation history in this room yet.\n\n"
                "Use `!semantic <query>` to analyze queries and build history.",
                markdown=True
            )
            return

        # Parse limit
        limit = 10
        if args:
            try:
                limit = min(20, max(1, int(args)))
            except ValueError:
                pass

        # Get last N entries
        queries = context.queries[-limit:]
        states = context.states[-limit:]
        shifts = context.topic_shifts

        response = f"""
**📜 Semantic Replay** (last {len(queries)} queries)

"""
        prev_cat = None
        for i, (query, state) in enumerate(zip(queries, states), 1):
            # Check if there was a shift to this state
            shift_indicator = ""
            if prev_cat and prev_cat != state.dominant_category:
                # Find if this was a recorded shift
                shift_indicator = " ⚡"

            truncated = query[:40] + ('...' if len(query) > 40 else '')
            response += f"**{i}.** `{truncated}`\n"
            response += f"   Category: **{state.dominant_category}**{shift_indicator} | "
            response += f"Momentum: {state.momentum:.2f} | Complexity: {state.complexity:.2f}\n\n"

            prev_cat = state.dominant_category

        # Summary
        total_shifts = len(shifts)
        avg_momentum = np.mean([s.momentum for s in states]) if states else 0
        coherence = "High" if total_shifts < len(queries) * 0.3 else "Medium" if total_shifts < len(queries) * 0.6 else "Low"

        response += f"""
---
**Summary:**
• Total queries: {len(context.queries)}
• Topic shifts: {total_shifts}
• Avg momentum: {avg_momentum:.2f}
• Conversation coherence: **{coherence}**
"""
        await self.bot.send_message(room.room_id, response, markdown=True)

    async def _suggest_action(self, room: MatrixRoom, event: RoomMessageText):
        """Suggest next action based on semantic context."""
        context = self._get_room_context(room.room_id)

        if not context.states:
            await self.bot.send_message(
                room.room_id,
                "**💡 Suggestions**\n\n"
                "No semantic context available yet.\n\n"
                "Use `!semantic <query>` first to build context.",
                markdown=True
            )
            return

        try:
            state = context.states[-1]
            query = context.queries[-1] if context.queries else "(unknown)"
            suggestions = []

            # Category-specific suggestions
            cat_suggestions = {
                "Creative": "Try a creative prompt or explore divergent ideas",
                "Cognitive": "Deep-dive with `!semantic tools` to find analysis tools",
                "Emotional": "Consider emotional context and user sentiment",
                "Temporal": "Add time-specific context or historical data",
                "Narrative": "Build a story arc or continue the narrative thread",
                "Core": "Clarify fundamentals or provide definitions",
            }
            if state.dominant_category in cat_suggestions:
                suggestions.append(f"📂 {cat_suggestions[state.dominant_category]}")

            # Momentum-based suggestions
            if state.momentum < 0.25:
                suggestions.append("🔄 Low momentum: Consider clarifying intent or reframing query")
            elif state.momentum > 0.75:
                suggestions.append("🎯 High momentum: Stay on this focused track")

            # Complexity-based suggestions
            if state.complexity > 0.8:
                suggestions.append("🧩 High complexity: Break into sub-questions")
            elif state.complexity < 0.2:
                suggestions.append("➕ Low complexity: May need more depth or specificity")

            # Shift-based suggestions
            if len(context.topic_shifts) >= 3 and len(context.queries) <= 5:
                suggestions.append("⚡ Frequent shifts: Consider focusing on one topic")

            # Tool selector suggestion
            if self.tool_selector:
                suggested_tool = self.tool_selector.suggest_tool(state)
                suggestions.append(f"🔧 Suggested tool: `{suggested_tool}`")

            # Bandit suggestion
            if self.bandit:
                expected = self.bandit.get_expected_rewards(state)
                best_tool = max(expected.items(), key=lambda x: x[1])
                suggestions.append(f"🎰 Bandit recommends: `{best_tool[0]}` (E[R]={best_tool[1]:.2f})")

            response = f"""
**💡 Suggestions for Next Action**

**Current Context:**
• Last query: `{query[:50]}{'...' if len(query) > 50 else ''}`
• Category: **{state.dominant_category}**
• Momentum: {state.momentum:.2f} | Complexity: {state.complexity:.2f}
• Conversation turn: {len(context.queries)}

**Recommendations:**
"""
            for i, suggestion in enumerate(suggestions, 1):
                response += f"{i}. {suggestion}\n"

            if not suggestions:
                response += "No specific suggestions at this time. Continue exploring!\n"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Suggest error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ Error: {str(e)}")

    def register_all(self):
        """Register all semantic handlers with the bot."""
        registry = HandlerRegistry()
        count = registry.register_instance(self)

        # Register with bot's handler dict for backward compatibility
        for command in registry.get_all_commands():
            spec = registry.get_handler(command)
            if spec:
                self.bot.register_handler(command, spec.handler)
                for alias in spec.aliases:
                    self.bot.register_handler(alias, spec.handler)

        logger.info(f"Registered {count} semantic handlers")
        return count


# ============================================================================
# Global Handler Instance
# ============================================================================

_semantic_handlers: Optional[SemanticMatrixHandlers] = None


def get_semantic_handlers() -> Optional[SemanticMatrixHandlers]:
    """Get the global semantic handlers instance."""
    return _semantic_handlers


def set_semantic_handlers(handlers: Optional[SemanticMatrixHandlers]) -> None:
    """Set the global semantic handlers instance."""
    global _semantic_handlers
    _semantic_handlers = handlers


# ============================================================================
# Integration Helper
# ============================================================================

def create_semantic_handlers(bot, embedder=None) -> SemanticMatrixHandlers:
    """
    Factory function to create and register semantic handlers.

    Args:
        bot: MatrixBot instance (can be None for testing)
        embedder: Optional embedding function

    Returns:
        SemanticMatrixHandlers instance (registered if bot provided)
    """
    handlers = SemanticMatrixHandlers(bot, embedder)
    if bot is not None:
        handlers.register_all()
    return handlers


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
Semantic State Matrix Bot Handlers (Phase 8)
=============================================

Commands available:
- !semantic <query>     - Analyze semantic state (244D -> 8D)
- !semantic shift       - Show topic shift detection
- !semantic tools [q]   - Show semantic tool suggestions
- !semantic bandit [q]  - Show Thompson Sampling adjustments
- !semantic help        - Show command help

Integration:
    from hololoom.apps.chatops.handlers.semantic_handlers import create_semantic_handlers

    bot = MatrixBot(config)
    semantic = create_semantic_handlers(bot, embedder=my_embedder)
    await bot.start()

Features:
- 244D semantic space → 8D policy features
- Topic shift detection across conversation
- Semantic-aware tool suggestions
- Thompson Sampling with category affinities
- Per-room conversation context tracking
""")
