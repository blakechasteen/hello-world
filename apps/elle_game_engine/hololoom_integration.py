"""
HoloLoom Integration for Elle Game Engine
==========================================

Replaces basic in-memory session storage with HoloLoom's enterprise-grade
knowledge graph and semantic memory systems.

Features:
- Knowledge Graph (Yarn Graph) for NPC relationships and world state
- Matryoshka embeddings for semantic conversation retrieval
- Bi-temporal edge tracking for conversation history evolution
- Persistent storage across sessions

Architecture:
- HoloLoomSessionStore: Main session storage using KG
- Semantic conversation retrieval using multi-scale embeddings
- Automatic entity extraction and relationship tracking
- Temporal threads for conversation history

Integration: 2025-11-16
"""

import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np

# HoloLoom imports
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

# Elle imports
from .session import (
    GameSession,
    ConversationExchange,
    NPCRelationship,
    SessionStore,
)


# ============================================================================
# HoloLoom-Backed Session Store
# ============================================================================

class HoloLoomSessionStore:
    """
    Knowledge Graph-backed session storage for Elle Game Engine.

    Stores:
    - Sessions as graph nodes
    - Conversations as temporal edges with entity relationships
    - NPC relationships as typed edges (LIKES, DISLIKES, TRUSTS)
    - World flags as graph properties
    - Semantic embeddings for conversation retrieval

    Advantages over InMemorySessionStore:
    - Persistent across restarts
    - Semantic search over conversation history
    - Entity relationship tracking (NPCs, items, locations)
    - Temporal queries ("What did player discuss with NPC X last week?")
    - Multi-hop reasoning (relationship graphs)
    """

    def __init__(
        self,
        kg_path: Optional[str] = None,
        enable_embeddings: bool = True,
        embedding_scales: List[int] = None,
    ):
        """
        Initialize HoloLoom session store.

        Args:
            kg_path: Path to save/load knowledge graph (default: sessions_kg.jsonl)
            enable_embeddings: Enable semantic embeddings for conversations
            embedding_scales: Matryoshka scales (default: [96, 192, 384])
        """
        self.kg_path = kg_path or "sessions_kg.jsonl"
        self.kg = KG()

        # Load existing graph if available
        kg_file = Path(self.kg_path)
        if kg_file.exists():
            try:
                self.kg = KG.load(self.kg_path)
            except Exception as e:
                print(f"Warning: Failed to load KG from {self.kg_path}: {e}")

        # Initialize embeddings for semantic conversation search
        self.enable_embeddings = enable_embeddings
        if enable_embeddings:
            scales = embedding_scales or [96, 192, 384]
            self.embeddings = MatryoshkaEmbeddings(sizes=scales)
            # Cache for conversation embeddings
            self._conversation_embeddings: Dict[str, Dict[int, np.ndarray]] = {}
        else:
            self.embeddings = None
            self._conversation_embeddings = {}

    # ========================================================================
    # SessionStore Protocol Implementation
    # ========================================================================

    def create_session(
        self,
        player_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> GameSession:
        """
        Create a new game session in the knowledge graph.

        Creates:
        - Session node with metadata
        - Player node (if player_id provided)
        - BELONGS_TO edge linking session to player
        """
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now().timestamp()

        # Create session object
        session = GameSession(
            session_id=session_id,
            player_id=player_id,
            created_at=now,
            last_accessed=now,
        )

        # Add session node to KG
        self.kg.G.add_node(
            f"session:{session_id}",
            node_type="session",
            session_id=session_id,
            player_id=player_id,
            created_at=now,
            last_accessed=now,
            max_history_size=session.max_history_size,
        )

        # Link to player if provided
        if player_id:
            # Ensure player node exists
            player_node = f"player:{player_id}"
            if player_node not in self.kg.G:
                self.kg.G.add_node(player_node, node_type="player", player_id=player_id)

            # Create BELONGS_TO edge
            edge = KGEdge(
                src=f"session:{session_id}",
                dst=player_node,
                type="BELONGS_TO",
                weight=1.0,
                metadata={"created_at": now},
            )
            self.kg.add_edge(edge)

        # Connect to time thread
        self.kg.connect_entity_to_time(
            entity=f"session:{session_id}",
            timestamp=datetime.now(),
            edge_type="CREATED_AT",
        )

        # Save to disk
        self.save()

        return session

    def get_session(self, session_id: str) -> Optional[GameSession]:
        """
        Retrieve session from knowledge graph.

        Reconstructs GameSession from:
        - Session node attributes
        - Conversation edges (EXCHANGE type)
        - NPC relationship edges (LIKES, TRUSTS, etc.)
        - World flag properties
        """
        session_node = f"session:{session_id}"

        if session_node not in self.kg.G:
            return None

        # Get session node data
        node_data = self.kg.G.nodes[session_node]

        # Reconstruct session object
        session = GameSession(
            session_id=session_id,
            player_id=node_data.get("player_id"),
            created_at=node_data.get("created_at", datetime.now().timestamp()),
            last_accessed=node_data.get("last_accessed", datetime.now().timestamp()),
            max_history_size=node_data.get("max_history_size", 10),
        )

        # Reconstruct conversation history from graph edges
        session.conversation_history = self._get_conversation_history(session_id)

        # Reconstruct NPC relationships
        session.npc_relationships = self._get_npc_relationships(session_id)

        # Reconstruct world flags
        session.world_flags = self._get_world_flags(session_id)

        return session

    def update_session(self, session: GameSession) -> None:
        """
        Update session in knowledge graph.

        Updates:
        - Session node metadata
        - Conversation edges (new exchanges)
        - NPC relationship edges (reputation changes)
        - World flags
        """
        session_node = f"session:{session.session_id}"

        if session_node not in self.kg.G:
            # Session doesn't exist - create it
            self.create_session(
                player_id=session.player_id, session_id=session.session_id
            )
            return

        # Update session metadata
        self.kg.G.nodes[session_node]["last_accessed"] = session.last_accessed
        self.kg.G.nodes[session_node]["max_history_size"] = session.max_history_size

        # Update conversation history
        self._update_conversation_history(session)

        # Update NPC relationships
        self._update_npc_relationships(session)

        # Update world flags
        self._update_world_flags(session)

        # Save to disk
        self.save()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete session from knowledge graph.

        Removes:
        - Session node
        - All conversation edges
        - All NPC relationship edges
        - World flag nodes
        """
        session_node = f"session:{session_id}"

        if session_node not in self.kg.G:
            return False

        # Remove session node (NetworkX automatically removes connected edges)
        self.kg.G.remove_node(session_node)

        # Save to disk
        self.save()

        return True

    def save(self) -> None:
        """Save knowledge graph to disk."""
        self.kg.save(self.kg_path)

    # ========================================================================
    # Private Helpers - Conversation History
    # ========================================================================

    def _get_conversation_history(self, session_id: str) -> List[ConversationExchange]:
        """
        Reconstruct conversation history from graph edges.

        Conversations are stored as temporal edges:
        - session -> EXCHANGE -> conversation_node
        - conversation_node contains: player_query, elle_response, timestamp, npc_id
        """
        session_node = f"session:{session_id}"
        exchanges = []

        # Find all EXCHANGE edges from session
        for _, dst, data in self.kg.G.out_edges(session_node, data=True):
            if data.get("type") == "EXCHANGE":
                # Get conversation node
                conv_data = self.kg.G.nodes.get(dst, {})

                if conv_data.get("node_type") == "conversation":
                    exchange = ConversationExchange(
                        player_query=conv_data.get("player_query", ""),
                        elle_response=conv_data.get("elle_response", ""),
                        timestamp=conv_data.get("timestamp", datetime.now().timestamp()),
                        npc_id=conv_data.get("npc_id"),
                    )
                    exchanges.append(exchange)

        # Sort by timestamp (oldest first)
        exchanges.sort(key=lambda e: e.timestamp)

        return exchanges

    def _update_conversation_history(self, session: GameSession) -> None:
        """
        Update conversation history in graph.

        Strategy:
        - Get existing conversation count
        - Add new exchanges as conversation nodes + EXCHANGE edges
        - Link conversations to NPCs if npc_id present
        - Generate embeddings for semantic search
        """
        session_node = f"session:{session.session_id}"
        existing_count = len(self._get_conversation_history(session.session_id))

        # Add new exchanges (those beyond existing count)
        new_exchanges = session.conversation_history[existing_count:]

        for i, exchange in enumerate(new_exchanges):
            conv_id = f"conv:{session.session_id}:{existing_count + i}"

            # Add conversation node
            self.kg.G.add_node(
                conv_id,
                node_type="conversation",
                player_query=exchange.player_query,
                elle_response=exchange.elle_response,
                timestamp=exchange.timestamp,
                npc_id=exchange.npc_id,
            )

            # Add EXCHANGE edge from session
            edge = KGEdge(
                src=session_node,
                dst=conv_id,
                type="EXCHANGE",
                weight=1.0,
                metadata={"timestamp": exchange.timestamp},
            )
            self.kg.add_edge(edge)

            # Link to NPC if present
            if exchange.npc_id:
                npc_node = f"npc:{exchange.npc_id}"

                # Ensure NPC node exists
                if npc_node not in self.kg.G:
                    self.kg.G.add_node(npc_node, node_type="npc", npc_id=exchange.npc_id)

                # Add TALKED_TO edge
                edge = KGEdge(
                    src=conv_id,
                    dst=npc_node,
                    type="TALKED_TO",
                    weight=1.0,
                    metadata={"timestamp": exchange.timestamp},
                )
                self.kg.add_edge(edge)

            # Connect to time thread
            self.kg.connect_entity_to_time(
                entity=conv_id,
                timestamp=datetime.fromtimestamp(exchange.timestamp),
                edge_type="OCCURRED_AT",
            )

            # Generate embeddings for semantic search
            if self.enable_embeddings:
                self._generate_conversation_embedding(conv_id, exchange)

    def _generate_conversation_embedding(
        self, conv_id: str, exchange: ConversationExchange
    ) -> None:
        """
        Generate multi-scale embeddings for conversation.

        Concatenates player query + Elle response for semantic search.
        Stores embeddings at all Matryoshka scales.
        """
        if not self.enable_embeddings or not self.embeddings:
            return

        # Ensure model is loaded
        self.embeddings._ensure_model_loaded()

        # Combine query + response for embedding
        text = f"Player: {exchange.player_query}\nElle: {exchange.elle_response}"

        try:
            # Generate base embedding
            import numpy as np

            base_emb = self.embeddings.encode([text])[0]  # Get first (and only) embedding

            # Store embeddings at all scales
            self._conversation_embeddings[conv_id] = {}
            for scale in self.embeddings.sizes:
                # Project to scale
                scale_emb = base_emb[:scale] if scale <= len(base_emb) else base_emb
                self._conversation_embeddings[conv_id][scale] = scale_emb

        except Exception as e:
            print(f"Warning: Failed to generate embedding for {conv_id}: {e}")

    # ========================================================================
    # Private Helpers - NPC Relationships
    # ========================================================================

    def _get_npc_relationships(self, session_id: str) -> Dict[str, NPCRelationship]:
        """
        Reconstruct NPC relationships from graph edges.

        NPC relationships stored as:
        - session -> HAS_RELATIONSHIP -> npc_relationship_node
        - npc_relationship_node contains: npc_id, reputation, interactions, etc.
        """
        session_node = f"session:{session_id}"
        relationships = {}

        # Find all HAS_RELATIONSHIP edges
        for _, dst, data in self.kg.G.out_edges(session_node, data=True):
            if data.get("type") == "HAS_RELATIONSHIP":
                rel_data = self.kg.G.nodes.get(dst, {})

                if rel_data.get("node_type") == "npc_relationship":
                    npc_id = rel_data.get("npc_id")
                    if npc_id:
                        relationship = NPCRelationship(
                            npc_id=npc_id,
                            reputation=rel_data.get("reputation", 0),
                            interactions=rel_data.get("interactions", 0),
                            last_mood=rel_data.get("last_mood"),
                            custom_flags=rel_data.get("custom_flags", {}),
                        )
                        relationships[npc_id] = relationship

        return relationships

    def _update_npc_relationships(self, session: GameSession) -> None:
        """
        Update NPC relationships in graph.

        Strategy:
        - For each NPC relationship in session
        - Create/update npc_relationship node
        - Add HAS_RELATIONSHIP edge from session
        - Add reputation-based edges (LIKES, DISLIKES, TRUSTS)
        """
        session_node = f"session:{session.session_id}"

        for npc_id, relationship in session.npc_relationships.items():
            rel_node = f"npc_rel:{session.session_id}:{npc_id}"

            # Add/update relationship node
            if rel_node in self.kg.G:
                # Update existing
                self.kg.G.nodes[rel_node]["reputation"] = relationship.reputation
                self.kg.G.nodes[rel_node]["interactions"] = relationship.interactions
                self.kg.G.nodes[rel_node]["last_mood"] = relationship.last_mood
                self.kg.G.nodes[rel_node]["custom_flags"] = relationship.custom_flags
            else:
                # Create new
                self.kg.G.add_node(
                    rel_node,
                    node_type="npc_relationship",
                    npc_id=npc_id,
                    reputation=relationship.reputation,
                    interactions=relationship.interactions,
                    last_mood=relationship.last_mood,
                    custom_flags=relationship.custom_flags,
                )

                # Add HAS_RELATIONSHIP edge
                edge = KGEdge(
                    src=session_node,
                    dst=rel_node,
                    type="HAS_RELATIONSHIP",
                    weight=1.0,
                )
                self.kg.add_edge(edge)

            # Add reputation-based semantic edges
            npc_node = f"npc:{npc_id}"
            if npc_node not in self.kg.G:
                self.kg.G.add_node(npc_node, node_type="npc", npc_id=npc_id)

            # LIKES edge (reputation > 20)
            if relationship.reputation > 20:
                edge = KGEdge(
                    src=session_node,
                    dst=npc_node,
                    type="LIKES",
                    weight=min(relationship.reputation / 100.0, 1.0),
                    metadata={"reputation": relationship.reputation},
                )
                self.kg.add_edge(edge)

            # DISLIKES edge (reputation < -20)
            elif relationship.reputation < -20:
                edge = KGEdge(
                    src=session_node,
                    dst=npc_node,
                    type="DISLIKES",
                    weight=min(abs(relationship.reputation) / 100.0, 1.0),
                    metadata={"reputation": relationship.reputation},
                )
                self.kg.add_edge(edge)

            # TRUSTS edge (reputation > 50)
            if relationship.reputation > 50:
                edge = KGEdge(
                    src=session_node,
                    dst=npc_node,
                    type="TRUSTS",
                    weight=min(relationship.reputation / 100.0, 1.0),
                    metadata={"reputation": relationship.reputation},
                )
                self.kg.add_edge(edge)

    # ========================================================================
    # Private Helpers - World Flags
    # ========================================================================

    def _get_world_flags(self, session_id: str) -> Dict[str, bool]:
        """
        Reconstruct world flags from graph edges.

        World flags stored as:
        - session -> HAS_FLAG -> flag_node
        - flag_node contains: flag_name, value (bool)
        """
        session_node = f"session:{session_id}"
        flags = {}

        # Find all HAS_FLAG edges
        for _, dst, data in self.kg.G.out_edges(session_node, data=True):
            if data.get("type") == "HAS_FLAG":
                flag_data = self.kg.G.nodes.get(dst, {})

                if flag_data.get("node_type") == "world_flag":
                    flag_name = flag_data.get("flag_name")
                    flag_value = flag_data.get("value", False)
                    if flag_name:
                        flags[flag_name] = flag_value

        return flags

    def _update_world_flags(self, session: GameSession) -> None:
        """
        Update world flags in graph.

        Strategy:
        - For each flag in session.world_flags
        - Create/update world_flag node
        - Add HAS_FLAG edge from session
        """
        session_node = f"session:{session.session_id}"

        for flag_name, flag_value in session.world_flags.items():
            flag_node = f"flag:{session.session_id}:{flag_name}"

            # Add/update flag node
            if flag_node in self.kg.G:
                # Update existing
                self.kg.G.nodes[flag_node]["value"] = flag_value
            else:
                # Create new
                self.kg.G.add_node(
                    flag_node,
                    node_type="world_flag",
                    flag_name=flag_name,
                    value=flag_value,
                )

                # Add HAS_FLAG edge
                edge = KGEdge(
                    src=session_node,
                    dst=flag_node,
                    type="HAS_FLAG",
                    weight=1.0,
                )
                self.kg.add_edge(edge)

    # ========================================================================
    # Semantic Search - Find Similar Conversations
    # ========================================================================

    def search_conversations(
        self,
        query: str,
        session_id: Optional[str] = None,
        npc_id: Optional[str] = None,
        limit: int = 5,
        scale: int = 192,
    ) -> List[tuple[ConversationExchange, float]]:
        """
        Semantic search over conversation history.

        Uses Matryoshka embeddings to find conversations similar to query.

        Args:
            query: Text query (e.g., "conversations about the quest")
            session_id: Optional filter by session
            npc_id: Optional filter by NPC
            limit: Maximum results
            scale: Embedding scale to use (96=fast, 384=quality)

        Returns:
            List of (ConversationExchange, similarity_score) tuples
        """
        if not self.enable_embeddings or not self.embeddings:
            return []

        # Ensure model is loaded
        self.embeddings._ensure_model_loaded()

        import numpy as np

        # Generate query embedding
        query_emb = self.embeddings.encode([query])[0]
        query_emb_scaled = query_emb[:scale] if scale <= len(query_emb) else query_emb

        # Find conversation nodes
        conv_nodes = [
            node
            for node, data in self.kg.G.nodes(data=True)
            if data.get("node_type") == "conversation"
        ]

        # Apply filters
        if session_id:
            session_node = f"session:{session_id}"
            # Filter to conversations in this session
            conv_nodes = [
                node
                for node in conv_nodes
                if node.startswith(f"conv:{session_id}:")
            ]

        if npc_id:
            # Filter to conversations with this NPC
            npc_node = f"npc:{npc_id}"
            conv_nodes = [
                node
                for node in conv_nodes
                if any(
                    dst == npc_node and data.get("type") == "TALKED_TO"
                    for _, dst, data in self.kg.G.out_edges(node, data=True)
                )
            ]

        # Calculate similarities
        results = []
        for conv_id in conv_nodes:
            # Get cached embedding
            if conv_id not in self._conversation_embeddings:
                continue

            conv_emb = self._conversation_embeddings[conv_id].get(scale)
            if conv_emb is None:
                continue

            # Cosine similarity
            similarity = np.dot(query_emb_scaled, conv_emb) / (
                np.linalg.norm(query_emb_scaled) * np.linalg.norm(conv_emb) + 1e-8
            )

            # Reconstruct exchange
            conv_data = self.kg.G.nodes[conv_id]
            exchange = ConversationExchange(
                player_query=conv_data.get("player_query", ""),
                elle_response=conv_data.get("elle_response", ""),
                timestamp=conv_data.get("timestamp", datetime.now().timestamp()),
                npc_id=conv_data.get("npc_id"),
            )

            results.append((exchange, float(similarity)))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return results[:limit]

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about stored sessions."""
        stats = {
            "total_sessions": sum(
                1
                for _, data in self.kg.G.nodes(data=True)
                if data.get("node_type") == "session"
            ),
            "total_conversations": sum(
                1
                for _, data in self.kg.G.nodes(data=True)
                if data.get("node_type") == "conversation"
            ),
            "total_npcs": sum(
                1
                for _, data in self.kg.G.nodes(data=True)
                if data.get("node_type") == "npc"
            ),
            "total_relationships": sum(
                1
                for _, data in self.kg.G.nodes(data=True)
                if data.get("node_type") == "npc_relationship"
            ),
            "kg_nodes": self.kg.G.number_of_nodes(),
            "kg_edges": self.kg.G.number_of_edges(),
        }
        return stats

    def __len__(self) -> int:
        """Number of active sessions."""
        return sum(
            1
            for _, data in self.kg.G.nodes(data=True)
            if data.get("node_type") == "session"
        )
