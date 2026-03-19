"""
HoloLoom Phase 3: Knowledge Sharing
November 2025

Export, share, and distribute knowledge between users and teams.
Supports multiple export formats and sharing workflows.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"           # Full JSON export
    JSON_LD = "json-ld"     # Linked Data format
    MARKDOWN = "markdown"   # Human-readable markdown
    CSV = "csv"             # Tabular data
    RDF = "rdf"             # RDF triples


class ShareScope(Enum):
    """Scope of sharing."""
    USER = "user"           # Single user
    TEAM = "team"           # Team members
    ORGANIZATION = "org"    # Organization-wide
    PUBLIC = "public"       # Public link


@dataclass
class SharedKnowledge:
    """Shared knowledge bundle."""
    share_id: str
    owner_id: str
    title: str
    description: str = ""
    scope: ShareScope = ShareScope.TEAM
    target_ids: set[str] = field(default_factory=set)  # User/team IDs depending on scope
    node_ids: set[str] = field(default_factory=set)    # Knowledge nodes included
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    access_count: int = 0
    access_token: str | None = None  # For public links
    metadata: dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if share has expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "share_id": self.share_id,
            "owner_id": self.owner_id,
            "title": self.title,
            "description": self.description,
            "scope": self.scope.value,
            "target_ids": list(self.target_ids),
            "node_ids": list(self.node_ids),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "access_token": self.access_token,
            "metadata": self.metadata
        }


@dataclass
class ExportResult:
    """Result of knowledge export."""
    format: ExportFormat
    content: str
    node_count: int
    edge_count: int
    size_bytes: int
    checksum: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class KnowledgeSharing:
    """
    Manages knowledge sharing and export.

    Features:
    - Export knowledge in multiple formats
    - Share with users, teams, or publicly
    - Track share access
    - Manage share expiration
    """

    def __init__(self, storage_path: str | None = None):
        """
        Initialize knowledge sharing.

        Args:
            storage_path: Path to persist data
        """
        self.storage_path = storage_path
        self.shares: dict[str, SharedKnowledge] = {}
        self._token_to_share: dict[str, str] = {}  # token -> share_id
        self._counter = 0

        # External dependencies (set by integrator)
        self.get_knowledge_graph = None  # () -> graph
        self.get_node_content = None     # (node_id) -> content
        self.get_node_edges = None       # (node_id) -> List[edges]
        self.check_access = None         # (node_id, user_id) -> bool

        if storage_path:
            self._load()

    def _generate_id(self) -> str:
        """Generate share ID."""
        self._counter += 1
        return f"share_{self._counter:06d}"

    def _generate_token(self) -> str:
        """Generate access token."""
        import secrets
        return secrets.token_urlsafe(32)

    def _compute_checksum(self, content: str) -> str:
        """Compute content checksum."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # === Export ===

    def export_nodes(self, node_ids: list[str], format: ExportFormat = ExportFormat.JSON,
                     user_id: str = None, include_edges: bool = True) -> ExportResult:
        """
        Export knowledge nodes to specified format.

        Args:
            node_ids: Node IDs to export
            format: Export format
            user_id: User requesting export (for access check)
            include_edges: Include relationships
        """
        # Filter by access if user specified
        if user_id and self.check_access:
            node_ids = [nid for nid in node_ids if self.check_access(nid, user_id)]

        # Gather content
        nodes = []
        edges = []

        for nid in node_ids:
            if self.get_node_content:
                content = self.get_node_content(nid)
            else:
                content = {"id": nid}

            nodes.append({"id": nid, **content} if isinstance(content, dict) else {"id": nid, "content": content})

            if include_edges and self.get_node_edges:
                node_edges = self.get_node_edges(nid)
                edges.extend(node_edges)

        # Format output
        if format == ExportFormat.JSON:
            output = self._export_json(nodes, edges)
        elif format == ExportFormat.JSON_LD:
            output = self._export_jsonld(nodes, edges)
        elif format == ExportFormat.MARKDOWN:
            output = self._export_markdown(nodes, edges)
        elif format == ExportFormat.CSV:
            output = self._export_csv(nodes, edges)
        elif format == ExportFormat.RDF:
            output = self._export_rdf(nodes, edges)
        else:
            raise ValueError(f"Unknown format: {format}")

        return ExportResult(
            format=format,
            content=output,
            node_count=len(nodes),
            edge_count=len(edges),
            size_bytes=len(output.encode()),
            checksum=self._compute_checksum(output)
        )

    def _export_json(self, nodes: list[dict], edges: list) -> str:
        """Export as JSON."""
        return json.dumps({
            "nodes": nodes,
            "edges": [{"source": e.source, "target": e.target, "type": e.edge_type}
                     if hasattr(e, 'source') else e
                     for e in edges],
            "exported_at": datetime.now().isoformat()
        }, indent=2)

    def _export_jsonld(self, nodes: list[dict], edges: list) -> str:
        """Export as JSON-LD."""
        graph = []
        for node in nodes:
            ld_node = {
                "@id": f"hololoom:node/{node['id']}",
                "@type": "Knowledge",
                "content": node.get("content", "")
            }
            graph.append(ld_node)

        return json.dumps({
            "@context": {
                "hololoom": "https://hololoom.ai/schema/",
                "Knowledge": "hololoom:Knowledge",
                "relates_to": {"@id": "hololoom:relates_to", "@type": "@id"}
            },
            "@graph": graph
        }, indent=2)

    def _export_markdown(self, nodes: list[dict], edges: list) -> str:
        """Export as Markdown."""
        lines = []
        lines.append("# HoloLoom Knowledge Export")
        lines.append(f"\n*Exported: {datetime.now().isoformat()}*\n")
        lines.append(f"**Nodes**: {len(nodes)} | **Edges**: {len(edges)}\n")
        lines.append("---\n")

        lines.append("## Knowledge Nodes\n")
        for node in nodes:
            lines.append(f"### {node['id']}\n")
            content = node.get("content", "")
            if isinstance(content, dict):
                for k, v in content.items():
                    lines.append(f"- **{k}**: {v}")
            else:
                lines.append(f"{content}\n")
            lines.append("")

        if edges:
            lines.append("## Relationships\n")
            lines.append("| Source | Relation | Target |")
            lines.append("|--------|----------|--------|")
            for e in edges:
                if hasattr(e, 'source'):
                    lines.append(f"| {e.source} | {e.edge_type} | {e.target} |")
                elif isinstance(e, dict):
                    lines.append(f"| {e.get('source', '?')} | {e.get('type', '?')} | {e.get('target', '?')} |")

        return "\n".join(lines)

    def _export_csv(self, nodes: list[dict], edges: list) -> str:
        """Export as CSV."""
        lines = []
        # Nodes
        lines.append("type,id,content")
        for node in nodes:
            content = node.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            content = str(content).replace(",", ";").replace("\n", " ")[:100]
            lines.append(f"node,{node['id']},{content}")

        # Edges
        for e in edges:
            if hasattr(e, 'source'):
                lines.append(f"edge,{e.source}->{e.target},{e.edge_type}")

        return "\n".join(lines)

    def _export_rdf(self, nodes: list[dict], edges: list) -> str:
        """Export as RDF/N-Triples."""
        lines = []
        prefix = "https://hololoom.ai/node/"

        for node in nodes:
            subject = f"<{prefix}{node['id']}>"
            lines.append(f'{subject} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{prefix}Knowledge> .')
            content = str(node.get("content", "")).replace('"', '\\"')[:200]
            lines.append(f'{subject} <{prefix}content> "{content}" .')

        for e in edges:
            if hasattr(e, 'source'):
                s = f"<{prefix}{e.source}>"
                o = f"<{prefix}{e.target}>"
                p = f"<{prefix}{e.edge_type.lower()}>"
                lines.append(f"{s} {p} {o} .")

        return "\n".join(lines)

    # === Sharing ===

    def share(self, node_ids: list[str], owner_id: str, title: str,
              scope: ShareScope = ShareScope.TEAM,
              target_ids: list[str] = None,
              description: str = "",
              expires_at: datetime = None) -> SharedKnowledge:
        """
        Create a knowledge share.

        Args:
            node_ids: Nodes to share
            owner_id: User creating share
            title: Share title
            scope: Sharing scope
            target_ids: Target users/teams (depends on scope)
            description: Optional description
            expires_at: Optional expiration
        """
        share_id = self._generate_id()

        # Generate token for public shares
        access_token = None
        if scope == ShareScope.PUBLIC:
            access_token = self._generate_token()

        share = SharedKnowledge(
            share_id=share_id,
            owner_id=owner_id,
            title=title,
            description=description,
            scope=scope,
            target_ids=set(target_ids or []),
            node_ids=set(node_ids),
            expires_at=expires_at,
            access_token=access_token
        )

        self.shares[share_id] = share
        if access_token:
            self._token_to_share[access_token] = share_id

        self._save()
        return share

    def get_share(self, share_id: str) -> SharedKnowledge | None:
        """Get share by ID."""
        share = self.shares.get(share_id)
        if share and not share.is_expired():
            return share
        return None

    def get_share_by_token(self, token: str) -> SharedKnowledge | None:
        """Get share by public token."""
        share_id = self._token_to_share.get(token)
        if share_id:
            return self.get_share(share_id)
        return None

    def can_access_share(self, share_id: str, user_id: str) -> bool:
        """Check if user can access a share."""
        share = self.shares.get(share_id)
        if not share or share.is_expired():
            return False

        # Owner always has access
        if share.owner_id == user_id:
            return True

        if share.scope == ShareScope.PUBLIC:
            return True

        if share.scope == ShareScope.USER:
            return user_id in share.target_ids

        if share.scope == ShareScope.TEAM:
            # Would need team membership check from external
            return user_id in share.target_ids

        if share.scope == ShareScope.ORGANIZATION:
            return True  # Simplified - would need org check

        return False

    def access_share(self, share_id: str, user_id: str) -> ExportResult | None:
        """Access a share and return its content."""
        share = self.shares.get(share_id)
        if not share or share.is_expired():
            return None

        if not self.can_access_share(share_id, user_id):
            return None

        # Track access
        share.access_count += 1
        self._save()

        # Export content
        return self.export_nodes(
            list(share.node_ids),
            ExportFormat.JSON,
            user_id=share.owner_id  # Use owner for access check
        )

    def list_shares_by_user(self, user_id: str) -> list[SharedKnowledge]:
        """List shares created by user."""
        return [
            share for share in self.shares.values()
            if share.owner_id == user_id and not share.is_expired()
        ]

    def list_shares_for_user(self, user_id: str) -> list[SharedKnowledge]:
        """List shares accessible to user."""
        return [
            share for share in self.shares.values()
            if self.can_access_share(share.share_id, user_id)
        ]

    def delete_share(self, share_id: str, user_id: str) -> bool:
        """Delete a share."""
        share = self.shares.get(share_id)
        if not share or share.owner_id != user_id:
            return False

        if share.access_token:
            del self._token_to_share[share.access_token]
        del self.shares[share_id]
        self._save()
        return True

    # === Import ===

    def import_json(self, json_content: str, user_id: str) -> tuple[int, int]:
        """
        Import knowledge from JSON.

        Returns tuple of (nodes_imported, edges_imported).
        Note: Actual import to graph requires external integration.
        """
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError:
            return (0, 0)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Here we would actually import to the graph
        # For now, just return counts
        return (len(nodes), len(edges))

    # === Persistence ===

    def _save(self):
        """Save to storage."""
        if not self.storage_path:
            return

        data = {
            "shares": {sid: s.to_dict() for sid, s in self.shares.items()},
            "counter": self._counter
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load from storage."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            self._counter = data.get("counter", 0)

            for sid, sdata in data.get("shares", {}).items():
                share = SharedKnowledge(
                    share_id=sdata["share_id"],
                    owner_id=sdata["owner_id"],
                    title=sdata["title"],
                    description=sdata.get("description", ""),
                    scope=ShareScope(sdata["scope"]),
                    target_ids=set(sdata.get("target_ids", [])),
                    node_ids=set(sdata.get("node_ids", [])),
                    created_at=datetime.fromisoformat(sdata["created_at"]) if "created_at" in sdata else datetime.now(),
                    expires_at=datetime.fromisoformat(sdata["expires_at"]) if sdata.get("expires_at") else None,
                    access_count=sdata.get("access_count", 0),
                    access_token=sdata.get("access_token"),
                    metadata=sdata.get("metadata", {})
                )
                self.shares[sid] = share
                if share.access_token:
                    self._token_to_share[share.access_token] = sid

        except FileNotFoundError:
            pass


# Quick test
if __name__ == "__main__":
    print("Testing Knowledge Sharing...")

    sharing = KnowledgeSharing()

    # Mock functions
    sharing.get_node_content = lambda nid: {"content": f"Content of {nid}"}
    sharing.get_node_edges = lambda nid: []

    # Export test
    print("\n=== Export Test ===")
    result = sharing.export_nodes(
        ["thompson_sampling", "bayesian_methods"],
        ExportFormat.MARKDOWN
    )
    print(f"Format: {result.format.value}")
    print(f"Nodes: {result.node_count}, Size: {result.size_bytes} bytes")
    print("\nContent preview:")
    print(result.content[:500])

    # Share test
    print("\n=== Share Test ===")
    share = sharing.share(
        node_ids=["thompson_sampling", "bayesian_methods"],
        owner_id="alice",
        title="Bayesian Methods Overview",
        scope=ShareScope.TEAM,
        target_ids=["bob", "charlie"],
        description="Key concepts in Bayesian inference"
    )
    print(f"Created share: {share.share_id}")
    print(f"Title: {share.title}")
    print(f"Scope: {share.scope.value}")

    # Access check
    print(f"\nBob can access: {sharing.can_access_share(share.share_id, 'bob')}")
    print(f"Dave can access: {sharing.can_access_share(share.share_id, 'dave')}")

    # Public share
    public_share = sharing.share(
        node_ids=["thompson_sampling"],
        owner_id="alice",
        title="Public Thompson Sampling",
        scope=ShareScope.PUBLIC
    )
    print(f"\nPublic share token: {public_share.access_token[:20]}...")

    # Export formats
    print("\n=== Export Formats ===")
    for fmt in ExportFormat:
        result = sharing.export_nodes(["test_node"], fmt)
        print(f"{fmt.value}: {result.size_bytes} bytes")
