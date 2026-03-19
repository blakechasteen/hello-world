"""
MemFS — Virtual filesystem view over LiteMemoryBus (borrowed from Letta MemFS).

A read-only materialization layer that renders bus contents as markdown files
with YAML frontmatter. Used for:
  1. Human inspection of memory state
  2. LLM system prompt injection (progressive disclosure)
  3. Cross-tool interop (export as .md files)

Does NOT own data — reads from bus._items on demand.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .lite_bus import LiteMemoryBus, StoredItem
    from .render_cache import RenderCache

__all__ = ["MemoryFile", "MemFS"]


@dataclass
class MemoryFile:
    """A memory item rendered as markdown with YAML frontmatter."""
    path: str                       # Virtual path, e.g. "entity/ingredient/2026-03_a1b2c3d4.md"
    frontmatter: dict[str, Any]     # importance, memory_type, token_estimate, etc.
    body: str                       # Content as markdown
    token_estimate: int             # From StoredItem.token_estimate

    def render(self) -> str:
        """Render as markdown string with YAML frontmatter."""
        fm_lines = ["---"]
        for k, v in self.frontmatter.items():
            fm_lines.append(f"{k}: {v}")
        fm_lines.append("---")
        return "\n".join(fm_lines) + "\n" + self.body

    def render_tokens(self) -> int:
        """Estimate tokens for the rendered output."""
        # Frontmatter adds ~overhead per field
        overhead = len(self.frontmatter) * 3 + 4  # --- lines + field labels
        return self.token_estimate + overhead


class MemFS:
    """Virtual filesystem view over LiteMemoryBus.

    Reads from bus._items, materializes on demand.
    The file tree serves as progressive disclosure index (Letta idea).
    """

    def __init__(self, bus: "LiteMemoryBus", render_cache: Optional["RenderCache"] = None):
        self._bus = bus
        self._render_cache = render_cache

    def tree(self, max_depth: int = 2) -> str:
        """Return file tree string for system prompt injection.

        Groups items by memory_type → entity_type.
        Shows item counts and approximate token totals per group.
        """
        if self._render_cache:
            cached = self._render_cache.get_tree(self._bus.generation)
            if cached is not None:
                return cached

        # Group items: memory_type → entity_type → list of items
        groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for item in self._bus._items.values():
            mt = item.memory_type or "unknown"
            et = item.entity_type or "general"
            groups[mt][et].append(item)

        lines = ["memory/"]
        for mt in sorted(groups.keys()):
            subtypes = groups[mt]
            mt_total = sum(
                len(items) for items in subtypes.values()
            )
            mt_tokens = sum(
                item.token_estimate for items in subtypes.values() for item in items
            )
            lines.append(f"  {mt}/ ({mt_total} items, ~{mt_tokens} tokens)")

            if max_depth >= 2:
                for et in sorted(subtypes.keys()):
                    items = subtypes[et]
                    et_tokens = sum(i.token_estimate for i in items)
                    lines.append(f"    {et}/ ({len(items)} items, ~{et_tokens} tokens)")

        result = "\n".join(lines)
        if self._render_cache:
            self._render_cache.put_tree(self._bus.generation, result)
        return result

    def render(self, item_id: str) -> MemoryFile | None:
        """Render single StoredItem as MemoryFile with YAML frontmatter."""
        item = self._bus._items.get(item_id)
        if not item:
            return None

        return MemoryFile(
            path=self._path_for(item),
            frontmatter={
                "importance": round(item.importance, 2),
                "memory_type": item.memory_type,
                "token_estimate": item.token_estimate,
                "access_count": item.access_count,
                "last_accessed": item.last_accessed or "never",
            },
            body=item.content,
            token_estimate=item.token_estimate,
        )

    def render_pinned(self, max_tokens: int = 2000) -> list[MemoryFile]:
        """Render high-importance items that should be pinned to context.

        Selection: importance > 0.7 AND memory_type in ('config', 'entity')
        Sorted by importance desc, token-budget limited.
        """
        candidates = [
            item for item in self._bus._items.values()
            if item.importance > 0.7
            and item.memory_type in ("config", "entity", "factual")
        ]
        # Sort by importance descending
        candidates.sort(key=lambda x: x.importance, reverse=True)

        files = []
        budget_remaining = max_tokens
        for item in candidates:
            mf = self.render(item.id)
            if mf and mf.render_tokens() <= budget_remaining:
                files.append(mf)
                budget_remaining -= mf.render_tokens()
            if budget_remaining <= 0:
                break
        return files

    def render_for_prompt(self, query: str, token_budget: int) -> str:
        """Progressive disclosure for LLM system prompts.

        Structure:
        1. File tree header (always, ~estimate from tree)
        2. Pinned files rendered in full (high importance, up to 40% of budget)
        3. Remaining budget: top-k files by keyword match to query
        4. Everything else: just paths in the tree (no content)
        """
        if self._render_cache:
            cached = self._render_cache.get_prompt(query, token_budget, self._bus.generation)
            if cached is not None:
                return cached

        parts = []

        # 1. Tree header
        tree_str = self.tree()
        tree_tokens = max(1, len(tree_str) // 4)
        parts.append(tree_str)
        remaining = token_budget - tree_tokens

        if remaining <= 0:
            return tree_str

        # 2. Pinned files (up to 40% of budget)
        pinned_budget = int(remaining * 0.4)
        pinned = self.render_pinned(max_tokens=pinned_budget)
        if pinned:
            parts.append("\n## Pinned Memories")
            for mf in pinned:
                rendered = mf.render()
                parts.append(rendered)
                remaining -= mf.render_tokens()

        # 3. Query-relevant files (keyword match, remaining budget)
        if query and remaining > 50:
            keywords = query.lower().split()
            pinned_paths = {mf.path for mf in pinned}
            scored = []
            for item in self._bus._items.values():
                # Skip already-pinned items
                if self._path_for(item) in pinned_paths:
                    continue
                content_lower = item.content.lower()
                hits = sum(1 for kw in keywords if kw in content_lower)
                if hits > 0:
                    scored.append((hits / len(keywords), item))

            scored.sort(key=lambda x: x[0], reverse=True)
            if scored:
                parts.append("\n## Relevant Memories")
                for _, item in scored:
                    mf = self.render(item.id)
                    if mf and mf.render_tokens() <= remaining:
                        parts.append(mf.render())
                        remaining -= mf.render_tokens()
                    if remaining <= 50:
                        break

        result = "\n\n".join(parts)
        if self._render_cache:
            self._render_cache.put_prompt(query, token_budget, self._bus.generation, result)
        return result

    def all_files(self) -> list[MemoryFile]:
        """Render all items as MemoryFiles."""
        return [
            self.render(item_id)
            for item_id in self._bus._items
        ]

    @staticmethod
    def _path_for(item: "StoredItem") -> str:
        """Generate virtual path: {memory_type}/{entity_type or general}/{date}_{id[:8]}.md"""
        mt = item.memory_type or "unknown"
        et = item.entity_type or "general"
        date_prefix = item.timestamp[:10] if item.timestamp else "undated"
        return f"{mt}/{et}/{date_prefix}_{item.id[:8]}.md"
