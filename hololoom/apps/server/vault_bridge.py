"""
Vault Bridge — Search + federation write for Promptly.

Gives Promptly direct filesystem access to Blake's PARA vault for context,
and federation protocol access for proposing notes to the inbox.

Read path: keyword search + file read (no external dependencies)
Write path: propose_note() drops into 00_Inbox/{agent}/ or departments/{dept}/inbox/{agent}/
Federation v2: quality scoring, promotion tracking, adaptive frequency.
Department support: multi-tier promotion (dept inbox → dept PARA → vault inbox).
"""

import logging
import os
import re
import shutil
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Vault root — configurable via env, defaults to Blake's OneDrive vault
VAULT_ROOT = Path(
    os.environ.get(
        "PROMPTLY_VAULT_ROOT",
        Path.home() / "OneDrive" / "Documents" / "notes",
    )
)

# Directories to skip when searching
SKIP_DIRS = {
    ".git", ".obsidian", ".claude", ".vault_cache", "__pycache__",
    "node_modules", ".venv", "vault_mcp", "components", "services",
    "scratch",  # department worker ephemeral output
}

# Department name validation: lowercase, alphanumeric + hyphens, 1-32 chars
DEPT_NAME_RE = re.compile(r'^[a-z][a-z0-9-]{0,31}$')

SKIP_FILES = {"notes_app.py", "db_handler.py", "setup_schema.py", "requirements.txt"}

PARA_SECTIONS = ("00_Inbox", "01_Projects", "02_Areas", "03_Resources", "04_Archives")


def _is_available() -> bool:
    """Check if the vault is accessible."""
    return VAULT_ROOT.is_dir() and (VAULT_ROOT / "CLAUDE.md").exists()


def _collect_notes() -> list[Path]:
    """Collect all markdown files in the vault, skipping system dirs."""
    if not _is_available():
        return []
    notes = []
    for md in VAULT_ROOT.rglob("*.md"):
        # Skip files in excluded directories
        parts = set(md.relative_to(VAULT_ROOT).parts)
        if parts & SKIP_DIRS:
            continue
        if md.name in SKIP_FILES:
            continue
        if md.name.startswith("_") and md.name != "_index.md":
            continue
        notes.append(md)
    return notes


def search(query: str, top_k: int = 5) -> list[dict]:
    """Keyword search over vault notes. Returns matches with snippets.

    Simple but effective: splits query into terms, scores by term frequency
    in title + content. No external dependencies.
    """
    if not _is_available():
        logger.debug("Vault not available at %s", VAULT_ROOT)
        return []

    terms = [t.lower() for t in re.split(r'\W+', query) if len(t) >= 2]
    if not terms:
        return []

    scored = []
    for note in _collect_notes():
        try:
            content = note.read_text(encoding="utf-8", errors="replace").lower()
            rel = str(note.relative_to(VAULT_ROOT)).replace("\\", "/")
            title = note.stem.replace("-", " ")

            # Score: title matches worth 3x, content matches worth 1x
            score = 0
            for term in terms:
                if term in title:
                    score += 3
                score += content.count(term)

            if score > 0:
                # Extract snippet around first match
                snippet = _extract_snippet(content, terms[0])
                scored.append({
                    "path": rel,
                    "title": title,
                    "score": score,
                    "snippet": snippet,
                })
        except OSError:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def read_note(rel_path: str) -> str | None:
    """Read a vault note by relative path. Returns None if not found."""
    if not _is_available():
        return None
    clean = rel_path.replace("\\", "/").lstrip("/")
    full = (VAULT_ROOT / clean).resolve()

    # Security: stay within vault
    try:
        full.relative_to(VAULT_ROOT.resolve())
    except ValueError:
        return None

    if not full.exists() or not full.is_file():
        return None

    return full.read_text(encoding="utf-8", errors="replace")


def list_notes(section: str | None = None) -> list[dict]:
    """List vault notes, optionally filtered by PARA section."""
    if not _is_available():
        return []

    results = []
    for note in _collect_notes():
        rel = str(note.relative_to(VAULT_ROOT)).replace("\\", "/")
        if section and not rel.startswith(section):
            continue
        results.append({
            "path": rel,
            "title": note.stem.replace("-", " "),
            "section": rel.split("/")[0] if "/" in rel else "",
        })

    results.sort(key=lambda x: x["path"])
    return results


def propose_note(
    title: str,
    content: str,
    links: list[str] | None = None,
    agent: str = "promptly",
    department: str | None = None,
) -> dict:
    """Propose a note to the vault inbox (or department inbox) via federation protocol.

    When department is None: writes to 00_Inbox/{agent}/ (original behavior).
    When department is set: writes to departments/{dept}/inbox/{agent}/.

    Quality-scored (0-1). Rejected if score < 0.3. Rate-limited if
    agent's promotion rate is below 10%.

    Returns: {"status": "proposed", "path": "..."} or {"error": "..."}
    """
    # Validate department name if provided
    if department and not DEPT_NAME_RE.match(department):
        return {"error": f"Invalid department name: {department}", "code": "invalid_department"}
    if not _is_available():
        return {"error": "Vault not available", "code": "unavailable"}

    # Federation v2: check rate limiting
    try:
        tracker = get_tracker()
        tracker.check_promotions()
        if not tracker.should_propose(agent):
            rate = tracker.get_promotion_rate(agent)
            return {
                "error": f"Rate-limited: {agent} promotion rate is {rate:.0%} (<10%)",
                "code": "rate_limited",
                "promotion_rate": rate,
            }
    except Exception:
        tracker = None  # Proceed without tracking if DB fails

    # Validate content substance
    body = content.strip()
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) >= 3:
            body = parts[2].strip()
    if len(body) < 50:
        return {
            "error": f"Content too thin ({len(body)} chars). Notes must be substantive.",
            "code": "insufficient_content",
        }

    # Compute inbox path: department inbox or vault inbox
    if department:
        inbox = VAULT_ROOT / "departments" / department / "inbox" / agent
    else:
        inbox = VAULT_ROOT / "00_Inbox" / agent
    existing_titles = []
    if inbox.is_dir():
        existing_titles = [f.stem.replace("-", " ") for f in inbox.glob("*.md")]
    score = _quality_score(title, content, existing_titles)
    if score < 0.3:
        return {
            "error": f"Quality score too low ({score:.2f}). Needs more substance or novelty.",
            "code": "low_quality",
            "quality_score": score,
        }

    # Generate kebab-case filename
    slug = re.sub(r'[^a-z0-9\s-]', '', title.lower())
    slug = re.sub(r'[\s_]+', '-', slug).strip('-')
    if not slug:
        return {"error": "Title must contain alphanumeric characters", "code": "invalid_title"}

    # Create inbox directory
    inbox.mkdir(parents=True, exist_ok=True)

    # Handle collision
    target = inbox / f"{slug}.md"
    if target.exists():
        stamp = date.today().strftime("%Y%m%d")
        target = inbox / f"{slug}-{stamp}.md"
        counter = 1
        while target.exists():
            target = inbox / f"{slug}-{stamp}-{counter}.md"
            counter += 1

    # Build note
    dept_line = f"department: {department}\n" if department else ""
    frontmatter = (
        f"---\ncreated: {date.today().isoformat()}\n"
        f"proposed_by: {agent}\n{dept_line}---\n\n"
    )
    heading = f"# {title.strip()}\n\n"
    note = frontmatter + heading + content.strip() + "\n"

    target.write_text(note, encoding="utf-8")
    rel = str(target.relative_to(VAULT_ROOT)).replace("\\", "/")

    # Track proposal
    if tracker:
        try:
            tracker.record_proposal(agent, title, rel, score, department=department)
        except Exception:
            pass  # Don't fail the write because tracking broke

    logger.info("Proposed vault note: %s (score=%.2f, agent=%s)", rel, score, agent)
    return {"status": "proposed", "path": rel, "title": title, "quality_score": score}


def promote_within_department(dept: str, source_rel: str, dest_subdir: str = "observations") -> dict:
    """Move a note from department inbox to department PARA.

    source_rel: path relative to VAULT_ROOT (e.g. "departments/farm/inbox/farm-worker/note.md")
    dest_subdir: target subdir under para/ (e.g. "observations", "resources")
    """
    if not _is_available():
        return {"error": "Vault not available", "code": "unavailable"}
    if not DEPT_NAME_RE.match(dept):
        return {"error": f"Invalid department name: {dept}", "code": "invalid_department"}

    source = VAULT_ROOT / source_rel
    if not source.exists():
        return {"error": f"Source not found: {source_rel}", "code": "not_found"}

    dest_dir = VAULT_ROOT / "departments" / dept / "para" / dest_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    shutil.move(str(source), str(dest))
    rel = str(dest.relative_to(VAULT_ROOT)).replace("\\", "/")
    logger.info("Promoted within dept %s: %s → %s", dept, source_rel, rel)
    return {"status": "promoted", "path": rel}


def promote_to_vault(dept: str, source_rel: str) -> dict:
    """Move a note from department PARA to vault inbox for human review.

    Destination: 00_Inbox/{dept}-head/{filename}
    """
    if not _is_available():
        return {"error": "Vault not available", "code": "unavailable"}
    if not DEPT_NAME_RE.match(dept):
        return {"error": f"Invalid department name: {dept}", "code": "invalid_department"}

    source = VAULT_ROOT / source_rel
    if not source.exists():
        return {"error": f"Source not found: {source_rel}", "code": "not_found"}

    dest_dir = VAULT_ROOT / "00_Inbox" / f"{dept}-head"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    shutil.move(str(source), str(dest))
    rel = str(dest.relative_to(VAULT_ROOT)).replace("\\", "/")
    logger.info("Promoted to vault: %s → %s", source_rel, rel)
    return {"status": "promoted_to_vault", "path": rel}


def vault_context_block(query: str, max_notes: int = 3) -> str:
    """Search the vault and format results as a context block for the system prompt.

    Returns empty string if no relevant notes found.
    """
    results = search(query, top_k=max_notes)
    if not results:
        return ""

    blocks = []
    for r in results:
        if r["score"] < 2:  # Skip weak matches
            continue
        content = read_note(r["path"])
        if not content:
            continue
        # Truncate long notes
        if len(content) > 1500:
            content = content[:1500] + "\n...(truncated)"
        blocks.append(f"### {r['path']}\n{content}")

    if not blocks:
        return ""

    return (
        "\n\n---\n## Vault Context (from your notes)\n"
        "The following notes from your PARA vault may be relevant:\n\n"
        + "\n\n".join(blocks)
        + "\n---\n"
    )


def _extract_snippet(content: str, term: str, window: int = 150) -> str:
    """Extract a snippet around the first occurrence of a term."""
    idx = content.find(term)
    if idx == -1:
        return content[:window]
    start = max(0, idx - window // 2)
    end = min(len(content), idx + window // 2)
    snippet = content[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet


# ============================================================================
# Federation v2: Quality Scoring + Promotion Tracking
# ============================================================================

_ACTION_WORDS = frozenset(
    "add remove update create build fix check test deploy review schedule order "
    "plant harvest prep bake measure track clean organize plan start finish".split()
)


def _log2(x: float) -> float:
    """Safe log2 that returns 0 for x <= 0."""
    import math
    return math.log2(x) if x > 0 else 0.0


def _term_entropy(words: list[str]) -> float:
    """Shannon entropy of term frequency distribution.

    Higher entropy = more diverse vocabulary = less repetitive content.
    Uses natural log2 (bits).
    """
    if not words:
        return 0.0
    from collections import Counter
    counts = Counter(words)
    total = len(words)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * _log2(p)
    return entropy


def _quality_score(title: str, content: str, existing_titles: list[str]) -> float:
    """Score a proposal 0.0-1.0 based on length, novelty, and actionability."""
    body = content.strip()
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) >= 3:
            body = parts[2].strip()

    # Length component (0-0.3): log scale
    body_len = len(body)
    if body_len < 50:
        length_score = 0.0
    elif body_len < 200:
        length_score = 0.1
    elif body_len < 500:
        length_score = 0.2
    else:
        length_score = 0.3

    # Novelty component (0-0.4): Shannon entropy of term distribution
    # High entropy = diverse vocabulary = more novel content
    # Low overlap with existing titles = higher novelty
    title_words = set(re.split(r'\W+', title.lower())) - {""}
    body_words_list = [w for w in re.split(r'\W+', body.lower()) if len(w) >= 2]

    if not existing_titles or not title_words:
        novelty_score = 0.3  # No comparisons = moderate novelty
    else:
        # Term frequency entropy of the candidate body
        body_entropy = _term_entropy(body_words_list) if body_words_list else 0.0
        # Normalize to [0, 1] — max entropy is log2(vocab_size)
        vocab_size = len(set(body_words_list)) if body_words_list else 1
        max_entropy = _log2(vocab_size) if vocab_size > 1 else 1.0
        normalized_entropy = min(1.0, body_entropy / max_entropy) if max_entropy > 0 else 0.0

        # Title overlap penalty (Jaccard, kept for near-duplicate detection)
        max_overlap = 0.0
        for existing in existing_titles:
            existing_words = set(re.split(r'\W+', existing.lower())) - {""}
            if not existing_words:
                continue
            intersection = title_words & existing_words
            union = title_words | existing_words
            jaccard = len(intersection) / len(union) if union else 0.0
            max_overlap = max(max_overlap, jaccard)

        # Blend: 60% entropy (vocabulary diversity) + 40% title novelty (duplicate detection)
        novelty_score = 0.4 * (0.6 * normalized_entropy + 0.4 * (1.0 - max_overlap))

    # Actionability component (0-0.3): action verbs, numbers, dates
    body_lower = body.lower()
    body_words = set(re.split(r'\W+', body_lower)) - {""}
    action_count = len(body_words & _ACTION_WORDS)
    has_numbers = bool(re.search(r'\d+', body))
    has_dates = bool(re.search(r'\d{4}-\d{2}', body))

    action_score = min(0.3, 0.05 * action_count + (0.05 if has_numbers else 0) +
                       (0.05 if has_dates else 0))

    return min(1.0, length_score + novelty_score + action_score)


FEDERATION_DB_PATH = os.environ.get(
    "VAULT_FEDERATION_DB",
    str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "vault_federation.db"),
)


class FederationTracker:
    """Tracks vault note proposals, promotions, and per-agent frequency."""

    def __init__(self, db_path: str = FEDERATION_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS federation_proposals (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                title TEXT NOT NULL,
                path TEXT NOT NULL,
                quality_score REAL NOT NULL,
                proposed_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS federation_stats (
                agent TEXT PRIMARY KEY,
                total_proposed INTEGER DEFAULT 0,
                total_promoted INTEGER DEFAULT 0,
                total_rejected INTEGER DEFAULT 0,
                current_frequency TEXT DEFAULT 'normal'
            )
        """)
        self._conn.commit()

        # Migration for existing DBs: add department column
        try:
            self._conn.execute(
                "ALTER TABLE federation_proposals ADD COLUMN department TEXT DEFAULT NULL"
            )
            self._conn.commit()
        except Exception:
            pass  # Column already exists

    def record_proposal(
        self, agent: str, title: str, path: str, score: float,
        department: str | None = None,
    ) -> str:
        """Record a new proposal. Returns proposal ID."""
        prop_id = str(uuid.uuid4())[:12]
        self._conn.execute(
            "INSERT INTO federation_proposals "
            "(id, agent, title, path, quality_score, proposed_at, department) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (prop_id, agent, title, path, score, datetime.now().isoformat(), department),
        )
        self._conn.execute(
            "INSERT INTO federation_stats (agent, total_proposed) VALUES (?, 1) "
            "ON CONFLICT(agent) DO UPDATE SET total_proposed = total_proposed + 1",
            (agent,),
        )
        self._conn.commit()
        return prop_id

    def check_promotions(self) -> int:
        """Scan inbox for promoted/expired proposals. Returns count of newly resolved."""
        rows = self._conn.execute(
            "SELECT id, agent, path, proposed_at FROM federation_proposals "
            "WHERE status = 'pending'"
        ).fetchall()

        resolved = 0
        for row in rows:
            full_path = VAULT_ROOT / row["path"]
            if full_path.exists():
                continue  # Still in inbox — pending

            # File gone: promoted or expired?
            try:
                proposed = datetime.fromisoformat(row["proposed_at"])
                age_days = (datetime.now() - proposed).days
            except (ValueError, TypeError):
                age_days = 0

            if age_days > 30:
                status = "expired"
            else:
                status = "promoted"
                self._conn.execute(
                    "UPDATE federation_stats SET total_promoted = total_promoted + 1 "
                    "WHERE agent = ?",
                    (row["agent"],),
                )

            self._conn.execute(
                "UPDATE federation_proposals SET status = ? WHERE id = ?",
                (status, row["id"]),
            )
            resolved += 1

        if resolved:
            self._conn.commit()
            self._update_frequency()
        return resolved

    def _update_frequency(self) -> None:
        """Update frequency tier for all agents based on promotion rate."""
        rows = self._conn.execute("SELECT * FROM federation_stats").fetchall()
        for row in rows:
            rate = self.get_promotion_rate(row["agent"])
            if row["total_proposed"] < 5:
                freq = "normal"  # Not enough data
            elif rate > 0.5:
                freq = "high"
            elif rate < 0.1:
                freq = "low"
            else:
                freq = "normal"
            self._conn.execute(
                "UPDATE federation_stats SET current_frequency = ? WHERE agent = ?",
                (freq, row["agent"]),
            )
        self._conn.commit()

    def get_promotion_rate(self, agent: str) -> float:
        """Get promotion rate for an agent (0.0-1.0)."""
        row = self._conn.execute(
            "SELECT total_proposed, total_promoted FROM federation_stats WHERE agent = ?",
            (agent,),
        ).fetchone()
        if not row or row["total_proposed"] == 0:
            return 0.0
        return row["total_promoted"] / row["total_proposed"]

    def should_propose(self, agent: str) -> bool:
        """Whether this agent should propose right now (frequency gating)."""
        row = self._conn.execute(
            "SELECT total_proposed, current_frequency FROM federation_stats WHERE agent = ?",
            (agent,),
        ).fetchone()
        if not row or row["total_proposed"] < 5:
            return True  # Not enough data — allow
        return row["current_frequency"] != "low"

    def get_stats(self, agent: str) -> dict:
        """Get federation stats for an agent."""
        self.check_promotions()
        row = self._conn.execute(
            "SELECT * FROM federation_stats WHERE agent = ?", (agent,),
        ).fetchone()
        if not row:
            return {
                "agent": agent, "total_proposed": 0, "total_promoted": 0,
                "total_rejected": 0, "promotion_rate": 0.0, "frequency": "normal",
            }
        return {
            "agent": agent,
            "total_proposed": row["total_proposed"],
            "total_promoted": row["total_promoted"],
            "total_rejected": row["total_rejected"],
            "promotion_rate": self.get_promotion_rate(agent),
            "frequency": row["current_frequency"],
        }


# Federation tracker singleton
_tracker: FederationTracker | None = None


def get_tracker() -> FederationTracker:
    global _tracker
    if _tracker is None:
        _tracker = FederationTracker()
    return _tracker
