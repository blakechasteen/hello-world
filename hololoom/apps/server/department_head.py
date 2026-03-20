"""
Department Head — LLM judge for department inbox curation.

Scans a department's inbox, evaluates each note against a promotion rubric,
and takes action: promote to dept PARA, return with feedback, or escalate.

Hybrid scoring: mechanical pre-checks for countable criteria (link count,
body length) gate the LLM call. LLM scores only the subjective terms.

Thompson Sampling: per-(dept, rubric_term) bandits learn which rubric weights
predict successful vault promotions over time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import APIRouter

from hololoom.apps.server import vault_bridge
from hololoom.apps.server.vault_bridge import VAULT_ROOT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/department", tags=["department-head"])


# ============================================================================
# Rubric
# ============================================================================

@dataclass
class RubricTerm:
    name: str
    weight: int
    description: str
    mechanical: bool = False  # True = checked by code, not LLM


FARM_RUBRIC = [
    RubricTerm("links",      20, "References >= 2 existing notes via [[wiki links]]", mechanical=True),
    RubricTerm("substance",  20, "Body contains >= 200 chars of non-frontmatter text", mechanical=True),
    RubricTerm("claim",      25, "Makes a specific claim, not just a topic label"),
    RubricTerm("connects",   15, "Relates to existing department or vault knowledge"),
    RubricTerm("actionable", 10, "Contains dates, quantities, or concrete observations"),
    RubricTerm("novel",      10, "Not a near-duplicate of existing department knowledge"),
]

# Thresholds
PROMOTE_THRESHOLD = 70
RETURN_THRESHOLD = 40


def get_rubric(dept: str) -> list[RubricTerm]:
    """Get the rubric for a department. Loads from _index.md if available."""
    loaded = _load_rubric_from_index(dept)
    if loaded:
        return loaded
    return FARM_RUBRIC  # fallback


# Rubric cache: dept → rubric terms
_rubric_cache: dict[str, list[RubricTerm]] = {}


def _load_rubric_from_index(dept: str) -> list[RubricTerm] | None:
    """Parse rubric from departments/{dept}/_index.md.

    Looks for a '## Promotion Rubric' section with numbered items like:
    1. **Makes a claim** (not just a topic label) — weight 25
    """
    if dept in _rubric_cache:
        return _rubric_cache[dept]

    index_path = VAULT_ROOT / "departments" / dept / "_index.md"
    if not index_path.exists():
        return None

    try:
        content = index_path.read_text(encoding="utf-8", errors="replace")

        # Find rubric section
        rubric_section = ""
        in_rubric = False
        for line in content.splitlines():
            if line.strip().startswith("## Promotion Rubric"):
                in_rubric = True
                continue
            elif line.strip().startswith("## ") and in_rubric:
                break
            elif in_rubric:
                rubric_section += line + "\n"

        if not rubric_section:
            return None

        # Parse numbered rubric items
        # Pattern: N. **Name** (description) — weight NN
        terms = []
        mechanical_names = {"links", "substance", "link", "body_length"}

        for line in rubric_section.splitlines():
            line = line.strip()
            if not line or not line[0].isdigit():
                continue

            # Extract weight
            weight = 10  # default
            if "weight" in line.lower():
                import re
                weight_match = re.search(r'weight\s+(\d+)', line, re.IGNORECASE)
                if weight_match:
                    weight = int(weight_match.group(1))

            # Extract name from **bold** text
            bold_match = re.search(r'\*\*(.+?)\*\*', line)
            if not bold_match:
                continue

            raw_name = bold_match.group(1).strip()
            # Convert to snake_case key
            name = raw_name.lower().replace(" ", "_").replace(">=", "gte")
            # Simplify common patterns
            if "claim" in name:
                name = "claim"
            elif "link" in name:
                name = "links"
            elif "substanti" in name:
                name = "substance"
            elif "connect" in name:
                name = "connects"
            elif "action" in name:
                name = "actionable"
            elif "novel" in name or "duplicate" in name:
                name = "novel"
            elif "seasonal" in name or "timing" in name:
                name = "seasonal_relevance"

            # Extract description (everything between ** and — or end)
            desc_match = re.search(r'\*\*.*?\*\*\s*(?:\(([^)]+)\)|(.+?)(?:\s*—|\s*$))', line)
            description = ""
            if desc_match:
                description = (desc_match.group(1) or desc_match.group(2) or "").strip()
            if not description:
                description = raw_name

            is_mechanical = name in mechanical_names

            terms.append(RubricTerm(
                name=name,
                weight=weight,
                description=description,
                mechanical=is_mechanical,
            ))

        if terms:
            _rubric_cache[dept] = terms
            logger.info("Loaded rubric for %s: %d terms from _index.md", dept, len(terms))
            return terms

    except Exception as e:
        logger.warning("Failed to load rubric for %s: %s", dept, e)

    return None


def list_departments() -> list[str]:
    """List all departments that have _index.md files."""
    dept_root = VAULT_ROOT / "departments"
    if not dept_root.is_dir():
        return []
    return [
        d.name for d in sorted(dept_root.iterdir())
        if d.is_dir() and (d / "_index.md").exists()
    ]


# ============================================================================
# Thompson Sampling — learn which rubric terms predict success
# ============================================================================

HEAD_BANDIT_PATH = Path(VAULT_ROOT) / "departments" / ".head_bandit_state.json"


@dataclass
class BanditArm:
    """Beta distribution for one (dept, rubric_term) pair."""
    successes: float = 1.0  # alpha — start uniform
    failures: float = 1.0   # beta — start uniform

    def sample(self) -> float:
        return float(np.random.beta(self.successes, self.failures))

    def update(self, reward: float) -> None:
        # Try natural gradient update (geometry-aware)
        try:
            from hololoom.apps.server.math_extensions import (
                HAS_NATURAL_GRADIENT,
                beta_natural_gradient_update,
            )
            if HAS_NATURAL_GRADIENT:
                self.successes, self.failures = beta_natural_gradient_update(
                    self.successes, self.failures, reward,
                )
                return
        except Exception:
            pass

        # Standard update (fallback)
        if reward > 0:
            self.successes += reward
        else:
            self.failures += abs(reward)

    @property
    def mean(self) -> float:
        return self.successes / (self.successes + self.failures)


class HeadBandit:
    """Thompson Sampling over rubric term effectiveness per department.

    Learns which rubric scores best predict successful vault promotions.
    Arms are keyed by (dept, term_name). After human review:
    - Note kept in vault → terms that scored high get reward
    - Note expired/rejected → terms that scored high get penalty
    """

    def __init__(self, persist_path: Path = HEAD_BANDIT_PATH):
        self.arms: dict[tuple[str, str], BanditArm] = {}
        self.persist_path = persist_path
        self._load()

    def get_arm(self, dept: str, term: str) -> BanditArm:
        key = (dept, term)
        if key not in self.arms:
            self.arms[key] = BanditArm()
        return self.arms[key]

    def weight_adjustment(self, dept: str, rubric: list[RubricTerm]) -> dict[str, float]:
        """Sample weight multipliers from the bandit.

        Returns {term_name: multiplier} where multiplier is sampled from
        the Beta distribution. Terms that predict success get higher multipliers.
        """
        adjustments = {}
        for term in rubric:
            arm = self.get_arm(dept, term.name)
            # Blend bandit sample with base weight: 70% base, 30% learned
            sample = arm.sample()
            adjustments[term.name] = 0.7 + 0.3 * sample  # range [0.7, 1.0]
        return adjustments

    def record_outcome(self, dept: str, term_scores: dict[str, float], success: bool) -> None:
        """Record promotion outcome. Update arms based on which terms scored high.

        High-scoring terms that predicted a success get rewarded.
        High-scoring terms that predicted a failure get penalized.
        """
        for term_name, score in term_scores.items():
            arm = self.get_arm(dept, term_name)
            # Normalized score [0, 1]
            normalized = score / 100.0
            if success:
                # Reward proportional to how high the term scored
                arm.update(normalized * 0.5)
            else:
                # Penalize proportional to how high the term scored
                # (high score on a bad note = bad signal)
                arm.update(-normalized * 0.5)
        self._save()

    def stats(self, dept: str) -> dict[str, dict]:
        """Return bandit state for a department."""
        result = {}
        for (d, term), arm in self.arms.items():
            if d == dept:
                result[term] = {
                    "successes": round(arm.successes, 2),
                    "failures": round(arm.failures, 2),
                    "mean": round(arm.mean, 3),
                }
        return result

    def _save(self) -> None:
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                f"{d}:{t}": {"s": arm.successes, "f": arm.failures}
                for (d, t), arm in self.arms.items()
            }
            tmp = self.persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.persist_path)
        except OSError as e:
            logger.warning("Failed to save head bandit state: %s", e)

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            for key, vals in data.items():
                dept, term = key.split(":", 1)
                self.arms[(dept, term)] = BanditArm(
                    successes=vals["s"], failures=vals["f"],
                )
            logger.info("Loaded head bandit: %d arms", len(self.arms))
        except Exception as e:
            logger.warning("Failed to load head bandit state: %s", e)


# Singleton
_head_bandit: HeadBandit | None = None


def get_head_bandit() -> HeadBandit:
    global _head_bandit
    if _head_bandit is None:
        _head_bandit = HeadBandit()
    return _head_bandit


# ============================================================================
# Contextual Bandit — 470-dim FGTS (upgrades simple HeadBandit)
# ============================================================================

try:
    from hololoom.core.warp.math.contextual_bandit import ContextualOperationSelector
    HAS_CONTEXTUAL_BANDIT = True
except ImportError:
    HAS_CONTEXTUAL_BANDIT = False

_contextual_selector: object | None = None  # lazy init


def get_contextual_selector() -> object | None:
    """Get the contextual bandit selector. Returns None if unavailable."""
    global _contextual_selector
    if not HAS_CONTEXTUAL_BANDIT:
        return None
    if _contextual_selector is None:
        subjective_terms = [t.name for t in FARM_RUBRIC if not t.mechanical]
        _contextual_selector = ContextualOperationSelector(
            operations=subjective_terms,
            exploration_coef=0.5,
        )
        logger.info("Contextual bandit initialized: %d operations, 470-dim FGTS", len(subjective_terms))
    return _contextual_selector


def contextual_weight_adjustment(
    dept: str,
    candidate_claim: str,
    rubric: list[RubricTerm],
    mechanical_scores: dict,
) -> dict[str, float]:
    """Get weight adjustments from the contextual bandit.

    Uses 470-dim context (claim text, mechanical scores, dept context)
    to select which rubric terms matter most for THIS note.

    Falls back to simple HeadBandit if contextual bandit unavailable.
    """
    selector = get_contextual_selector()
    if selector is None:
        # Fallback to simple HeadBandit
        return get_head_bandit().weight_adjustment(dept, rubric)

    # Use the contextual selector to rank terms
    # The "select" picks which term is MOST important in this context
    try:
        _selected, metadata = selector.select(
            query_text=candidate_claim,
            intent="promotion",
            budget_remaining=50.0,
            required_confidence=0.5,
            quality_preference=sum(mechanical_scores.values()) / max(1, len(mechanical_scores)) / 100.0,
        )

        # Convert per-term UCB scores to weight multipliers
        scores = metadata.get("scores", {})
        if not scores:
            return get_head_bandit().weight_adjustment(dept, rubric)

        # Normalize scores to [0.7, 1.3] range (allows both boost and dampen)
        min_score = min(scores.values())
        max_score = max(scores.values())
        score_range = max_score - min_score if max_score != min_score else 1.0

        adjustments = {}
        for term in rubric:
            if term.name in scores:
                normalized = (scores[term.name] - min_score) / score_range
                adjustments[term.name] = 0.7 + 0.6 * normalized  # [0.7, 1.3]
            else:
                adjustments[term.name] = 1.0  # neutral for mechanical terms

        return adjustments

    except Exception as e:
        logger.warning("Contextual bandit failed, falling back to HeadBandit: %s", e)
        return get_head_bandit().weight_adjustment(dept, rubric)


def contextual_update(candidate_claim: str, term_scores: dict[str, float], success: bool) -> None:
    """Update contextual bandit after observing promotion outcome."""
    selector = get_contextual_selector()
    if selector is None:
        return

    try:
        reward = 1.0 if success else -0.5
        # Update each term's bandit with the outcome
        for term_name, score in term_scores.items():
            if term_name in selector.fgts.operations:
                # Weight reward by how confident the term was
                weighted_reward = reward * (score / 100.0)
                selector.update(
                    operation=term_name,
                    query_text=candidate_claim,
                    intent="promotion",
                    reward=weighted_reward,
                    cost=1.0,
                )
    except Exception as e:
        logger.warning("Contextual bandit update failed: %s", e)


# ============================================================================
# Regret Minimization — Hedge for evaluation strategy learning
# ============================================================================

try:
    from hololoom.core.warp.math.decision.regret_minimization import Hedge
    HAS_HEDGE = True
except ImportError:
    HAS_HEDGE = False

# Three evaluation strategies: strict (high thresholds), balanced (default), generous (low thresholds)
EVAL_STRATEGIES = ["strict", "balanced", "generous"]
STRATEGY_THRESHOLD_MODS = {
    "strict":    {"promote": 80, "return": 50},   # higher bar
    "balanced":  {"promote": 70, "return": 40},   # default
    "generous":  {"promote": 60, "return": 30},   # lower bar
}

_hedge: object | None = None


def get_hedge():
    """Get the Hedge regret minimizer for evaluation strategy selection."""
    global _hedge
    if not HAS_HEDGE:
        return None
    if _hedge is None:
        _hedge = Hedge(n_actions=len(EVAL_STRATEGIES), time_horizon=500)
        logger.info("Hedge regret minimizer initialized: %d strategies", len(EVAL_STRATEGIES))
    return _hedge


def select_strategy() -> tuple:
    """Select evaluation strategy via Hedge. Returns (strategy_name, thresholds)."""
    hedge = get_hedge()
    if hedge is None:
        return "balanced", STRATEGY_THRESHOLD_MODS["balanced"]

    idx = hedge.choose()
    strategy = EVAL_STRATEGIES[idx]
    return strategy, STRATEGY_THRESHOLD_MODS[strategy]


def update_strategy_regret(strategy: str, success: bool) -> None:
    """Update Hedge with observed outcome.

    Loss = 1.0 for bad outcomes (promoted something that got rejected,
    or returned something that should have been promoted).
    """
    hedge = get_hedge()
    if hedge is None:
        return

    # Loss vector: chosen strategy gets loss based on outcome
    losses = np.zeros(len(EVAL_STRATEGIES))
    idx = EVAL_STRATEGIES.index(strategy) if strategy in EVAL_STRATEGIES else 1
    losses[idx] = 0.0 if success else 1.0
    hedge.update(losses)


# ============================================================================
# Mechanical pre-checks
# ============================================================================

_WIKI_LINK_RE = re.compile(r'\[\[([^\]]+)\]\]')


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    if content.strip().startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return content.strip()


def _count_wiki_links(content: str) -> int:
    """Count [[wiki links]] in content."""
    return len(_WIKI_LINK_RE.findall(content))


def mechanical_scores(content: str, rubric: list[RubricTerm]) -> dict:
    """Score mechanical rubric terms. Returns {term_name: score (0-100)}."""
    body = _strip_frontmatter(content)
    scores = {}

    for term in rubric:
        if not term.mechanical:
            continue

        if term.name == "links":
            link_count = _count_wiki_links(content)
            scores["links"] = 100 if link_count >= 2 else max(0, link_count * 40)

        elif term.name == "substance":
            body_len = len(body)
            if body_len >= 200:
                scores["substance"] = 100
            elif body_len >= 100:
                scores["substance"] = 50
            else:
                scores["substance"] = max(0, int(body_len / 2))

    return scores


def mechanical_feedback(scores: dict) -> list[str]:
    """Generate specific feedback for failed mechanical checks."""
    feedback = []
    if scores.get("links", 100) < 100:
        feedback.append("Add [[wiki links]] to at least 2 existing notes.")
    if scores.get("substance", 100) < 100:
        feedback.append("Expand the note body to at least 200 characters of substantive content.")
    return feedback


# ============================================================================
# LLM evaluation
# ============================================================================

def _build_evaluation_prompt(
    note_content: str,
    rubric: list[RubricTerm],
    dept_context: list[dict],
    vault_context: list[dict],
    contradiction_context: str = "",
) -> list[dict]:
    """Build the evaluation prompt for the LLM judge."""
    subjective_terms = [t for t in rubric if not t.mechanical]

    rubric_text = "\n".join(
        f"- **{t.name}** (weight {t.weight}): {t.description}"
        for t in subjective_terms
    )

    context_notes = ""
    if dept_context:
        context_notes += "\n## Existing Department Knowledge\n"
        for n in dept_context[:5]:
            context_notes += f"- [{n['title']}]({n['path']}): {n.get('snippet', '')[:100]}\n"

    if vault_context:
        context_notes += "\n## Existing Vault Knowledge\n"
        for n in vault_context[:3]:
            context_notes += f"- [{n['title']}]({n['path']}): {n.get('snippet', '')[:100]}\n"

    system_prompt = f"""You are a Department Head evaluating a note proposed for promotion into department knowledge.

Score each criterion 0-100 based on the note content and existing context.
{context_notes}
{contradiction_context}

## Scoring Criteria
{rubric_text}

## Response Format
Respond with ONLY valid JSON, no other text:
{{
  "scores": {{"claim": <0-100>, "connects": <0-100>, "actionable": <0-100>, "novel": <0-100>}},
  "reason": "<1-2 sentence justification>",
  "feedback": "<specific improvement suggestions if score is low>"
}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Evaluate this note:\n\n{note_content}"},
    ]


def _parse_llm_response(text: str) -> dict | None:
    """Parse LLM JSON response. Returns None if unparseable."""
    # Strip markdown code blocks if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Try direct parse
    try:
        data = json.loads(text)
        if "scores" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from mixed text
    match = re.search(r'\{[\s\S]*"scores"[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def compute_weighted_score(
    mechanical: dict,
    llm_scores: dict,
    rubric: list[RubricTerm],
    dept: str | None = None,
) -> float:
    """Compute final weighted score from mechanical + LLM scores.

    When dept is provided, applies Thompson Sampling weight adjustments
    learned from past promotion outcomes.
    """
    # Get weight adjustments — contextual bandit (primary) or HeadBandit (fallback)
    adjustments = {}
    if dept:
        try:
            # Extract claim for contextual features
            claim = ""
            body_text = mechanical.get("_body", "")  # set by caller if available
            if body_text:
                claim = body_text.split(".")[0] if body_text else ""
            adjustments = contextual_weight_adjustment(dept, claim, rubric, mechanical)
        except Exception:
            pass  # No adjustment if both bandits fail

    total_weight = 0.0
    score = 0.0

    for term in rubric:
        if term.mechanical:
            raw = mechanical.get(term.name, 0)
        else:
            raw = llm_scores.get(term.name, 0)

        adjusted_weight = term.weight * adjustments.get(term.name, 1.0)
        total_weight += adjusted_weight
        score += (raw / 100.0) * adjusted_weight

    return (score / total_weight) * 100 if total_weight > 0 else 0.0


# ============================================================================
# Evaluation endpoint
# ============================================================================

# ============================================================================
# Batch Deduplication — JS divergence on term distributions
# ============================================================================

try:
    from hololoom.core.warp.math.information.information_theory import InformationMetrics
    HAS_BATCH_JS = True
except ImportError:
    HAS_BATCH_JS = False


def _dedup_candidates(candidates: list[Path], js_threshold: float = 0.05) -> tuple:
    """Detect and remove near-duplicate proposals via batch JS divergence.

    Converts each note's term frequencies to a probability distribution,
    then computes pairwise JS divergence. Notes with JS < threshold are
    near-duplicates — keep the longer one, skip the rest.

    Returns (filtered_candidates, n_duplicates_removed).
    """
    if not HAS_BATCH_JS or len(candidates) < 2:
        return candidates, 0

    from collections import Counter

    # Build term distributions for each candidate
    all_words = []
    note_words = []
    for md_file in candidates:
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace").lower()
            content = _strip_frontmatter(content)
            words = [w for w in re.split(r'\W+', content) if len(w) >= 3]
            note_words.append(words)
            all_words.extend(words)
        except OSError:
            note_words.append([])

    if not all_words:
        return candidates, 0

    # Build shared vocabulary
    vocab = sorted(set(all_words))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    n_vocab = len(vocab)

    # Convert to distributions
    distributions = np.zeros((len(candidates), n_vocab), dtype=np.float64)
    for i, words in enumerate(note_words):
        counts = Counter(words)
        for w, c in counts.items():
            if w in vocab_idx:
                distributions[i, vocab_idx[w]] = c
        # Normalize to probability distribution
        total = distributions[i].sum()
        if total > 0:
            distributions[i] /= total
        else:
            distributions[i] = 1.0 / n_vocab  # uniform if empty

    # Add smoothing
    distributions = np.clip(distributions, 1e-10, None)
    distributions /= distributions.sum(axis=1, keepdims=True)

    # Compute pairwise JS divergence
    metrics = InformationMetrics(base=2)
    js_matrix = metrics.batch_js_divergence(distributions)

    # Find duplicates: for each pair with JS < threshold, keep the longer note
    keep = set(range(len(candidates)))
    for i in range(len(candidates)):
        if i not in keep:
            continue
        for j in range(i + 1, len(candidates)):
            if j not in keep:
                continue
            if js_matrix[i, j] < js_threshold:
                # Keep the longer note
                len_i = len(note_words[i])
                len_j = len(note_words[j])
                remove = j if len_i >= len_j else i
                keep.discard(remove)
                logger.info(
                    "Dedup: %s ~ %s (JS=%.4f, removing shorter)",
                    candidates[i].name, candidates[j].name, js_matrix[i, j],
                )

    filtered = [candidates[i] for i in sorted(keep)]
    removed = len(candidates) - len(filtered)
    return filtered, removed


@dataclass
class EvaluationResult:
    path: str
    action: str  # "promote", "return", "escalate", "skip"
    score: float
    reason: str
    feedback: str | None = None
    explanation: dict | None = None  # causal chain + counterfactuals


@router.post("/{dept}/feedback")
async def record_feedback(dept: str, note_title: str, success: bool, scores: dict | None = None):
    """Record human review outcome for Thompson Sampling learning.

    Called when a human promotes (success=True) or rejects (success=False)
    a note that was auto-promoted to the vault inbox.

    Args:
        dept: Department name
        note_title: Title of the note (for logging)
        success: True if human kept the note, False if rejected
        scores: Optional dict of {term_name: score} from the original evaluation
    """
    if not scores:
        return {"status": "skipped", "reason": "No scores provided — nothing to learn from"}

    bandit = get_head_bandit()
    bandit.record_outcome(dept, scores, success)

    # Also update contextual bandit
    contextual_update(note_title, scores, success)

    logger.info(
        "Head bandit feedback: dept=%s title='%s' success=%s terms=%s",
        dept, note_title[:50], success, list(scores.keys()),
    )

    response = {
        "status": "recorded",
        "dept": dept,
        "success": success,
        "bandit_stats": bandit.stats(dept),
    }

    # Add contextual bandit stats if available
    selector = get_contextual_selector()
    if selector:
        response["contextual_stats"] = selector.get_stats()

    return response


@router.get("/{dept}/bandit")
async def bandit_stats(dept: str):
    """Get current Thompson Sampling state for a department's rubric terms."""
    bandit = get_head_bandit()
    response = {"dept": dept, "arms": bandit.stats(dept)}

    selector = get_contextual_selector()
    if selector:
        response["contextual"] = selector.get_stats()

    return response


@router.post("/{dept}/evaluate")
async def evaluate_inbox(dept: str):
    """Scan department inbox. Evaluate each note against rubric. Promote, return, or escalate."""
    if not vault_bridge._is_available():
        return {"error": "Vault not available", "evaluated": 0}

    dept_dir = VAULT_ROOT / "departments" / dept
    if not dept_dir.is_dir():
        return {"error": f"Department not found: {dept}", "evaluated": 0}

    inbox_dir = dept_dir / "inbox"
    if not inbox_dir.is_dir():
        return {"evaluated": 0, "promoted": 0, "returned": 0, "escalated": 0}

    rubric = get_rubric(dept)
    results = []

    # Collect all candidate notes
    candidates = []
    for worker_dir in inbox_dir.iterdir():
        if not worker_dir.is_dir():
            continue
        for md_file in worker_dir.glob("*.md"):
            candidates.append(md_file)

    if not candidates:
        return {"evaluated": 0, "promoted": 0, "returned": 0, "escalated": 0}

    # Dedup: detect near-duplicate proposals via batch JS divergence
    duplicates_skipped = 0
    try:
        candidates, duplicates_skipped = _dedup_candidates(candidates)
    except Exception as e:
        logger.debug("Dedup unavailable: %s", e)

    # Schedule: order candidates by urgency (EDF) instead of FIFO
    try:
        from hololoom.apps.server.math_extensions_v2 import schedule_inbox_evaluation
        note_infos = [
            {
                "path": str(c),
                "estimated_cost": 5,
                "urgency": min(1.0, (time.time() - c.stat().st_mtime) / 86400),  # older = more urgent
            }
            for c in candidates
        ]
        ordered = schedule_inbox_evaluation(note_infos)
        candidates = [candidates[note_infos.index(n)] for n in ordered]
    except Exception:
        pass  # FIFO fallback

    # Get model router + LLM caller
    from hololoom.apps.server.model_router import get_router
    from hololoom.apps.server.promptly.llm import call_model

    model_router = get_router()

    for md_file in candidates:
        try:
            result = await _evaluate_single(
                md_file, dept, rubric, model_router, call_model,
            )
            results.append(result)
        except Exception as e:
            logger.exception("Failed to evaluate %s: %s", md_file.name, e)
            results.append(EvaluationResult(
                path=str(md_file.relative_to(VAULT_ROOT)),
                action="skip",
                score=0,
                reason=f"Evaluation error: {e}",
            ))

    summary = {
        "evaluated": len(results),
        "promoted": sum(1 for r in results if r.action == "promote"),
        "returned": sum(1 for r in results if r.action == "return"),
        "escalated": sum(1 for r in results if r.action == "escalate"),
        "skipped": sum(1 for r in results if r.action == "skip"),
        "duplicates_skipped": duplicates_skipped,
        "details": [
            {
                "path": r.path, "action": r.action, "score": round(r.score, 1),
                "reason": r.reason, "explanation": r.explanation,
            }
            for r in results
        ],
    }
    # Record pipeline health metrics
    try:
        from hololoom.apps.server.math_extensions_v2 import get_pipeline_health
        health = get_pipeline_health()
        if health:
            summary["pipeline_health"] = health
    except Exception:
        pass

    # v5 deep math: Kalman filter all scores, detect regime shifts, stability
    all_scores = [r.score for r in results if r.score > 0]
    if len(all_scores) >= 3:
        try:
            from hololoom.apps.server.math_extensions_v5 import (
                detect_quality_regime_shift,
                evaluation_diversity,
                kalman_score_filter,
                pipeline_stability_report,
                score_tail_analysis,
            )

            # Kalman-filtered scores (denoise LLM noise)
            filtered = kalman_score_filter(all_scores)
            summary["kalman_filtered_scores"] = [round(s, 1) for s in filtered]

            # Regime shift detection
            regime = detect_quality_regime_shift(all_scores)
            if regime:
                summary["regime_analysis"] = regime

            # Stability report
            stability = pipeline_stability_report(all_scores)
            if stability:
                summary["stability"] = stability

            # Tail risk
            tails = score_tail_analysis(all_scores)
            if tails:
                summary["tail_risk"] = tails

            # Evaluation diversity (are we scoring across topics or stuck?)
            score_dist = np.array(all_scores)
            score_dist = np.histogram(score_dist, bins=10, range=(0, 100))[0].astype(float)
            diversity = evaluation_diversity(score_dist)
            if diversity:
                summary["diversity"] = diversity

        except Exception as e:
            logger.debug("v5 deep math unavailable: %s", e)

    return summary


async def _evaluate_single(
    md_file: Path,
    dept: str,
    rubric: list[RubricTerm],
    model_router,
    call_model,
) -> EvaluationResult:
    """Evaluate a single candidate note."""
    content = md_file.read_text(encoding="utf-8", errors="replace")
    rel_path = str(md_file.relative_to(VAULT_ROOT)).replace("\\", "/")

    # Step 1: Mechanical pre-checks
    mech = mechanical_scores(content, rubric)
    mech_issues = mechanical_feedback(mech)

    # If mechanical checks fail hard, return immediately (no LLM call needed)
    mech_total = sum(mech.values()) / max(1, len(mech))
    if mech_total < 30:
        feedback_text = "## Head Feedback\n\n" + "\n".join(f"- {f}" for f in mech_issues)
        _append_feedback(md_file, feedback_text)
        return EvaluationResult(
            path=rel_path,
            action="return",
            score=mech_total,
            reason="Failed mechanical pre-checks",
            feedback="; ".join(mech_issues),
        )

    # Step 1b: Hensel lifting — decide if LLM call is necessary
    # If mechanical score is far from all thresholds, the coarse check
    # is conclusive and the finer LLM evaluation will agree (Hensel guarantee)
    try:
        from hololoom.apps.server.math_extensions_v3 import hensel_lift_decision
        strategy_name, strategy_thresholds = select_strategy()
        lift = hensel_lift_decision(
            mech_total,
            promote_threshold=strategy_thresholds.get("promote", PROMOTE_THRESHOLD),
            return_threshold=strategy_thresholds.get("return", RETURN_THRESHOLD),
        )
        if lift.skip_llm and mech_total >= strategy_thresholds.get("promote", PROMOTE_THRESHOLD):
            # Mechanical score is conclusively above promote threshold
            result = vault_bridge.promote_within_department(dept, rel_path, "observations")
            promoted_path = result.get("path", rel_path)
            asyncio.create_task(_trigger_gate(dept, promoted_path))
            logger.info("Hensel skip (promote): %s score=%.0f (%s)", md_file.name, mech_total, lift.reason)
            return EvaluationResult(
                path=promoted_path, action="promote", score=mech_total,
                reason=f"Hensel lift: {lift.reason}",
            )
        elif lift.skip_llm and mech_total <= strategy_thresholds.get("return", RETURN_THRESHOLD):
            # Mechanical score is conclusively below return threshold
            logger.info("Hensel skip (escalate): %s score=%.0f (%s)", md_file.name, mech_total, lift.reason)
            escalated_dir = VAULT_ROOT / "00_Inbox" / f"{dept}-escalated"
            escalated_dir.mkdir(parents=True, exist_ok=True)
            dest = escalated_dir / md_file.name
            shutil.move(str(md_file), str(dest))
            return EvaluationResult(
                path=str(dest.relative_to(VAULT_ROOT)).replace("\\", "/"),
                action="escalate", score=mech_total,
                reason=f"Hensel lift: {lift.reason}",
            )
        # else: lift.skip_llm is False, proceed to LLM evaluation
    except Exception:
        pass  # Hensel unavailable, proceed normally

    # Step 2: Gather context for LLM evaluation
    body = _strip_frontmatter(content)
    first_sentence = body.split(".")[0] if body else ""

    dept_context = vault_bridge.search(first_sentence, top_k=5)
    vault_context = vault_bridge.search(first_sentence, top_k=3)

    # Step 2b: Check for contradictions (uses 9b, fast)
    from hololoom.apps.server.contradiction import (
        check_contradictions,
        check_logical_consistency,
        format_contradiction_warnings,
    )

    contradiction_context = ""
    try:
        signals = await check_contradictions(
            content, dept, model_router, call_model,
        )
        if signals:
            contradiction_context = format_contradiction_warnings(signals)
            logger.info(
                "Found %d contradiction(s) for %s", len(signals), md_file.name,
            )
    except Exception as e:
        logger.warning("Contradiction check failed for %s: %s", md_file.name, e)

    # Step 2c: SAT-based logical consistency check (complements embedding check)
    try:
        logic_warning = await check_logical_consistency(content, dept)
        if logic_warning:
            contradiction_context += logic_warning
    except Exception as e:
        logger.debug("Logic consistency check unavailable: %s", e)

    # Step 3: Route to 30b model
    try:
        decision = await model_router.route(
            first_sentence, speed_weight=0.3, require_tags={"mid"},
        )
        model_spec = decision.model
    except Exception as e:
        logger.warning("Model routing failed, skipping LLM evaluation: %s", e)
        return EvaluationResult(
            path=rel_path,
            action="skip",
            score=0,
            reason=f"30b model unavailable: {e}",
        )

    # Step 4: LLM evaluation with retry on garbled JSON
    messages = _build_evaluation_prompt(
        content, rubric, dept_context, vault_context, contradiction_context,
    )
    llm_scores = None

    for attempt in range(2):  # retry once
        try:
            response = await call_model(messages, model_spec, temperature=0.3)
            response_text = response.get("message", {}).get("content", "")
            parsed = _parse_llm_response(response_text)
            if parsed and "scores" in parsed:
                llm_scores = parsed
                break
            logger.warning(
                "Attempt %d: garbled JSON from Head (model=%s): %s",
                attempt + 1, model_spec.id, response_text[:200],
            )
            if attempt == 0:
                # Retry with stricter prompt
                messages[-1]["content"] += (
                    "\n\nIMPORTANT: Respond with ONLY valid JSON. "
                    "No explanation, no markdown, just the JSON object."
                )
        except Exception as e:
            logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)

    # If LLM failed after retry, return with raw feedback
    if llm_scores is None:
        feedback_text = "## Head Feedback\n\nEvaluation could not complete — LLM response was unparseable. Note stays in inbox for next review cycle."
        _append_feedback(md_file, feedback_text)
        return EvaluationResult(
            path=rel_path,
            action="return",
            score=0,
            reason="LLM response unparseable after retry",
        )

    # Step 5: Compute weighted score (with Thompson Sampling adjustment)
    final_score = compute_weighted_score(mech, llm_scores["scores"], rubric, dept=dept)
    reason = llm_scores.get("reason", "")
    feedback = llm_scores.get("feedback", "")

    # Step 5b: Generate causal explanation
    explanation = None
    try:
        from hololoom.apps.server.promotion_causal import feature_importance, trace_causal_chain

        observation = {**mech, **llm_scores.get("scores", {})}
        observation["mechanical_score"] = sum(mech.values()) / max(1, len(mech))
        observation["llm_score"] = sum(llm_scores["scores"].values()) / max(1, len(llm_scores["scores"]))

        action_name = (
            "promote" if final_score >= PROMOTE_THRESHOLD
            else "return" if final_score >= RETURN_THRESHOLD
            else "escalate"
        )

        causal = trace_causal_chain(observation, action_name, final_score)
        importance = feature_importance(
            observation,
            {t.name: t.weight for t in rubric},
        )

        explanation = {
            "action": causal.action,
            "primary_causes": causal.primary_causes,
            "counterfactuals": causal.counterfactuals,
            "chain": [
                {"var": link.variable, "value": round(link.value, 1), "contribution": link.contribution}
                for link in causal.causal_chain
            ],
            "feature_importance": [{"feature": f, "importance": imp} for f, imp in importance[:5]],
        }
    except Exception as e:
        logger.debug("Causal explanation unavailable: %s", e)

    # Step 6: Select evaluation strategy via Hedge (adjusts thresholds)
    strategy_name, strategy_thresholds = select_strategy()
    promote_thresh = strategy_thresholds.get("promote", PROMOTE_THRESHOLD)
    return_thresh = strategy_thresholds.get("return", RETURN_THRESHOLD)

    # Take action based on score + strategy thresholds
    if final_score >= promote_thresh:
        result = vault_bridge.promote_within_department(dept, rel_path, "observations")
        promoted_path = result.get("path", rel_path)

        # Trigger async gate check for vault promotion (SecOps + EA)
        asyncio.create_task(_trigger_gate(dept, promoted_path))

        return EvaluationResult(
            path=promoted_path,
            action="promote",
            score=final_score,
            reason=reason,
            explanation=explanation,
        )
    elif final_score >= return_thresh:
        all_feedback = []
        if mech_issues:
            all_feedback.extend(mech_issues)
        if feedback:
            all_feedback.append(feedback)
        feedback_text = (
            f"## Head Feedback\n\n"
            f"Score: {final_score:.0f}/100 (needs {promote_thresh} to promote, strategy: {strategy_name})\n\n"
            + "\n".join(f"- {f}" for f in all_feedback)
        )
        _append_feedback(md_file, feedback_text)
        return EvaluationResult(
            path=rel_path,
            action="return",
            score=final_score,
            reason=reason,
            feedback="; ".join(all_feedback),
            explanation=explanation,
        )
    else:
        # Score < 40: escalate to human (note is too weak for dept but maybe
        # the worker's intent is worth human review)
        escalated_dir = VAULT_ROOT / "00_Inbox" / f"{dept}-escalated"
        escalated_dir.mkdir(parents=True, exist_ok=True)
        dest = escalated_dir / md_file.name
        shutil.move(str(md_file), str(dest))
        escalated_rel = str(dest.relative_to(VAULT_ROOT)).replace("\\", "/")
        logger.info("Escalated %s to %s (score=%.0f)", rel_path, escalated_rel, final_score)
        return EvaluationResult(
            path=escalated_rel,
            action="escalate",
            score=final_score,
            reason=reason,
            feedback=feedback,
        )


def _append_feedback(md_file: Path, feedback_text: str) -> None:
    """Append feedback section to a note file."""
    try:
        existing = md_file.read_text(encoding="utf-8", errors="replace")
        # Don't stack feedback sections — replace if one exists
        if "## Head Feedback" in existing:
            existing = re.sub(
                r'## Head Feedback\n.*?(?=\n## |\Z)',
                '',
                existing,
                flags=re.DOTALL,
            ).rstrip()
        md_file.write_text(
            existing.rstrip() + "\n\n" + feedback_text + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("Failed to append feedback to %s: %s", md_file, e)


async def _trigger_gate(dept: str, note_path: str) -> None:
    """Trigger the oversight gate asynchronously for vault promotion.

    Note is already in dept PARA (trusted at department level).
    Gate decides whether it also goes to the vault.
    Runs in background — never blocks the Head's evaluation loop.
    """
    try:
        from hololoom.apps.server.oversight import gate_check
        result = await gate_check(dept=dept, note_path=note_path)
        logger.info(
            "Gate result for %s: %s — %s",
            note_path, result.get("action"), result.get("reason", ""),
        )
    except Exception as e:
        logger.warning(
            "Gate check failed for %s (note stays in dept PARA): %s",
            note_path, e,
        )
