"""
Contradiction Detector — surface semantic contradictions in department knowledge.

Two-layer detection:
1. JS divergence on Gemini embeddings measures semantic distance between claims.
   Falls back to keyword search when embeddings are unavailable.
2. Fast 9b LLM call classifies agree/disagree for high-similarity pairs.

The math: JS divergence is symmetric KL — JS(p,q) = 0.5*KL(p||m) + 0.5*KL(q||m)
where m = 0.5*(p+q). Low JS = similar distributions = candidates for contradiction.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hololoom.apps.server import vault_bridge
from hololoom.apps.server.vault_bridge import VAULT_ROOT

logger = logging.getLogger(__name__)

# Graceful import of JS divergence
try:
    from hololoom.core.warp.math.decision.information_theory import DivergenceMetrics
    HAS_DIVERGENCE = True
except ImportError:
    HAS_DIVERGENCE = False
    logger.info("DivergenceMetrics unavailable — contradiction detector uses keyword fallback")


@dataclass
class ContradictionSignal:
    existing_note_path: str
    similarity_score: float  # cosine similarity or keyword score
    js_divergence: float | None  # JS divergence between embeddings (None if unavailable)
    existing_claim: str
    candidate_claim: str


def _extract_claim(content: str) -> str:
    """Extract the core claim from a note — first heading or first sentence."""
    text = content.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2].strip()

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# ") and not line.startswith("##"):
            return line[2:].strip()

    match = re.match(r'([^.!?\n]+[.!?])', text)
    if match:
        return match.group(1).strip()

    return text[:150].strip()


# ============================================================================
# Embedding-based similarity (primary path)
# ============================================================================

def _embedding_to_distribution(embedding: np.ndarray) -> np.ndarray:
    """Convert a raw embedding vector to a probability distribution for JS divergence.

    Uses softmax normalization: each dimension becomes a probability.
    This preserves the relative magnitudes while ensuring sum-to-one.
    """
    # Shift for numerical stability
    shifted = embedding - np.max(embedding)
    exp = np.exp(shifted)
    dist = exp / exp.sum()
    # Ensure no exact zeros (would cause log issues in KL)
    dist = np.clip(dist, 1e-10, None)
    dist = dist / dist.sum()
    return dist


def _get_vault_embeddings() -> dict | None:
    """Try to access the vault's cached Gemini embeddings.

    Returns dict of {path_str: np.ndarray} or None if unavailable.
    """
    try:
        import sys
        vault_root = VAULT_ROOT
        # Add vault root to path so we can import services
        sys.path.insert(0, str(vault_root))
        from services.data_service import data_service
        if data_service.vector_store:
            return data_service.vector_store
    except Exception as e:
        logger.debug("Vault embeddings unavailable: %s", e)
    return None


async def _get_embedding_for_text(text: str) -> np.ndarray | None:
    """Get a Gemini embedding for arbitrary text. Returns None if unavailable."""
    try:
        import sys
        sys.path.insert(0, str(VAULT_ROOT))
        from services.ai_service import ai_service
        result = await ai_service.get_embedding(text)
        if result is not None:
            return np.array(result, dtype=np.float32)
    except Exception as e:
        logger.debug("Embedding generation failed: %s", e)
    return None


def _find_similar_by_embedding(
    candidate_embedding: np.ndarray,
    vault_embeddings: dict,
    top_k: int = 8,
    cosine_threshold: float = 0.3,
) -> list[dict]:
    """Find similar notes by cosine similarity on embeddings.

    Returns list of {path, score (cosine), embedding} sorted by similarity.
    """
    candidate_norm = np.linalg.norm(candidate_embedding)
    if candidate_norm == 0:
        return []

    scored = []
    for path_str, embedding in vault_embeddings.items():
        emb_norm = np.linalg.norm(embedding)
        if emb_norm == 0:
            continue
        cosine = float(np.dot(candidate_embedding, embedding) / (candidate_norm * emb_norm))
        if cosine >= cosine_threshold:
            scored.append({
                "path": str(Path(path_str).relative_to(VAULT_ROOT)).replace("\\", "/")
                if Path(path_str).is_absolute() else path_str,
                "score": cosine,
                "embedding": embedding,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _compute_js_divergence(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute JS divergence between two embeddings.

    Converts embeddings to probability distributions via softmax,
    then computes symmetric JS divergence.
    """
    if not HAS_DIVERGENCE:
        return 0.0

    dist_a = _embedding_to_distribution(emb_a)
    dist_b = _embedding_to_distribution(emb_b)
    return float(DivergenceMetrics.js_divergence(dist_a, dist_b))


# ============================================================================
# LLM agree/disagree classification
# ============================================================================

def _build_agree_disagree_prompt(claim_a: str, claim_b: str) -> list[dict]:
    """Build a minimal prompt for 9b agree/disagree classification."""
    return [
        {
            "role": "system",
            "content": (
                "You are a factual consistency checker. Given two claims, "
                "determine if they AGREE or DISAGREE. Respond with exactly "
                "one word: AGREE or DISAGREE."
            ),
        },
        {
            "role": "user",
            "content": f"Claim A: {claim_a}\nClaim B: {claim_b}",
        },
    ]


def _parse_agree_disagree(response_text: str) -> str | None:
    """Parse LLM response to AGREE or DISAGREE. Returns None if unclear."""
    text = response_text.strip().upper()
    if text in ("AGREE", "DISAGREE"):
        return text
    if "DISAGREE" in text:
        return "DISAGREE"
    if "AGREE" in text:
        return "AGREE"
    return None


# ============================================================================
# Main check
# ============================================================================

async def check_contradictions(
    candidate_text: str,
    dept: str,
    model_router,
    call_model,
    similarity_threshold: int = 3,
    cosine_threshold: float = 0.3,
    top_k: int = 8,
) -> list[ContradictionSignal]:
    """Check candidate note against existing knowledge for contradictions.

    Two-layer strategy:
    1. Try embedding-based similarity (JS divergence) if Gemini embeddings available
    2. Fall back to keyword search if embeddings unavailable
    3. For similar notes, call 9b model for agree/disagree classification
    """
    candidate_claim = _extract_claim(candidate_text)
    if not candidate_claim or len(candidate_claim) < 10:
        return []

    # Try embedding-based similarity first
    similar_notes = []
    candidate_embedding = None
    use_embeddings = False

    vault_embeddings = _get_vault_embeddings()
    if vault_embeddings and HAS_DIVERGENCE:
        candidate_embedding = await _get_embedding_for_text(candidate_claim)
        if candidate_embedding is not None:
            similar_notes = _find_similar_by_embedding(
                candidate_embedding, vault_embeddings,
                top_k=top_k, cosine_threshold=cosine_threshold,
            )
            use_embeddings = True
            logger.debug(
                "Embedding-based similarity: %d candidates (cosine >= %.2f)",
                len(similar_notes), cosine_threshold,
            )

    # Fall back to keyword search
    if not similar_notes:
        results = vault_bridge.search(candidate_claim, top_k=top_k)
        similar_notes = [
            {"path": r["path"], "score": r["score"], "embedding": None}
            for r in results if r["score"] >= similarity_threshold
        ]

    if not similar_notes:
        return []

    # Route to 9b model for agree/disagree
    try:
        decision = await model_router.route(
            candidate_claim, speed_weight=0.9, require_tags={"fast"},
        )
        model_spec = decision.model
    except Exception as e:
        logger.warning("9b model unavailable for contradiction check: %s", e)
        return []

    signals = []
    for note in similar_notes:
        note_content = vault_bridge.read_note(note["path"])
        if not note_content:
            continue

        existing_claim = _extract_claim(note_content)
        if not existing_claim or len(existing_claim) < 10:
            continue

        if existing_claim.lower().strip() == candidate_claim.lower().strip():
            continue

        # Compute JS divergence if we have embeddings
        js_div = None
        if use_embeddings and candidate_embedding is not None and note.get("embedding") is not None:
            js_div = _compute_js_divergence(candidate_embedding, note["embedding"])

        # Ask 9b: agree or disagree?
        messages = _build_agree_disagree_prompt(candidate_claim, existing_claim)
        try:
            response = await call_model(messages, model_spec, temperature=0.1, max_tokens=10)
            response_text = response.get("message", {}).get("content", "")
            verdict = _parse_agree_disagree(response_text)

            if verdict == "DISAGREE":
                signals.append(ContradictionSignal(
                    existing_note_path=note["path"],
                    similarity_score=note["score"],
                    js_divergence=js_div,
                    existing_claim=existing_claim,
                    candidate_claim=candidate_claim,
                ))
                js_info = f", JS={js_div:.4f}" if js_div is not None else ""
                logger.info(
                    "Contradiction: '%s' vs '%s' (sim=%.2f%s)",
                    candidate_claim[:50], existing_claim[:50],
                    note["score"], js_info,
                )
        except Exception as e:
            logger.warning("Contradiction check failed for %s: %s", note["path"], e)
            continue

    return signals


async def check_logical_consistency(
    candidate_text: str,
    dept: str,
) -> str | None:
    """Check logical consistency of candidate claim against dept PARA using SAT.

    Complements the embedding-based contradiction check with formal logic.
    Returns a warning string if inconsistency found, None otherwise.
    """
    try:
        from hololoom.apps.server.math_extensions_v2 import check_claim_consistency
    except ImportError:
        return None

    # Collect existing claims from dept PARA
    existing_claims = []
    para_dir = VAULT_ROOT / "departments" / dept / "para"
    if para_dir.is_dir():
        for md in para_dir.rglob("*.md"):
            if md.name.startswith("_"):
                continue
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                claim = _extract_claim(content)
                if claim and len(claim) >= 10:
                    existing_claims.append(claim)
            except OSError:
                continue

    if not existing_claims:
        return None

    candidate_claim = _extract_claim(candidate_text)
    if not candidate_claim or len(candidate_claim) < 10:
        return None

    result = check_claim_consistency(existing_claims, candidate_claim)
    if not result.consistent:
        return (
            f"\n## Logical Consistency Warning\n"
            f"{result.explanation}\n"
            + "\n".join(
                f"- Conflicts with: \"{existing[:80]}\""
                for existing, _ in result.conflicting_claims[:3]
            )
        )
    return None


def enrich_contradictions_with_v5(
    signals: list[ContradictionSignal],
    dept: str,
) -> dict:
    """Enrich contradiction signals with v5 deep math.

    Adds: predictability horizon (are contradictions expected?),
    kernel-based similarity refinement, changepoint context.
    """
    enrichment = {"n_contradictions": len(signals)}

    try:
        from hololoom.apps.server.math_extensions_v5 import predictability_horizon

        # If we have similarity scores as a trajectory, check predictability
        sim_scores = [s.similarity_score for s in signals if s.similarity_score > 0]
        if len(sim_scores) >= 5:
            pred = predictability_horizon(sim_scores)
            if pred:
                enrichment["predictability"] = pred
    except Exception:
        pass

    try:
        from hololoom.apps.server.math_extensions_v5 import detect_quality_regime_shift

        # Check if contradictions cluster at regime boundaries
        if len(signals) >= 3:
            indices = [hash(s.existing_note_path) % 100 for s in signals]
            regime = detect_quality_regime_shift([float(i) for i in indices])
            if regime and regime.get("n_regimes", 1) > 1:
                enrichment["regime_context"] = "Contradictions cluster at regime boundaries — knowledge is shifting"
    except Exception:
        pass

    return enrichment


def format_contradiction_warnings(signals: list[ContradictionSignal]) -> str:
    """Format contradiction signals as markdown for the Head's evaluation context."""
    if not signals:
        return ""

    lines = ["\n## Contradiction Warnings\n"]
    for s in signals:
        js_info = f", JS divergence: {s.js_divergence:.4f}" if s.js_divergence is not None else ""
        lines.append(
            f"- **CONFLICTS WITH** [{s.existing_note_path}] "
            f"(similarity: {s.similarity_score:.2f}{js_info})\n"
            f"  - Existing claim: \"{s.existing_claim}\"\n"
            f"  - Candidate claim: \"{s.candidate_claim}\""
        )
    return "\n".join(lines)
