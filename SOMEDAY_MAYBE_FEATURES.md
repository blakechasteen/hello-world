# Someday/Maybe Features - Deferred/Overkill

**Status**: Deferred
**Reason**: Good ideas but overkill for current scope, or unclear ROI.

This document tracks features from the original proposal that are **NOT** included in [AGENTIC_INTEGRATION_PROPOSAL.md](AGENTIC_INTEGRATION_PROPOSAL.md) but may be valuable later.

---

## Category 1: Advanced Embedding Verification (Overkill)

### 1A. Nightly Auto-Ablation

**Original Proposal**:
> Nightly auto-ablation (stop-wording, chunk size, overlap) to find better chunking regimes.

**Why Deferred**:
- **Complexity**: Requires automated experiment management, result tracking, statistical significance testing
- **Unclear ROI**: Chunking parameters are relatively stable; manual tuning is sufficient
- **Resource intensive**: Nightly full re-embedding + quality evaluation is expensive
- **Better alternative**: Manual A/B testing when performance issues identified

**Reconsider When**:
- Performance degradation observed across multiple datasets
- Team has dedicated MLOps infrastructure
- Clear hypothesis about chunking improvements

**Estimated Effort**: 3-4 weeks (experiment framework + analysis)

---

### 1B. PII Guardrails Before Embedding

**Original Proposal**:
> PII guardrail before embedding; hash+mask pipeline with opt-out tags.

**Why Deferred**:
- **Scope creep**: PII detection is a separate compliance concern
- **Domain-specific**: Requirements vary wildly (GDPR, HIPAA, internal policies)
- **Better handled upstream**: PII should be removed at data ingestion, not embedding time
- **Alternative exists**: Use external tools (Presidio, spaCy NER) before SpinningWheel

**Reconsider When**:
- Specific compliance requirement emerges
- Processing user-generated content with PII
- Legal/security team mandates embedding-level protection

**Estimated Effort**: 2-3 weeks (detection + masking pipeline)

**Recommended Approach** (if needed):
```python
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()

def scrub_pii(text: str) -> str:
    results = analyzer.analyze(text, entities=["EMAIL", "PHONE", "SSN"])
    # Mask or hash detected PII
    return masked_text

# Use before embedding
texts = [scrub_pii(s.text) for s in shards]
embeddings = embedder.encode(texts)
```

---

### 1C. Mahalanobis Distance Outlier Detection

**Original Proposal**:
> OOD score (e.g., Mahalanobis distance in embedding space) → quarantine bucket.

**Why Deferred**:
- **Statistical complexity**: Requires estimating covariance matrix, handling high dimensions
- **Noisy in practice**: Embeddings don't follow clean Gaussian distributions
- **Simpler alternatives exist**: Cosine similarity to cluster centroids is more interpretable
- **Unclear what to do with outliers**: Quarantine criteria poorly defined

**Reconsider When**:
- Adversarial inputs become a real threat
- Clear procedure for handling quarantined items
- Statistical foundation validated on real data

**Estimated Effort**: 2 weeks (implementation + validation)

**Simpler Alternative** (use this instead):
```python
# Detect outliers via cosine similarity to k-nearest neighbors
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=10).fit(embeddings)
distances, _ = nbrs.kneighbors(embeddings)
avg_distances = distances.mean(axis=1)

# Flag items far from neighbors
threshold = np.percentile(avg_distances, 95)
outliers = np.where(avg_distances > threshold)[0]
```

---

### 1D. Blue/Green Index Deployments

**Original Proposal**:
> Blue/green index deploy: build index_next, run smoke tests, cutover pointer, keep index_prev for rollback.

**Why Deferred**:
- **Premature optimization**: No evidence of deployment issues yet
- **Infrastructure burden**: Requires index versioning, cutover logic, rollback procedures
- **Better served by backups**: Simple index snapshots + restore is sufficient
- **Qdrant/Neo4j have their own HA**: Use native replication instead

**Reconsider When**:
- Embedding updates cause production outages
- Zero-downtime requirement emerges
- Managing multiple production indexes

**Estimated Effort**: 3 weeks (versioning + deployment automation)

**Simpler Alternative**:
```bash
# Snapshot-based rollback
docker exec qdrant-container /bin/bash -c "qdrant-cli snapshot create"
# If issues: restore snapshot
docker exec qdrant-container /bin/bash -c "qdrant-cli snapshot restore <id>"
```

---

### 1E. Weekly Automated "Vector Health Reports"

**Original Proposal**:
> Weekly "Vector Health Report" auto-posted to ReflectionCore with pass/fail gates.

**Why Deferred**:
- **Manual-first approach**: Better to run reports manually until workflow proven
- **Unclear consumer**: Who reads these reports? What actions do they trigger?
- **Notification fatigue**: Automated reports often ignored
- **Better alternative**: On-demand dashboards + alerts for critical issues

**Reconsider When**:
- Team routinely runs manual health checks
- Clear action items from reports
- Stakeholder demand for regular updates

**Estimated Effort**: 1 week (templating + automation)

**Alternative** (manual cron):
```bash
# Run weekly via cron
0 0 * * 0 python scripts/generate_vector_health_report.py --email team@example.com
```

---

## Category 2: Advanced Agentic Features (Defer Until Proven)

### 2A. Multi-Agent Debate/Verification

**Original Proposal** (implied):
> Multiple agents verify each other's outputs, debate contradictions.

**Why Deferred**:
- **Complexity explosion**: Coordinating multiple agents, resolving conflicts
- **Latency**: N agents → N× latency
- **Unclear benefit**: Single agent with verification loop is simpler and faster
- **Research territory**: Lacks established best practices

**Reconsider When**:
- Single-agent verification proves insufficient
- Research on multi-agent verification matures
- Latency is acceptable (>5s per query)

**Estimated Effort**: 4-6 weeks (coordination protocol + evaluation)

---

### 2B. Semantic Diff for Embedding Migrations

**Original Proposal**:
> Migration playbook for dimension changes (shadow-write both; dual-query A/B for a week).

**Why Deferred**:
- **Rare scenario**: Dimension changes are infrequent (model upgrades)
- **Complex orchestration**: Dual-write logic, query routing, result comparison
- **Better served by regression tests**: Use gold set validation instead

**Reconsider When**:
- Frequent model/dimension changes
- High-risk production environment
- Team has A/B testing infrastructure

**Estimated Effort**: 2-3 weeks (shadow-write + analysis)

**Simpler Alternative**:
```python
# Test new dimensions on gold set before cutover
metrics_old = compute_quality_metrics(embedder_old, gold_set)
metrics_new = compute_quality_metrics(embedder_new, gold_set)

# Compare before switching
assert metrics_new.recall_at_5 >= metrics_old.recall_at_5 * 0.95
```

---

### 2C. Goal Branch/Merge for Contradictory Evidence

**Original Proposal**:
> Branch/merge rules when new evidence contradicts prior embeddings.

**Why Deferred**:
- **Conceptually unclear**: What does "branch" mean for embeddings?
- **Version control for knowledge?**: Interesting idea but no proven design
- **Better handled by**: Temporal versioning (keep old + new, let retrieval decide)

**Reconsider When**:
- Clear use case for knowledge versioning
- Research on temporal knowledge graphs advances
- Team has bandwidth for exploratory work

**Estimated Effort**: 6-8 weeks (design + implementation)

---

### 2D. Fine-Grained Human-in-the-Loop Controls

**Original Proposal**:
> Autonomy dial affects: query expansion radius, allowable domains, and cutover from draft to publish.

**Why Deferred**:
- **UI complexity**: Requires sliders, domain allowlists, approval workflows
- **Premature**: Build autonomous system first, add controls when issues emerge
- **Better via config**: Simple config flags are sufficient initially

**Reconsider When**:
- Autonomous actions cause issues
- Multiple users with different autonomy needs
- UI/UX resources available

**Estimated Effort**: 3-4 weeks (UI + backend controls)

---

## Category 3: Meta-Learning & Advanced Analytics (Research)

### 3A. Citation Integrity Scores

**Original Proposal**:
> Citation Integrity Score = redundancy across sources × agreement with gold/constraints.

**Why Deferred**:
- **Metric design unclear**: How to compute "agreement with gold"?
- **Requires gold constraints**: Need labeled dataset of valid/invalid citations
- **Research territory**: No established formula

**Reconsider When**:
- Citation quality becomes an issue
- Gold dataset available
- Clear definition of "integrity"

**Estimated Effort**: 4-6 weeks (metric design + validation)

---

### 3B. Feature Attribution for Failed Queries

**Original Proposal**:
> Log feature attributions for failed queries; feed misfires into next fine-tuning/eval set.

**Why Deferred**:
- **Requires fine-tuning**: HoloLoom uses pre-trained models, not fine-tuned
- **Explainability complexity**: Feature attribution is non-trivial for embeddings
- **Better via inspection**: Manual analysis of failures is more actionable

**Reconsider When**:
- Fine-tuning embeddings becomes standard
- Clear pipeline for using misfires
- SHAP/LIME tooling integrated

**Estimated Effort**: 3-4 weeks (attribution + pipeline)

---

### 3C. Isotonic/Platt Calibration for Cosine→Confidence

**Original Proposal**:
> Cosine→confidence calibration curve (isotonic or Platt on validation pairs).

**Why Deferred**:
- **Already have confidence**: From policy network output
- **Unclear improvement**: Cosine similarity doesn't need calibration for ranking
- **Better for classification**: Calibration is for binary/multi-class prediction

**Reconsider When**:
- Need probabilistic confidence from retrieval alone (no policy)
- Validation pairs available
- Calibration improves downstream decisions

**Estimated Effort**: 1-2 weeks (calibration + evaluation)

---

## Category 4: Infrastructure (Premature)

### 4A. UMAP/PCA Snapshots per Run

**Original Proposal**:
> UMAP/PCA snapshots per run with anchors highlighted; store as artifacts.

**Why Deferred**:
- **Storage overhead**: Large embedding spaces → large visualizations
- **Unclear value**: Who looks at these regularly?
- **On-demand is better**: Generate visualizations when debugging issues

**Reconsider When**:
- Regular need for embedding visualization
- Storage is cheap
- Automated anomaly detection via visual inspection

**Estimated Effort**: 1 week (visualization pipeline)

**Alternative** (on-demand):
```python
from sklearn.manifold import UMAP
import matplotlib.pyplot as plt

# Generate when needed
reducer = UMAP(n_components=2)
embedding_2d = reducer.fit_transform(embeddings)
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1])
plt.savefig(f"embedding_viz_{run_id}.png")
```

---

### 4B. Cross-Run Regression Test Suite

**Original Proposal**:
> V2 must meet or beat V1 on all primary metrics.

**Why Deferred**:
- **Partially covered**: Quality metrics (Phase 2) handle this
- **"All metrics" too strict**: Some tradeoffs are acceptable
- **Manual review needed**: Automated pass/fail is brittle

**Reconsider When**:
- Frequent breaking regressions
- Clear primary/secondary metric hierarchy
- CI/CD pipeline mature

**Estimated Effort**: 1-2 weeks (test harness)

**Note**: Phase 2 quality metrics provide foundation - this is just formalization.

---

## Summary: What We're NOT Building (And Why)

| Feature | Reason | Reconsider When |
|---------|--------|-----------------|
| Nightly auto-ablation | Unclear ROI, resource intensive | Performance issues emerge |
| PII guardrails | Scope creep, domain-specific | Compliance requirement |
| Mahalanobis outliers | Statistical complexity, noisy | Adversarial threats real |
| Blue/green deploys | Premature, use snapshots | Zero-downtime needed |
| Weekly auto-reports | Manual-first, unclear consumer | Reports proven valuable |
| Multi-agent debate | Complexity explosion | Single-agent insufficient |
| Semantic diff migration | Rare scenario, complex | Frequent model changes |
| Goal branch/merge | Conceptually unclear | Use case emerges |
| HITL autonomy dial | Premature UI work | Autonomous issues arise |
| Citation integrity | Metric unclear | Quality issues emerge |
| Feature attribution | No fine-tuning pipeline | Fine-tuning standard |
| Calibration curves | Better alternatives exist | Probabilistic needs |
| UMAP auto-snapshots | Storage overhead | Regular need proven |

---

## Guiding Principle

> "Build the simplest thing that could possibly work. Add complexity only when proven necessary through real usage."

- Start with **Phase 1-3** from [AGENTIC_INTEGRATION_PROPOSAL.md](AGENTIC_INTEGRATION_PROPOSAL.md)
- **Validate** with real usage for 3-6 months
- **Revisit** this document based on actual pain points

---

## How to Use This Document

1. **Before adding features**: Check if it's in SOMEDAY_MAYBE
2. **If yes**: Understand why it was deferred, validate conditions
3. **If no**: Consider adding it here first, get team alignment
4. **Quarterly review**: Revisit deferred features, promote if conditions met

---

## Related Documents

- [AGENTIC_INTEGRATION_PROPOSAL.md](AGENTIC_INTEGRATION_PROPOSAL.md) - What we ARE building
- [ALIGNMENT_FRAMEWORK_INTEGRATION.md](ALIGNMENT_FRAMEWORK_INTEGRATION.md) - Safety framework
- [RECURSIVE_LEARNING_COMPLETE.md](RECURSIVE_LEARNING_COMPLETE.md) - Existing learning system