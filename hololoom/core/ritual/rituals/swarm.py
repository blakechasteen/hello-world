"""
Swarm rituals — cross-agent weekly digest.

weekly_digest: TEMPORAL_WEEKLY — each domain agent contributes a summary,
MasterWeaver merges into a CrossDomainInsightBobbin.
"""


from ..liturgy import LiturgySpec
from ..policies import LiturgyFailurePolicy
from ..types import RitualResult


def _merge_domain_summaries(results: dict[str, RitualResult]) -> RitualResult:
    """
    Merge domain summaries from all participants into a cross-domain insight.
    MasterWeaver uses this to identify cross-cutting patterns.
    """
    summaries = {}
    for role, result in results.items():
        summaries[role] = result.produced.get("DomainSummaryBobbin", {})

    cross_domain = {
        "summaries": summaries,
        "participant_count": len(results),
        "cross_cutting_patterns": [],  # wire to pattern detection
    }

    return RitualResult(
        success=True,
        produced={"CrossDomainInsightBobbin": cross_domain},
        confidence_delta=0.0,
        notes=f"Weekly digest merged from {len(results)} participants.",
    )


WEEKLY_DIGEST = LiturgySpec(
    name="weekly_digest",
    participants=["terragraph", "beekeeping_weaverlet", "sous"],
    quorum=0.5,
    sync_point="TEMPORAL_WEEKLY",
    contributions={
        "terragraph": "DomainSummaryBobbin",
        "beekeeping_weaverlet": "DomainSummaryBobbin",
        "sous": "DomainSummaryBobbin",
    },
    consensus_output="CrossDomainInsightBobbin",
    failure_policy=LiturgyFailurePolicy.PARTIAL_QUORUM,
    merge_fn=_merge_domain_summaries,
)
