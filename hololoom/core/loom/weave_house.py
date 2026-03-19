"""
WeaveHouse - The Collective of Looms
====================================

A composite loom that orchestrates multiple perspectives with consensus
and auto-exploration of disagreement zones.

Philosophy:
"Each loom sees different truths. Together, they weave a fabric
richer than any single perspective. When they disagree, we explore deeper."

Date: December 2025
Author: HoloLoom Architecture Team
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from hololoom.fabric.fabric import Fabric
from hololoom.fabric.spacetime import WeavingTrace
from hololoom.loom.base_loom import BaseLoom
from hololoom.loom.consensus import LoomConsensus, Resolution, Tension
from hololoom.loom.protocol import COLLECTIVE, DreamInsight, Loom
from hololoom.protocols.department import (
    ConfidenceMetadata,
    DepartmentRequest,
    DepartmentResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class ExplorationResult:
    """Result of exploring a tension zone."""
    tension: Tension
    resolution: Resolution
    exploration_fabrics: list[Fabric]
    depth_reached: int
    resolved: bool


@dataclass
class WeaveResult:
    """Complete result of a WeaveHouse weaving cycle."""
    final_fabric: Fabric
    individual_fabrics: list[Fabric]
    explorations: list[ExplorationResult]
    total_duration_ms: float
    looms_used: list[str]


class WeaveHouse(BaseLoom):
    """
    A collective of Looms with consensus and auto-exploration.

    The WeaveHouse orchestrates multiple looms (perspectives), synthesizes
    their outputs through consensus, and automatically explores zones of
    significant disagreement.

    Attributes:
        looms: List of Loom instances to weave together
        consensus: LoomConsensus engine for synthesis
        exploration_depth: How deep to explore tensions (default: 2)
        tension_threshold: Minimum tension severity to trigger exploration (default: 0.3)
    """

    def __init__(
        self,
        looms: list[Loom],
        consensus: LoomConsensus | None = None,
        exploration_depth: int = 2,
        tension_threshold: float = 0.3,
        enable_parallel: bool = True,
        **kwargs
    ):
        """
        Initialize the WeaveHouse.

        Args:
            looms: List of Loom instances to orchestrate
            consensus: LoomConsensus engine (creates default if None)
            exploration_depth: Maximum depth for tension exploration
            tension_threshold: Minimum tension severity to explore
            enable_parallel: Whether to weave looms in parallel
            **kwargs: Arguments passed to BaseLoom
        """
        super().__init__(**kwargs)

        self.looms = looms
        self.consensus = consensus or LoomConsensus()
        self.exploration_depth = exploration_depth
        self.tension_threshold = tension_threshold
        self.enable_parallel = enable_parallel

        # Compute collective identity
        self._collective_blind_spots = self._compute_collective_blind_spots()

        # Track exploration history for learning
        self._exploration_history: list[ExplorationResult] = []

    @property
    def perspective(self) -> str:
        """The collective perspective."""
        return COLLECTIVE

    @property
    def blind_spots(self) -> list[str]:
        """
        Collective blind spots - intersection of all loom blind spots.

        Only things ALL looms are blind to remain as collective blind spots.
        """
        return self._collective_blind_spots

    def _compute_collective_blind_spots(self) -> list[str]:
        """
        Compute intersection of all loom blind spots.

        Returns:
            List of blind spots shared by ALL looms
        """
        if not self.looms:
            return []

        # Start with first loom's blind spots
        common: set[str] = set(self.looms[0].blind_spots)

        # Intersect with each subsequent loom
        for loom in self.looms[1:]:
            common &= set(loom.blind_spots)

        return list(common)

    def admits_ignorance(self, query: str) -> list[str]:
        """
        Collect ignorance admissions from all looms.

        The collective is transparent about what each perspective doesn't know.

        Args:
            query: The query to assess

        Returns:
            Aggregated list of knowledge gaps from all looms
        """
        all_ignorance = []

        for loom in self.looms:
            loom_ignorance = loom.admits_ignorance(query)
            for gap in loom_ignorance:
                all_ignorance.append(f"[{loom.perspective}] {gap}")

        # Add collective-level ignorance
        if len(self.looms) < 3:
            all_ignorance.append("[collective] Limited perspectives (< 3 looms)")

        return all_ignorance

    def falsifiable_by(self, claim: str) -> list[str]:
        """
        Collect falsifiers from all looms.

        Each perspective contributes what would change its mind.

        Args:
            claim: The claim to assess

        Returns:
            Aggregated list of falsifiers from all looms
        """
        all_falsifiers = []

        for loom in self.looms:
            loom_falsifiers = loom.falsifiable_by(claim)
            for falsifier in loom_falsifiers[:2]:  # Limit per loom
                all_falsifiers.append(f"[{loom.perspective}] {falsifier}")

        return all_falsifiers

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute the collective weaving process.

        1. Parallel weaving - all looms work simultaneously
        2. Initial consensus synthesis
        3. Auto-explore tensions if significant disagreement
        4. Return synthesized fabric with full provenance

        Args:
            request: Department request with query

        Returns:
            Response with synthesized collective fabric
        """
        start_time = datetime.now()
        query = request.query

        # Step 1: Parallel weaving - all looms work simultaneously
        if self.enable_parallel:
            responses = await asyncio.gather(*[
                loom.execute(request) for loom in self.looms
            ], return_exceptions=True)
        else:
            responses = []
            for loom in self.looms:
                try:
                    response = await loom.execute(request)
                    responses.append(response)
                except Exception as e:
                    responses.append(e)

        # Extract fabrics from successful responses
        fabrics = []
        failed_looms = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.warning(f"Loom {self.looms[i].perspective} failed: {response}")
                failed_looms.append(self.looms[i].perspective)
            else:
                fabric = response.metadata.get('fabric')
                if fabric:
                    fabrics.append(fabric)
                else:
                    # Create fabric from response if not present
                    fabric = self._create_fabric_from_response(
                        response, self.looms[i].perspective
                    )
                    fabrics.append(fabric)

        # Step 2: Initial consensus
        if not fabrics:
            # All looms failed - return error fabric
            return self._create_error_response(
                "All looms failed", failed_looms, start_time
            )

        synthesis = await self.consensus.synthesize(fabrics)

        # Step 3: Auto-explore tensions if significant disagreement
        explorations = []
        if self.consensus.has_significant_tensions(synthesis, self.tension_threshold):
            synthesis, explorations = await self._explore_tensions(
                synthesis,
                fabrics,
                request,
                depth=self.exploration_depth
            )

        # Calculate duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Update metrics
        self._update_metrics(synthesis.confidence, duration_ms)

        # Store exploration history for dreaming
        self._exploration_history.extend(explorations)

        # Create final fabric with full provenance
        final_fabric = self._create_collective_fabric(
            synthesis=synthesis,
            individual_fabrics=fabrics,
            explorations=explorations,
            query=query,
            duration_ms=duration_ms,
            failed_looms=failed_looms
        )

        return DepartmentResponse(
            task_id=getattr(request, 'task_id', uuid4()),
            result=synthesis.response,
            confidence=ConfidenceMetadata.from_score(
                synthesis.confidence,
                justification=[f"Collective of {len(fabrics)} looms, {len(explorations)} explorations"],
                sources=["weave_house"],
            ),
            metadata={
                "perspective": self.perspective,
                "fabric": final_fabric,
                "individual_fabrics": fabrics,
                "explorations": explorations,
                "failed_looms": failed_looms,
                "looms_used": [f.perspective for f in fabrics],
            },
        )

    async def _explore_tensions(
        self,
        synthesis: Fabric,
        fabrics: list[Fabric],
        request: DepartmentRequest,
        depth: int
    ) -> tuple[Fabric, list[ExplorationResult]]:
        """
        Recursively explore disagreement zones.

        Args:
            synthesis: Current synthesis fabric
            fabrics: Individual loom fabrics
            request: Original request
            depth: Remaining exploration depth

        Returns:
            Tuple of (updated synthesis, exploration results)
        """
        if depth == 0:
            return synthesis, []

        explorations = []
        tensions = synthesis.metadata.get("tensions", [])

        for tension_dict in tensions:
            # Convert dict back to Tension if needed
            if isinstance(tension_dict, dict):
                tension = Tension(
                    loom_a=tension_dict.get("loom_a", ""),
                    loom_b=tension_dict.get("loom_b", ""),
                    claim_a=tension_dict.get("claim_a", ""),
                    claim_b=tension_dict.get("claim_b", ""),
                    severity=tension_dict.get("severity", 0.0),
                )
            else:
                tension = tension_dict

            # Skip low-severity tensions
            if tension.severity < self.tension_threshold:
                continue

            # Generate targeted query about the disagreement
            exploration_query = self._generate_tension_query(tension)

            # Get disagreeing looms
            disagreeing_looms = self._get_disagreeing_looms(tension, fabrics)

            if not disagreeing_looms:
                continue

            # Create exploration request
            exploration_request = DepartmentRequest(
                query=exploration_query,
                metadata={
                    "is_exploration": True,
                    "original_query": request.query,
                    "tension": tension,
                }
            )

            # Deeper exploration with disagreeing looms
            exploration_fabrics = []
            for loom in disagreeing_looms:
                try:
                    response = await loom.execute(exploration_request)
                    fabric = response.metadata.get('fabric')
                    if fabric:
                        exploration_fabrics.append(fabric)
                except Exception as e:
                    logger.warning(f"Exploration failed for {loom.perspective}: {e}")

            if not exploration_fabrics:
                continue

            # Try to resolve with more context
            resolution = await self.consensus.attempt_resolution(
                tension, exploration_fabrics
            )

            exploration_result = ExplorationResult(
                tension=tension,
                resolution=resolution,
                exploration_fabrics=exploration_fabrics,
                depth_reached=self.exploration_depth - depth + 1,
                resolved=resolution.resolved,
            )
            explorations.append(exploration_result)

            if resolution.resolved:
                synthesis = self._incorporate_resolution(synthesis, resolution)
            else:
                # Irreducible uncertainty - mark it explicitly
                synthesis = self._mark_irreducible(synthesis, tension)

        return synthesis, explorations

    def _generate_tension_query(self, tension: Tension) -> str:
        """
        Generate a targeted query to explore a tension.

        Args:
            tension: The tension to explore

        Returns:
            Exploration query string
        """
        return (
            f"Resolve this disagreement: "
            f"'{tension.claim_a}' vs '{tension.claim_b}'. "
            f"What evidence supports each position? "
            f"Is there a synthesis or must we accept irreducible uncertainty?"
        )

    def _get_disagreeing_looms(
        self,
        tension: Tension,
        fabrics: list[Fabric]
    ) -> list[Loom]:
        """
        Get looms involved in a tension.

        Args:
            tension: The tension to get looms for
            fabrics: Available fabrics

        Returns:
            List of disagreeing looms
        """
        disagreeing = []
        perspectives = {tension.loom_a, tension.loom_b}

        for loom in self.looms:
            if loom.perspective in perspectives:
                disagreeing.append(loom)

        return disagreeing

    def _incorporate_resolution(
        self,
        synthesis: Fabric,
        resolution: Resolution
    ) -> Fabric:
        """
        Incorporate a successful resolution into synthesis.

        Args:
            synthesis: Current synthesis
            resolution: Successful resolution

        Returns:
            Updated synthesis fabric
        """
        # Update response with resolution
        updated_response = synthesis.response
        if resolution.consensus_claim:
            updated_response += f"\n\n[Resolved] {resolution.consensus_claim}"

        # Increase confidence due to resolution
        new_confidence = min(synthesis.confidence + 0.1, 1.0)

        # Update tensions in metadata
        tensions = synthesis.metadata.get("tensions", [])
        updated_tensions = [
            t for t in tensions
            if not (t.get("loom_a") in resolution.supporting_looms and
                    t.get("loom_b") in resolution.supporting_looms)
        ]

        return Fabric(
            query_text=synthesis.query_text,
            response=updated_response,
            tool_used=synthesis.tool_used,
            trace=synthesis.trace,
            perspective=synthesis.perspective,
            confidence=new_confidence,
            epistemic_confidence=synthesis.epistemic_confidence,
            blind_spots=synthesis.blind_spots,
            admits_ignorance=synthesis.admits_ignorance,
            falsifiable_by=synthesis.falsifiable_by,
            tensions_with=synthesis.tensions_with,
            agreement_with=synthesis.agreement_with,
            metadata={
                **synthesis.metadata,
                "tensions": updated_tensions,
                "resolutions": synthesis.metadata.get("resolutions", []) + [resolution],
            },
        )

    def _mark_irreducible(self, synthesis: Fabric, tension: Tension) -> Fabric:
        """
        Mark a tension as irreducible uncertainty.

        Args:
            synthesis: Current synthesis
            tension: Irreducible tension

        Returns:
            Updated synthesis with marked irreducible
        """
        # Add to admits_ignorance
        new_ignorance = synthesis.admits_ignorance + [
            f"Irreducible: {tension.loom_a} vs {tension.loom_b}"
        ]

        # Reduce epistemic confidence
        new_epistemic = max(synthesis.epistemic_confidence - 0.1, 0.3)

        return Fabric(
            query_text=synthesis.query_text,
            response=synthesis.response,
            tool_used=synthesis.tool_used,
            trace=synthesis.trace,
            perspective=synthesis.perspective,
            confidence=synthesis.confidence,
            epistemic_confidence=new_epistemic,
            blind_spots=synthesis.blind_spots,
            admits_ignorance=new_ignorance,
            falsifiable_by=synthesis.falsifiable_by,
            tensions_with=synthesis.tensions_with,
            agreement_with=synthesis.agreement_with,
            metadata={
                **synthesis.metadata,
                "irreducible_tensions": synthesis.metadata.get("irreducible_tensions", []) + [tension],
            },
        )

    def _create_fabric_from_response(
        self,
        response: DepartmentResponse,
        perspective: str
    ) -> Fabric:
        """
        Create a Fabric from a DepartmentResponse.

        Args:
            response: The department response
            perspective: The loom's perspective

        Returns:
            Fabric wrapping the response
        """
        now = datetime.now()
        return Fabric(
            query_text=response.metadata.get("query", "") if response.metadata else "",
            response=response.result,
            tool_used=perspective,
            confidence=response.confidence.score if response.confidence else 0.5,
            trace=WeavingTrace(start_time=now, end_time=now, duration_ms=0.0),
            perspective=perspective,
            epistemic_confidence=0.5,
            blind_spots=[],
            admits_ignorance=[],
            falsifiable_by=[],
            tensions_with={},
            agreement_with={},
            metadata=response.metadata or {},
        )

    def _create_collective_fabric(
        self,
        synthesis: Fabric,
        individual_fabrics: list[Fabric],
        explorations: list[ExplorationResult],
        query: str,
        duration_ms: float,
        failed_looms: list[str]
    ) -> Fabric:
        """
        Create the final collective fabric with full provenance.

        Args:
            synthesis: Synthesized fabric
            individual_fabrics: Individual loom fabrics
            explorations: Exploration results
            query: Original query
            duration_ms: Total duration
            failed_looms: List of failed loom names

        Returns:
            Complete collective fabric
        """
        return Fabric(
            query_text=query,
            response=synthesis.response,
            tool_used="weave_house",
            confidence=synthesis.confidence,
            trace=synthesis.trace,
            perspective=COLLECTIVE,
            epistemic_confidence=synthesis.epistemic_confidence,
            blind_spots=self.blind_spots,
            admits_ignorance=synthesis.admits_ignorance,
            falsifiable_by=synthesis.falsifiable_by,
            tensions_with=synthesis.tensions_with,
            agreement_with=synthesis.agreement_with,
            metadata={
                **synthesis.metadata,
                "query": query,
                "duration_ms": duration_ms,
                "looms_count": len(individual_fabrics),
                "explorations_count": len(explorations),
                "failed_looms": failed_looms,
                "individual_perspectives": [f.perspective for f in individual_fabrics],
                "exploration_summary": [
                    {
                        "tension": e.tension.loom_a + " vs " + e.tension.loom_b,
                        "resolved": e.resolved,
                        "depth": e.depth_reached,
                    }
                    for e in explorations
                ],
            },
        )

    def _create_error_response(
        self,
        error_message: str,
        failed_looms: list[str],
        start_time: datetime
    ) -> DepartmentResponse:
        """
        Create an error response when all looms fail.

        Args:
            error_message: Error description
            failed_looms: List of failed loom names
            start_time: When execution started

        Returns:
            Error department response
        """
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        return DepartmentResponse(
            task_id=uuid4(),
            result=f"Error: {error_message}",
            confidence=ConfidenceMetadata.from_score(
                0.0,
                justification=[f"All looms failed: {', '.join(failed_looms)}"],
                sources=["weave_house"],
            ),
            metadata={
                "perspective": self.perspective,
                "error": error_message,
                "failed_looms": failed_looms,
                "duration_ms": duration_ms,
            },
        )

    # Dreaming interface

    async def dream_share(self) -> list[DreamInsight]:
        """
        Share exploration insights during collective dreaming.

        The WeaveHouse shares insights about tensions and resolutions.
        """
        insights = await super().dream_share()

        # Add exploration insights
        for exploration in self._exploration_history[-10:]:  # Last 10
            if exploration.resolved:
                insights.append(DreamInsight(
                    source_loom=COLLECTIVE,
                    insight_type="discovery",
                    content={
                        "type": "tension_resolution",
                        "tension": f"{exploration.tension.loom_a} vs {exploration.tension.loom_b}",
                        "resolution": exploration.resolution.consensus_claim,
                    },
                    confidence=exploration.resolution.confidence,
                    shareable=True,
                ))
            else:
                insights.append(DreamInsight(
                    source_loom=COLLECTIVE,
                    insight_type="correction",
                    content={
                        "type": "irreducible_tension",
                        "tension": f"{exploration.tension.loom_a} vs {exploration.tension.loom_b}",
                        "claims": [exploration.tension.claim_a, exploration.tension.claim_b],
                    },
                    confidence=0.5,
                    shareable=True,
                ))

        return insights

    async def dream_receive(self, insight: DreamInsight) -> None:
        """
        Receive and distribute insights to member looms.

        The WeaveHouse forwards insights to appropriate looms.
        """
        await super().dream_receive(insight)

        # Forward to relevant looms
        for loom in self.looms:
            if insight.source_loom != loom.perspective:
                await loom.dream_receive(insight)

    def get_loom(self, perspective: str) -> Loom | None:
        """
        Get a specific loom by perspective.

        Args:
            perspective: The loom perspective to find

        Returns:
            The loom or None if not found
        """
        for loom in self.looms:
            if loom.perspective == perspective:
                return loom
        return None

    def add_loom(self, loom: Loom) -> None:
        """
        Add a loom to the collective.

        Args:
            loom: The loom to add
        """
        self.looms.append(loom)
        self._collective_blind_spots = self._compute_collective_blind_spots()

    def remove_loom(self, perspective: str) -> bool:
        """
        Remove a loom from the collective.

        Args:
            perspective: The perspective of the loom to remove

        Returns:
            True if removed, False if not found
        """
        for i, loom in enumerate(self.looms):
            if loom.perspective == perspective:
                self.looms.pop(i)
                self._collective_blind_spots = self._compute_collective_blind_spots()
                return True
        return False


def create_weave_house(
    looms: list[Loom] | None = None,
    exploration_depth: int = 2,
    tension_threshold: float = 0.3,
    **kwargs
) -> WeaveHouse:
    """
    Factory function to create a WeaveHouse.

    Args:
        looms: List of looms (creates all 5 core looms if None)
        exploration_depth: Maximum exploration depth
        tension_threshold: Minimum tension to explore
        **kwargs: Additional arguments

    Returns:
        Configured WeaveHouse
    """
    if looms is None:
        from hololoom.loom.core_looms import create_all_looms
        looms = create_all_looms(**kwargs)

    consensus = LoomConsensus()

    return WeaveHouse(
        looms=looms,
        consensus=consensus,
        exploration_depth=exploration_depth,
        tension_threshold=tension_threshold,
        **kwargs
    )


__all__ = ['WeaveHouse', 'WeaveResult', 'ExplorationResult', 'create_weave_house']
