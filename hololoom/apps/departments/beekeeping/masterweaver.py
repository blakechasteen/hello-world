#!/usr/bin/env python3
"""
MasterWeaver Department - Beekeeping entity extraction and knowledge management.

This department provides domain-specific entity extraction for beekeeping operations:
- Extracts entities (queen, workers, hive, brood, etc.)
- Validates against beekeeping taxonomy
- Supports multiple extraction strategies (LLM, pattern-based, hybrid)
- Calibrates confidence based on extraction quality

Supported Tasks:
- "extract_entities": Extract entities from beekeeping text
- "validate_taxonomy": Validate entities against known taxonomy
- "enrich_knowledge": Enrich extracted entities with relationships

Design Philosophy:
"Domain expertise in, structured knowledge out."
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import BaseDepartment
from ..protocol import (
    ConfidenceMetadata,
    DepartmentConfig,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class ExtractedEntity:
    """Entity extracted from text."""
    text: str
    entity_type: str
    category: str  # e.g., "colony_members", "hive_components"
    confidence: float
    source: str  # "llm", "pattern", "hybrid"
    context: list[str]
    metadata: dict[str, Any]


@dataclass
class EntityRelationship:
    """Relationship between two entities."""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    justification: str


# ============================================================================
# MasterWeaver Department
# ============================================================================

class MasterWeaverDepartment(BaseDepartment):
    """
    MasterWeaver Department - Beekeeping entity extraction.

    This department extracts and validates beekeeping-specific entities
    from text, using a combination of:
    1. LLM-based extraction (Ollama + OpenAI fallback)
    2. Pattern-based extraction (regex + taxonomy)
    3. Hybrid (both methods combined)

    Example:

        # Create department
        async with MasterWeaverDepartment() as dept:
            # Extract entities
            request = DepartmentRequest(
                task_id="extract_001",
                task_type="extract_entities",
                parameters={
                    "text": "Inspected hive today. Found 5 frames of capped brood. Queen is laying well.",
                    "strategy": "hybrid"  # "llm", "pattern", or "hybrid"
                }
            )

            response = await dept.execute(request)

            # Verify extraction quality
            verification = await dept.verify(response)

            # Refine if needed
            if not verification.sufficient:
                refined = await dept.refine(request, response, verification)
    """

    def __init__(
        self,
        taxonomy_path: Path | None = None,
        enable_llm: bool = True,
        ollama_endpoint: str = "http://localhost:11434",
        openai_api_key: str | None = None,
        dept_config: DepartmentConfig | None = None
    ):
        """
        Initialize MasterWeaver Department.

        Args:
            taxonomy_path: Path to taxonomy.json (defaults to package taxonomy)
            enable_llm: Enable LLM-based extraction
            ollama_endpoint: Ollama API endpoint
            openai_api_key: OpenAI API key (fallback)
            dept_config: Department configuration
        """
        # Initialize base department
        super().__init__(
            name="masterweaver",
            domain="beekeeping",
            version="1.0.0",
            supported_tasks=[
                "extract_entities",
                "validate_taxonomy",
                "enrich_knowledge"
            ],
            confidence_range=(0.70, 0.95),
            config=dept_config or DepartmentConfig(name="masterweaver", domain="beekeeping")
        )

        # Load taxonomy
        if taxonomy_path is None:
            taxonomy_path = Path(__file__).parent / "taxonomy.json"

        self.taxonomy_path = taxonomy_path
        self.taxonomy: dict[str, Any] = {}

        # LLM configuration
        self.enable_llm = enable_llm
        self.ollama_endpoint = ollama_endpoint
        self.openai_api_key = openai_api_key

        # LLM clients (lazy initialization)
        self._ollama_client: Any | None = None
        self._openai_client: Any | None = None

        # Pattern cache
        self._compiled_patterns: dict[str, re.Pattern] = {}

        # Extraction statistics
        self._extraction_stats = {
            "llm_extractions": 0,
            "pattern_extractions": 0,
            "hybrid_extractions": 0,
            "entities_extracted": 0,
            "validation_failures": 0
        }

        logger.info(
            f"MasterWeaver Department initialized (LLM: {enable_llm}, "
            f"taxonomy: {taxonomy_path})"
        )

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize department (load taxonomy, compile patterns)."""
        if self._initialized:
            return

        # Load taxonomy
        logger.info(f"Loading taxonomy from {self.taxonomy_path}")
        with open(self.taxonomy_path) as f:
            self.taxonomy = json.load(f)

        # Compile regex patterns
        logger.info("Compiling entity patterns")
        self._compile_patterns()

        # Initialize LLM clients if enabled
        if self.enable_llm:
            await self._initialize_llm_clients()

        await super().initialize()

    async def close(self) -> None:
        """Cleanup resources."""
        # Close LLM clients
        if self._ollama_client is not None:
            # Ollama client doesn't require explicit cleanup
            self._ollama_client = None

        if self._openai_client is not None:
            # OpenAI client doesn't require explicit cleanup
            self._openai_client = None

        await super().close()

    # ========================================================================
    # Core Department Methods
    # ========================================================================

    async def execute(
        self,
        request: DepartmentRequest
    ) -> DepartmentResponse:
        """
        Execute a beekeeping entity extraction task.

        Supported task types:
        - "extract_entities": Extract entities from text
        - "validate_taxonomy": Validate entities against taxonomy
        - "enrich_knowledge": Enrich entities with relationships

        Args:
            request: Standardized request with task details

        Returns:
            DepartmentResponse with extracted entities and confidence
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        # Validate task type
        if request.task_type not in self.supported_tasks:
            raise ValueError(
                f"Unsupported task type: {request.task_type}. "
                f"Supported: {self.supported_tasks}"
            )

        # Route to specific handler
        try:
            if request.task_type == "extract_entities":
                result, confidence = await self._extract_entities(request)
            elif request.task_type == "validate_taxonomy":
                result, confidence = await self._validate_taxonomy(request)
            elif request.task_type == "enrich_knowledge":
                result, confidence = await self._enrich_knowledge(request)
            else:
                raise ValueError(f"Unknown task type: {request.task_type}")

            # Build learning signals
            learning_signals = {
                "task_type": request.task_type,
                "outcome": "success",
                "confidence": confidence.score,
                "entities_extracted": len(result.get("entities", []))
            }

            # Build response
            response = DepartmentResponse(
                task_id=request.task_id,
                result=result,
                confidence=confidence,
                detail_level=request.context_preference,
                reasoning=self._build_reasoning(result, request) if request.context_preference != "minimal" else None,
                learning_signals=learning_signals,
                session_state=await self._build_session_state(request.session_id, result),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

            # Record success
            self._record_request(
                success=True,
                latency_ms=response.execution_time_ms,
                confidence=confidence.score
            )

            return response

        except Exception as e:
            # Record error
            self._record_error(e)

            # Return error response
            error_confidence = ConfidenceMetadata.from_score(
                0.0,
                justification=[],
                uncertainty_sources=[f"Execution failed: {str(e)}"]
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._record_request(success=False, latency_ms=latency_ms, confidence=0.0)

            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": str(e)},
                confidence=error_confidence,
                execution_time_ms=latency_ms
            )

    async def verify(
        self,
        response: DepartmentResponse
    ) -> VerificationResult:
        """
        Verify extraction quality using DS-STAR pattern.

        Checks:
        1. Confidence validity (score ≥ 0.70 for beekeeping)
        2. Entity count (at least 1 entity extracted)
        3. Taxonomy validation (all entities match known types)

        Args:
            response: Response to verify

        Returns:
            VerificationResult with pass/fail checks and refinement suggestions
        """
        # Check 1: Confidence threshold
        confidence_threshold = 0.70  # Lower for beekeeping (domain-specific)
        confidence_valid = response.confidence.score >= confidence_threshold

        # Check 2: Entity count
        result = response.result
        entities = result.get("entities", [])
        has_entities = len(entities) > 0

        # Check 3: Taxonomy validation
        all_validated = True
        if entities:
            for entity in entities:
                if not entity.get("validated", True):
                    all_validated = False
                    break

        # Determine if sufficient
        sufficient = confidence_valid and has_entities and all_validated

        # Generate refinement suggestions if insufficient
        refinement_suggestions = None
        if not sufficient:
            refinement_suggestions = {}

            if not confidence_valid:
                refinement_suggestions["use_llm"] = True
                refinement_suggestions["reason"] = "Low confidence - try LLM extraction"

            if not has_entities:
                refinement_suggestions["expand_patterns"] = True
                refinement_suggestions["reason"] = "No entities found - expand pattern matching"

            if not all_validated:
                refinement_suggestions["strict_validation"] = False
                refinement_suggestions["reason"] = "Unknown entities - relax validation"

        # Identify alternative paths
        alternative_paths = []
        if not sufficient:
            if response.confidence.score < 0.6:
                alternative_paths.append("llm_extraction")
            if response.confidence.score < 0.5:
                alternative_paths.append("manual_review")

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=confidence_valid,
            reasoning_sound=has_entities and all_validated,
            alternative_paths=alternative_paths,
            refinement_suggestions=refinement_suggestions,
            escalation_needed=response.confidence.score < 0.4,
            escalation_reason="Very low confidence" if response.confidence.score < 0.4 else None,
            verifier_confidence=0.85  # High confidence in beekeeping verification
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """
        Refine extraction based on verification feedback.

        Refinement strategies:
        1. Switch to LLM extraction if pattern-based failed
        2. Expand pattern matching if no entities found
        3. Relax validation if unknown entities detected

        Args:
            request: Original request
            prior_response: Previous insufficient response
            verification: Verification results with suggestions

        Returns:
            Refined DepartmentResponse with improved confidence
        """
        logger.info(
            f"Refining extraction for task {request.task_id} "
            f"(prior confidence: {prior_response.confidence.score:.2f})"
        )

        # Apply refinement strategy
        refined_params = request.parameters.copy()
        suggestions = verification.refinement_suggestions or {}

        if suggestions.get("use_llm"):
            # Switch to LLM extraction
            refined_params["strategy"] = "llm"

        if suggestions.get("expand_patterns"):
            # Enable fuzzy matching
            refined_params["fuzzy_match"] = True

        if "strict_validation" in suggestions:
            # Relax validation
            refined_params["strict_validation"] = suggestions["strict_validation"]

        # Create refined request
        refined_request = DepartmentRequest(
            task_id=request.task_id + "_refined",
            task_type=request.task_type,
            parameters=refined_params,
            confidence_expected=request.confidence_expected,
            context_preference=request.context_preference,
            privacy_level=request.privacy_level,
            session_id=request.session_id,
            parent_task_id=request.task_id,
            timeout_ms=request.timeout_ms
        )

        # Execute with refined parameters
        refined_response = await self.execute(refined_request)

        # Add refinement metadata
        refined_response.reasoning = refined_response.reasoning or {}
        refined_response.reasoning["refinement_applied"] = suggestions

        return refined_response

    # ========================================================================
    # Entity Extraction Methods
    # ========================================================================

    async def _extract_entities(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Extract entities using specified strategy."""
        text = request.parameters.get("text", "")
        strategy = request.parameters.get("strategy", "hybrid")
        strict_validation = request.parameters.get("strict_validation", True)

        if not text:
            raise ValueError("Missing required parameter: 'text'")

        # Extract based on strategy
        if strategy == "llm" and self.enable_llm:
            entities = await self._extract_entities_llm(text)
            self._extraction_stats["llm_extractions"] += 1
        elif strategy == "pattern":
            entities = self._extract_entities_pattern(text)
            self._extraction_stats["pattern_extractions"] += 1
        elif strategy == "hybrid":
            # Combine both methods
            pattern_entities = self._extract_entities_pattern(text)
            if self.enable_llm:
                llm_entities = await self._extract_entities_llm(text)
                entities = self._merge_entities(pattern_entities, llm_entities)
            else:
                entities = pattern_entities
            self._extraction_stats["hybrid_extractions"] += 1
        else:
            raise ValueError(f"Unknown extraction strategy: {strategy}")

        # Validate entities against taxonomy
        validated_entities = self._validate_entities(entities, strict=strict_validation)

        # Calculate confidence
        confidence = self._calculate_extraction_confidence(
            validated_entities,
            strategy,
            len(text)
        )

        # Build result
        result = {
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type,
                    "category": e.category,
                    "confidence": e.confidence,
                    "source": e.source,
                    "context": e.context,
                    "validated": e.metadata.get("validated", True)
                }
                for e in validated_entities
            ],
            "entity_count": len(validated_entities),
            "strategy": strategy,
            "text_length": len(text)
        }

        self._extraction_stats["entities_extracted"] += len(validated_entities)

        return result, confidence

    async def _validate_taxonomy(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Validate entities against taxonomy."""
        entities = request.parameters.get("entities", [])

        if not entities:
            raise ValueError("Missing required parameter: 'entities'")

        # Validate each entity
        validation_results = []
        for entity_data in entities:
            entity = ExtractedEntity(
                text=entity_data.get("text", ""),
                entity_type=entity_data.get("type", ""),
                category=entity_data.get("category", ""),
                confidence=entity_data.get("confidence", 0.5),
                source=entity_data.get("source", "unknown"),
                context=entity_data.get("context", []),
                metadata={}
            )

            is_valid, reason = self._is_valid_entity(entity)
            validation_results.append({
                "entity": entity_data,
                "valid": is_valid,
                "reason": reason
            })

        # Calculate confidence
        valid_count = sum(1 for r in validation_results if r["valid"])
        total_count = len(validation_results)
        validation_rate = valid_count / total_count if total_count > 0 else 0.0

        confidence = ConfidenceMetadata.from_score(
            validation_rate,
            justification=[f"{valid_count}/{total_count} entities validated"],
            uncertainty_sources=[] if validation_rate > 0.8 else ["Low validation rate"]
        )

        result = {
            "validation_results": validation_results,
            "valid_count": valid_count,
            "total_count": total_count,
            "validation_rate": validation_rate
        }

        return result, confidence

    async def _enrich_knowledge(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Enrich entities with relationships."""
        entities = request.parameters.get("entities", [])

        if not entities:
            raise ValueError("Missing required parameter: 'entities'")

        # Extract relationships between entities
        relationships = self._extract_relationships(entities)

        # Calculate confidence
        confidence_score = 0.80  # Base confidence for relationship extraction
        if len(relationships) == 0:
            confidence_score = 0.50  # Lower if no relationships found

        confidence = ConfidenceMetadata.from_score(
            confidence_score,
            justification=[f"Extracted {len(relationships)} relationships"],
            uncertainty_sources=[] if len(relationships) > 0 else ["No relationships found"]
        )

        result = {
            "entities": entities,
            "relationships": [
                {
                    "source": r.source_entity,
                    "target": r.target_entity,
                    "type": r.relationship_type,
                    "confidence": r.confidence,
                    "justification": r.justification
                }
                for r in relationships
            ],
            "relationship_count": len(relationships)
        }

        return result, confidence

    # ========================================================================
    # Pattern-Based Extraction
    # ========================================================================

    def _extract_entities_pattern(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using regex patterns from taxonomy."""
        entities = []

        # Extract from each category
        for category, entity_types in self.taxonomy.get("entity_types", {}).items():
            for entity_type, entity_data in entity_types.items():
                # Get compiled patterns
                patterns = self._get_compiled_patterns(entity_type, entity_data)

                for pattern in patterns:
                    matches = pattern.finditer(text.lower())

                    for match in matches:
                        entity_text = match.group(0)

                        # Calculate confidence
                        base_confidence = entity_data.get("confidence_base", 0.75)
                        context_boost = self._calculate_context_boost(
                            text,
                            match.start(),
                            match.end(),
                            entity_data.get("context_boost", [])
                        )

                        confidence = min(base_confidence + context_boost, 1.0)

                        # Extract context (surrounding words)
                        context = self._extract_context(text, match.start(), match.end())

                        entities.append(ExtractedEntity(
                            text=entity_text,
                            entity_type=entity_type,
                            category=category,
                            confidence=confidence,
                            source="pattern",
                            context=context,
                            metadata={"pattern_matched": True}
                        ))

        return entities

    def _compile_patterns(self) -> None:
        """Compile regex patterns from taxonomy."""
        for category, entity_types in self.taxonomy.get("entity_types", {}).items():
            for entity_type, entity_data in entity_types.items():
                patterns = entity_data.get("patterns", [])
                self._compiled_patterns[entity_type] = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in patterns
                ]

    def _get_compiled_patterns(
        self,
        entity_type: str,
        entity_data: dict[str, Any]
    ) -> list[re.Pattern]:
        """Get compiled patterns for entity type."""
        if entity_type in self._compiled_patterns:
            return self._compiled_patterns[entity_type]

        # Fallback: compile on-demand
        patterns = entity_data.get("patterns", [])
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        self._compiled_patterns[entity_type] = compiled
        return compiled

    def _calculate_context_boost(
        self,
        text: str,
        start: int,
        end: int,
        boost_terms: list[str]
    ) -> float:
        """Calculate confidence boost based on context."""
        if not boost_terms:
            return 0.0

        # Extract surrounding context (50 chars before/after)
        context_start = max(0, start - 50)
        context_end = min(len(text), end + 50)
        context = text[context_start:context_end].lower()

        # Count boost terms in context
        boost_count = sum(1 for term in boost_terms if term.lower() in context)

        # Calculate boost (max 0.15)
        return min(boost_count * 0.05, 0.15)

    def _extract_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 30
    ) -> list[str]:
        """Extract context words around entity."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end]

        # Split into words
        words = re.findall(r'\w+', context)
        return words

    # ========================================================================
    # LLM-Based Extraction
    # ========================================================================

    async def _extract_entities_llm(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using LLM (Ollama with OpenAI fallback)."""
        if not self.enable_llm:
            logger.warning("LLM extraction disabled")
            return []

        # Try Ollama first
        try:
            if self._ollama_client is None:
                await self._initialize_llm_clients()

            if self._ollama_client is not None:
                return await self._extract_with_ollama(text)
        except Exception as e:
            logger.warning(f"Ollama extraction failed: {e}")

        # Fallback to OpenAI
        try:
            if self._openai_client is not None:
                return await self._extract_with_openai(text)
        except Exception as e:
            logger.warning(f"OpenAI extraction failed: {e}")

        # If both fail, return empty list
        logger.error("LLM extraction failed (both Ollama and OpenAI)")
        return []

    async def _initialize_llm_clients(self) -> None:
        """Initialize LLM clients (Ollama + OpenAI)."""
        # Try Ollama
        try:
            import ollama
            self._ollama_client = ollama.Client(host=self.ollama_endpoint)
            logger.info(f"Ollama client initialized (endpoint: {self.ollama_endpoint})")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama client: {e}")
            self._ollama_client = None

        # Try OpenAI
        if self.openai_api_key:
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self._openai_client = None

    async def _extract_with_ollama(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using Ollama."""
        # Build prompt
        prompt = self._build_extraction_prompt(text)

        # Call Ollama
        response = self._ollama_client.generate(
            model="llama2",  # Or other model
            prompt=prompt
        )

        # Parse response
        entities = self._parse_llm_response(response.get("response", ""), source="llm")
        return entities

    async def _extract_with_openai(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using OpenAI."""
        # Build prompt
        prompt = self._build_extraction_prompt(text)

        # Call OpenAI
        response = self._openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a beekeeping expert that extracts entities from text."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse response
        entities = self._parse_llm_response(
            response.choices[0].message.content,
            source="llm"
        )
        return entities

    def _build_extraction_prompt(self, text: str) -> str:
        """Build extraction prompt for LLM."""
        entity_types = []
        for category, types in self.taxonomy.get("entity_types", {}).items():
            entity_types.extend(types.keys())

        prompt = f"""Extract beekeeping entities from the following text.

Entity types to look for: {', '.join(entity_types)}

Text:
{text}

For each entity found, provide:
- Entity text
- Entity type
- Confidence (0.0-1.0)

Format as JSON:
[{{"text": "...", "type": "...", "confidence": 0.85}}]
"""
        return prompt

    def _parse_llm_response(
        self,
        response: str,
        source: str
    ) -> list[ExtractedEntity]:
        """Parse LLM response into entities."""
        entities = []

        try:
            # Try to parse as JSON
            import json
            parsed = json.loads(response)

            for item in parsed:
                entity_text = item.get("text", "")
                entity_type = item.get("type", "")
                confidence = item.get("confidence", 0.85)

                # Find category
                category = self._find_entity_category(entity_type)

                entities.append(ExtractedEntity(
                    text=entity_text,
                    entity_type=entity_type,
                    category=category,
                    confidence=confidence,
                    source=source,
                    context=[],
                    metadata={"llm_extracted": True}
                ))

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return entities

    # ========================================================================
    # Validation and Calibration
    # ========================================================================

    def _validate_entities(
        self,
        entities: list[ExtractedEntity],
        strict: bool = True
    ) -> list[ExtractedEntity]:
        """Validate entities against taxonomy."""
        validated = []

        for entity in entities:
            is_valid, reason = self._is_valid_entity(entity)

            if is_valid or not strict:
                # Calibrate confidence based on validation
                calibrated_confidence = self._calibrate_confidence(
                    entity,
                    is_valid
                )
                entity.confidence = calibrated_confidence
                entity.metadata["validated"] = is_valid
                entity.metadata["validation_reason"] = reason
                validated.append(entity)
            else:
                self._extraction_stats["validation_failures"] += 1

        return validated

    def _is_valid_entity(
        self,
        entity: ExtractedEntity
    ) -> tuple[bool, str]:
        """Check if entity is valid according to taxonomy."""
        # Find entity in taxonomy
        for category, entity_types in self.taxonomy.get("entity_types", {}).items():
            if entity.entity_type in entity_types:
                return True, "Known entity type"

            # Check aliases
            for entity_type, entity_data in entity_types.items():
                aliases = entity_data.get("aliases", [])
                if entity.text.lower() in [a.lower() for a in aliases]:
                    # Update entity type to canonical name
                    entity.entity_type = entity_type
                    entity.category = category
                    return True, "Matched via alias"

        return False, "Unknown entity type"

    def _calibrate_confidence(
        self,
        entity: ExtractedEntity,
        is_valid: bool
    ) -> float:
        """Calibrate confidence based on extraction method and validation."""
        base_confidence = entity.confidence

        calibration = self.taxonomy.get("confidence_calibration", {})

        # Apply source-specific calibration
        if entity.source == "llm":
            llm_cal = calibration.get("llm_extraction", {})
            if is_valid:
                base_confidence += llm_cal.get("boost_if_multiple_sources", 0.0)
            else:
                base_confidence += llm_cal.get("penalty_if_ambiguous", 0.0)

        elif entity.source == "pattern":
            pattern_cal = calibration.get("pattern_matching", {})
            if len(entity.context) > 0:
                base_confidence += pattern_cal.get("boost_if_context_match", 0.0)
            else:
                base_confidence += pattern_cal.get("penalty_if_no_context", 0.0)

        # Apply taxonomy validation calibration
        taxonomy_cal = calibration.get("taxonomy_validation", {})
        if is_valid:
            base_confidence += taxonomy_cal.get("known_entity", 0.0)
        else:
            base_confidence += taxonomy_cal.get("unknown_entity", 0.0)

        return max(0.0, min(base_confidence, 1.0))

    def _calculate_extraction_confidence(
        self,
        entities: list[ExtractedEntity],
        strategy: str,
        text_length: int
    ) -> ConfidenceMetadata:
        """Calculate overall extraction confidence."""
        if not entities:
            return ConfidenceMetadata.from_score(
                0.3,
                justification=[],
                uncertainty_sources=["No entities extracted"]
            )

        # Average entity confidence
        avg_confidence = sum(e.confidence for e in entities) / len(entities)

        # Justification
        justification = [
            f"Extracted {len(entities)} entities using {strategy} strategy",
            f"Average entity confidence: {avg_confidence:.2f}"
        ]

        # Uncertainty sources
        uncertainty_sources = []
        if avg_confidence < 0.75:
            uncertainty_sources.append("Low average entity confidence")
        if len(entities) < 2:
            uncertainty_sources.append("Few entities extracted")

        return ConfidenceMetadata.from_score(
            avg_confidence,
            justification=justification,
            uncertainty_sources=uncertainty_sources
        )

    # ========================================================================
    # Entity Merging and Relationships
    # ========================================================================

    def _merge_entities(
        self,
        pattern_entities: list[ExtractedEntity],
        llm_entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        """Merge entities from pattern and LLM extraction."""
        merged = []
        seen_texts: set[str] = set()

        # Add pattern entities first (higher precision)
        for entity in pattern_entities:
            if entity.text.lower() not in seen_texts:
                merged.append(entity)
                seen_texts.add(entity.text.lower())

        # Add LLM entities not already found
        for entity in llm_entities:
            if entity.text.lower() not in seen_texts:
                merged.append(entity)
                seen_texts.add(entity.text.lower())

        return merged

    def _extract_relationships(
        self,
        entities: list[dict[str, Any]]
    ) -> list[EntityRelationship]:
        """Extract relationships between entities."""
        relationships = []

        # Get relationship types from taxonomy
        rel_types = self.taxonomy.get("relationship_types", {})

        # Simple heuristic relationships based on entity types
        for i, source in enumerate(entities):
            for j, target in enumerate(entities):
                if i >= j:
                    continue

                # Check for known relationships
                rel_type, confidence, justification = self._infer_relationship(
                    source,
                    target,
                    rel_types
                )

                if rel_type:
                    relationships.append(EntityRelationship(
                        source_entity=source.get("text", ""),
                        target_entity=target.get("text", ""),
                        relationship_type=rel_type,
                        confidence=confidence,
                        justification=justification
                    ))

        return relationships

    def _infer_relationship(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
        rel_types: dict[str, Any]
    ) -> tuple[str | None, float, str]:
        """Infer relationship between two entities."""
        source_type = source.get("type", "")
        target_type = target.get("type", "")

        # Heuristic rules for common relationships
        if source_type == "worker" and target_type == "wax":
            return "PRODUCES", 0.90, "Workers produce wax"

        if source_type == "queen" and target_type == "brood":
            return "PRODUCES", 0.95, "Queen produces brood"

        if source_type == "frame" and target_type == "hive":
            return "PART_OF", 0.95, "Frame is part of hive"

        if source_type == "varroa" and target_type in ["worker", "drone", "queen"]:
            return "AFFECTS", 0.90, "Varroa affects colony health"

        # No relationship found
        return None, 0.0, ""

    def _find_entity_category(self, entity_type: str) -> str:
        """Find category for entity type."""
        for category, entity_types in self.taxonomy.get("entity_types", {}).items():
            if entity_type in entity_types:
                return category
        return "unknown"

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _build_reasoning(
        self,
        result: dict[str, Any],
        request: DepartmentRequest
    ) -> dict[str, Any]:
        """Build reasoning dict for response."""
        return {
            "extraction_strategy": result.get("strategy", "unknown"),
            "entities_extracted": result.get("entity_count", 0),
            "text_length": result.get("text_length", 0),
            "statistics": self._extraction_stats.copy()
        }

    async def _build_session_state(
        self,
        session_id: str | None,
        result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build session state update."""
        if not session_id:
            return None

        return {
            "entities_extracted": result.get("entity_count", 0),
            "last_extraction_strategy": result.get("strategy", "unknown")
        }
