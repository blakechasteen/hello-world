"""
The Comb - Deterministic Structure Alignment
============================================
The Reed that beats the probabilistic weft into a deterministic structure.

Philosophy:
In weaving, the Comb (or Reed) pushes the newly inserted weft thread tightly
against the woven fabric. It transforms the loose, potential energy of the
thread into the rigid, durable structure of the cloth.

In HoloLoom, The Comb projects the high-dimensional, fuzzy thought-space
of the LLM onto a rigid, lower-dimensional manifold (a Schema).

It is the mechanism of Crystallization:
    Plasma (Logits) -> The Comb (Schema) -> Crystal (Structured Object)

Usage:
    class SearchIntent(BaseModel):
        query: str
        confidence: float

    comb = TheComb(llm)
    intent = await comb.align("Find PPO papers", SearchIntent)
"""

import json
import logging
from typing import Type, TypeVar, Optional, Any, Dict, Generic
from pydantic import BaseModel, ValidationError

from HoloLoom.awareness.llm_integration import LLMProtocol

# Generic type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class TheComb:
    """
    The Comb aligns probabilistic energy (LLM output) into geometric structure (Schema).
    """

    def __init__(self, llm: LLMProtocol, max_retries: int = 2):
        """
        Initialize The Comb.

        Args:
            llm: The source of probabilistic energy (the model).
            max_retries: How many times to "beat" (retry) if alignment fails.
        """
        self.llm = llm
        self.max_retries = max_retries

    async def align(
        self,
        energy: str,
        geometry: Type[T],
        context: Optional[str] = None
    ) -> T:
        """
        Aligns the raw 'energy' (prompt/intent) into the 'geometry' (Schema).

        Args:
            energy: The user's intent or raw input text.
            geometry: The Pydantic model class defining the desired shape.
            context: Optional background context to inform the alignment.

        Returns:
            An instance of T, guaranteed to be valid.

        Raises:
            ValueError: If alignment fails after max_retries.
        """
        # 1. The Mold: Extract the JSON Schema from the Pydantic model
        schema = geometry.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        # 2. The Setup: Construct the System Prompt
        # We frame this not as a request, but as a constraint.
        system_prompt = (
            "You are a Structural Alignment Engine. "
            "Your goal is to map the user's intent into the provided JSON Schema.\n\n"
            f"TARGET SCHEMA:\n{schema_str}\n\n"
            "RULES:\n"
            "1. Output ONLY valid JSON.\n"
            "2. Adhere strictly to the field types and descriptions.\n"
            "3. Do not include markdown formatting (```json).\n"
            "4. If the intent is ambiguous, make the most reasonable assumption based on the schema."
        )

        user_prompt = f"INTENT: {energy}"
        if context:
            user_prompt = f"CONTEXT:\n{context}\n\n{user_prompt}"

        # 3. The Beat: Iterative refinement
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # The Cast: Pour logits into the mold
                # We assume the LLM supports a 'format' param or we enforce it via prompt
                # Note: We pass format="json" if the underlying provider supports it (e.g. Ollama)
                response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.1,  # Low temp for structural rigidity
                    format="json"     # Force JSON mode if supported
                )

                # Clean the output (remove potential markdown wrappers)
                raw_json = self._clean_json(response.content)

                # The Crystallization: Validate against the schema
                # This performs type coercion and validation
                instance = geometry.model_validate_json(raw_json)
                
                logger.debug(f"Aligned energy into {geometry.__name__} (Attempt {attempt+1})")
                return instance

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(f"Alignment snagged on attempt {attempt+1}: {e}")
                
                # The Correction: Feed the error back into the system
                # "The thread snagged. Align it again."
                user_prompt += (
                    f"\n\nPREVIOUS ATTEMPT FAILED:\n"
                    f"Error: {str(e)}\n"
                    f"Please correct the JSON structure and try again."
                )

        # If we reach here, the thread is broken.
        raise ValueError(
            f"Failed to align energy into {geometry.__name__} after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _clean_json(self, content: str) -> str:
        """
        Cleans the raw string output to ensure pure JSON.
        Removes markdown code blocks if present.
        """
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
