"""
Dual-Stream Response Generation with Compositional Awareness

Generates both internal reasoning and external response streams,
both guided by unified compositional awareness context.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import time

from HoloLoom.awareness.compositional_awareness import (
    CompositionalAwarenessLayer,
    UnifiedAwarenessContext,
    format_awareness_for_prompt
)

# LLM integration (optional)
try:
    from HoloLoom.awareness.llm_integration import LLMProtocol, LLMResponse
    LLM_AVAILABLE = True
except ImportError:
    LLMProtocol = None
    LLMResponse = None
    LLM_AVAILABLE = False


@dataclass
class DualStreamResponse:
    """Response containing both internal and external streams"""
    
    query: str
    awareness_context: UnifiedAwarenessContext
    internal_stream: str
    external_stream: str
    generation_time_ms: float
    
    def format_for_display(self, show_internal: bool = True) -> str:
        """Format response for display"""
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"QUERY: {self.query}")
        lines.append("=" * 80)
        lines.append("")
        
        if show_internal:
            lines.append("[ INTERNAL REASONING ]")
            lines.append("-" * 80)
            lines.append(self.internal_stream)
            lines.append("")
        
        lines.append("[ RESPONSE ]")
        lines.append("-" * 80)
        lines.append(self.external_stream)
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"Generated in {self.generation_time_ms:.1f}ms")
        lines.append("=" * 80)
        
        return "\n".join(lines)


class DualStreamGenerator:
    """
    Generates awareness-guided dual-stream responses.
    
    Both internal reasoning and external response are informed by
    the same unified compositional awareness context.
    """
    
    def __init__(
        self,
        awareness_layer: CompositionalAwarenessLayer,
        llm_generator = None  # Optional: actual LLM for generation
    ):
        """
        Initialize dual-stream generator.

        Args:
            awareness_layer: Compositional awareness layer
            llm_generator: Optional LLM for actual generation (else use templates)
                          Can be OllamaLLM, AnthropicLLM, or OpenAILLM
        """
        self.awareness = awareness_layer
        self.llm = llm_generator
        self.use_llm = llm_generator is not None and hasattr(llm_generator, 'generate')
    
    async def generate(
        self,
        query: str,
        show_internal: bool = True,
        use_llm: Optional[bool] = None
    ) -> DualStreamResponse:
        """
        Generate dual-stream response with awareness guidance.
        
        Args:
            query: User query
            show_internal: Whether to show internal reasoning
            use_llm: Whether to use actual LLM (None = auto-detect from init)
            
        Returns:
            DualStreamResponse with both streams
        """
        
        start_time = time.time()
        
        # Auto-detect LLM usage if not specified
        if use_llm is None:
            use_llm = self.use_llm
        
        # 1. Generate unified awareness context
        awareness_ctx = await self.awareness.get_unified_context(query)
        
        # 2. Generate internal reasoning stream
        internal_reasoning = await self._generate_internal_stream(
            query,
            awareness_ctx,
            use_llm=use_llm
        )
        
        # 3. Generate external response stream
        external_response = await self._generate_external_stream(
            query,
            awareness_ctx,
            internal_reasoning,
            use_llm=use_llm
        )
        
        # 4. Update awareness from generation (feedback loop)
        await self.awareness.update_from_generation(
            query=query,
            internal_reasoning=internal_reasoning,
            external_response=external_response,
            user_feedback={'success': True}  # Assume success for now
        )
        
        generation_time = (time.time() - start_time) * 1000  # ms
        
        return DualStreamResponse(
            query=query,
            awareness_context=awareness_ctx,
            internal_stream=internal_reasoning,
            external_stream=external_response,
            generation_time_ms=generation_time
        )
    
    async def _generate_internal_stream(
        self,
        query: str,
        awareness_ctx: UnifiedAwarenessContext,
        use_llm: bool = False
    ) -> str:
        """Generate internal reasoning stream with awareness guidance"""

        # If using LLM, generate actual reasoning
        if use_llm and self.llm:
            try:
                prompt = build_internal_prompt(query, awareness_ctx)
                response = await self.llm.generate(
                    prompt=prompt,
                    system_prompt="You are an AI assistant analyzing your own reasoning process. Explain your thought process step by step.",
                    max_tokens=400,
                    temperature=0.7
                )
                return response.content
            except Exception as e:
                # Fall back to template if LLM fails
                print(f"Warning: LLM generation failed, using template: {e}")
                pass

        # Template-based internal reasoning (fallback)
        lines = []
        lines.append("[AWARENESS-GUIDED REASONING]")
        lines.append("")

        # Confidence analysis
        conf = awareness_ctx.confidence
        if conf.uncertainty_level < 0.3:
            lines.append("✓ High confidence detected")
            lines.append(f"  - Cache status: {conf.query_cache_status}")
            lines.append(f"  - Pattern familiarity: {awareness_ctx.patterns.seen_count}× seen")
            lines.append(f"  - Uncertainty: {conf.uncertainty_level:.2f} (LOW)")
        else:
            lines.append("⚠️ Low confidence detected")
            lines.append(f"  - Cache status: {conf.query_cache_status}")
            lines.append(f"  - Uncertainty: {conf.uncertainty_level:.2f} (HIGH)")
            if conf.knowledge_gap_detected:
                lines.append("  - Knowledge gap: Novel query pattern")

        lines.append("")

        # Structural analysis
        struct = awareness_ctx.structural
        lines.append("Structural Analysis:")
        lines.append(f"  - Type: {struct.phrase_type}")
        if struct.is_question:
            lines.append(f"  - Question type: {struct.question_type}")
            lines.append(f"  - Expected response: {struct.suggested_response_type}")
        lines.append("")

        # Pattern analysis
        patterns = awareness_ctx.patterns
        lines.append("Pattern Analysis:")
        lines.append(f"  - Domain: {patterns.domain}/{patterns.subdomain}")
        lines.append(f"  - Confidence: {patterns.confidence:.2f}")
        lines.append("")

        # Strategy selection
        internal = awareness_ctx.internal_guidance
        lines.append("Selected Strategy:")
        lines.append(f"  - Reasoning: {internal.reasoning_structure}")
        if internal.high_confidence_shortcuts:
            lines.append(f"  - Shortcuts: {', '.join(internal.high_confidence_shortcuts)}")
        if internal.low_confidence_checks:
            lines.append(f"  - Checks: {', '.join(internal.low_confidence_checks)}")

        if internal.should_ask_clarification:
            lines.append("  - ⚠️ RECOMMENDATION: Ask for clarification")

        return "\n".join(lines)
    
    async def _generate_external_stream(
        self,
        query: str,
        awareness_ctx: UnifiedAwarenessContext,
        internal_reasoning: str,
        use_llm: bool = False
    ) -> str:
        """Generate external response stream with awareness guidance"""

        external = awareness_ctx.external_guidance
        conf = awareness_ctx.confidence
        patterns = awareness_ctx.patterns

        # If using LLM, generate actual response
        if use_llm and self.llm:
            try:
                prompt = build_external_prompt(query, awareness_ctx, internal_reasoning)
                response = await self.llm.generate(
                    prompt=prompt,
                    system_prompt="You are a helpful AI assistant. Follow the guidance provided about confidence levels and tone.",
                    max_tokens=500,
                    temperature=0.7
                )
                return response.content
            except Exception as e:
                # Fall back to templates if LLM fails
                print(f"Warning: LLM generation failed, using template: {e}")
                pass

        # Template-based responses (fallback)
        # If clarification needed, generate clarification question
        if external.clarification_needed:
            return self._generate_clarification_response(query, awareness_ctx)

        # Otherwise, generate appropriate response based on confidence and domain
        if conf.uncertainty_level < 0.3:
            # High confidence: Direct, confident response
            return self._generate_confident_response(query, awareness_ctx)
        elif conf.uncertainty_level < 0.6:
            # Medium confidence: Hedged but informative
            return self._generate_hedged_response(query, awareness_ctx)
        else:
            # Low confidence: Acknowledge uncertainty
            return self._generate_uncertain_response(query, awareness_ctx)
    
    def _generate_clarification_response(
        self,
        query: str,
        awareness_ctx: UnifiedAwarenessContext
    ) -> str:
        """Generate clarification question for uncertain queries"""
        
        conf = awareness_ctx.confidence
        
        return (
            f"I notice I'm not familiar with some concepts in your question. "
            f"{conf.suggested_clarification or 'Could you provide more context?'}"
        )
    
    def _generate_confident_response(
        self,
        query: str,
        awareness_ctx: UnifiedAwarenessContext
    ) -> str:
        """Generate confident response for familiar queries"""
        
        patterns = awareness_ctx.patterns
        struct = awareness_ctx.structural
        
        # Example template (in production, would use actual LLM)
        if "ball" in query.lower() and "red" in query.lower():
            if struct.expects_definition:
                return (
                    "A red ball is a spherical toy or sports equipment, "
                    "commonly used in games like dodgeball and kickball."
                )
            elif struct.expects_explanation:
                return (
                    "Red balls are popular in sports and games because the color "
                    "provides high visibility against most backgrounds, making them "
                    "easier to track during play."
                )
        
        # Generic confident response
        return (
            f"Based on {patterns.seen_count} similar queries, I can confidently say "
            f"this relates to the {patterns.domain} domain. "
            f"[This would be a detailed, direct answer in production.]"
        )
    
    def _generate_hedged_response(
        self,
        query: str,
        awareness_ctx: UnifiedAwarenessContext
    ) -> str:
        """Generate hedged response for medium-confidence queries"""
        
        external = awareness_ctx.external_guidance
        hedging = external.appropriate_hedging[0] if external.appropriate_hedging else "generally"
        
        patterns = awareness_ctx.patterns
        
        return (
            f"Based on the patterns I've seen, this {hedging} relates to "
            f"{patterns.domain}. While I have some familiarity with this topic, "
            f"I want to note that my confidence is moderate. "
            f"[Production would provide more detailed hedged answer here.]"
        )
    
    def _generate_uncertain_response(
        self,
        query: str,
        awareness_ctx: UnifiedAwarenessContext
    ) -> str:
        """Generate response acknowledging high uncertainty"""
        
        conf = awareness_ctx.confidence
        
        nearest = ""
        if conf.nearest_known_patterns:
            patterns_str = ", ".join([p[0] for p in conf.nearest_known_patterns[:2]])
            nearest = f" The closest concepts I'm familiar with are: {patterns_str}."
        
        return (
            f"I notice this query is quite novel to me (uncertainty: {conf.uncertainty_level:.2f}).{nearest} "
            f"Could you provide more context or clarify what you're asking about?"
        )


# ============================================================================
# Prompt Templates for LLM Integration (when using actual LLMs)
# ============================================================================

def build_internal_prompt(
    query: str,
    awareness_ctx: UnifiedAwarenessContext
) -> str:
    """Build prompt for internal reasoning stream"""
    
    awareness_text = format_awareness_for_prompt(awareness_ctx)
    
    return f"""You are an AI assistant with compositional awareness. Think through this query step by step.

{awareness_text}

Query: {query}

Think through your reasoning:
1. What does the awareness context tell you?
2. What's your confidence level and why?
3. What strategy should you use?
4. Should you answer directly or ask for clarification?

Internal reasoning:"""


def build_external_prompt(
    query: str,
    awareness_ctx: UnifiedAwarenessContext,
    internal_reasoning: str
) -> str:
    """Build prompt for external response stream"""
    
    external = awareness_ctx.external_guidance
    
    prompt_parts = [
        "You are an AI assistant. Based on the following context and reasoning, generate an appropriate response.",
        "",
        "[YOUR INTERNAL REASONING]",
        internal_reasoning,
        "",
        "[RESPONSE GUIDELINES]",
        f"- Tone: {external.confidence_tone}",
        f"- Structure: {external.response_structure}",
        f"- Length: {external.expected_length}",
    ]
    
    if external.appropriate_hedging:
        prompt_parts.append(f"- Use hedging: {', '.join(external.appropriate_hedging)}")
    
    if external.should_acknowledge_uncertainty:
        prompt_parts.append("- IMPORTANT: Acknowledge uncertainty")
    
    if external.clarification_needed:
        prompt_parts.append("- IMPORTANT: Ask for clarification, don't answer directly")
    
    prompt_parts.extend([
        "",
        f"Query: {query}",
        "",
        "Your response:"
    ])
    
    return "\n".join(prompt_parts)
