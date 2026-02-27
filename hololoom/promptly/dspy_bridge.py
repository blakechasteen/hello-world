#!/usr/bin/env python3
"""
DSPy Bridge for HoloLoom

Integrates DSPy's systematic prompt engineering with HoloLoom's knowledge system.

Key Features:
- DSPy signature definitions from HoloLoom patterns
- Automatic optimization using HoloLoom's memory as training data
- Compositional cache integration for faster execution
- Semantic grounding via Matryoshka embeddings

Architecture:
    DSPy Signatures → HoloLoom Memory → Optimized Prompts
                      ↓
    Promptly Workflows → DSPy Programs → Responses

Usage:
    from HoloLoom.promptly.dspy_bridge import DSPyHoloLoom, create_signature

    # Create bridge
    bridge = DSPyHoloLoom(config=Config.fused())

    # Define signature
    sig = create_signature(
        "Answer questions using relevant context",
        inputs=["question", "context"],
        outputs=["answer", "confidence"]
    )

    # Optimize from HoloLoom memory
    optimized = await bridge.optimize_from_memory(
        signature=sig,
        memory_query="question answering examples",
        metric="accuracy"
    )

    # Execute
    result = await bridge.execute(optimized, question="What is Thompson Sampling?")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
import json

# Type imports
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.fabric.spacetime import Spacetime

# Optional DSPy import with graceful degradation
try:
    import dspy
    from dspy.signatures import Signature
    from dspy.teleprompt import BootstrapFewShot, MIPROv2
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logging.warning("DSPy not available - install with: pip install dspy-ai")


@dataclass
class DSPySignature:
    """Wrapper for DSPy signature with HoloLoom context"""
    name: str
    description: str
    inputs: List[str]
    outputs: List[str]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal DSPy signature (lazy-loaded)
    _dspy_sig: Optional[Any] = None

    def to_dspy_signature(self) -> Any:
        """Convert to DSPy signature class"""
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy not available")

        if self._dspy_sig is None:
            # Build signature class dynamically
            sig_fields = {}

            # Add inputs
            for input_name in self.inputs:
                sig_fields[input_name] = dspy.InputField(desc=f"Input: {input_name}")

            # Add outputs
            for output_name in self.outputs:
                sig_fields[output_name] = dspy.OutputField(desc=f"Output: {output_name}")

            # Create signature class
            self._dspy_sig = type(
                self.name,
                (dspy.Signature,),
                {
                    "__doc__": self.description,
                    **sig_fields
                }
            )

        return self._dspy_sig


@dataclass
class DSPyProgram:
    """Optimized DSPy program with provenance"""
    signature: DSPySignature
    program: Any  # dspy.Module
    optimizer: str
    training_examples: int
    validation_score: float
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DSPyOptimizationConfig:
    """Configuration for DSPy optimization"""
    optimizer: str = "bootstrap"  # "bootstrap", "mipro", "copro"
    num_threads: int = 4
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    max_rounds: int = 1
    metric: Optional[Callable] = None
    teacher_settings: Dict[str, Any] = field(default_factory=dict)


class DSPyHoloLoom:
    """
    Bridge between DSPy and HoloLoom.

    Enables:
    1. Creating DSPy signatures from HoloLoom patterns
    2. Optimizing DSPy programs using HoloLoom memory as training data
    3. Executing DSPy programs with HoloLoom context
    4. Caching optimized prompts in compositional cache
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        lm_model: str = "openai/gpt-4o-mini",
        lm_api_key: Optional[str] = None
    ):
        """
        Initialize DSPy-HoloLoom bridge.

        Args:
            config: HoloLoom configuration
            lm_model: Language model to use. Supported formats:
                - "ollama/llama3.2:3b" - Local Ollama (free, private)
                - "openai/gpt-4o-mini" - OpenAI (API key required)
                - "anthropic/claude-3-sonnet" - Anthropic (API key required)
            lm_api_key: API key (or use environment variable OPENAI_API_KEY/ANTHROPIC_API_KEY)
                Not required for Ollama (uses OLLAMA_HOST env var, default: http://localhost:11434)
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy not available - install with: pip install dspy-ai")

        self.config = config or Config.fused()

        # Initialize DSPy language model (DSPy 3.0+ uses LM class)
        if lm_model.startswith("ollama/"):
            # Ollama local inference: ollama/llama3.2:3b
            model_name = lm_model.split("/", 1)[1]
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.lm = dspy.LM(
                model=f"ollama/{model_name}",
                api_base=ollama_host,
                max_tokens=4096
            )
            logging.info(f"Using Ollama model: {model_name} at {ollama_host}")
        elif lm_model.startswith("openai/"):
            model_name = lm_model.split("/", 1)[1]
            self.lm = dspy.LM(model=model_name, api_key=lm_api_key, max_tokens=4096)
        elif lm_model.startswith("anthropic/"):
            model_name = lm_model.split("/", 1)[1]
            self.lm = dspy.LM(model=model_name, api_key=lm_api_key, max_tokens=4096)
        else:
            # Default to OpenAI
            self.lm = dspy.LM(model=lm_model, api_key=lm_api_key, max_tokens=4096)

        # Set as default LM
        dspy.settings.configure(lm=self.lm)

        # Cache for optimized programs
        self.program_cache: Dict[str, DSPyProgram] = {}

        # HoloLoom integration (lazy-loaded)
        self._orchestrator = None
        self._memory = None

        logging.info(f"DSPyHoloLoom initialized with model: {lm_model}")

    async def _get_orchestrator(self):
        """Lazy-load HoloLoom orchestrator"""
        if self._orchestrator is None:
            from HoloLoom.weaving_orchestrator import WeavingOrchestrator
            from HoloLoom.memory.backend_factory import create_memory_backend

            # Create memory backend
            self._memory = await create_memory_backend(self.config)

            # Create orchestrator
            self._orchestrator = WeavingOrchestrator(
                cfg=self.config,
                memory=self._memory
            )

        return self._orchestrator

    async def create_signature_from_pattern(
        self,
        pattern_name: str,
        description: str,
        inputs: List[str],
        outputs: List[str]
    ) -> DSPySignature:
        """
        Create a DSPy signature from a HoloLoom pattern.

        Args:
            pattern_name: Name of the pattern
            description: Human-readable description
            inputs: List of input field names
            outputs: List of output field names

        Returns:
            DSPySignature instance
        """
        sig = DSPySignature(
            name=pattern_name,
            description=description,
            inputs=inputs,
            outputs=outputs,
            metadata={"source": "hololoom_pattern"}
        )

        logging.info(f"Created signature: {pattern_name} ({len(inputs)} inputs → {len(outputs)} outputs)")

        return sig

    async def fetch_training_examples_from_memory(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch training examples from HoloLoom memory.

        Args:
            query: Search query to find relevant examples
            limit: Maximum number of examples

        Returns:
            List of training examples as dicts
        """
        orchestrator = await self._get_orchestrator()

        # Search memory
        search_query = Query(text=query)
        spacetime = await orchestrator.weave(search_query)

        # Extract examples from response
        examples = []

        # Parse response for examples (this is domain-specific)
        # For now, create synthetic examples from memories
        if hasattr(spacetime, 'context') and spacetime.context:
            for memory in spacetime.context[:limit]:
                example = {
                    "question": memory.get("query", ""),
                    "answer": memory.get("response", ""),
                    "context": memory.get("context", "")
                }

                if example["question"] and example["answer"]:
                    examples.append(example)

        logging.info(f"Fetched {len(examples)} training examples from memory")

        return examples

    async def optimize_from_memory(
        self,
        signature: DSPySignature,
        memory_query: str,
        optimization_config: Optional[DSPyOptimizationConfig] = None,
        validation_split: float = 0.2
    ) -> DSPyProgram:
        """
        Optimize a DSPy program using HoloLoom memory as training data.

        Args:
            signature: DSPy signature to optimize
            memory_query: Query to fetch training examples
            optimization_config: Optimization configuration
            validation_split: Fraction of data for validation

        Returns:
            Optimized DSPy program
        """
        opt_config = optimization_config or DSPyOptimizationConfig()

        # Fetch training examples
        examples = await self.fetch_training_examples_from_memory(
            memory_query,
            limit=opt_config.max_labeled_demos + 10
        )

        if not examples:
            raise ValueError(f"No training examples found for query: {memory_query}")

        # Convert to DSPy examples
        dspy_signature = signature.to_dspy_signature()
        dspy_examples = []

        for ex in examples:
            # Map fields to signature inputs/outputs
            example_dict = {}

            for input_field in signature.inputs:
                if input_field in ex:
                    example_dict[input_field] = ex[input_field]

            for output_field in signature.outputs:
                if output_field in ex:
                    example_dict[output_field] = ex[output_field]

            if len(example_dict) >= len(signature.inputs) + len(signature.outputs):
                dspy_examples.append(dspy.Example(**example_dict).with_inputs(*signature.inputs))

        logging.info(f"Converted {len(dspy_examples)} examples for optimization")

        # Split train/validation
        split_idx = int(len(dspy_examples) * (1 - validation_split))
        train_examples = dspy_examples[:split_idx]
        val_examples = dspy_examples[split_idx:]

        # Create base program (Predict module)
        base_program = dspy.Predict(dspy_signature)

        # Select optimizer
        if opt_config.optimizer == "bootstrap":
            optimizer = BootstrapFewShot(
                metric=opt_config.metric,
                max_bootstrapped_demos=opt_config.max_bootstrapped_demos,
                max_labeled_demos=opt_config.max_labeled_demos,
                teacher_settings=opt_config.teacher_settings
            )
        elif opt_config.optimizer == "mipro":
            optimizer = MIPRO(
                metric=opt_config.metric,
                num_threads=opt_config.num_threads,
                init_temperature=1.0
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.optimizer}")

        # Optimize
        logging.info(f"Optimizing with {opt_config.optimizer}...")

        optimized_program = optimizer.compile(
            base_program,
            trainset=train_examples,
            valset=val_examples if val_examples else None
        )

        # Evaluate
        validation_score = 0.0
        if val_examples and opt_config.metric:
            scores = [opt_config.metric(ex, optimized_program(ex)) for ex in val_examples]
            validation_score = sum(scores) / len(scores) if scores else 0.0

        # Create program wrapper
        program = DSPyProgram(
            signature=signature,
            program=optimized_program,
            optimizer=opt_config.optimizer,
            training_examples=len(train_examples),
            validation_score=validation_score,
            created_at=str(asyncio.get_event_loop().time()),
            metadata={
                "memory_query": memory_query,
                "val_examples": len(val_examples)
            }
        )

        # Cache
        self.program_cache[signature.name] = program

        logging.info(
            f"Optimization complete: {signature.name} "
            f"(score={validation_score:.3f}, examples={len(train_examples)})"
        )

        return program

    async def execute(
        self,
        program: DSPyProgram,
        use_hololoom_context: bool = True,
        **inputs
    ) -> Dict[str, Any]:
        """
        Execute an optimized DSPy program.

        Args:
            program: Optimized DSPy program
            use_hololoom_context: Augment inputs with HoloLoom context
            **inputs: Input values matching signature

        Returns:
            Output dict with results
        """
        # Augment with HoloLoom context if requested
        if use_hololoom_context and "question" in inputs:
            orchestrator = await self._get_orchestrator()

            # Get relevant context
            query = Query(text=inputs["question"])
            spacetime = await orchestrator.weave(query)

            # Add context to inputs
            if "context" in program.signature.inputs and spacetime.context:
                context_str = "\n".join(
                    mem.get("content", "") for mem in spacetime.context[:3]
                )
                inputs["context"] = context_str

        # Execute DSPy program
        try:
            result = program.program(**inputs)

            # Convert to dict
            output = {}
            for field in program.signature.outputs:
                if hasattr(result, field):
                    output[field] = getattr(result, field)

            logging.info(f"Executed {program.signature.name}: {list(output.keys())}")

            return output

        except Exception as e:
            logging.error(f"Execution failed: {e}")
            return {"error": str(e)}

    async def save_program(self, program: DSPyProgram, path: Path):
        """Save optimized program to disk"""
        program.program.save(str(path))

        # Save metadata
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "signature": {
                    "name": program.signature.name,
                    "description": program.signature.description,
                    "inputs": program.signature.inputs,
                    "outputs": program.signature.outputs
                },
                "optimizer": program.optimizer,
                "training_examples": program.training_examples,
                "validation_score": program.validation_score,
                "created_at": program.created_at,
                "metadata": program.metadata
            }, f, indent=2)

        logging.info(f"Saved program to {path}")

    async def load_program(self, path: Path) -> DSPyProgram:
        """Load optimized program from disk"""
        # Load metadata
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'r') as f:
            meta = json.load(f)

        # Recreate signature
        sig = DSPySignature(
            name=meta["signature"]["name"],
            description=meta["signature"]["description"],
            inputs=meta["signature"]["inputs"],
            outputs=meta["signature"]["outputs"]
        )

        # Load program
        dspy_sig = sig.to_dspy_signature()
        loaded_program = dspy.Predict(dspy_sig)
        loaded_program.load(str(path))

        program = DSPyProgram(
            signature=sig,
            program=loaded_program,
            optimizer=meta["optimizer"],
            training_examples=meta["training_examples"],
            validation_score=meta["validation_score"],
            created_at=meta["created_at"],
            metadata=meta.get("metadata", {})
        )

        logging.info(f"Loaded program from {path}")

        return program


def create_signature(
    description: str,
    inputs: List[str],
    outputs: List[str],
    name: Optional[str] = None
) -> DSPySignature:
    """
    Convenience function to create a DSPy signature.

    Args:
        description: What the signature does
        inputs: Input field names
        outputs: Output field names
        name: Optional name (auto-generated from description)

    Returns:
        DSPySignature instance
    """
    if name is None:
        # Generate name from description
        name = description.split()[0].capitalize() + "Signature"

    return DSPySignature(
        name=name,
        description=description,
        inputs=inputs,
        outputs=outputs
    )


# Common signatures for HoloLoom tasks

class QuestionAnswering(DSPySignature):
    """Answer questions using relevant context"""
    name = "QuestionAnswering"
    description = "Answer questions using relevant context from memory"
    inputs = ["question", "context"]
    outputs = ["answer", "confidence"]


class MemorySynthesis(DSPySignature):
    """Synthesize information from multiple memory shards"""
    name = "MemorySynthesis"
    description = "Synthesize coherent response from multiple memory fragments"
    inputs = ["query", "memories"]
    outputs = ["synthesis", "sources"]


class ReasoningChain(DSPySignature):
    """Multi-step reasoning with justification"""
    name = "ReasoningChain"
    description = "Multi-step reasoning with intermediate justifications"
    inputs = ["question", "context"]
    outputs = ["reasoning_steps", "final_answer", "confidence"]


# Example usage and demo
async def demo_dspy_hololoom():
    """Demonstrate DSPy-HoloLoom integration"""

    print("🎯 DSPy-HoloLoom Integration Demo\n")

    # Initialize bridge
    bridge = DSPyHoloLoom(
        config=Config.fused(),
        lm_model="openai/gpt-4o-mini"
    )

    # 1. Create signature
    print("1️⃣  Creating DSPy signature...")
    sig = create_signature(
        "Answer technical questions accurately",
        inputs=["question", "context"],
        outputs=["answer", "confidence"]
    )
    print(f"   Created: {sig.name} ({len(sig.inputs)} inputs → {len(sig.outputs)} outputs)\n")

    # 2. Optimize from memory
    print("2️⃣  Optimizing from HoloLoom memory...")
    try:
        program = await bridge.optimize_from_memory(
            signature=sig,
            memory_query="technical Q&A examples",
            optimization_config=DSPyOptimizationConfig(
                optimizer="bootstrap",
                max_bootstrapped_demos=3,
                max_labeled_demos=10
            )
        )
        print(f"   Optimized! Score: {program.validation_score:.3f}\n")
    except ValueError as e:
        print(f"   Skipped optimization: {e}\n")
        # Create unoptimized program for demo
        base_program = dspy.Predict(sig.to_dspy_signature())
        program = DSPyProgram(
            signature=sig,
            program=base_program,
            optimizer="none",
            training_examples=0,
            validation_score=0.0,
            created_at=str(asyncio.get_event_loop().time())
        )

    # 3. Execute
    print("3️⃣  Executing optimized program...")
    result = await bridge.execute(
        program,
        question="What is Thompson Sampling?",
        use_hololoom_context=True
    )
    print(f"   Answer: {result.get('answer', 'N/A')[:100]}...")
    print(f"   Confidence: {result.get('confidence', 'N/A')}\n")

    # 4. Save program
    print("4️⃣  Saving optimized program...")
    save_path = Path("./optimized_qa_program.json")
    await bridge.save_program(program, save_path)
    print(f"   Saved to: {save_path}\n")

    print("✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_dspy_hololoom())
