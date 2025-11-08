# Question Answering (Basic) - Starter Template

A simple question-answering system using DSPy + HoloLoom.

## What This Does

- Takes a question as input
- Retrieves relevant context from knowledge base
- Generates accurate, context-grounded answer
- Returns confidence score

## Quick Start

### 1. Set Up Environment

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Install dependencies (if not already installed)
pip install dspy-ai
```

### 2. Run the Demo

```bash
cd examples/starter_templates/qa_basic
python qa_basic.py
```

**Expected output**:
```
>>> Promptly Starter Template: Question Answering (Basic)
======================================================================

Step 1: Creating Q&A signature...
>> SUCCESS: Signature created
   Inputs: ['question', 'context']
   Outputs: ['answer', 'confidence']

Step 2: Creating DSPy program...
>> SUCCESS: Program created (unoptimized)

Step 3: Testing with sample questions...

Question 1: What is Thompson Sampling?
Context: Thompson Sampling is a Bayesian approach...
Answer: Thompson Sampling is a Bayesian approach to the multi-armed bandit problem...
Confidence: 0.95

Question 2: How do Matryoshka embeddings work?
Context: Matryoshka embeddings are multi-scale representations...
Answer: Matryoshka embeddings work by providing multiple scales of representation...
Confidence: 0.92

...
```

## How It Works

### Architecture

```
User Question
     ↓
Retrieve Context (simple keyword matching)
     ↓
DSPy Signature (question + context → answer + confidence)
     ↓
DSPy Program (unoptimized Predict)
     ↓
Answer + Confidence
```

### Key Components

**1. Signature Definition**
```python
signature = create_signature(
    "Answer technical questions accurately using the provided context",
    inputs=["question", "context"],
    outputs=["answer", "confidence"]
)
```

**2. Knowledge Base**
```python
KNOWLEDGE_BASE = [
    MemoryShard(content="...", metadata={...}),
    # Add your own shards here
]
```

**3. Simple Retrieval**
```python
def retrieve_context(question, knowledge_base):
    # Simple keyword matching
    # In production, use HoloLoom's semantic retrieval
    ...
```

**4. Program Creation**
```python
program = dspy.Predict(signature.to_dspy_signature())
result = program(question=question, context=context)
```

## Customization

### Add Your Own Knowledge Base

Replace the sample knowledge base with your own:

```python
KNOWLEDGE_BASE = [
    MemoryShard(
        content="Your content here",
        metadata={"topic": "your_topic", "type": "definition"}
    ),
    MemoryShard(
        content="More content...",
        metadata={"topic": "another_topic", "type": "example"}
    ),
]
```

### Change the Task Instruction

Modify the signature instruction:

```python
signature = create_signature(
    "Your custom instruction here",
    inputs=["question", "context"],
    outputs=["answer", "confidence"]
)
```

### Add More Output Fields

```python
signature = create_signature(
    "Answer questions with sources",
    inputs=["question", "context"],
    outputs=["answer", "confidence", "sources", "reasoning"]
)
```

## Optimization (Next Level)

### 1. Add Training Examples

Create `examples.json`:
```json
[
  {
    "question": "What is X?",
    "context": "X is a technique that...",
    "answer": "X is a technique that solves Y by doing Z.",
    "confidence": "0.95"
  }
]
```

### 2. Optimize with BootstrapFewShot

```python
from dspy.teleprompt import BootstrapFewShot

# Define metric
def qa_metric(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# Create optimizer
optimizer = BootstrapFewShot(metric=qa_metric, max_bootstrapped_demos=3)

# Load examples
import json
with open("examples.json") as f:
    examples = json.load(f)

# Convert to DSPy format
trainset = [dspy.Example(**ex).with_inputs("question", "context") for ex in examples]

# Optimize
optimized_program = optimizer.compile(program, trainset=trainset)
```

### 3. Optimize from HoloLoom Memory

```python
from HoloLoom.promptly import DSPyHoloLoom
from HoloLoom.config import Config

# Create bridge
bridge = DSPyHoloLoom(config=Config.fused())

# Optimize using memory
optimized = await bridge.optimize_from_memory(
    signature=signature,
    memory_query="qa_examples",  # Query HoloLoom memory
    optimization_config=config
)

# Use optimized program
result = optimized(question=question, context=context)
```

## Metrics and Evaluation

### Add Custom Metrics

```python
from HoloLoom.promptly.metrics_system import MetricsEvaluator, MetricType

# Create evaluator
evaluator = MetricsEvaluator(
    metrics=[
        MetricType.ACCURACY,
        MetricType.COMPLETENESS,
        MetricType.CLARITY
    ],
    threshold=0.8
)

# Evaluate
result = evaluator.evaluate(
    example={"answer": "Expected answer"},
    prediction={"answer": "Predicted answer"},
    context={"question": question, "context": context}
)

print(f"Overall Score: {result.overall_score}")
print(f"Passed: {result.passed}")
```

## Integration with HoloLoom

### Use HoloLoom's Memory System

```python
from HoloLoom import HoloLoom
from HoloLoom.config import Config

# Create HoloLoom instance
async with HoloLoom(config=Config.fused()) as loom:
    # Experience (form memories)
    for shard in KNOWLEDGE_BASE:
        await loom.experience(shard.content)

    # Recall (retrieve context)
    memories = await loom.recall(question)
    context = " ".join([m.content for m in memories])

    # Use with Q&A
    result = program(question=question, context=context)
```

### Use HoloLoom's Orchestrator

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query

async with WeavingOrchestrator(cfg=config, shards=KNOWLEDGE_BASE) as orchestrator:
    # Full weaving cycle
    spacetime = await orchestrator.weave(Query(text=question))

    print(f"Response: {spacetime.response}")
    print(f"Confidence: {spacetime.confidence}")
    print(f"Context used: {spacetime.metadata.get('context_used')}")
```

## Workflow Definition (YAML)

See `qa_workflow.yaml` for declarative workflow definition:

```yaml
version: "1.0"
name: qa_basic
description: Basic question answering with context retrieval

steps:
  - name: retrieve
    module: memory_retrieval
    inputs:
      query: "{input.question}"
    outputs:
      - context

  - name: answer
    module: qa_predict
    inputs:
      question: "{input.question}"
      context: "{retrieve.context}"
    outputs:
      - answer
      - confidence
```

Run with workflow adapter:

```python
from HoloLoom.promptly import DSPyWorkflowAdapter

adapter = DSPyWorkflowAdapter(config=Config.fast())
workflow = await adapter.load_workflow("qa_workflow.yaml")
result = await workflow.execute({"question": "What is X?"})
```

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"

**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Issue: "No relevant context found"

**Solution**:
1. Add more examples to KNOWLEDGE_BASE
2. Improve retrieval (use semantic search instead of keywords)
3. Check question phrasing

### Issue: Low confidence scores

**Solution**:
1. Add training examples
2. Optimize with BootstrapFewShot
3. Improve context quality
4. Use more specific task instruction

### Issue: Incorrect answers

**Solution**:
1. Check context relevance
2. Add examples showing correct reasoning
3. Optimize program with metric-based optimizer
4. Use chain-of-thought prompting

## Next Steps

1. **Add more examples**: Expand KNOWLEDGE_BASE with domain-specific content
2. **Optimize**: Use BootstrapFewShot or MIPRO optimizers
3. **Add metrics**: Implement custom evaluation metrics
4. **Create workflow**: Chain multiple steps (retrieve → reason → verify)
5. **Deploy**: Integrate with your application

## Learning Resources

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [HoloLoom Master Index](../../../MASTER_INDEX.md)
- [Promptly Quick Start](../../../QUICK_START_GUIDE.md)
- [Architecture Guide](../../../ARCHITECTURE_6_PROBLEMS.md)

## Contributing

Found a bug? Want to improve this template?

1. Open an issue on GitHub
2. Submit a pull request
3. Share your improvements with the community

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

---

**Happy building! 🚀**

Part of Promptly - The Universal AI Reliability Layer
Open source (MIT): https://github.com/yourusername/promptly
