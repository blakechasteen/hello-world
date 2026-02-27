#!/usr/bin/env python3
"""
DSPy-Promptly Integration Demo

Demonstrates the complete DSPy-HoloLoom-Promptly integration:
1. Creating DSPy signatures from hololoom patterns
2. Optimizing programs using HoloLoom memory
3. Building workflows with DSPy steps
4. Executing end-to-end pipelines

Run:
    PYTHONPATH=. python demos/demo_dspy_promptly_integration.py
"""

import asyncio
import logging
from pathlib import Path

from hololoom.config import Config
from hololoom.promptly.dspy_bridge import DSPyHoloLoom, create_signature
from hololoom.promptly.dspy_workflow_adapter import (
    DSPyWorkflowAdapter,
    create_qa_workflow,
    create_research_workflow
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


async def demo_1_basic_signature():
    """Demo 1: Creating and using basic DSPy signatures"""
    print("\n" + "="*70)
    print("DEMO 1: Basic DSPy Signature Creation")
    print("="*70 + "\n")

    # Initialize bridge
    bridge = DSPyHoloLoom(
        config=Config.fused(),
        lm_model="openai/gpt-4o-mini"
    )

    # Create signature
    sig = create_signature(
        description="Explain technical concepts clearly",
        inputs=["concept", "context"],
        outputs=["explanation", "examples"],
        name="ConceptExplainer"
    )

    print(f"📝 Created signature: {sig.name}")
    print(f"   Inputs: {', '.join(sig.inputs)}")
    print(f"   Outputs: {', '.join(sig.outputs)}")
    print(f"   Description: {sig.description}\n")

    # Create unoptimized program for demo
    import dspy
    dspy_sig = sig.to_dspy_signature()
    program = dspy.Predict(dspy_sig)

    # Execute
    print("🚀 Executing program...")
    result = program(
        concept="Thompson Sampling",
        context="Bayesian bandit algorithms for exploration/exploitation"
    )

    print(f"\n✅ Result:")
    print(f"   Explanation: {result.explanation[:150]}...")
    if hasattr(result, 'examples'):
        print(f"   Examples: {result.examples[:100]}...")

    return bridge


async def demo_2_optimization_from_memory(bridge: DSPyHoloLoom):
    """Demo 2: Optimizing programs using HoloLoom memory"""
    print("\n" + "="*70)
    print("DEMO 2: Optimization from hololoom Memory")
    print("="*70 + "\n")

    # Create signature for Q&A
    sig = create_signature(
        description="Answer questions accurately using provided context",
        inputs=["question", "context"],
        outputs=["answer", "confidence"],
        name="AccurateQA"
    )

    print(f"📝 Signature: {sig.name}")
    print(f"   Will optimize using HoloLoom memory...\n")

    # Note: This will attempt to fetch examples from memory
    # In a real system, you'd have training data in HoloLoom
    print("⚠️  Note: This demo requires training examples in HoloLoom memory")
    print("   Skipping optimization for demo purposes\n")

    # Create unoptimized version for demo
    import dspy
    from hololoom.promptly.dspy_bridge import DSPyProgram

    dspy_sig = sig.to_dspy_signature()
    base_program = dspy.Predict(dspy_sig)

    program = DSPyProgram(
        signature=sig,
        program=base_program,
        optimizer="none",
        training_examples=0,
        validation_score=0.0,
        created_at=str(asyncio.get_event_loop().time())
    )

    # Execute
    print("🚀 Executing unoptimized program...")
    result = await bridge.execute(
        program,
        question="What is the difference between ε-greedy and Thompson Sampling?",
        context="Both are bandit algorithms. ε-greedy explores randomly with probability ε. Thompson Sampling samples from posterior distributions.",
        use_hololoom_context=False
    )

    print(f"\n✅ Result:")
    print(f"   Answer: {result.get('answer', 'N/A')[:200]}...")
    print(f"   Confidence: {result.get('confidence', 'N/A')}")

    return program


async def demo_3_workflow_creation(bridge: DSPyHoloLoom):
    """Demo 3: Creating DSPy workflows"""
    print("\n" + "="*70)
    print("DEMO 3: Creating DSPy Workflows")
    print("="*70 + "\n")

    # Create adapter
    adapter = DSPyWorkflowAdapter(bridge)

    # Create QA workflow
    print("📋 Creating Question-Answering workflow...")
    qa_workflow = await create_qa_workflow(adapter)

    print(f"\n✅ Workflow created: {qa_workflow.name}")
    print(f"   Description: {qa_workflow.description}")
    print(f"   Steps: {len(qa_workflow.steps)}\n")

    for i, step in enumerate(qa_workflow.steps, 1):
        print(f"   {i}. {step.step_id}")
        print(f"      Signature: {step.signature_name}")
        print(f"      Inputs: {', '.join(step.inputs.keys())}")
        print(f"      Outputs: {', '.join(step.outputs)}")
        print(f"      Optimize: {step.optimize}")
        print()

    return adapter, qa_workflow


async def demo_4_workflow_execution(adapter: DSPyWorkflowAdapter, workflow):
    """Demo 4: Executing DSPy workflows"""
    print("\n" + "="*70)
    print("DEMO 4: Executing DSPy Workflow")
    print("="*70 + "\n")

    # Execute workflow
    print(f"🚀 Executing workflow: {workflow.name}")
    print(f"   Query: 'What is Thompson Sampling and when should I use it?'\n")

    result = await adapter.execute_workflow(
        workflow,
        initial_inputs={
            "query": "What is Thompson Sampling and when should I use it?"
        },
        optimize_on_the_fly=False
    )

    print(f"\n✅ Execution complete!")
    print(f"   Success: {result['success']}")
    print(f"   Steps executed: {len(result['trace'])}\n")

    # Show execution trace
    print("📊 Execution trace:")
    for trace_item in result["trace"]:
        status = "✓" if trace_item["success"] else "✗"
        duration = trace_item.get("duration_ms", 0)
        print(f"   {status} {trace_item['step_id']:<15} {duration:>6.1f}ms")

        if trace_item["success"]:
            outputs = trace_item.get("outputs", {})
            for key, value in outputs.items():
                if isinstance(value, str):
                    preview = value[:80] + "..." if len(value) > 80 else value
                    print(f"      └─ {key}: {preview}")
        else:
            print(f"      └─ Error: {trace_item.get('error', 'Unknown')}")

    print()

    # Final answer
    if result['success'] and 'answer.answer' in result['context']:
        print("💡 Final Answer:")
        print(f"   {result['context']['answer.answer'][:300]}...")
        print()

    return result


async def demo_5_statistics(adapter: DSPyWorkflowAdapter):
    """Demo 5: Workflow execution statistics"""
    print("\n" + "="*70)
    print("DEMO 5: Execution Statistics")
    print("="*70 + "\n")

    stats = adapter.get_execution_statistics()

    print("📊 Statistics:")
    print(f"   Total executions: {stats['total_executions']}")
    print(f"   Successful: {stats['successful_executions']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Total steps: {stats['total_steps']}")
    print(f"   Successful steps: {stats['successful_steps']}")
    print(f"   Step success rate: {stats['step_success_rate']:.1%}")
    print(f"   Avg duration: {stats['avg_duration_ms']:.1f}ms")
    print()


async def demo_6_workflow_persistence(adapter: DSPyWorkflowAdapter, workflow):
    """Demo 6: Saving and loading workflows"""
    print("\n" + "="*70)
    print("DEMO 6: Workflow Persistence")
    print("="*70 + "\n")

    # Save workflow
    save_path = Path("./demo_qa_workflow.yaml")
    await adapter.save_workflow(workflow, save_path)
    print(f"💾 Saved workflow to: {save_path}\n")

    # Load workflow
    loaded = await adapter.load_workflow(save_path)
    print(f"📂 Loaded workflow: {loaded.name}")
    print(f"   Steps: {len(loaded.steps)}")
    print(f"   Matches original: {loaded.name == workflow.name}\n")


async def demo_7_custom_workflow():
    """Demo 7: Creating custom workflows"""
    print("\n" + "="*70)
    print("DEMO 7: Custom Workflow Creation")
    print("="*70 + "\n")

    bridge = DSPyHoloLoom(config=Config.fused(), lm_model="openai/gpt-4o-mini")
    adapter = DSPyWorkflowAdapter(bridge)

    # Register custom signatures
    adapter.register_signature(create_signature(
        "Extract key entities from text",
        inputs=["text"],
        outputs=["entities", "entity_types"],
        name="EntityExtraction"
    ))

    adapter.register_signature(create_signature(
        "Summarize text concisely",
        inputs=["text"],
        outputs=["summary", "key_points"],
        name="TextSummarization"
    ))

    adapter.register_signature(create_signature(
        "Generate insights from entities and summary",
        inputs=["entities", "summary"],
        outputs=["insights", "recommendations"],
        name="InsightGeneration"
    ))

    # Create custom workflow
    custom_workflow = adapter.create_workflow(
        name="Text_Analysis_Pipeline",
        description="Extract entities, summarize, and generate insights",
        steps=[
            {
                "step_id": "extract_entities",
                "signature": "EntityExtraction",
                "inputs": {"text": "{input_text}"},
                "outputs": ["entities", "entity_types"]
            },
            {
                "step_id": "summarize",
                "signature": "TextSummarization",
                "inputs": {"text": "{input_text}"},
                "outputs": ["summary", "key_points"]
            },
            {
                "step_id": "generate_insights",
                "signature": "InsightGeneration",
                "inputs": {
                    "entities": "{extract_entities.entities}",
                    "summary": "{summarize.summary}"
                },
                "outputs": ["insights", "recommendations"]
            }
        ]
    )

    print(f"✅ Created custom workflow: {custom_workflow.name}")
    print(f"   Description: {custom_workflow.description}")
    print(f"   Steps: {len(custom_workflow.steps)}\n")

    for i, step in enumerate(custom_workflow.steps, 1):
        print(f"   {i}. {step.step_id} ({step.signature_name})")

    print()

    # Execute on sample text
    sample_text = """
    HoloLoom is a neural decision-making system that combines multi-scale Matryoshka embeddings
    with knowledge graph memory and Thompson Sampling exploration. It implements a complete
    9-layer weaving architecture with Phase 5 compositional caching for 10-300x speedup.
    """

    print("🚀 Executing custom workflow on sample text...\n")

    result = await adapter.execute_workflow(
        custom_workflow,
        initial_inputs={"input_text": sample_text.strip()},
        optimize_on_the_fly=False
    )

    print(f"✅ Execution complete: {result['success']}\n")

    if result['success']:
        print("📊 Results:")
        if 'extract_entities.entities' in result['context']:
            entities = result['context']['extract_entities.entities']
            print(f"   Entities: {entities[:150]}...")

        if 'summarize.summary' in result['context']:
            summary = result['context']['summarize.summary']
            print(f"   Summary: {summary[:150]}...")

        if 'generate_insights.insights' in result['context']:
            insights = result['context']['generate_insights.insights']
            print(f"   Insights: {insights[:150]}...")

    print()


async def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🎯 DSPy-Promptly-HoloLoom Integration Demos")
    print("="*70)

    try:
        # Demo 1: Basic signature
        bridge = await demo_1_basic_signature()

        # Demo 2: Optimization (note: requires training data)
        program = await demo_2_optimization_from_memory(bridge)

        # Demo 3: Workflow creation
        adapter, workflow = await demo_3_workflow_creation(bridge)

        # Demo 4: Workflow execution
        result = await demo_4_workflow_execution(adapter, workflow)

        # Demo 5: Statistics
        await demo_5_statistics(adapter)

        # Demo 6: Persistence
        await demo_6_workflow_persistence(adapter, workflow)

        # Demo 7: Custom workflow
        await demo_7_custom_workflow()

        print("\n" + "="*70)
        print("✅ All demos completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
