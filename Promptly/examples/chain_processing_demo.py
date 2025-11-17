#!/usr/bin/env python3
"""
Chain Processing Demo

Demonstrates advanced chain processing capabilities including:
- Conditional logic
- Parallel execution
- Loop patterns
- Retry mechanisms
- Data transformation
- Execution tracing and visualization
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from promptly.chain_dsl import ChainDSL, ChainExecutionContext
from promptly.chain_visualization import visualize_chain, visualize_trace
from promptly.chain_tracing import create_tracer
from promptly.plugins.processors import (
    ConditionalProcessor,
    ParallelProcessor,
    LoopProcessor,
    RetryProcessor,
    TransformProcessor
)


def demo_1_conditional_processor():
    """Demo: Conditional processor with pattern matching"""
    print("\n" + "="*60)
    print("DEMO 1: Conditional Processor")
    print("="*60)

    processor = ConditionalProcessor()

    # Test data
    test_input = {"score": 0.85, "category": "high_quality"}

    # Configuration
    config = {
        "conditions": [
            {
                "type": "numeric",
                "field": "score",
                "operator": "gt",
                "value": 0.7,
                "action": {
                    "type": "set",
                    "field": "result",
                    "value": "Excellent quality!"
                }
            },
            {
                "type": "keyword",
                "field": "category",
                "keywords": ["high_quality"],
                "action": {
                    "type": "merge",
                    "data": {"badge": "gold"}
                }
            }
        ]
    }

    result = processor.process(test_input, config)
    print(f"Input: {test_input}")
    print(f"Result: {result}")


def demo_2_parallel_processor():
    """Demo: Parallel execution with aggregation"""
    print("\n" + "="*60)
    print("DEMO 2: Parallel Processor")
    print("="*60)

    processor = ParallelProcessor()

    # Mock executor
    def mock_executor(prompt, inputs):
        return f"Processed: {prompt[:30]}..."

    processor.set_executor(mock_executor)

    # Test data
    test_input = {"query": "What is the weather?"}

    # Configuration
    config = {
        "tasks": [
            {"name": "task1", "prompt": "Check weather database for {query}"},
            {"name": "task2", "prompt": "Check weather API for {query}"},
            {"name": "task3", "prompt": "Check weather cache for {query}"}
        ],
        "aggregation": "concat",
        "timeout": 5.0,
        "max_workers": 3
    }

    result = processor.process(test_input, config)
    print(f"Input: {test_input}")
    print(f"Success count: {result['success_count']}/{result['total_count']}")
    print(f"Aggregated result: {result['aggregated_result'][:100]}...")


def demo_3_loop_processor():
    """Demo: Loop processor with iteration"""
    print("\n" + "="*60)
    print("DEMO 3: Loop Processor")
    print("="*60)

    processor = LoopProcessor()

    # Test data
    test_input = {"items": [1, 2, 3, 4, 5], "multiplier": 2}

    # Configuration - map operation
    config = {
        "loop_type": "map",
        "items": "items",
        "action": {
            "value": "item * 2"
        },
        "map_field": "output"
    }

    result = processor.process(test_input, config)
    print(f"Input: {test_input}")
    print(f"Mapped results: {result['mapped_results']}")


def demo_4_retry_processor():
    """Demo: Retry with exponential backoff"""
    print("\n" + "="*60)
    print("DEMO 4: Retry Processor")
    print("="*60)

    processor = RetryProcessor()

    # Mock executor that fails twice then succeeds
    attempt_count = [0]

    def flaky_executor(action, data):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            return {"error": f"Attempt {attempt_count[0]} failed", "success": False}
        return {"result": "Success!", "success": True}

    processor._executor = flaky_executor

    # Test data
    test_input = {"task": "process_data"}

    # Configuration
    config = {
        "max_attempts": 5,
        "backoff_strategy": "exponential",
        "initial_delay": 0.1,
        "action": {"value": "test"}
    }

    result = processor.process(test_input, config)
    print(f"Input: {test_input}")
    print(f"Success: {result['success']}")
    print(f"Attempts: {result['attempts']}")
    print(f"Retry history: {len(result['retry_history'])} retries")


def demo_5_transform_processor():
    """Demo: Data extraction and transformation"""
    print("\n" + "="*60)
    print("DEMO 5: Transform Processor")
    print("="*60)

    processor = TransformProcessor()

    # Test data with JSON
    test_input = {
        "output": '{"name": "John", "age": 30, "city": "New York"}'
    }

    # Configuration - extract JSON
    config = {
        "transform_type": "extract",
        "method": "json",
        "source_field": "output",
        "target_field": "extracted"
    }

    result = processor.process(test_input, config)
    print(f"Input: {test_input['output']}")
    print(f"Extracted: {result['extracted']}")

    # Test data for regex extraction
    test_input2 = {
        "output": "The price is $99.99 and discount is 20%"
    }

    config2 = {
        "transform_type": "extract",
        "method": "regex",
        "pattern": r'\$(\d+\.\d+)',
        "source_field": "output",
        "target_field": "price"
    }

    result2 = processor.process(test_input2, config2)
    print(f"\nInput: {test_input2['output']}")
    print(f"Extracted price: ${result2['price']}")


def demo_6_chain_dsl():
    """Demo: Chain DSL workflow"""
    print("\n" + "="*60)
    print("DEMO 6: Chain DSL Workflow")
    print("="*60)

    # Create simple chain in YAML format
    chain_yaml = """
name: simple_workflow
description: Simple data processing workflow
version: "1.0"

variables:
  threshold: 0.5

steps:
  - name: extract_data
    type: transform
    config:
      transform_type: extract
      method: json
      source_field: input
      target_field: data

  - name: validate_data
    type: transform
    depends_on: [extract_data]
    config:
      transform_type: validate
      source_field: data
      validation_rules:
        - type: required

  - name: process_data
    type: conditional
    depends_on: [validate_data]
    config:
      conditions:
        - type: keyword
          field: is_valid
          keywords: ["true", "True"]
          action:
            type: set
            field: result
            value: "Data is valid!"
      default:
        type: set
        field: result
        value: "Data validation failed"
"""

    # Parse and execute chain
    dsl = ChainDSL()
    chain_def = dsl.load_chain_from_string(chain_yaml)

    print(f"Chain: {chain_def['name']}")
    print(f"Description: {chain_def['description']}")
    print(f"Steps: {len(chain_def['steps'])}")

    # Validate chain
    validation = dsl.validate_chain(chain_def)
    print(f"\nValidation: {'✓ Valid' if validation['valid'] else '✗ Invalid'}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")


def demo_7_visualization():
    """Demo: Chain visualization"""
    print("\n" + "="*60)
    print("DEMO 7: Chain Visualization")
    print("="*60)

    # Create simple chain
    chain_yaml = """
name: visualization_demo
steps:
  - name: input_processor
    type: transform
    config:
      transform_type: extract

  - name: parallel_tasks
    type: parallel
    depends_on: [input_processor]
    config:
      tasks: []

  - name: conditional_branch
    type: conditional
    depends_on: [parallel_tasks]
    config:
      conditions: []

  - name: final_output
    type: transform
    depends_on: [conditional_branch]
    config:
      transform_type: aggregate
"""

    dsl = ChainDSL()
    chain_def = dsl.load_chain_from_string(chain_yaml)

    # Generate visualizations
    print("\nMermaid Diagram:")
    print("-" * 40)
    mermaid = visualize_chain(chain_def, format="mermaid")
    print(mermaid[:500] + "..." if len(mermaid) > 500 else mermaid)

    print("\n\nASCII Representation:")
    print("-" * 40)
    ascii_viz = visualize_chain(chain_def, format="ascii")
    print(ascii_viz)


def demo_8_execution_tracing():
    """Demo: Execution tracing"""
    print("\n" + "="*60)
    print("DEMO 8: Execution Tracing")
    print("="*60)

    tracer = create_tracer(trace_level="standard")

    # Simulate chain execution
    tracer.start_execution()

    # Step 1
    tracer.start_step("extract_data", "transform", {"input": "test"})
    import time
    time.sleep(0.1)
    tracer.end_step("extract_data", "completed", {"data": "extracted"})

    # Step 2
    tracer.start_step("process_data", "conditional", {"data": "extracted"})
    time.sleep(0.05)
    tracer.end_step("process_data", "completed", {"result": "processed"})

    # Step 3 (with error)
    tracer.start_step("finalize", "transform", {"result": "processed"})
    time.sleep(0.02)
    tracer.end_step("finalize", "failed", error="Validation failed")

    tracer.end_execution()

    # Print summary
    print("\nExecution Summary:")
    print("-" * 40)
    summary = tracer.get_summary()
    print(f"Total duration: {summary['total_duration']:.3f}s")
    print(f"Total steps: {summary['total_steps']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")

    print("\nStep Details:")
    for step in summary['steps']:
        status_symbol = "✓" if step['status'] == 'completed' else "✗"
        duration = f"{step['duration']:.3f}s" if step['duration'] else "N/A"
        print(f"{status_symbol} {step['name']} ({step['type']}) - {duration}")

    # Print trace timeline
    print("\n" + visualize_trace(tracer.get_trace(), format="timeline"))


def demo_9_load_workflow():
    """Demo: Load and visualize example workflows"""
    print("\n" + "="*60)
    print("DEMO 9: Example Workflows")
    print("="*60)

    workflow_dir = os.path.join(os.path.dirname(__file__), 'workflows')

    workflows = ['rag_pipeline.yaml', 'ab_testing.yaml', 'multi_agent.yaml']

    for workflow_file in workflows:
        workflow_path = os.path.join(workflow_dir, workflow_file)

        if os.path.exists(workflow_path):
            print(f"\n{workflow_file}:")
            print("-" * 40)

            try:
                dsl = ChainDSL()
                chain_def = dsl.load_chain(workflow_path)

                print(f"Name: {chain_def['name']}")
                print(f"Description: {chain_def['description']}")
                print(f"Steps: {len(chain_def['steps'])}")

                # Validate
                validation = dsl.validate_chain(chain_def)
                print(f"Valid: {validation['valid']}")

                if validation['errors']:
                    print(f"Errors: {validation['errors'][:2]}")

            except Exception as e:
                print(f"Error loading workflow: {e}")
        else:
            print(f"\nWorkflow not found: {workflow_path}")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("PROMPTLY ADVANCED CHAIN PROCESSING DEMO")
    print("="*60)

    demos = [
        demo_1_conditional_processor,
        demo_2_parallel_processor,
        demo_3_loop_processor,
        demo_4_retry_processor,
        demo_5_transform_processor,
        demo_6_chain_dsl,
        demo_7_visualization,
        demo_8_execution_tracing,
        demo_9_load_workflow
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\nError in {demo.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
