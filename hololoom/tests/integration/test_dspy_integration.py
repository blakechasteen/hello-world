#!/usr/bin/env python3
"""
Integration tests for DSPy-HoloLoom-Promptly integration

Tests:
1. DSPy signature creation and conversion
2. Basic program execution
3. Workflow creation and execution
4. Input/output mapping in workflows
5. Workflow persistence
6. Error handling

Run:
    pytest hololoom/tests/integration/test_dspy_integration.py -v
"""

import pytest
import asyncio
from pathlib import Path
import tempfile

from hololoom.config import Config
from hololoom.promptly.dspy_bridge import (
    DSPyHoloLoom,
    DSPySignature,
    create_signature,
    DSPY_AVAILABLE
)
from hololoom.promptly.dspy_workflow_adapter import (
    DSPyWorkflowAdapter,
    DSPyWorkflowStep,
    create_qa_workflow
)


# Skip all tests if DSPy not available
pytestmark = pytest.mark.skipif(
    not DSPY_AVAILABLE,
    reason="DSPy not available - install with: pip install dspy-ai"
)


@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def bridge():
    """DSPy bridge instance"""
    config = Config.bare()  # Use BARE for faster tests
    bridge = DSPyHoloLoom(
        config=config,
        lm_model="openai/gpt-4o-mini"
    )
    yield bridge


@pytest.fixture
async def adapter(bridge):
    """DSPy workflow adapter instance"""
    adapter = DSPyWorkflowAdapter(bridge)
    yield adapter


class TestDSPySignature:
    """Test DSPy signature creation and conversion"""

    def test_create_signature(self):
        """Test basic signature creation"""
        sig = create_signature(
            description="Test signature",
            inputs=["input1", "input2"],
            outputs=["output1"],
            name="TestSig"
        )

        assert sig.name == "TestSig"
        assert sig.description == "Test signature"
        assert sig.inputs == ["input1", "input2"]
        assert sig.outputs == ["output1"]

    def test_signature_auto_naming(self):
        """Test automatic name generation"""
        sig = create_signature(
            description="Answer questions accurately",
            inputs=["question"],
            outputs=["answer"]
        )

        assert sig.name == "AnswerSignature"

    def test_signature_to_dspy(self):
        """Test conversion to DSPy signature class"""
        sig = create_signature(
            description="Test conversion",
            inputs=["x", "y"],
            outputs=["z"],
            name="ConversionTest"
        )

        dspy_sig = sig.to_dspy_signature()

        # Should be a class
        assert isinstance(dspy_sig, type)
        assert dspy_sig.__name__ == "ConversionTest"

    def test_dataclass_signature(self):
        """Test dataclass-based signature"""
        sig = DSPySignature(
            name="DataclassTest",
            description="Test dataclass",
            inputs=["a", "b"],
            outputs=["c"]
        )

        assert sig.name == "DataclassTest"
        assert len(sig.inputs) == 2
        assert len(sig.outputs) == 1
        assert sig.examples == []


class TestDSPyBridge:
    """Test DSPy-HoloLoom bridge functionality"""

    @pytest.mark.asyncio
    async def test_bridge_initialization(self, bridge):
        """Test bridge initializes correctly"""
        assert bridge is not None
        assert bridge.config is not None
        assert bridge.lm is not None
        assert isinstance(bridge.program_cache, dict)

    @pytest.mark.asyncio
    async def test_create_signature_from_pattern(self, bridge):
        """Test creating signature from hololoom pattern"""
        sig = await bridge.create_signature_from_pattern(
            pattern_name="TestPattern",
            description="Test pattern signature",
            inputs=["input1"],
            outputs=["output1"]
        )

        assert sig.name == "TestPattern"
        assert sig.metadata["source"] == "hololoom_pattern"

    @pytest.mark.asyncio
    async def test_fetch_training_examples_graceful(self, bridge):
        """Test fetching training examples handles no data gracefully"""
        # Should return empty list if no examples found
        examples = await bridge.fetch_training_examples_from_memory(
            query="nonexistent_examples_xyz123",
            limit=10
        )

        assert isinstance(examples, list)
        # May be empty, which is expected


class TestDSPyWorkflowAdapter:
    """Test DSPy workflow adapter"""

    def test_adapter_initialization(self, adapter):
        """Test adapter initializes correctly"""
        assert adapter is not None
        assert isinstance(adapter.signatures, dict)
        assert isinstance(adapter.programs, dict)
        assert isinstance(adapter.execution_history, list)

    def test_register_signature(self, adapter):
        """Test signature registration"""
        sig = create_signature(
            "Test registration",
            inputs=["x"],
            outputs=["y"],
            name="RegTest"
        )

        adapter.register_signature(sig)

        assert "RegTest" in adapter.signatures
        assert adapter.signatures["RegTest"] == sig

    @pytest.mark.asyncio
    async def test_create_simple_workflow(self, adapter):
        """Test creating a simple workflow"""
        # Register signatures
        adapter.register_signature(create_signature(
            "Step 1",
            inputs=["input"],
            outputs=["output"],
            name="Step1"
        ))

        adapter.register_signature(create_signature(
            "Step 2",
            inputs=["input"],
            outputs=["output"],
            name="Step2"
        ))

        # Create workflow
        workflow = adapter.create_workflow(
            name="SimpleWorkflow",
            description="Test workflow",
            steps=[
                {
                    "step_id": "step1",
                    "signature": "Step1",
                    "inputs": {"input": "{query}"},
                    "outputs": ["output"]
                },
                {
                    "step_id": "step2",
                    "signature": "Step2",
                    "inputs": {"input": "{step1.output}"},
                    "outputs": ["output"]
                }
            ]
        )

        assert workflow.name == "SimpleWorkflow"
        assert len(workflow.steps) == 2
        assert workflow.steps[0].step_id == "step1"
        assert workflow.steps[1].step_id == "step2"

    @pytest.mark.asyncio
    async def test_workflow_input_resolution(self, adapter):
        """Test resolving workflow inputs from context"""
        context = {
            "query": "test query",
            "step1.output": "intermediate result"
        }

        input_spec = {
            "field1": "{query}",
            "field2": "{step1.output}",
            "field3": "literal value"
        }

        resolved = adapter._resolve_inputs(input_spec, context)

        assert resolved["field1"] == "test query"
        assert resolved["field2"] == "intermediate result"
        assert resolved["field3"] == "literal value"

    @pytest.mark.asyncio
    async def test_workflow_missing_context_key(self, adapter):
        """Test handling missing context keys"""
        context = {"existing_key": "value"}

        input_spec = {
            "field1": "{existing_key}",
            "field2": "{missing_key}"
        }

        resolved = adapter._resolve_inputs(input_spec, context)

        assert resolved["field1"] == "value"
        assert resolved["field2"] == ""  # Missing key → empty string

    @pytest.mark.asyncio
    async def test_workflow_persistence(self, adapter, temp_dir):
        """Test saving and loading workflows"""
        # Register signature
        adapter.register_signature(create_signature(
            "Test sig",
            inputs=["x"],
            outputs=["y"],
            name="TestSig"
        ))

        # Create workflow
        original = adapter.create_workflow(
            name="PersistenceTest",
            description="Test persistence",
            steps=[
                {
                    "step_id": "test_step",
                    "signature": "TestSig",
                    "inputs": {"x": "{input}"},
                    "outputs": ["y"]
                }
            ]
        )

        # Save
        save_path = temp_dir / "test_workflow.yaml"
        await adapter.save_workflow(original, save_path)

        assert save_path.exists()

        # Load
        loaded = await adapter.load_workflow(save_path)

        assert loaded.name == original.name
        assert loaded.description == original.description
        assert len(loaded.steps) == len(original.steps)
        assert loaded.steps[0].step_id == original.steps[0].step_id

    @pytest.mark.asyncio
    async def test_execution_statistics_empty(self, adapter):
        """Test statistics with no executions"""
        stats = adapter.get_execution_statistics()

        assert stats["total_executions"] == 0

    @pytest.mark.asyncio
    async def test_execution_statistics_with_data(self, adapter):
        """Test statistics after adding execution data"""
        # Simulate execution history
        adapter.execution_history.append({
            "workflow": "test",
            "trace": [
                {"success": True, "duration_ms": 100},
                {"success": True, "duration_ms": 150}
            ]
        })

        adapter.execution_history.append({
            "workflow": "test2",
            "trace": [
                {"success": False, "duration_ms": 50}
            ]
        })

        stats = adapter.get_execution_statistics()

        assert stats["total_executions"] == 2
        assert stats["successful_executions"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["total_steps"] == 3
        assert stats["successful_steps"] == 2


class TestWorkflowExecution:
    """Test workflow execution (integration tests)"""

    @pytest.mark.asyncio
    async def test_single_step_workflow_execution(self, adapter):
        """Test executing a single-step workflow"""
        # Register signature
        adapter.register_signature(create_signature(
            "Echo input",
            inputs=["text"],
            outputs=["result"],
            name="Echo"
        ))

        # Create workflow
        workflow = adapter.create_workflow(
            name="EchoWorkflow",
            description="Simple echo",
            steps=[
                {
                    "step_id": "echo",
                    "signature": "Echo",
                    "inputs": {"text": "{input}"},
                    "outputs": ["result"]
                }
            ]
        )

        # Execute (Note: This will call the actual LLM, so it's an integration test)
        # In CI/CD, you might want to skip this or use mocks
        result = await adapter.execute_workflow(
            workflow,
            initial_inputs={"input": "test message"},
            optimize_on_the_fly=False
        )

        assert "trace" in result
        assert len(result["trace"]) == 1
        # Success depends on LLM availability
        # assert result["trace"][0]["success"]


class TestErrorHandling:
    """Test error handling in DSPy integration"""

    @pytest.mark.asyncio
    async def test_missing_signature_error(self, adapter):
        """Test error when referencing unregistered signature"""
        with pytest.raises(ValueError, match="not registered"):
            adapter.create_workflow(
                name="BadWorkflow",
                description="Missing sig",
                steps=[
                    {
                        "step_id": "bad",
                        "signature": "NonexistentSignature",
                        "inputs": {},
                        "outputs": []
                    }
                ]
            )

    @pytest.mark.asyncio
    async def test_workflow_execution_error_handling(self, adapter):
        """Test workflow continues/stops appropriately on errors"""
        # Register signature
        adapter.register_signature(create_signature(
            "Test sig",
            inputs=["x"],
            outputs=["y"],
            name="TestSigError"
        ))

        # Create workflow with invalid input reference
        workflow = adapter.create_workflow(
            name="ErrorWorkflow",
            description="Test errors",
            steps=[
                {
                    "step_id": "step1",
                    "signature": "TestSigError",
                    "inputs": {"x": "{nonexistent_input}"},
                    "outputs": ["y"]
                }
            ]
        )

        # Execute - should handle error gracefully
        result = await adapter.execute_workflow(
            workflow,
            initial_inputs={"actual_input": "value"},
            optimize_on_the_fly=False
        )

        # Should have trace even with errors
        assert "trace" in result
        # Workflow might fail, but shouldn't crash
        assert isinstance(result["success"], bool)


# Integration test fixtures
@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
