
import asyncio
import shutil
import tempfile
import unittest
import os
from pathlib import Path
from datetime import datetime

# Mocking parts of the system for integration test without full LLM
from HoloLoom.fabric.spacetime import Spacetime, Artifact, ArtifactType, WeavingTrace
from HoloLoom.fabric.materializer import Materializer
from HoloLoom.tools.executor import ToolExecutor
from HoloLoom.protocols.types import Query, Context

class TestMaterializationPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.materializer = Materializer(root_dir=self.test_dir)
        self.executor = ToolExecutor(enable_skills=False) # Disable actual skills bridge loading for unit test

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_artifact_extraction_and_materialization(self):
        """
        Simulates:
        1. LLM producing text with code blocks (Software Engineer skill output)
        2. ToolExecutor extracting artifacts
        3. Spacetime carrying artifacts
        4. Materializer writing to disk
        """ 
        
        # 1. Simulated LLM Response (what software-engineer skill would produce)
        llm_output = """
Here is the requested calculator application.

First, the core logic:
```python
# filename: src/calculator.py
class Calculator:
    def add(self, a, b):
        return a + b
```

And a test file:
```python
# filename: tests/test_calc.py
import unittest
from src.calculator import Calculator

class TestCalc(unittest.TestCase):
    def test_add(self):
        c = Calculator()
        assert c.add(1, 1) == 2
```

And a README:
```markdown
# filename: README.md
# Simple Calculator
Run tests with `python -m unittest`
```
"""
        
        # 2. Extraction via ToolExecutor
        # (We use the internal method directly or simulating an execution result)
        artifacts_dicts = self.executor._extract_artifacts(llm_output)
        
        # Verify extraction
        self.assertEqual(len(artifacts_dicts), 3)
        self.assertEqual(artifacts_dicts[0]['name'], 'calculator.py')
        self.assertEqual(artifacts_dicts[0]['destination_path'], 'src/calculator.py')
        
        # Convert back to Artifact objects for Spacetime
        artifacts = [Artifact.from_dict(d) for d in artifacts_dicts]
        
        # 3. Create Spacetime fabric
        trace = WeavingTrace(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=100.0
        )
        
        spacetime = Spacetime(
            query_text="Build a calculator",
            response=llm_output,
            tool_used="software-engineer",
            confidence=0.95,
            trace=trace,
            artifacts=artifacts
        )
        
        # 4. Materialization
        paths = self.materializer.materialize(spacetime)
        
        # Verify files on disk
        self.assertTrue((Path(self.test_dir) / "src/calculator.py").exists())
        self.assertTrue((Path(self.test_dir) / "tests/test_calc.py").exists())
        self.assertTrue((Path(self.test_dir) / "README.md").exists())
        
        # Check content
        with open(Path(self.test_dir) / "src/calculator.py", 'r') as f:
            content = f.read()
            self.assertIn("class Calculator:", content)
            
        print(f"\n[SUCCESS] Materialized {len(paths['code'])} code files and {len(paths['document'])} docs/other.")

if __name__ == "__main__":
    unittest.main()
