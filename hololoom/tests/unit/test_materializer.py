
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from datetime import datetime

from hololoom.fabric.spacetime import Spacetime, Artifact, ArtifactType, WeavingTrace
from hololoom.fabric.materializer import Materializer

class TestMaterializer(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.materializer = Materializer(root_dir=self.test_dir)

    def tearDown(self):
        # Cleanup
        shutil.rmtree(self.test_dir)

    def test_materialize_code_artifact(self):
        """Test writing a code artifact."""
        # Create dummy trace
        trace = WeavingTrace(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=10.0
        )
        
        # Create artifact
        code_content = "print('Hello World')"
        artifact = Artifact(
            name="hello.py",
            type=ArtifactType.CODE,
            content=code_content,
            destination_path="src/hello.py"
        )
        
        # Create Spacetime
        spacetime = Spacetime(
            query_text="test",
            response="done",
            tool_used="test_tool",
            confidence=1.0,
            trace=trace,
            artifacts=[artifact]
        )
        
        # Materialize
        results = self.materializer.materialize(spacetime)
        
        # Verify
        self.assertIn("code", results)
        self.assertEqual(len(results["code"]), 1)
        
        expected_path = Path(self.test_dir) / "src/hello.py"
        self.assertTrue(expected_path.exists())
        
        with open(expected_path, "r") as f:
            content = f.read()
        self.assertEqual(content, code_content)

    def test_materialize_binary_artifact(self):
        """Test writing a binary artifact."""
        trace = WeavingTrace(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=10.0
        )
        
        # Create binary content
        binary_content = b'\x00\x01\x02\x03'
        artifact = Artifact(
            name="image.png",
            type=ArtifactType.IMAGE,
            content=binary_content,
            destination_path="assets/image.png"
        )
        
        spacetime = Spacetime(
            query_text="test",
            response="done",
            tool_used="test_tool",
            confidence=1.0,
            trace=trace,
            artifacts=[artifact]
        )
        
        results = self.materializer.materialize(spacetime)
        
        self.assertIn("image", results)
        expected_path = Path(self.test_dir) / "assets/image.png"
        self.assertTrue(expected_path.exists())
        
        with open(expected_path, "rb") as f:
            content = f.read()
        self.assertEqual(content, binary_content)

if __name__ == "__main__":
    unittest.main()
