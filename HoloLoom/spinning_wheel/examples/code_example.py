#!/usr/bin/env python3
"""
CodeSpinner Example
===================

Demonstrates how to use CodeSpinner to ingest:
1. Code files
2. Git diffs
3. Repository structures

Usage:
    python code_example.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.spinning_wheel import CodeSpinner, CodeSpinnerConfig
from HoloLoom.spinning_wheel.code import spin_code_file, spin_git_diff, spin_repository


async def example_1_code_file():
    """Example 1: Ingest a Python code file."""
    print("\n=== Example 1: Code File Ingestion ===\n")

    # Sample Python code
    code_content = '''
import numpy as np
from typing import List

class DataProcessor:
    """Process numerical data."""

    def __init__(self, name: str):
        self.name = name
        self.data = []

    def add_data(self, values: List[float]):
        """Add data points."""
        self.data.extend(values)

    def compute_mean(self) -> float:
        """Compute mean of data."""
        return np.mean(self.data)

def process_dataset(path: str) -> DataProcessor:
    """Load and process dataset."""
    processor = DataProcessor("dataset")
    # Load data...
    return processor
'''

    # Method 1: Using convenience function
    shards = await spin_code_file(
        path='src/data_processor.py',
        content=code_content,
        language='python',
        chunk_by=None  # Single shard for entire file
    )

    print(f"Created {len(shards)} shard(s)")
    for shard in shards:
        print(f"\nShard ID: {shard.id}")
        print(f"Entities: {shard.entities}")
        print(f"Imports: {shard.metadata.get('imports', [])}")
        print(f"Language: {shard.metadata['language']}")
        print(f"Text preview: {shard.text[:200]}...")


async def example_2_chunked_code():
    """Example 2: Chunk code by functions."""
    print("\n=== Example 2: Chunked Code (by function) ===\n")

    code_content = '''
def hello():
    """Say hello."""
    print("Hello, world!")

def goodbye():
    """Say goodbye."""
    print("Goodbye!")

def greet(name: str):
    """Greet someone by name."""
    print(f"Hello, {name}!")
'''

    config = CodeSpinnerConfig(
        chunk_by='function',  # Split into separate function shards
        extract_entities=True,
        extract_imports=True
    )

    spinner = CodeSpinner(config)
    shards = await spinner.spin({
        'type': 'file',
        'path': 'greetings.py',
        'content': code_content
    })

    print(f"Created {len(shards)} function shard(s)")
    for shard in shards:
        print(f"\nShard ID: {shard.id}")
        print(f"Function: {shard.metadata.get('function_name', 'N/A')}")
        print(f"Entities: {shard.entities}")


async def example_3_git_diff():
    """Example 3: Ingest a git diff."""
    print("\n=== Example 3: Git Diff Ingestion ===\n")

    # Sample git diff output
    git_diff = '''diff --git a/src/main.py b/src/main.py
index 1234567..abcdefg 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1,5 +1,6 @@
 import sys
+import logging

 def main():
-    print("Hello")
+    logging.info("Application started")
     return 0

new file mode 100644
diff --git a/src/utils.py b/src/utils.py
--- /dev/null
+++ b/src/utils.py
@@ -0,0 +1,3 @@
+def helper():
+    """Helper function."""
+    pass
'''

    shards = await spin_git_diff(
        diff=git_diff,
        commit_sha='abc123def456',
        author='developer@example.com',
        message='Add logging and helper utilities'
    )

    print(f"Created {len(shards)} shard(s) from diff")
    for shard in shards:
        print(f"\nShard ID: {shard.id}")
        print(f"Type: {shard.metadata['type']}")
        if shard.metadata['type'] == 'diff_summary':
            print(f"Files changed: {shard.metadata['files_changed']}")
            print(f"Insertions: +{shard.metadata['insertions']}")
            print(f"Deletions: -{shard.metadata['deletions']}")
        else:
            print(f"File: {shard.metadata.get('file_path', 'N/A')}")
            print(f"Change type: {shard.metadata.get('change_type', 'N/A')}")


async def example_4_repository_structure():
    """Example 4: Ingest repository structure."""
    print("\n=== Example 4: Repository Structure ===\n")

    # Sample repository file list
    repo_files = [
        'README.md',
        'setup.py',
        'src/__init__.py',
        'src/main.py',
        'src/utils.py',
        'src/models/base.py',
        'src/models/neural.py',
        'tests/test_main.py',
        'tests/test_utils.py',
    ]

    shards = await spin_repository(
        root_path='/home/user/my-project',
        files=repo_files
    )

    print(f"Created {len(shards)} shard(s) for repository")
    for shard in shards:
        print(f"\nShard ID: {shard.id}")
        print(f"Languages: {shard.metadata['languages']}")
        print(f"File count: {shard.metadata['file_count']}")
        print(f"\nStructure preview:\n{shard.text[:300]}...")


async def example_5_multi_language():
    """Example 5: Multi-language code detection."""
    print("\n=== Example 5: Multi-Language Detection ===\n")

    languages = {
        'app.py': 'def main(): pass',
        'server.js': 'function main() {}',
        'Main.java': 'public class Main {}',
        'main.go': 'func main() {}',
        'lib.rs': 'fn main() {}',
    }

    spinner = CodeSpinner()

    for filename, code in languages.items():
        shards = await spinner.spin({
            'type': 'file',
            'path': filename,
            'content': code
        })

        detected_lang = shards[0].metadata['language']
        print(f"{filename} -> {detected_lang}")


async def example_6_import_extraction():
    """Example 6: Extract imports and dependencies."""
    print("\n=== Example 6: Import/Dependency Extraction ===\n")

    code_samples = {
        'Python': '''
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Dict
''',
        'JavaScript': '''
import React from 'react';
import { useState, useEffect } from 'react';
const axios = require('axios');
''',
        'Go': '''
import (
    "fmt"
    "net/http"
    "github.com/gorilla/mux"
)
''',
    }

    config = CodeSpinnerConfig(extract_imports=True)
    spinner = CodeSpinner(config)

    for lang_name, code in code_samples.items():
        ext = {'Python': '.py', 'JavaScript': '.js', 'Go': '.go'}[lang_name]

        shards = await spinner.spin({
            'type': 'file',
            'path': f'example{ext}',
            'content': code
        })

        imports = shards[0].metadata.get('imports', [])
        print(f"\n{lang_name}:")
        for imp in imports:
            print(f"  - {imp}")


async def main():
    """Run all examples."""
    print("CodeSpinner Examples")
    print("=" * 60)

    await example_1_code_file()
    await example_2_chunked_code()
    await example_3_git_diff()
    await example_4_repository_structure()
    await example_5_multi_language()
    await example_6_import_extraction()

    print("\n" + "=" * 60)
    print("All examples completed!")


if __name__ == '__main__':
    asyncio.run(main())
