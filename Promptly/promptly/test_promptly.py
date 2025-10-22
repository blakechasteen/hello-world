#!/usr/bin/env python3
"""
Promptly Test Suite
Comprehensive tests for all functionality
"""

import os
import sys
import shutil
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from promptly import Promptly

def cleanup():
    """Remove test directory"""
    test_dir = Path('.promptly_test')
    if test_dir.exists():
        shutil.rmtree(test_dir)

def test_init():
    """Test repository initialization"""
    print("Testing: Repository initialization...")
    cleanup()
    
    os.makedirs('.promptly_test')
    os.chdir('.promptly_test')
    
    promptly = Promptly()
    result = promptly.init()
    
    assert "Initialized" in result
    assert Path('.promptly').exists()
    assert Path('.promptly/promptly.db').exists()
    assert Path('.promptly/prompts').exists()
    assert Path('.promptly/chains').exists()
    
    print("✓ Repository initialization works")

def test_add_prompt():
    """Test adding prompts"""
    print("Testing: Adding prompts...")
    
    promptly = Promptly()
    
    # Add first prompt
    result = promptly.add('test-prompt', 'Test content: {input}')
    assert 'test-prompt' in result
    assert 'v1' in result
    
    # Add second version
    result = promptly.add('test-prompt', 'Updated content: {input}')
    assert 'v2' in result
    
    print("✓ Adding prompts works")

def test_get_prompt():
    """Test retrieving prompts"""
    print("Testing: Getting prompts...")
    
    promptly = Promptly()
    
    # Get latest version
    prompt = promptly.get('test-prompt')
    assert prompt is not None
    assert prompt['name'] == 'test-prompt'
    assert prompt['version'] == 2
    assert 'Updated content' in prompt['content']
    
    # Get specific version
    prompt_v1 = promptly.get('test-prompt', version=1)
    assert prompt_v1['version'] == 1
    assert 'Test content' in prompt_v1['content']
    
    print("✓ Getting prompts works")

def test_list_prompts():
    """Test listing prompts"""
    print("Testing: Listing prompts...")
    
    promptly = Promptly()
    
    # Add more prompts
    promptly.add('prompt-a', 'Content A')
    promptly.add('prompt-b', 'Content B')
    
    prompts = promptly.list_prompts()
    assert len(prompts) >= 3
    
    names = [p['name'] for p in prompts]
    assert 'test-prompt' in names
    assert 'prompt-a' in names
    assert 'prompt-b' in names
    
    print("✓ Listing prompts works")

def test_branching():
    """Test branch operations"""
    print("Testing: Branching...")
    
    promptly = Promptly()
    
    # Create branch
    result = promptly.branch('feature-branch')
    assert 'feature-branch' in result
    
    # Switch to branch
    result = promptly.checkout('feature-branch')
    assert 'feature-branch' in result
    
    # Verify current branch
    current = promptly._get_current_branch()
    assert current == 'feature-branch'
    
    # Add prompt on feature branch
    promptly.add('feature-prompt', 'Feature content')
    
    # Switch back to main
    promptly.checkout('main')
    
    # Verify feature-prompt not on main
    prompt = promptly.get('feature-prompt')
    assert prompt is None
    
    # Switch to feature and verify it exists
    promptly.checkout('feature-branch')
    prompt = promptly.get('feature-prompt')
    assert prompt is not None
    
    print("✓ Branching works")

def test_log():
    """Test commit history"""
    print("Testing: Commit history...")
    
    promptly = Promptly()
    promptly.checkout('main')
    
    # Get history
    history = promptly.log(limit=10)
    assert len(history) > 0
    
    # Get history for specific prompt
    history = promptly.log(name='test-prompt', limit=5)
    assert len(history) == 2  # We added 2 versions
    
    print("✓ Commit history works")

def test_metadata():
    """Test metadata storage"""
    print("Testing: Metadata...")
    
    promptly = Promptly()
    
    metadata = {
        'tags': ['test', 'example'],
        'model': 'claude-sonnet-4',
        'temperature': 0.7
    }
    
    promptly.add('meta-prompt', 'Content with metadata', metadata=metadata)
    
    prompt = promptly.get('meta-prompt')
    assert prompt['metadata'] == metadata
    assert 'test' in prompt['metadata']['tags']
    
    print("✓ Metadata works")

def test_chains():
    """Test chain operations"""
    print("Testing: Chains...")
    
    promptly = Promptly()
    promptly.checkout('main')
    
    # Add prompts for chain
    promptly.add('step1', 'Step 1: {input}')
    promptly.add('step2', 'Step 2: {output}')
    promptly.add('step3', 'Step 3: {output}')
    
    # Create chain
    result = promptly.create_chain(
        'test-chain',
        ['step1', 'step2', 'step3'],
        'Test chain description'
    )
    assert 'test-chain' in result
    
    # Execute chain (without model function)
    results = promptly.execute_chain('test-chain', {'input': 'test'})
    assert len(results) == 3
    assert results[0]['step'] == 'step1'
    
    print("✓ Chains work")

def test_evaluation():
    """Test evaluation framework"""
    print("Testing: Evaluation...")
    
    promptly = Promptly()
    promptly.checkout('main')
    
    # Add prompt for evaluation
    promptly.add('eval-prompt', 'Process: {text}')
    
    # Define test cases
    test_cases = [
        {
            'id': 'test-1',
            'inputs': {'text': 'input 1'},
            'expected': 'output 1'
        },
        {
            'id': 'test-2',
            'inputs': {'text': 'input 2'},
            'expected': 'output 2'
        }
    ]
    
    # Run evaluation without model function
    results = promptly.eval_prompt('eval-prompt', test_cases)
    assert len(results) == 2
    
    print("✓ Evaluation works")

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Promptly Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_init()
        test_add_prompt()
        test_get_prompt()
        test_list_prompts()
        test_branching()
        test_log()
        test_metadata()
        test_chains()
        test_evaluation()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        os.chdir('..')
        cleanup()
    
    return True

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
