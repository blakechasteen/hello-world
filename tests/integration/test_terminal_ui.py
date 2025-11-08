#!/usr/bin/env python3
"""
Terminal UI Integration Test
============================
Verify that terminal UI works correctly with WeavingOrchestrator.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.terminal_ui import TerminalUI
from HoloLoom.loom.command import PatternCard


async def test_ui_creation():
    """Test 1: UI creation and initialization"""
    print("\n" + "="*70)
    print("TEST 1: UI Creation")
    print("="*70)
    
    try:
        from HoloLoom.memory.backend_factory import create_memory_backend
        
        config = Config.bare()
        memory = await create_memory_backend(config)
        orchestrator = WeavingOrchestrator(cfg=config, memory=memory)
        ui = TerminalUI(orchestrator)
        
        print(" UI created successfully")
        print(f" Conversation history: {len(ui.conversation_history)} entries")
        print(f" Stages defined: {len(ui.stages)}")
        print(" TEST 1 PASSED")
        return True
    except Exception as e:
        print(f" TEST 1 FAILED: {e}")
        return False


async def test_banner_display():
    """Test 2: Banner display"""
    print("\n" + "="*70)
    print("TEST 2: Banner Display")
    print("="*70)
    
    try:
        from HoloLoom.memory.backend_factory import create_memory_backend
        
        config = Config.bare()
        memory = await create_memory_backend(config)
        orchestrator = WeavingOrchestrator(cfg=config, memory=memory)
        ui = TerminalUI(orchestrator)
        
        ui.print_banner()
        print("\n Banner displayed successfully")
        print(" TEST 2 PASSED")
        return True
    except Exception as e:
        print(f" TEST 2 FAILED: {e}")
        return False


async def test_basic_weave():
    """Test 3: Basic weave with display"""
    print("\n" + "="*70)
    print("TEST 3: Basic Weave with Display")
    print("="*70)
    
    try:
        from HoloLoom.memory.backend_factory import create_memory_backend
        
        config = Config.bare()
        memory = await create_memory_backend(config)
        orchestrator = WeavingOrchestrator(
            cfg=config,
            memory=memory,
            enable_complexity_auto_detect=True
        )
        ui = TerminalUI(orchestrator)
        
        # Simple query
        result = await ui.weave_with_display("hi", show_trace=False)
        
        print(f"\n Result obtained: {type(result)}")
        print(f" History entries: {len(ui.conversation_history)}")
        print(" TEST 3 PASSED")
        return True
    except Exception as e:
        print(f" TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_history_display():
    """Test 4: History and stats display"""
    print("\n" + "="*70)
    print("TEST 4: History and Stats")
    print("="*70)
    
    try:
        from HoloLoom.memory.backend_factory import create_memory_backend
        
        config = Config.bare()
        memory = await create_memory_backend(config)
        orchestrator = WeavingOrchestrator(
            cfg=config,
            memory=memory,
            enable_complexity_auto_detect=True
        )
        ui = TerminalUI(orchestrator)
        
        # Run a few queries
        queries = ["hi", "what is AI?", "explain machine learning"]
        for query in queries:
            await ui.weave_with_display(query, show_trace=False)
        
        print("\n=== Showing History ===")
        ui.show_history()
        
        print("\n=== Showing Stats ===")
        ui.show_stats()
        
        print("\n TEST 4 PASSED")
        return True
    except Exception as e:
        print(f" TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_trace_display():
    """Test 5: Trace visualization"""
    print("\n" + "="*70)
    print("TEST 5: Trace Display")
    print("="*70)
    
    try:
        from HoloLoom.memory.backend_factory import create_memory_backend
        
        config = Config.fast()
        memory = await create_memory_backend(config)
        orchestrator = WeavingOrchestrator(
            cfg=config,
            memory=memory,
            enable_complexity_auto_detect=True
        )
        ui = TerminalUI(orchestrator)
        
        # Query with trace
        result = await ui.weave_with_display(
            "what is neural processing?",
            show_trace=True
        )
        
        print("\n Trace displayed successfully")
        print(" TEST 5 PASSED")
        return True
    except Exception as e:
        print(f" TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\nTERMINAL UI INTEGRATION TESTS")
    print("="*70)
    
    tests = [
        test_ui_creation,
        test_banner_display,
        test_basic_weave,
        test_history_display,
        test_trace_display,
    ]
    
    results = []
    for test_func in tests:
        try:
            passed = await test_func()
            results.append((test_func.__name__, passed))
        except Exception as e:
            print(f"\nTest {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"{status} - {name}")
    
    print()
    print(f"Results: {passed_count}/{total_count} tests passed ({(passed_count/total_count)*100:.1f}%)")
    
    if passed_count == total_count:
        print("\nALL TESTS PASSED! Terminal UI is working correctly!")
        return 0
    else:
        print(f"\n{total_count - passed_count} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
