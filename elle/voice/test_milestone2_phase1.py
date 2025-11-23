"""
Test Script for Voice UX Milestone 2 Phase 1

Tests:
- Command Grammar Parser
- Structured command handling
- Navigation history
- Thread operations (brief responses)
- Task operations (placeholders)

Created: November 22, 2025
"""

import asyncio
from elle.voice.command_parser import CommandGrammarParser, CommandType
from elle.voice.assistant import VoiceAssistant


def test_command_parser():
    """Test CommandGrammarParser"""
    print("\n" + "="*70)
    print("TEST 1: Command Grammar Parser")
    print("="*70)

    parser = CommandGrammarParser()

    test_cases = [
        # Navigation
        ("back", CommandType.BACK, {}),
        ("next", CommandType.NEXT, {}),
        ("home", CommandType.HOME, {}),

        # Thread shortcuts
        ("t3", CommandType.THREAD_SWITCH, {"thread_id": 3}),
        ("#2", CommandType.THREAD_SWITCH, {"thread_id": 2}),
        ("threads", CommandType.THREAD_LIST, {}),
        ("+ greenhouse", CommandType.THREAD_CREATE, {"topic": "greenhouse"}),
        ("- 4", CommandType.THREAD_DELETE, {"thread_id": 4}),

        # Task operations
        ("run analyze", CommandType.TASK_RUN, {"task_name": "analyze"}),
        ("> build", CommandType.TASK_RUN, {"task_name": "build"}),
        ("stop", CommandType.TASK_STOP, {}),
        ("pause", CommandType.TASK_PAUSE, {}),
        ("c", CommandType.TASK_RESUME, {}),
        ("?", CommandType.TASK_STATUS, {}),

        # Query operations
        ("@Thompson", CommandType.ENTITY_LOOKUP, {"entity": "Thompson"}),
        ("find Matryoshka", CommandType.SEARCH, {"query": "Matryoshka"}),

        # Conversational fallback
        ("What's the weather like?", CommandType.CONVERSATIONAL, {"text": "What's the weather like?"}),
    ]

    passed = 0
    failed = 0

    for text, expected_type, expected_params in test_cases:
        commands = parser.parse(text)
        cmd = commands[0]

        if cmd.command_type == expected_type and cmd.parameters == expected_params:
            print(f"  [OK] '{text}' -> {expected_type.value}")
            passed += 1
        else:
            print(f"  [FAIL] '{text}'")
            print(f"        Expected: {expected_type.value} {expected_params}")
            print(f"        Got: {cmd.command_type.value} {cmd.parameters}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")


def test_command_chaining():
    """Test command chaining"""
    print("\n" + "="*70)
    print("TEST 2: Command Chaining")
    print("="*70)

    parser = CommandGrammarParser()

    test_cases = [
        ("t3; run analyze", 2, [CommandType.THREAD_SWITCH, CommandType.TASK_RUN]),
        ("new research; > search", 2, [CommandType.THREAD_CREATE, CommandType.TASK_RUN]),
        ("threads; #2; ?", 3, [CommandType.THREAD_LIST, CommandType.THREAD_SWITCH, CommandType.TASK_STATUS]),
    ]

    passed = 0
    failed = 0

    for text, expected_count, expected_types in test_cases:
        commands = parser.parse(text)

        if len(commands) == expected_count:
            types_match = all(cmd.command_type == expected_types[i] for i, cmd in enumerate(commands))
            if types_match:
                print(f"  [OK] '{text}' -> {expected_count} commands")
                passed += 1
            else:
                print(f"  [FAIL] '{text}' - wrong types")
                failed += 1
        else:
            print(f"  [FAIL] '{text}' - expected {expected_count}, got {len(commands)}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")


async def test_assistant_integration():
    """Test VoiceAssistant integration"""
    print("\n" + "="*70)
    print("TEST 3: VoiceAssistant Integration")
    print("="*70)

    assistant = VoiceAssistant(
        whisper_model="tiny",
        tts_rate=150,
        use_llm_parser=False,
        verbose=False
    )

    await assistant.initialize()

    test_commands = [
        # Thread creation
        ("+ baking", "Created baking"),
        ("+ biochar", "Created biochar"),
        ("+ greenhouse", "Created greenhouse"),

        # Thread listing
        ("threads", "3 threads"),

        # Thread switching
        ("t1", "Thread 1"),
        ("t2", "Thread 2"),

        # Navigation (after switching)
        ("back", "Back to"),  # Should go back to thread 1

        # Thread delete
        ("- 3", "Deleted greenhouse"),

        # Task operations (placeholders)
        ("run analyze", "Running analyze"),
        ("stop", "Stopped"),
        ("pause", "Paused"),
        ("c", "Resumed"),
        ("?", "No tasks running"),
    ]

    passed = 0
    failed = 0

    for text, expected_substring in test_commands:
        try:
            response = await assistant.process_voice_input(text)

            if expected_substring.lower() in response.lower():
                print(f"  [OK] '{text}' -> '{response}'")
                passed += 1
            else:
                print(f"  [FAIL] '{text}'")
                print(f"        Expected substring: '{expected_substring}'")
                print(f"        Got: '{response}'")
                failed += 1
        except Exception as e:
            print(f"  [ERROR] '{text}' raised exception: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("VOICE UX MILESTONE 2 PHASE 1 - TEST SUITE")
    print("="*70)

    # Test 1: Command Parser
    test_command_parser()

    # Test 2: Command Chaining
    test_command_chaining()

    # Test 3: Assistant Integration
    await test_assistant_integration()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
