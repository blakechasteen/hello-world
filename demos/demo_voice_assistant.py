#!/usr/bin/env python
"""
Demo: Elle Core Voice Assistant

Comprehensive demonstration of voice control system:
- Speech-to-text with Whisper
- Natural language parsing
- Text-to-speech responses
- Wake word detection
- Complete voice workflow

Created: 2025-11-15
Author: Blake Chasteen
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from elle.voice import (
    WhisperSTT,
    LLMParser,
    TextToSpeech,
    VoiceGender,
    WakeWordDetector,
    CommandVariationGenerator,
    VoiceAssistant
)


async def demo_wake_word_detection():
    """Demo: Wake word detection"""
    print("\n" + "="*70)
    print("DEMO 1: WAKE WORD DETECTION")
    print("="*70)

    detector = WakeWordDetector()

    test_inputs = [
        ("hey elle, show bread SOP", True),
        ("elle, update temperature", True),
        ("ok elle, start baking", True),
        ("hello there, what's up?", False),
        ("hey el", True),           # Fuzzy match
        ("ella", True),             # Fuzzy match
        ("what time is it?", False),
    ]

    print("\nDetecting wake words in various inputs:\n")

    for text, expected_result in test_inputs:
        detected, confidence = detector.detect(text)
        status = "✓" if detected == expected_result else "✗"

        print(f"{status} Input: '{text}'")
        print(f"  Detected: {detected} (confidence: {confidence:.1%})")

        if detected:
            command = detector.extract_command(text)
            if command:
                print(f"  Command: '{command}'")
        print()

    print("✓ Wake word detection demo complete\n")


async def demo_nlp_parsing():
    """Demo: Natural language parsing"""
    print("\n" + "="*70)
    print("DEMO 2: NATURAL LANGUAGE PARSING")
    print("="*70)

    parser = LLMParser(use_llm=False)  # Pattern matching only

    print("\n1. SOP Update Commands (Duration):")
    print("-" * 70)

    sop_updates = CommandVariationGenerator.get_sop_update_variations(
        "bread", 50, "minutes"
    )

    for text in sop_updates:
        cmd = parser.parse(text)
        print(f"\nInput: '{text}'")
        print(f"  Command Type: {cmd.command_type}")
        print(f"  Entity: {cmd.entity}")
        print(f"  Intent: {cmd.intent}")
        print(f"  Value: {cmd.parameters.get('value')} {cmd.parameters.get('unit')}")
        print(f"  Confidence: {cmd.confidence:.1%}")

    print("\n\n2. Query Commands (Knowledge):")
    print("-" * 70)

    query_variations = CommandVariationGenerator.get_query_variations(
        "inoculation ratio"
    )

    for text in query_variations:
        cmd = parser.parse(text)
        print(f"\nInput: '{text}'")
        print(f"  Command Type: {cmd.command_type}")
        print(f"  Intent: {cmd.action}")
        print(f"  Query: {cmd.parameters.get('query_text')}")

    print("\n\n3. Task Commands:")
    print("-" * 70)

    task_commands = [
        ("start baking bread", "task_start"),
        ("finish baking, made 24 loaves for $180", "task_end"),
        ("pause the timer", "task_pause"),
        ("resume", "task_resume"),
    ]

    for text, expected_type in task_commands:
        cmd = parser.parse(text)
        status = "✓" if cmd.command_type == expected_type else "✗"
        print(f"\n{status} Input: '{text}'")
        print(f"  Command Type: {cmd.command_type}")
        print(f"  Action: {cmd.action}")

    print("\n✓ NLP parsing demo complete\n")


async def demo_text_to_speech():
    """Demo: Text-to-speech"""
    print("\n" + "="*70)
    print("DEMO 3: TEXT-TO-SPEECH")
    print("="*70)

    tts = TextToSpeech(rate=150, volume=0.9, voice_gender=VoiceGender.FEMALE)

    print("\n1. Available Voices:")
    print("-" * 70)

    voices = tts.get_voices()
    if voices:
        for i, voice in enumerate(voices, 1):
            print(f"  {i}. {voice['name']}")
    else:
        print("  No voices available (pyttsx3 not installed)")

    print("\n2. Demonstration Phrases:")
    print("-" * 70)

    phrases = [
        "Hello! I am Elle, your farm and kitchen assistant.",
        "Ready to help with your operations.",
        "What can I help you with today?",
    ]

    for i, phrase in enumerate(phrases, 1):
        print(f"\n[{i}] Speaking: \"{phrase}\"")
        success = await tts.speak(phrase, wait=True)
        if not success:
            print("    (pyttsx3 not available - would speak this text)")

    print("\n3. Speech Rate Variation:")
    print("-" * 70)

    rates = [80, 150, 200]
    for rate in rates:
        tts.set_rate(rate)
        text = f"This is {rate} words per minute."
        print(f"\n  Rate {rate} wpm: \"{text}\"")
        await tts.speak(text, wait=True)

    print("\n✓ Text-to-speech demo complete\n")


async def demo_complete_workflow():
    """Demo: Complete workflow"""
    print("\n" + "="*70)
    print("DEMO 4: COMPLETE VOICE WORKFLOW")
    print("="*70)

    print("\nInitializing Elle Voice Assistant...")

    assistant = VoiceAssistant(
        sop_dir="elle/sops",
        whisper_model="tiny",
        tts_rate=150,
        use_llm_parser=False,
        verbose=True
    )

    await assistant.initialize()

    print("\n✓ Assistant initialized\n")

    # Simulate workflow
    test_commands = [
        ("show bread SOP", "Requesting SOP information"),
        ("what's the proofing duration?", "Querying knowledge"),
        ("update bread SOP: increase proofing to 50 minutes", "Updating SOP"),
        ("start baking bread", "Starting task"),
        ("pause", "Pausing task"),
        ("resume", "Resuming task"),
    ]

    print("\nSimulating voice workflow:")
    print("-" * 70)

    for text, description in test_commands:
        print(f"\n[{description}]")
        print(f"You: {text}")

        response = await assistant.process_voice_input(text)
        print(f"Elle: {response}")

        # Simulate TTS
        if assistant.tts.pyttsx3_available:
            await assistant.tts.speak(response)
        else:
            print(f"       [Voice would say: {response}]")

    print("\n✓ Complete workflow demo finished\n")


async def demo_command_variations():
    """Demo: Test command variations"""
    print("\n" + "="*70)
    print("DEMO 5: COMMAND VARIATION ROBUSTNESS")
    print("="*70)

    parser = LLMParser(use_llm=False)

    print("\nTesting robustness to natural language variations...\n")

    # Test 1: Duration changes (same intent, different wording)
    print("Test 1: Duration Changes (Different Wordings)")
    print("-" * 70)

    duration_variations = [
        "increase proofing time to 45 minutes",
        "make proofing longer, 45 minutes",
        "bump proofing up to 45 minutes",
        "set proofing duration to 45 minutes",
        "change proofing to 45 minutes",
        "proofing should be 45 minutes",
    ]

    for text in duration_variations:
        cmd = parser.parse(f"update bread SOP: {text}")
        value = cmd.parameters.get("value")
        unit = cmd.parameters.get("unit")
        status = "✓" if value == 45 and unit == "minutes" else "✗"
        print(f"  {status} '{text}'")

    # Test 2: Temperature changes
    print("\nTest 2: Temperature Changes (Different Wordings)")
    print("-" * 70)

    temp_variations = [
        "increase temperature to 450 degrees",
        "heat it up to 450",
        "raise temperature to 450 F",
        "set temp to 450 degrees",
        "change oven to 450",
    ]

    for text in temp_variations:
        cmd = parser.parse(f"update bread SOP: {text}")
        value = cmd.parameters.get("value_degrees") or cmd.parameters.get("value")
        status = "✓" if value == 450 else "✗"
        print(f"  {status} '{text}'")

    # Test 3: Task commands
    print("\nTest 3: Task Commands (Different Wordings)")
    print("-" * 70)

    task_variations = [
        ("start baking", "task_start"),
        ("begin baking bread", "task_start"),
        ("finish task", "task_end"),
        ("complete baking", "task_end"),
        ("pause timer", "task_pause"),
        ("hold it", "task_pause"),
        ("continue", "task_resume"),
        ("go again", "task_resume"),
    ]

    for text, expected_type in task_variations:
        cmd = parser.parse(text)
        status = "✓" if cmd.command_type == expected_type else "✗"
        print(f"  {status} '{text}' → {cmd.command_type}")

    print("\n✓ Command variation demo complete\n")


async def interactive_test():
    """Interactive testing mode"""
    print("\n" + "="*70)
    print("INTERACTIVE VOICE TESTING")
    print("="*70)

    print("\nEnter voice commands to test parsing:")
    print("  Examples:")
    print("    - 'show bread SOP'")
    print("    - 'update bread SOP: increase proofing to 50 minutes'")
    print("    - 'what is the inoculation ratio?'")
    print("    - 'start baking bread'")
    print("  Type 'quit' to exit\n")

    assistant = VoiceAssistant(
        sop_dir="elle/sops",
        whisper_model="tiny",
        use_llm_parser=False,
        verbose=False
    )

    await assistant.initialize()

    while True:
        try:
            text = input("You> ").strip()

            if not text:
                continue

            if text.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            response = await assistant.process_voice_input(text)
            print(f"Elle> {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


async def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("ELLE CORE VOICE ASSISTANT - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nThis demo showcases all voice features:")
    print("  1. Wake word detection")
    print("  2. Natural language parsing")
    print("  3. Text-to-speech")
    print("  4. Complete voice workflow")
    print("  5. Command variation robustness")
    print("  6. Interactive testing\n")

    # Run demos
    await demo_wake_word_detection()
    await demo_nlp_parsing()
    await demo_text_to_speech()
    await demo_complete_workflow()
    await demo_command_variations()

    # Interactive test
    print("\n" + "="*70)
    print("Would you like to test interactive mode?")
    print("="*70)

    try:
        response = input("\nEnter interactive mode? (y/n): ").strip().lower()
        if response in ["y", "yes"]:
            await interactive_test()
    except KeyboardInterrupt:
        print("\n\nSkipping interactive mode.")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Test voice mode: python elle/voice/assistant.py voice")
    print("  2. Test interactive: python elle/voice/assistant.py interactive")
    print("  3. Read documentation: elle/voice/README.md")
    print("  4. Check Elle Core: elle/__init__.py for full integration\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
