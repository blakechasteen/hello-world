#!/usr/bin/env python3
"""
Voice-Enhanced Scratchpad Demo
===============================

Demonstrates conversational scratchpad with:
- TTS feedback for confirmations
- Clarity requests for ambiguous data
- Read-back verification
- Safety confirmations (HiTL)

Author: Claude Code
Date: November 2025
"""

import asyncio
from hololoom.spinningWheel.voice_scratchpad import VoiceScratchpad, VoicePrompt, VoiceResponse


# ============================================================================
# Demo Scenarios
# ============================================================================

async def demo_basic_voice_feedback():
    """Demo 1: Basic voice feedback"""
    print("=" * 70)
    print("DEMO 1: Basic Voice Feedback")
    print("=" * 70)

    scratchpad = VoiceScratchpad(enable_voice=True)

    # Capture voice prompts
    voice_prompts = []

    def on_prompt(prompt: VoicePrompt):
        voice_prompts.append(prompt)
        print(f"\n[TTS] {prompt.text}")
        if prompt.expected_response:
            print(f"[Expected] {prompt.expected_response}")

    scratchpad.register_voice_callbacks(on_prompt, lambda r: None)

    # Start recording
    await scratchpad.start_recording('recipe')

    # Transcription
    await scratchpad.update_from_transcription(
        "Recipe for chocolate chip cookies. Serves 24. "
        "1 cup butter, 2 cups flour, 2 cups chocolate chips."
    )

    print(f"\n[Voice Prompts Generated]: {len(voice_prompts)}")
    print(f"[State Valid]: {scratchpad.get_current_state()['is_valid']}")


async def demo_clarity_requests():
    """Demo 2: Clarity requests for ambiguous data"""
    print("\n" + "=" * 70)
    print("DEMO 2: Clarity Requests")
    print("=" * 70)

    scratchpad = VoiceScratchpad(
        enable_voice=True,
        enable_clarity_requests=True
    )

    voice_prompts = []

    def on_prompt(prompt: VoicePrompt):
        voice_prompts.append(prompt)
        print(f"\n[TTS] {prompt.text}")
        if prompt.prompt_type == 'clarification':
            print(f"[Type] Clarity Request")
        if prompt.choices:
            print(f"[Choices] {', '.join(prompt.choices)}")

    scratchpad.register_voice_callbacks(on_prompt, lambda r: None)

    # Start with budget entry
    await scratchpad.start_recording('budget')

    # Ambiguous amount (missing decimal)
    await scratchpad.update_from_transcription(
        "Spent forty five dollars and ninety nine cents at the grocery store."
    )

    print(f"\n[Clarity Requests]: {sum(1 for p in voice_prompts if p.prompt_type == 'clarification')}")


async def demo_required_field_guidance():
    """Demo 3: Guided completion for required fields"""
    print("\n" + "=" * 70)
    print("DEMO 3: Required Field Guidance")
    print("=" * 70)

    scratchpad = VoiceScratchpad(enable_voice=True)

    voice_prompts = []

    def on_prompt(prompt: VoicePrompt):
        voice_prompts.append(prompt)
        if prompt.prompt_type == 'guidance':
            print(f"\n[TTS 🔊] {prompt.text}")

    scratchpad.register_voice_callbacks(on_prompt, lambda r: None)

    # Start SOP (has many required fields)
    await scratchpad.start_recording('sop')

    # Incomplete transcription
    await scratchpad.update_from_transcription(
        "SOP for winter hive inspection."
    )

    # Should prompt for missing required fields
    guidance_prompts = [p for p in voice_prompts if p.prompt_type == 'guidance']
    print(f"\n[Guidance Prompts]: {len(guidance_prompts)}")

    # Check what's missing
    state = scratchpad.get_current_state()
    print(f"[Validation Errors]: {state['validation_errors']}")


async def demo_readback_verification():
    """Demo 4: Read-back verification before save"""
    print("\n" + "=" * 70)
    print("DEMO 4: Read-Back Verification")
    print("=" * 70)

    scratchpad = VoiceScratchpad(
        enable_voice=True,
        enable_readback=True
    )

    voice_prompts = []

    def on_prompt(prompt: VoicePrompt):
        voice_prompts.append(prompt)
        if prompt.prompt_type == 'readback':
            print(f"\n[TTS 🔊 Reading Back]:")
            print(f"  {prompt.text[:200]}...")
        elif prompt.prompt_type == 'confirmation':
            print(f"\n[TTS 🔊 Confirming]: {prompt.text}")

    scratchpad.register_voice_callbacks(on_prompt, lambda r: None)

    # Complete recipe
    await scratchpad.start_recording('recipe')
    await scratchpad.update_from_transcription(
        "Recipe for chocolate chip cookies. Serves 24. "
        "Prep time 15 minutes. Cook time 12 minutes. "
        "Ingredients: 1 cup butter, 2 cups flour, 2 cups chocolate chips."
    )

    # Finalize (triggers read-back)
    try:
        shard = await scratchpad.finalize(skip_readback=False)
        print(f"\n[Finalized]: {shard.id}")
    except ValueError as e:
        print(f"\n[Cancelled]: {e}")

    readback_prompts = [p for p in voice_prompts if p.prompt_type == 'readback']
    print(f"[Read-Back Prompts]: {len(readback_prompts)}")


async def demo_safety_confirmations():
    """Demo 5: Safety confirmations for high-risk entries"""
    print("\n" + "=" * 70)
    print("DEMO 5: Safety Confirmations (HiTL)")
    print("=" * 70)

    scratchpad = VoiceScratchpad(
        enable_voice=True,
        enable_safety_confirmations=True
    )

    voice_prompts = []

    def on_prompt(prompt: VoicePrompt):
        voice_prompts.append(prompt)
        if 'risk' in prompt.text.lower():
            print(f"\n[TTS 🔊 Safety Alert]: {prompt.text}")
        if prompt.prompt_type == 'confirmation' and 'sure' in prompt.text.lower():
            print(f"[TTS 🔊 Requires Confirmation]: {prompt.text}")

    scratchpad.register_voice_callbacks(on_prompt, lambda r: None)

    # High-risk: Financial entry
    await scratchpad.start_recording('expense')
    await scratchpad.update_from_transcription(
        "Expense from Amazon for $5000 worth of equipment. Receipt AMZ-12345."
    )

    # Manual completion
    scratchpad.update_field('vendor', 'Amazon')
    scratchpad.update_field('category', 'equipment')
    scratchpad.update_field('description', 'Office equipment')

    # Finalize (should trigger safety confirmation)
    try:
        shard = await scratchpad.finalize()
        print(f"\n[Saved]: {shard.id}")
    except ValueError as e:
        print(f"\n[Blocked]: {e}")

    safety_prompts = [p for p in voice_prompts if 'risk' in p.text.lower()]
    print(f"[Safety Prompts]: {len(safety_prompts)}")


async def demo_conversation_flow():
    """Demo 6: Complete conversation flow"""
    print("\n" + "=" * 70)
    print("DEMO 6: Complete Conversation Flow")
    print("=" * 70)

    scratchpad = VoiceScratchpad(
        enable_voice=True,
        enable_clarity_requests=True,
        enable_readback=True
    )

    print("\n[Simulating voice conversation]\n")
    print("=" * 70)

    voice_log = []

    def on_prompt(prompt: VoicePrompt):
        voice_log.append(('TTS', prompt.text, prompt.prompt_type))
        icon = {
            'guidance': '💬',
            'clarification': '[QUESTION]',
            'confirmation': '[STOP]',
            'readback': '📖'
        }.get(prompt.prompt_type, '🔊')
        print(f"\n{icon} TTS: {prompt.text}")

    scratchpad.register_voice_callbacks(on_prompt, lambda r: None)

    # User starts
    print("\n👤 USER: [Starts recording for bee inspection]")
    await scratchpad.start_recording('bee_inspection')

    # User speaks
    print("\n👤 USER: 'Inspected Hive tree today. Temperature was minus five. "
          "Colony weak. Need feeding.'")

    await scratchpad.update_from_transcription(
        "Inspected Hive tree today. Temperature was minus five degrees. "
        "Colony looks weak. Will need feeding soon."
    )

    # Manual clarification (simulated)
    print("\n👤 USER: 'I meant Hive 3'")
    scratchpad.update_field('hive_id', 'Hive 3')

    # Finalize
    print("\n👤 USER: [Clicks finalize]")
    try:
        shard = await scratchpad.finalize()
        print(f"\n[OK] SAVED: {shard.id}")
    except Exception:
        print("\n[FAIL] CANCELLED")

    print("\n" + "=" * 70)
    print(f"[Total TTS Interactions]: {len(voice_log)}")
    print("\nConversation breakdown:")
    for i, (speaker, text, ptype) in enumerate(voice_log, 1):
        print(f"  {i}. [{ptype.upper()}] {text[:60]}...")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("VOICE-ENHANCED SCRATCHPAD DEMO")
    print("=" * 70)
    print("\nDemonstrating conversational voice interactions:")
    print("- TTS feedback and guidance")
    print("- Clarity requests for ambiguous data")
    print("- Read-back verification")
    print("- Safety confirmations (HiTL)")
    print()

    await demo_basic_voice_feedback()
    await demo_clarity_requests()
    await demo_required_field_guidance()
    await demo_readback_verification()
    await demo_safety_confirmations()
    await demo_conversation_flow()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("✓ TTS prompts for guidance")
    print("✓ Clarity requests for ambiguous values")
    print("✓ Required field reminders")
    print("✓ Read-back verification")
    print("✓ Safety confirmations for high-risk actions")
    print("✓ Natural conversation flow")
    print("\nNext Steps:")
    print("1. Start voice API: python HoloLoom/server/voice_scratchpad_api.py")
    print("2. Use with TTS-enabled web UI")
    print("3. Experience fully conversational data entry!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
