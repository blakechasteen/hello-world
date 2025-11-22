"""
Voice UX Milestone 1 Demo - Conversational Threading

Demonstrates:
1. Thread creation via voice commands
2. Thread switching
3. Thread listing
4. Thread summarization
5. Thread-aware HoloLoom queries (with context)

Created: November 20, 2025
Status: Milestone 1 Complete
"""

import asyncio
from elle.voice.llm_parser import LLMParser
from elle.voice.threads import ThreadManager


def demo_llm_parser_thread_commands():
    """Demo: LLM Parser recognizing thread commands"""
    print("\n" + "="*70)
    print("DEMO 1: LLM Parser - Thread Command Recognition")
    print("="*70)

    parser = LLMParser(use_llm=False)  # Pattern matching only

    # Test thread commands
    test_commands = [
        # Thread creation
        "start a new thread for baking bread",
        "create thread about biochar",
        "open a new thread on greenhouse planning",

        # Thread switching
        "switch to thread baking",
        "go back to biochar thread",
        "activate baking thread",

        # Thread listing
        "list my threads",
        "show threads",
        "what threads do I have",

        # Thread summarize
        "summarize baking thread",
        "summarize active thread",
        "summarize current",
    ]

    print("\nParsing thread commands:")
    for text in test_commands:
        cmd = parser.parse(text)
        print(f"\n  Input:  '{text}'")
        print(f"  Parsed: {cmd.command_type}")
        if cmd.parameters:
            print(f"  Params: {cmd.parameters}")

    print("\n" + "="*70)


async def demo_thread_management():
    """Demo: Thread Manager CRUD operations"""
    print("\n" + "="*70)
    print("DEMO 2: Thread Manager - Create, Switch, List, Summarize")
    print("="*70)

    manager = ThreadManager(storage_path="elle/data/demo_threads.json")

    # Create threads
    print("\n1. Creating threads via voice commands:")
    print("  Voice: 'Start a new thread for baking bread'")
    thread1 = manager.create_thread("Baking Bread", "Baking sourdough")
    print(f"  [OK] Created: {thread1.name}")

    print("\n  Voice: 'Create thread about biochar'")
    thread2 = manager.create_thread("Biochar", "Biochar production")
    print(f"  [OK] Created: {thread2.name}")

    print("\n  Voice: 'Open thread on greenhouse planning'")
    thread3 = manager.create_thread("Greenhouse", "Greenhouse setup")
    print(f"  [OK] Created: {thread3.name}")

    # Add messages to active thread (Greenhouse)
    print("\n2. Conversation in 'Greenhouse' thread:")
    manager.add_message_to_active("user", "What temperature for tomatoes?")
    manager.add_message_to_active("assistant", "Tomatoes thrive at 70-80F during day.")
    manager.add_message_to_active("user", "What about nighttime?")
    manager.add_message_to_active("assistant", "60-65F is ideal for nighttime.")
    print("  [OK] Added 4 messages to Greenhouse thread")

    # Switch to thread 1
    print("\n3. Switching threads via voice command:")
    print("  Voice: 'Switch to baking thread'")
    switched = manager.switch_thread("Baking")
    if switched:
        print(f"  [OK] Switched to: {switched.name}")

        # Add messages to Baking thread
        manager.add_message_to_active("user", "How long to proof sourdough?")
        manager.add_message_to_active("assistant", "Bulk: 4-6 hours. Final: 2-4 hours.")
        print("  [OK] Added 2 messages to Baking Bread thread")

    # List threads
    print("\n4. Listing threads via voice command:")
    print("  Voice: 'List my threads'")
    threads = manager.list_threads()
    for i, thread in enumerate(threads, 1):
        active_marker = "*" if thread.id == manager.active_thread_id else " "
        msg_count = len(thread.messages)
        print(f"  {active_marker} {i}. {thread.name} - {msg_count} messages")

    # Summarize active thread
    print("\n5. Summarizing thread via voice command:")
    print("  Voice: 'Summarize active thread'")
    summary = manager.get_active_thread().get_summary()
    print(f"\n{summary}\n")

    # Switch by partial name
    print("6. Fuzzy thread switching:")
    print("  Voice: 'Go back to greenhouse'")
    switched = manager.switch_thread("greenhouse")
    if switched:
        print(f"  [OK] Switched to: {switched.name}")

    # Summarize specific thread
    print("\n  Voice: 'Summarize greenhouse thread'")
    greenhouse = manager.get_thread_by_name("greenhouse")
    if greenhouse:
        print(f"\n{greenhouse.get_summary()}\n")

    print("="*70)


async def demo_thread_aware_context():
    """Demo: Thread context for HoloLoom queries"""
    print("\n" + "="*70)
    print("DEMO 3: Thread-Aware Context for HoloLoom Queries")
    print("="*70)

    manager = ThreadManager(storage_path="elle/data/demo_context_threads.json")

    # Create thread with conversation
    print("\n1. Creating thread with conversation history:")
    thread = manager.create_thread("Tomato Growing", "Growing tomatoes in greenhouse")
    manager.add_message_to_active("user", "What's the ideal temperature for tomatoes?")
    manager.add_message_to_active("assistant", "Tomatoes thrive at 70-80F during day.")
    manager.add_message_to_active("user", "What about watering?")
    manager.add_message_to_active("assistant", "Water deeply 1-2 times per week.")
    manager.add_message_to_active("user", "And fertilizing?")
    manager.add_message_to_active("assistant", "Use balanced fertilizer every 2 weeks.")
    print(f"  [OK] Created thread with {len(thread.messages)} messages")

    # Get thread context (last 10 messages)
    print("\n2. Extracting thread context for HoloLoom query:")
    recent_messages = thread.get_last_n_messages(10)
    context = [f"{msg.role}: {msg.content}" for msg in recent_messages]

    print("  Thread context (last 5 messages):")
    for msg_str in context[-5:]:
        print(f"    {msg_str}")

    # Simulate query with context
    print("\n3. Simulating HoloLoom query WITH thread context:")
    query = "How often?"
    print(f"  User query: '{query}'")
    print("\n  WITHOUT context: Query is ambiguous")
    print("  WITH context: Elle knows user is asking about fertilizing frequency")
    print("  Enriched query sent to HoloLoom:")
    enriched = f"Conversation context:\n" + "\n".join(context[-5:]) + f"\n\nCurrent question: {query}"
    print(f"\n{enriched}\n")
    print("  HoloLoom can now provide accurate answer using conversation history!")

    print("="*70)


async def demo_complete_interaction():
    """Demo: Complete voice interaction with threading"""
    print("\n" + "="*70)
    print("DEMO 4: Complete Voice Interaction (Simulated)")
    print("="*70)

    manager = ThreadManager(storage_path="elle/data/demo_complete.json")
    parser = LLMParser(use_llm=False)

    print("\nSimulated voice conversation:")
    print("-" * 70)

    # Conversation 1: Create thread
    print("\nUser:  'Hey Elle, start a new thread for baking bread'")
    cmd = parser.parse("start a new thread for baking bread")
    if cmd.command_type == "thread_create":
        topic = cmd.parameters.get("thread_topic")
        thread = manager.create_thread(topic, topic)
        print(f"Elle:  'Created new thread Baking bread. What would you like to discuss?'")
        print(f"       [Internal: Thread ID: {thread.id[:8]}, Active: {thread.name}]")

    # Conversation 2: Ask question in thread
    print("\nUser:  'How long should I proof sourdough?'")
    manager.add_message_to_active("user", "How long should I proof sourdough?")
    response = "Bulk fermentation: 4-6 hours. Final proof: 2-4 hours."
    manager.add_message_to_active("assistant", response)
    print(f"Elle:  '{response}'")
    print(f"       [Internal: Message stored in '{manager.get_active_thread().name}' thread]")

    # Conversation 3: Follow-up question (uses thread context!)
    print("\nUser:  'What about temperature?'")
    manager.add_message_to_active("user", "What about temperature?")
    print("       [Internal: HoloLoom receives thread context:]")
    context = manager.get_active_thread().get_last_n_messages(5)
    for msg in context:
        print(f"         {msg.role}: {msg.content}")
    response = "For proofing, 75-80F is ideal."
    manager.add_message_to_active("assistant", response)
    print(f"Elle:  '{response}'")

    # Conversation 4: Create another thread
    print("\nUser:  'Create a new thread about biochar'")
    cmd = parser.parse("create a new thread about biochar")
    if cmd.command_type == "thread_create":
        topic = cmd.parameters.get("thread_topic")
        thread = manager.create_thread(topic, topic)
        print(f"Elle:  'Created new thread Biochar. What would you like to discuss?'")

    # Conversation 5: List threads
    print("\nUser:  'List my threads'")
    cmd = parser.parse("list my threads")
    if cmd.command_type == "thread_list":
        threads = manager.list_threads()
        response = "Your threads:\n"
        for i, thread in enumerate(threads, 1):
            active = "*" if thread.id == manager.active_thread_id else " "
            response += f"  {active} {i}. {thread.name} ({len(thread.messages)} messages)\n"
        print(f"Elle:  {response}")

    # Conversation 6: Switch thread
    print("User:  'Switch to baking thread'")
    cmd = parser.parse("switch to baking thread")
    if cmd.command_type == "thread_switch":
        thread_name = cmd.parameters.get("thread_name")
        thread = manager.switch_thread(thread_name)
        if thread:
            print(f"Elle:  'Switched to thread Baking bread. {len(thread.messages)} messages in this thread.'")

    # Conversation 7: Summarize thread
    print("\nUser:  'Summarize this thread'")
    cmd = parser.parse("summarize this thread")
    if cmd.command_type == "thread_summarize":
        summary = manager.get_active_thread().get_summary()
        print(f"Elle:  '{summary}'")

    print("\n" + "-" * 70)
    print(f"\nFinal state: {len(manager.threads)} threads, active: {manager.get_active_thread().name}")
    print("="*70)


async def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("VOICE UX MILESTONE 1 - COMPLETE DEMO")
    print("Conversational Threading System")
    print("="*70)

    # Demo 1: LLM Parser
    demo_llm_parser_thread_commands()

    # Demo 2: Thread Manager
    await demo_thread_management()

    # Demo 3: Thread Context
    await demo_thread_aware_context()

    # Demo 4: Complete Interaction
    await demo_complete_interaction()

    print("\n" + "="*70)
    print("MILESTONE 1 FEATURES DEMONSTRATED:")
    print("  [OK] Thread creation via voice commands")
    print("  [OK] Thread switching (exact + fuzzy matching)")
    print("  [OK] Thread listing")
    print("  [OK] Thread summarization")
    print("  [OK] Thread-aware HoloLoom context")
    print("  [OK] Message persistence (JSON storage)")
    print("  [OK] Conversational continuity")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
