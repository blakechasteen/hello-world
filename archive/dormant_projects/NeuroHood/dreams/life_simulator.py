"""
Life Simulator - Interactive Neighborhood Experience

Watch and participate in the daily lives of neighborhood residents.

Modes:
- WATCH: Observe the simulation unfold automatically
- PARTICIPATE: Join conversations and interact with residents

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

import asyncio
import sys
from typing import Optional, List
from enum import Enum

from daytime_simulation import DaytimeSimulation, SimulationEvent, EventType
from conversation import Conversation, DialogueLine
from resident import Resident


class SimMode(Enum):
    WATCH = "watch"
    PARTICIPATE = "participate"
    PAUSED = "paused"


class LifeSimulator:
    """Interactive life simulation interface."""

    def __init__(self, dream_system=None):
        self.sim = DaytimeSimulation(dream_system)
        self.mode = SimMode.PAUSED
        self.running = True
        self.speed = 1.0  # Seconds between updates in watch mode

        # Current state
        self.current_conversation: Optional[Conversation] = None
        self.conversation_partner: Optional[str] = None
        self.response_options: List[str] = []

        # Event buffer for display
        self.event_buffer: List[str] = []
        self.max_buffer = 10

        # Register callbacks
        self.sim.add_event_callback(self._on_event)
        self.sim.add_conversation_callback(self._on_conversation)

    def _on_event(self, event: SimulationEvent):
        """Handle simulation events."""
        # Format event for display
        time_str = event.timestamp.strftime("%H:%M")

        if event.event_type == EventType.CONVERSATION:
            msg = f"[{time_str}] {event.description}"
        elif event.event_type == EventType.MOVEMENT:
            msg = f"[{time_str}] {event.description}"
        elif event.event_type == EventType.USER_ACTION:
            msg = f"[{time_str}] >> {event.description}"
        else:
            msg = f"[{time_str}] {event.description}"

        self.event_buffer.append(msg)
        if len(self.event_buffer) > self.max_buffer:
            self.event_buffer.pop(0)

    def _on_conversation(self, conversation: Conversation):
        """Handle conversation events in watch mode."""
        if self.mode == SimMode.WATCH:
            self._display_conversation(conversation)

    def _display_conversation(self, conversation: Conversation):
        """Display a conversation."""
        print(f"\n  {'~'*50}")
        print(f"  At {conversation.location}")
        print(f"  {'~'*50}")

        for line in conversation.lines:
            if line.action:
                print(f"    {line.action}")
            print(f"  {line.speaker_name}: \"{line.text}\"")

        print(f"  {'~'*50}\n")

    def clear_screen(self):
        """Clear terminal screen."""
        print("\033[2J\033[H", end="")

    def print_header(self):
        """Print simulation header."""
        print("=" * 60)
        print("  NEUROHOOD LIFE SIMULATOR")
        print("  " + "-" * 56)
        status = self.sim.get_status()
        for line in status.strip().split("\n"):
            print(line)
        print("=" * 60)

    def print_events(self):
        """Print recent events."""
        if self.event_buffer:
            print("\n  Recent Activity:")
            print("  " + "-" * 40)
            for event in self.event_buffer[-5:]:
                print(f"  {event}")

    def print_menu_main(self):
        """Print main menu."""
        print("\n  --- MAIN MENU ---")
        print()
        print("  [W] Watch Mode - Observe the neighborhood")
        print("  [P] Participate Mode - Join and interact")
        print("  [L] Look Around - See locations")
        print("  [R] Residents - View all residents")
        print("  [T] Advance Time - Skip ahead")
        print("  [Q] Quit")
        print()

    def print_menu_watch(self):
        """Print watch mode menu."""
        print("\n  [SPACE/ENTER] Next step")
        print("  [F] Fast forward (5 steps)")
        print("  [S] Slow down / Speed up")
        print("  [P] Switch to Participate")
        print("  [M] Main menu")

    def print_menu_participate(self):
        """Print participate mode menu."""
        print("\n  --- PARTICIPATE MODE ---")
        print()
        print("  [G] Go to a location")
        print("  [T] Talk to someone here")
        print("  [L] Look around")
        print("  [W] Switch to Watch mode")
        print("  [M] Main menu")

    def get_input(self, prompt: str = "> ", default: str = "") -> str:
        """Get user input."""
        try:
            result = input(f"  {prompt}").strip()
            return result if result else default
        except (EOFError, KeyboardInterrupt):
            return "q"

    def show_locations(self):
        """Show all available locations."""
        print("\n  --- LOCATIONS ---")
        print()

        tod = self.sim.state.time_of_day.value

        for loc_id, loc in self.sim.location_manager.locations.items():
            if loc.location_type.value != "home":
                status = "OPEN" if loc.is_open(tod) else "CLOSED"
                occupants = len(loc.current_occupants)
                print(f"  [{loc_id[:12]:12}] {loc.name:20} ({status}, {occupants} people)")

        print()

    def show_residents(self):
        """Show all residents."""
        print("\n  --- RESIDENTS ---")
        print()

        for r_id, resident in self.sim.residents.items():
            loc = self.sim.location_manager.get_resident_location(r_id)
            loc_name = loc.name if loc else "Unknown"
            mood = resident.emotional_state.mood_description()
            print(f"  {resident.name:15} | {loc_name:20} | {mood}")

        print()

    def go_to_location(self):
        """Move user to a location."""
        self.show_locations()
        loc_id = self.get_input("Go to (location id): ")

        if loc_id:
            result = self.sim.user_goes_to(loc_id)
            print(result)

    def look_around(self):
        """Look around current location."""
        if self.sim.user_location:
            print(self.sim.get_location_description(self.sim.user_location))
        else:
            print("\n  You haven't gone anywhere yet.")
            print("  Use [G] to go to a location first.")

    def talk_to_someone(self):
        """Initiate conversation with someone."""
        if not self.sim.user_location:
            print("\n  Go to a location first!")
            return

        # Find people here
        loc = self.sim.location_manager.get_location(self.sim.user_location)
        if not loc or not loc.current_occupants:
            print("\n  No one else is here.")
            return

        print("\n  Who would you like to talk to?")
        residents_here = []
        for i, r_id in enumerate(loc.current_occupants, 1):
            resident = self.sim.residents.get(r_id)
            if resident:
                residents_here.append(r_id)
                print(f"  [{i}] {resident.name}")

        choice = self.get_input("Choose: ")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(residents_here):
                r_id = residents_here[idx]
                self.start_conversation(r_id)
        except ValueError:
            pass

    def start_conversation(self, resident_id: str):
        """Start a conversation with a resident."""
        result = self.sim.user_talks_to(resident_id)
        if not result:
            print("\n  Can't talk to them right now.")
            return

        self.current_conversation, self.response_options = result
        self.conversation_partner = resident_id

        self.run_conversation()

    def run_conversation(self):
        """Run the conversation loop."""
        resident = self.sim.residents.get(self.conversation_partner)
        if not resident:
            return

        while self.current_conversation and not self.current_conversation.is_complete:
            self.clear_screen()
            print("\n" + "=" * 60)
            print(f"  CONVERSATION WITH {resident.name.upper()}")
            print("=" * 60)

            # Show conversation so far
            for line in self.current_conversation.lines[-6:]:
                if line.action:
                    print(f"\n    {line.action}")
                speaker = "You" if line.speaker_id == "user" else line.speaker_name
                print(f"\n  {speaker}: \"{line.text}\"")

            # Show options
            print("\n  " + "-" * 50)
            print("  Your response:")
            for i, option in enumerate(self.response_options, 1):
                # Truncate long options
                display = option[:55] + "..." if len(option) > 55 else option
                print(f"  [{i}] {display}")
            print("  [0] (Type your own response)")
            print("  [Q] End conversation")

            choice = self.get_input("\n  Choice: ")

            if choice.lower() == 'q':
                self.end_current_conversation()
                break

            try:
                idx = int(choice)
                if idx == 0:
                    # Custom response
                    custom = self.get_input("  Say: ")
                    if custom:
                        self.continue_conversation(custom)
                elif 1 <= idx <= len(self.response_options):
                    self.continue_conversation(self.response_options[idx - 1])
            except ValueError:
                # Treat as custom input
                if choice:
                    self.continue_conversation(choice)

    def continue_conversation(self, user_text: str):
        """Continue the conversation with user's response."""
        # Check for goodbye
        if "going" in user_text.lower() or "goodbye" in user_text.lower():
            self.end_current_conversation()
            return

        response, self.response_options = self.sim.user_says(
            self.current_conversation,
            user_text,
            self.conversation_partner,
        )

    def end_current_conversation(self):
        """End the current conversation."""
        resident = self.sim.residents.get(self.conversation_partner)

        if self.current_conversation and self.conversation_partner:
            self.sim.end_conversation(self.current_conversation, self.conversation_partner)

        print(f"\n  You wave goodbye to {resident.name if resident else 'them'}.")
        print("  Press Enter to continue...")
        self.get_input("")

        self.current_conversation = None
        self.conversation_partner = None
        self.response_options = []

    def run_watch_mode(self):
        """Run watch mode."""
        self.mode = SimMode.WATCH

        while self.mode == SimMode.WATCH:
            self.clear_screen()
            self.print_header()
            self.print_events()
            self.print_menu_watch()

            choice = self.get_input("Watch > ").lower()

            if choice in ['', ' ', 'n']:
                # Single step
                events = self.sim.simulate_step()
            elif choice == 'f':
                # Fast forward
                print("\n  Fast forwarding...")
                for _ in range(5):
                    self.sim.simulate_step()
            elif choice == 's':
                self.speed = 0.5 if self.speed == 1.0 else 1.0
                print(f"\n  Speed: {'Fast' if self.speed == 0.5 else 'Normal'}")
            elif choice == 'p':
                self.mode = SimMode.PARTICIPATE
            elif choice == 'm':
                self.mode = SimMode.PAUSED
                break

    def run_participate_mode(self):
        """Run participate mode."""
        self.mode = SimMode.PARTICIPATE

        # First go somewhere if not already
        if not self.sim.user_location:
            print("\n  You need to go somewhere first!")
            self.go_to_location()

        while self.mode == SimMode.PARTICIPATE:
            self.clear_screen()
            self.print_header()

            if self.sim.user_location:
                loc = self.sim.location_manager.get_location(self.sim.user_location)
                print(f"\n  You are at: {loc.name if loc else 'Unknown'}")

            self.print_events()
            self.print_menu_participate()

            choice = self.get_input("Participate > ").lower()

            if choice == 'g':
                self.go_to_location()
            elif choice == 't':
                self.talk_to_someone()
            elif choice == 'l':
                self.look_around()
                self.get_input("\n  Press Enter to continue...")
            elif choice == 'w':
                self.mode = SimMode.WATCH
                self.run_watch_mode()
            elif choice == 'm':
                self.mode = SimMode.PAUSED
                break

            # Advance time a bit
            self.sim.advance_time(5)

    def run(self):
        """Main simulation loop."""
        self.running = True

        # Welcome message
        self.clear_screen()
        print("\n" + "=" * 60)
        print("  WELCOME TO NEUROHOOD")
        print("=" * 60)
        print("""
  You're about to enter a living neighborhood where
  residents go about their daily lives, have conversations,
  form relationships, and dream.

  You can:
  - WATCH the neighborhood come alive
  - PARTICIPATE in conversations and events
  - EXPLORE different locations
  - BUILD relationships with residents

  Their dreams at night will reflect their daytime experiences.
""")
        print("=" * 60)
        self.get_input("\n  Press Enter to begin...")

        while self.running:
            self.clear_screen()
            self.print_header()
            self.print_events()
            self.print_menu_main()

            choice = self.get_input("Main > ").lower()

            if choice == 'w':
                self.run_watch_mode()
            elif choice == 'p':
                self.run_participate_mode()
            elif choice == 'l':
                self.show_locations()
                self.get_input("\n  Press Enter to continue...")
            elif choice == 'r':
                self.show_residents()
                self.get_input("\n  Press Enter to continue...")
            elif choice == 't':
                print("\n  Advancing time by 1 hour...")
                for _ in range(4):
                    self.sim.simulate_step()
            elif choice == 'q':
                print("\n  Goodbye!")
                self.running = False

        print("\n  Thanks for visiting NeuroHood!\n")


async def main():
    """Main entry point."""
    simulator = LifeSimulator()
    simulator.run()


if __name__ == "__main__":
    asyncio.run(main())
