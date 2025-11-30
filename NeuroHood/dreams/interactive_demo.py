"""
Interactive Dream System Demo

A terminal-based interface for exploring the Dream Consciousness System.
Generate dreams, record experiences, view analysis, and explore the
collective unconscious of the neighborhood.

Usage:
    python interactive_demo.py

Author: Claude (Phase 5 Implementation)
Date: November 2025
"""

import asyncio
import sys
from typing import Optional, Dict, List
from datetime import datetime

# Import dream system components
from dream_system import DreamSystem, create_dream_system, DreamSession
from narrative_generator import NarrativeArchetype, DreamMood
from dream_influence import InfluenceType
from dream_journal import DreamSignificance


class InteractiveDreamDemo:
    """Interactive terminal interface for the Dream System."""

    def __init__(self):
        self.system: Optional[DreamSystem] = None
        self.current_resident: str = "maya_001"
        self.residents = [
            "maya_001", "jorge_002", "elena_003",
            "carlos_004", "sofia_005", "diego_006"
        ]
        self.running = True

    async def initialize(self):
        """Initialize the dream system."""
        print("\n" + "=" * 60)
        print("  DREAM CONSCIOUSNESS SYSTEM")
        print("  Interactive Demo")
        print("=" * 60)
        print("\nInitializing...")

        self.system = await create_dream_system()
        print("System ready!\n")

    def print_menu(self):
        """Display main menu."""
        print("\n" + "-" * 40)
        print(f"  Current Resident: {self.current_resident}")
        print("-" * 40)
        print("  1. Generate Dream")
        print("  2. Record Waking Experience")
        print("  3. View Active Influences")
        print("  4. Analyze Dream History")
        print("  5. View Dream Journal")
        print("  6. Generate Collective Dream")
        print("  7. Neighborhood State")
        print("  8. Explore Symbols")
        print("  9. Switch Resident")
        print("  0. Exit")
        print("-" * 40)

    def get_input(self, prompt: str, default: str = "") -> str:
        """Get user input with optional default."""
        if default:
            result = input(f"{prompt} [{default}]: ").strip()
            return result if result else default
        return input(f"{prompt}: ").strip()

    def get_float(self, prompt: str, default: float = 0.5) -> float:
        """Get float input with validation."""
        while True:
            try:
                result = input(f"{prompt} [{default}]: ").strip()
                if not result:
                    return default
                value = float(result)
                if 0 <= value <= 1:
                    return value
                print("  Please enter a value between 0 and 1")
            except ValueError:
                print("  Invalid number, please try again")

    def get_emotions(self) -> Dict[str, float]:
        """Get emotional state from user."""
        print("\n  Enter emotional intensities (0-1, press Enter for default):")
        emotions = {}

        emotion_list = [
            ("fear", 0.3), ("hope", 0.5), ("joy", 0.3),
            ("grief", 0.2), ("anxiety", 0.3), ("connection", 0.4)
        ]

        for emotion, default in emotion_list:
            value = self.get_float(f"    {emotion.capitalize()}", default)
            if value > 0:
                emotions[emotion] = value

        return emotions if emotions else {"hope": 0.5, "anxiety": 0.3}

    async def generate_dream(self):
        """Generate a dream for the current resident."""
        print("\n=== GENERATE DREAM ===")

        # Get emotional state
        emotions = self.get_emotions()

        # Get consciousness level
        print("\n  Consciousness Level:")
        print("    0.0 = Highly personal (individual unconscious)")
        print("    0.5 = Balanced")
        print("    1.0 = Universal (collective unconscious)")
        consciousness = self.get_float("  Level", 0.5)

        # Optional: choose archetype
        print("\n  Choose archetype (or press Enter for auto):")
        archetypes = list(NarrativeArchetype)
        for i, arch in enumerate(archetypes, 1):
            print(f"    {i}. {arch.value.replace('_', ' ').title()}")
        print(f"    {len(archetypes)+1}. Auto-select")

        arch_choice = self.get_input("  Choice", str(len(archetypes)+1))
        try:
            idx = int(arch_choice) - 1
            archetype = archetypes[idx] if idx < len(archetypes) else None
        except (ValueError, IndexError):
            archetype = None

        print("\n  Generating dream...")

        # Generate the dream
        session = await self.system.generate_dream(
            resident_id=self.current_resident,
            emotional_state=emotions,
            consciousness_level=consciousness,
            archetype=archetype,
        )

        # Display result
        self.display_dream(session)

    def display_dream(self, session: DreamSession):
        """Display a generated dream."""
        narrative = session.narrative

        print("\n" + "=" * 60)
        print(f"  {narrative.title.upper()}")
        print("=" * 60)

        # Mood and archetype
        print(f"\n  A {narrative.mood.value} dream of {narrative.archetype.value}")
        print()

        # Display acts
        for act in narrative.acts:
            act_title = act.act_type.upper()
            print(f"  --- {act_title} ---")
            # Word wrap description
            desc = act.description
            while len(desc) > 55:
                split = desc[:55].rfind(' ')
                if split == -1:
                    split = 55
                print(f"  {desc[:split]}")
                desc = desc[split:].strip()
            if desc:
                print(f"  {desc}")

            if act.literary_echo:
                print(f"  *Echoing: {act.literary_echo}*")
            print()

        # Jungian interpretation
        print("  --- INTERPRETATION ---")
        interp = narrative.jungian_interpretation
        while len(interp) > 55:
            split = interp[:55].rfind(' ')
            if split == -1:
                split = 55
            print(f"  {interp[:split]}")
            interp = interp[split:].strip()
        if interp:
            print(f"  {interp}")

        # Symbols used
        print(f"\n  Symbols: {', '.join(narrative.symbols_used[:5])}")

        # Influences generated
        if session.influences:
            print(f"\n  Generated {len(session.influences)} influence(s):")
            for inf in session.influences:
                print(f"    - {inf.influence_type.value}: {inf.description[:40]}...")

        print("\n" + "=" * 60)

    async def record_experience(self):
        """Record a waking experience."""
        print("\n=== RECORD WAKING EXPERIENCE ===")

        # Experience type
        print("\n  Experience type:")
        types = ["conversation", "event", "emotion", "discovery", "conflict"]
        for i, t in enumerate(types, 1):
            print(f"    {i}. {t.capitalize()}")
        type_choice = self.get_input("  Choice", "1")
        try:
            exp_type = types[int(type_choice) - 1]
        except (ValueError, IndexError):
            exp_type = "event"

        # Description
        description = self.get_input("\n  Describe the experience")
        if not description:
            description = "Something meaningful happened today"

        # People involved
        print("\n  People involved (comma-separated, or Enter for none):")
        people_input = self.get_input("  Names", "")
        people = [p.strip() for p in people_input.split(",")] if people_input else []

        # Emotional impact
        print("\n  Emotional impact:")
        emotions = self.get_emotions()

        # Record it
        experience = self.system.record_waking_experience(
            resident_id=self.current_resident,
            experience_type=exp_type,
            description=description,
            emotional_impact=emotions,
            people_involved=people if people[0] else None,
        )

        print(f"\n  Experience recorded!")
        print(f"  Dream probability: {experience.dream_probability():.0%}")
        print(f"  Symbols evoked: {', '.join(experience.symbols_evoked[:5])}")

    async def view_influences(self):
        """View active dream influences."""
        print("\n=== ACTIVE INFLUENCES ===")

        influences = self.system.get_active_influences(self.current_resident)

        if not influences:
            print("\n  No active influences.")
            return

        print(f"\n  {len(influences)} active influence(s):\n")

        for i, inf in enumerate(influences, 1):
            remaining = inf.remaining_strength()
            hours_left = (inf.expires_at - datetime.now()).total_seconds() / 3600

            print(f"  {i}. {inf.influence_type.value.upper()}")
            print(f"     Strength: {'*' * int(remaining * 10)} ({remaining:.0%})")
            print(f"     {inf.description}")
            print(f"     Expires in: {hours_left:.1f} hours")

            if inf.affected_traits:
                traits = [f"{k}: {v:+.2f}" for k, v in list(inf.affected_traits.items())[:3]]
                print(f"     Traits: {', '.join(traits)}")
            print()

        # Current modifiers
        modifiers = self.system.get_trait_modifiers(self.current_resident)
        if modifiers:
            print("  Current Trait Modifiers:")
            for trait, mod in modifiers.items():
                bar = '+' * int(abs(mod) * 20) if mod > 0 else '-' * int(abs(mod) * 20)
                print(f"    {trait}: {bar} ({mod:+.2f})")

    async def analyze_history(self):
        """Show dream analysis."""
        print("\n=== DREAM ANALYSIS ===")

        days = int(self.get_input("\n  Analyze last N days", "30"))
        analysis = self.system.analyze_dreams(self.current_resident, days)

        if analysis.dream_count == 0:
            print("\n  No dreams recorded yet. Generate some dreams first!")
            return

        print(f"\n  Dreams analyzed: {analysis.dream_count}")
        if analysis.date_range[0] != analysis.date_range[1]:
            print(f"  Date range: {analysis.date_range[0].date()} to {analysis.date_range[1].date()}")

        # Development metrics
        print("\n  --- DEVELOPMENT METRICS ---")
        metrics = [
            ("Individuation", analysis.individuation_score),
            ("Shadow Integration", analysis.shadow_integration),
            ("Anima/Animus", analysis.anima_animus_development),
            ("Self-Awareness", analysis.self_awareness),
        ]
        for name, score in metrics:
            bar = '#' * int(score * 20) + '.' * (20 - int(score * 20))
            print(f"  {name:18} [{bar}] {score:.0%}")

        # Most common symbols
        if analysis.most_common_symbols:
            print("\n  --- TOP SYMBOLS ---")
            for symbol, count in analysis.most_common_symbols[:5]:
                print(f"    {symbol}: {count} appearances")

        # Emotional profile
        print(f"\n  --- EMOTIONAL PROFILE ---")
        print(f"  Dominant mood: {analysis.dominant_mood}")
        if analysis.emotional_baseline:
            for emotion, value in sorted(analysis.emotional_baseline.items(), key=lambda x: -x[1])[:3]:
                bar = '|' * int(value * 15)
                print(f"    {emotion}: {bar} ({value:.0%})")

        # Insights
        if analysis.insights:
            print("\n  --- INSIGHTS ---")
            for insight in analysis.insights[:3]:
                print(f"    * {insight[:60]}...")

        # Suggestions
        if analysis.suggestions:
            print("\n  --- SUGGESTIONS ---")
            for suggestion in analysis.suggestions[:2]:
                print(f"    > {suggestion[:60]}...")

    async def view_journal(self):
        """View dream journal entries."""
        print("\n=== DREAM JOURNAL ===")

        entries = self.system.journal.get_entries(self.current_resident, limit=10)

        if not entries:
            print("\n  No dreams recorded yet.")
            return

        print(f"\n  Last {len(entries)} dreams:\n")

        for i, entry in enumerate(entries, 1):
            sig_stars = '*' * entry.significance.value
            lucid = ' [LUCID]' if entry.is_lucid else ''
            recurring = ' [RECURRING]' if entry.is_recurring else ''

            print(f"  {i}. {entry.title} {sig_stars}{lucid}{recurring}")
            print(f"     {entry.timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"     Mood: {entry.mood} | Archetype: {entry.archetype or 'none'}")
            print(f"     Symbols: {', '.join(entry.symbols[:4])}")
            print()

        # Option to view full entry
        choice = self.get_input("\n  Enter number to view full entry (or Enter to skip)", "")
        if choice:
            try:
                idx = int(choice) - 1
                entry = entries[idx]
                print("\n" + "=" * 50)
                print(f"  {entry.title.upper()}")
                print("=" * 50)
                print(f"\n  {entry.narrative}")
                print(f"\n  Interpretation: {entry.interpretation}")
                if entry.personal_notes:
                    print(f"\n  Notes: {entry.personal_notes}")
            except (ValueError, IndexError):
                pass

    async def collective_dream(self):
        """Generate a collective dream."""
        print("\n=== COLLECTIVE DREAM ===")

        # Select participants
        print("\n  Available residents:")
        for i, r in enumerate(self.residents, 1):
            print(f"    {i}. {r}")

        print("\n  Select participants (comma-separated numbers):")
        selection = self.get_input("  Choices", "1,2,3")

        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            participants = [self.residents[i] for i in indices if 0 <= i < len(self.residents)]
        except ValueError:
            participants = self.residents[:3]

        if len(participants) < 2:
            print("  Need at least 2 participants!")
            return

        print(f"\n  Participants: {', '.join(participants)}")

        # Shared emotional state
        print("\n  Enter shared emotional state:")
        emotions = self.get_emotions()

        print("\n  Generating collective dream...")

        sessions = await self.system.generate_collective_dream(
            dreamer_ids=participants,
            shared_emotional_state=emotions,
            consciousness_level=0.8,
        )

        # Display shared dream
        self.display_dream(sessions[0])

        print(f"\n  This dream was shared by {len(participants)} residents.")
        print("  Each will be influenced by its themes.")

    async def neighborhood_state(self):
        """Show neighborhood state."""
        print("\n=== NEIGHBORHOOD STATE ===")

        state = self.system.get_system_state()

        print(f"\n  Dreams Today: {state.total_dreams_today}")
        print(f"  Collective Dreams: {state.collective_dream_count}")

        # Zeitgeist
        zeitgeist = state.neighborhood_zeitgeist
        print("\n  --- EMOTIONAL ZEITGEIST ---")
        emotions = {
            "Joy": zeitgeist.joy,
            "Fear": zeitgeist.fear,
            "Hope": zeitgeist.hope,
            "Grief": zeitgeist.grief,
            "Anxiety": zeitgeist.anxiety,
            "Connection": zeitgeist.connection,
        }
        for name, value in emotions.items():
            if abs(value) > 0.01:
                bar = '+' * int(value * 10) if value > 0 else '-' * int(abs(value) * 10)
                print(f"    {name}: {bar} ({value:+.2f})")

        print(f"\n  Emotional Temperature: {zeitgeist.emotional_temperature():.2f}")
        dominant = zeitgeist.dominant_emotion()
        if dominant:
            print(f"  Dominant Emotion: {dominant}")

        # Active archetypes
        if state.active_archetypes:
            print("\n  --- ACTIVE ARCHETYPES ---")
            for arch, count in state.active_archetypes.items():
                print(f"    {arch}: {count} dream(s)")

        # Neighborhood mood
        mood = state.neighborhood_mood
        active_moods = {k: v for k, v in mood.items() if abs(v) > 0.01}
        if active_moods:
            print("\n  --- NEIGHBORHOOD MOOD ---")
            for m, v in active_moods.items():
                bar = '>' * int(v * 20) if v > 0 else '<' * int(abs(v) * 20)
                print(f"    {m}: {bar} ({v:+.2f})")

    async def explore_symbols(self):
        """Explore the symbol database."""
        print("\n=== EXPLORE SYMBOLS ===")

        # Categories
        categories = [
            "trapped", "loss", "fear", "hope", "transformation",
            "guilt", "power", "conflict", "mystery", "connection"
        ]

        print("\n  Symbol Categories:")
        for i, cat in enumerate(categories, 1):
            print(f"    {i}. {cat.capitalize()}")

        choice = self.get_input("\n  Choose category (or Enter to search)", "")

        if choice:
            try:
                cat = categories[int(choice) - 1]
                symbols = self.system.collective.get_symbols_by_category(cat)
                print(f"\n  Symbols in '{cat}':")
                for s in symbols[:10]:
                    print(f"    - {s['symbol_id']}: {s.get('name', s['symbol_id'])}")
            except (ValueError, IndexError):
                pass
        else:
            # Search by name
            query = self.get_input("  Search symbol")
            if query:
                symbol = self.system.collective.get_symbol(query)
                if symbol:
                    print(f"\n  Found: {symbol['symbol_id'].upper()}")
                    print(f"  Category: {symbol.get('category', 'unknown')}")
                    print(f"  Description: {symbol.get('description', 'N/A')}")

                    if 'literary_references' in symbol:
                        refs = symbol['literary_references']
                        total = symbol.get('total_references', 0)
                        print(f"\n  Literary References ({total} total):")

                        for category, items in refs.items():
                            if items:
                                print(f"\n    {category.replace('_', ' ').title()}:")
                                for ref in items[:2]:  # Show first 2 per category
                                    title = ref.get('title', 'Unknown')
                                    author = ref.get('author', ref.get('artist', ref.get('source', '')))
                                    connection = ref.get('connection', '')[:60]
                                    print(f"      - {title} ({author})")
                                    if connection:
                                        print(f"        \"{connection}...\"")
                else:
                    print(f"  Symbol '{query}' not found")

        # Hot symbols
        print("\n  --- HOT SYMBOLS (Most Used) ---")
        hot = self.system.collective.get_hot_symbols(count=5)
        for symbol_id, heat_score in hot:
            print(f"    {symbol_id}: heat={heat_score:.2f}")

    def switch_resident(self):
        """Switch current resident."""
        print("\n=== SWITCH RESIDENT ===")

        print("\n  Available residents:")
        for i, r in enumerate(self.residents, 1):
            marker = " <--" if r == self.current_resident else ""
            print(f"    {i}. {r}{marker}")

        choice = self.get_input("\n  Choose resident", "1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.residents):
                self.current_resident = self.residents[idx]
                print(f"\n  Switched to: {self.current_resident}")
        except ValueError:
            pass

    async def run(self):
        """Main loop."""
        await self.initialize()

        while self.running:
            self.print_menu()
            choice = self.get_input("\n  Choice")

            try:
                if choice == "1":
                    await self.generate_dream()
                elif choice == "2":
                    await self.record_experience()
                elif choice == "3":
                    await self.view_influences()
                elif choice == "4":
                    await self.analyze_history()
                elif choice == "5":
                    await self.view_journal()
                elif choice == "6":
                    await self.collective_dream()
                elif choice == "7":
                    await self.neighborhood_state()
                elif choice == "8":
                    await self.explore_symbols()
                elif choice == "9":
                    self.switch_resident()
                elif choice == "0":
                    print("\n  Goodbye! Sweet dreams...")
                    self.running = False
                else:
                    print("\n  Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\n\n  Interrupted. Exiting...")
                self.running = False
            except Exception as e:
                print(f"\n  Error: {e}")
                print("  Please try again.")

        print()


async def main():
    demo = InteractiveDreamDemo()
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())