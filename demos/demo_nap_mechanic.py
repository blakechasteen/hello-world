"""
Demo: Nap Mechanic and Energy System
=====================================

Comprehensive demonstration of NeuroHood's energy system and dream triggers.

Scenario: Follow a resident through a full day/night cycle showing:
1. Morning: High energy, normal activities
2. Midday: Work depletes energy
3. Afternoon: Low energy, decision point for nap
4. Evening: More activities, energy drops further
5. Night: Critical energy, forced sleep with intense dream
6. Next morning: Recovery and vivid dreams

Shows:
- Real-time energy tracking
- Activity cost breakdown
- Dream trigger logic
- Quality metrics at different energy levels
- Historical trends
"""

import sys
from pathlib import Path

# Add NeuroHood to path
neurohoodpath = Path(__file__).parent.parent / "NeuroHood" / "dreams"
sys.path.insert(0, str(neurohoodpath))

from nap_mechanic import (
    NapMechanic,
    ActivityType,
    DreamType,
    create_nap_mechanic,
)


# ============================================================================
# Color codes for terminal output
# ============================================================================

class Colors:
    """ANSI color codes for pretty output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


def color_energy(energy: float) -> str:
    """Return color based on energy level."""
    if energy < 30:
        return Colors.RED + Colors.BOLD
    elif energy < 60:
        return Colors.YELLOW
    elif energy < 80:
        return Colors.CYAN
    else:
        return Colors.GREEN


def energy_bar(energy: float, width: int = 30) -> str:
    """Create a visual energy bar."""
    filled = int(width * (energy / 100.0))
    bar = "[" + "=" * filled + "-" * (width - filled) + "]"
    color = color_energy(energy)
    return f"{color}{bar}{Colors.RESET}"


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")


def print_subheader(text: str):
    """Print a subsection header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.DIM}{'- ' * 35}{Colors.RESET}")


def print_time(hour: int, minute: int = 0):
    """Print formatted time."""
    return f"{Colors.BOLD}{Colors.MAGENTA}{hour:02d}:{minute:02d}{Colors.RESET}"


# ============================================================================
# Demo: Day Cycle
# ============================================================================


def demo_day_cycle():
    """Simulate a complete day with energy and dream tracking."""
    print_header("NEUROHOOD NAP MECHANIC DEMO: Full Day Cycle")

    # Create resident
    resident = create_nap_mechanic("Alice")
    print(f"Resident: {Colors.BOLD}Alice{Colors.RESET}")
    print(f"Initial Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%\n")

    # Track timeline
    events = []

    # ========================================================================
    # MORNING (6:00-9:00)
    # ========================================================================
    print_subheader("MORNING: 6:00 AM - Wake Up (High Energy)")
    print(f"Time: {print_time(6, 0)}")
    print(f"Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")
    print("Status: Rested, ready for the day")
    print("Quality Metrics:")
    metrics = resident.get_dream_quality_metrics()
    for key, value in metrics.items():
        print(f"  • {key.capitalize()}: {value:.1%}")

    # Light morning routine (3 hours)
    resident.update_energy(hours_elapsed=3.0)
    print(f"\n{print_time(9, 0)} - After 3 hours of morning routine")
    print(f"Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")

    # ========================================================================
    # WORK DAY (9:00-17:00)
    # ========================================================================
    print_subheader("DAYTIME: 9:00 AM - 5:00 PM (Work)")
    print(f"Time: {print_time(9, 0)}")
    print(f"Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")

    # Intense work session
    print(f"\n{print_time(9, 0)} - Start work day")
    resident.deplete_energy(ActivityType.WORK)
    resident.update_energy(hours_elapsed=4.0)
    print(f"Energy after 4h work: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")

    # Lunch break (1 hour)
    print(f"\n{print_time(13, 0)} - Lunch break (rest)")
    resident.update_energy(hours_elapsed=1.0)  # Passive depletion continues
    print(f"Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")

    # Afternoon work (4 hours)
    print(f"\n{print_time(14, 0)} - Afternoon work")
    resident.deplete_energy(ActivityType.WORK)
    resident.update_energy(hours_elapsed=3.0)
    print(f"Energy after work: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")

    current_energy = resident.energy_state.current_energy
    print(f"\nEnergy Status: ", end="")
    if current_energy < 30:
        print(f"{Colors.RED}CRITICAL - Must sleep soon{Colors.RESET}")
    elif current_energy < 60:
        print(f"{Colors.YELLOW}LOW - Nap recommended{Colors.RESET}")
    elif current_energy < 80:
        print(f"{Colors.CYAN}MODERATE - Still functional{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}GOOD - Plenty of energy{Colors.RESET}")

    # ========================================================================
    # AFTERNOON NAP DECISION (17:00-18:30)
    # ========================================================================
    print_subheader("AFTERNOON: 5:00 PM - Nap Decision Point")
    print(f"Time: {print_time(17, 0)}")
    print(f"Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")
    print(f"Needs Sleep: {Colors.YELLOW}{resident.energy_state.needs_sleep}{Colors.RESET}")

    if resident.energy_state.current_energy < 60:
        print(f"\n{Colors.BOLD}Alice decides to take a nap...{Colors.RESET}")
        energy_before_nap = resident.energy_state.current_energy
        state, nap_event = resident.restore_energy(sleep_hours=1.5, quality_modifier=1.0)

        print(f"\nNap Details:")
        print(f"  Duration: {nap_event.duration_hours} hours")
        print(f"  Energy Before: {energy_bar(energy_before_nap)} {energy_before_nap:.1f}%")
        print(f"  Energy After: {energy_bar(state.current_energy)} {state.current_energy:.1f}%")
        print(f"  Restoration: {state.current_energy - energy_before_nap:.1f}%")

        if nap_event.dream_triggered:
            print(f"\n  {Colors.BOLD}Dream Triggered!{Colors.RESET}")
            print(f"  Dream Type: {Colors.MAGENTA}{nap_event.dream_type.value.upper()}{Colors.RESET}")
            print(f"  Intensity: {nap_event.dream_intensity:.1%}")
            intensity_bar = "=" * int(nap_event.dream_intensity * 20) + "-" * (20 - int(nap_event.dream_intensity * 20))
            print(f"  {Colors.MAGENTA}{intensity_bar}{Colors.RESET}")
        else:
            print(f"\n  No dream occurred")

    # ========================================================================
    # EVENING ACTIVITIES (18:30-21:00)
    # ========================================================================
    print_subheader("EVENING: 6:30 PM - Activities")
    print(f"Time: {print_time(18, 30)}")
    print(f"Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")

    # Social dinner (2.5 hours)
    print(f"\n{print_time(18, 30)} - Social dinner with friends")
    resident.deplete_energy(ActivityType.SOCIAL)
    resident.update_energy(hours_elapsed=2.5)
    print(f"Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")

    # Stress event
    print(f"\n{print_time(21, 0)} - Unexpected conflict")
    resident.deplete_energy(ActivityType.CONFLICT, multiplier=1.3)
    print(f"Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")
    print(f"Status: {Colors.RED}CRITICAL - Needs rest immediately{Colors.RESET}")

    # ========================================================================
    # NIGHT SLEEP (21:00-7:00 next morning)
    # ========================================================================
    print_subheader("NIGHT: 9:00 PM - Mandatory Sleep")
    print(f"Time: {print_time(21, 0)}")
    print(f"Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")

    print(f"\n{Colors.BOLD}Alice is exhausted and must sleep immediately...{Colors.RESET}")
    energy_before_sleep = resident.energy_state.current_energy
    state, sleep_event = resident.restore_energy(sleep_hours=8.0, quality_modifier=0.8)

    print(f"\nSleep Details:")
    print(f"  Duration: {sleep_event.duration_hours} hours")
    print(f"  Energy Before: {energy_bar(energy_before_sleep)} {energy_before_sleep:.1f}%")
    print(f"  Energy After: {energy_bar(state.current_energy)} {state.current_energy:.1f}%")
    print(f"  Restoration: {state.current_energy - energy_before_sleep:.1f}%")
    print(f"  Forced Sleep: {Colors.RED}{sleep_event.forced_sleep}{Colors.RESET}")

    if sleep_event.dream_triggered:
        print(f"\n  {Colors.BOLD}Powerful Dream Occurred!{Colors.RESET}")
        print(f"  Dream Type: {Colors.MAGENTA}{sleep_event.dream_type.value.upper()}{Colors.RESET}")
        print(f"  Intensity: {sleep_event.dream_intensity:.1%}")
        intensity_bar = "=" * int(sleep_event.dream_intensity * 20) + "-" * (20 - int(sleep_event.dream_intensity * 20))
        print(f"  {Colors.MAGENTA}{intensity_bar}{Colors.RESET}")

        print(f"\n  Dream Quality Metrics at Sleep Time:")
        metrics_during_sleep = resident.get_dream_quality_metrics()
        for key, value in metrics_during_sleep.items():
            bar = "=" * int(value * 10) + "-" * (10 - int(value * 10))
            print(f"    • {key.capitalize()}: {Colors.MAGENTA}{bar}{Colors.RESET} {value:.1%}")

    # ========================================================================
    # NEXT MORNING SUMMARY
    # ========================================================================
    print_subheader("NEXT MORNING: 7:00 AM - Recovery")
    print(f"Time: {print_time(7, 0)}")
    print(f"Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")
    print(f"Status: {Colors.GREEN}Recovered and ready for new day{Colors.RESET}")

    # Dream quality metrics after full recovery
    print(f"\nDream Quality Metrics:")
    metrics = resident.get_dream_quality_metrics()
    for key, value in metrics.items():
        bar = "=" * int(value * 10) + "-" * (10 - int(value * 10))
        print(f"  • {key.capitalize()}: {Colors.GREEN}{bar}{Colors.RESET} {value:.1%}")

    # ========================================================================
    # STATISTICS
    # ========================================================================
    print_subheader("Session Statistics")
    status = resident.get_status()
    print(f"Total Naps: {status['nap_count']}")
    print(f"Total Sleep Hours: {status['total_sleep_hours']:.1f}")
    print(f"Activities Logged: {len(resident.activity_history)}")

    if resident.nap_history:
        print(f"\nNap History:")
        for i, nap in enumerate(resident.nap_history, 1):
            print(f"  {i}. {nap.duration_hours:.1f}h sleep - "
                  f"Energy {nap.energy_before:.0f}% -> {nap.energy_after:.0f}% "
                  f"(Dream: {Colors.MAGENTA}{nap.dream_type.value if nap.dream_triggered else 'None'}{Colors.RESET})")

    # ========================================================================
    # ENERGY DYNAMICS EXPLANATION
    # ========================================================================
    print_subheader("Energy System Explanation")
    print(f"""
{Colors.BOLD}Base Mechanics:{Colors.RESET}
  • Base depletion: 2% per hour (passive exhaustion)
  • Work: -10% per session
  • Conflict: -15% per session
  • Social: -5% per session
  • Nap (1-2h): +20-30% restoration
  • Sleep (6-8h): +80-100% restoration

{Colors.BOLD}Dream Trigger Logic:{Colors.RESET}
  • Energy < 30%: Mandatory exhaustion dreams
  • 30-60% + sleep: Fragmented dreams
  • 60-80% + sleep: Normal dreams
  • > 80% + sleep: Vivid dreams
  • Minimum 1 hour rest for dream trigger

{Colors.BOLD}Dream Intensity (0.0-1.0):{Colors.RESET}
  • Clarity: Improves with energy (30-100%)
  • Vividness: Improves with energy
  • Emotional Intensity: Inverse to energy (low energy = high intensity)
  • Complexity: Moderate (0.3-0.7 range)
    """)

    print(f"\n{Colors.BOLD}{Colors.GREEN}Demo Complete!{Colors.RESET}")


# ============================================================================
# Demo: Stress Spiral
# ============================================================================


def demo_stress_spiral():
    """Demonstrate what happens under continuous stress."""
    print_header("DEMO 2: Stress Spiral and Recovery")

    resident = create_nap_mechanic("Bob")
    print(f"Resident: {Colors.BOLD}Bob{Colors.RESET}")
    print(f"Initial Energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%\n")

    print_subheader("Day 1: Multiple Conflicts")
    for i in range(3):
        print(f"\n  Conflict {i+1}:")
        print(f"    Before: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")
        resident.deplete_energy(ActivityType.CONFLICT, multiplier=1.5)
        print(f"    After:  {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")

    resident.update_energy(hours_elapsed=4.0)
    print(f"\n  Evening energy: {energy_bar(resident.energy_state.current_energy)} {resident.energy_state.current_energy:.1f}%")
    print(f"  Status: {Colors.RED}{resident.energy_state.needs_sleep and 'CRITICAL' or 'LOW'}{Colors.RESET}")

    print_subheader("Recovery: 10-Hour Sleep")
    energy_before = resident.energy_state.current_energy
    state, sleep_event = resident.restore_energy(sleep_hours=10.0, quality_modifier=1.2)
    print(f"\nEnergy: {energy_bar(energy_before)} {energy_before:.1f}% -> {energy_bar(state.current_energy)} {state.current_energy:.1f}%")
    print(f"Dream Type: {Colors.MAGENTA}{sleep_event.dream_type.value.upper()}{Colors.RESET}")
    print(f"Dream Intensity: {sleep_event.dream_intensity:.1%}")
    print(f"Status: {Colors.GREEN}Fully Recovered{Colors.RESET}")


# ============================================================================
# Demo: Energy Thresholds
# ============================================================================


def demo_energy_thresholds():
    """Show how different energy levels affect dreams and behavior."""
    print_header("DEMO 3: Energy Thresholds and Dream Characteristics")

    thresholds = [
        (95.0, "Excellent Energy"),
        (75.0, "Good Energy"),
        (50.0, "Medium Energy"),
        (35.0, "Low Energy"),
        (15.0, "Critical Energy"),
    ]

    for energy, description in thresholds:
        print_subheader(f"Energy Level: {description} ({energy:.0f}%)")

        resident = create_nap_mechanic(f"test_{energy}")
        resident.energy_state.current_energy = energy

        # Show metrics
        metrics = resident.get_dream_quality_metrics()
        print(f"Quality Metrics:")
        for key, value in metrics.items():
            bar = "=" * int(value * 15) + "-" * (15 - int(value * 15))
            print(f"  • {key.capitalize()}: {bar} {value:.1%}")

        # Simulate sleep and dream
        state, event = resident.restore_energy(sleep_hours=8.0)
        print(f"\nAfter 8-hour sleep:")
        print(f"  Energy After: {energy_bar(state.current_energy)} {state.current_energy:.1f}%")
        print(f"  Dream Type: {Colors.MAGENTA}{event.dream_type.value.upper()}{Colors.RESET}")
        print(f"  Dream Intensity: {event.dream_intensity:.1%}")
        print()


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all demos."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print(r"   _   __           __  __           __            ")
    print(r"  / | / /__ ___  __/ /_/ /___  ____ _/ / ____  _____")
    print(r" /  |/ / _ `/ / / / __/ / __ \/ __ `/ / / __ \/ ___/")
    print(r"/ /|  / /_/ / /_/ / /_/ / /_/ / /_/ / / / /_/ / /    ")
    print(r"/_/ |_/\__,_/\__,_/\__/_/\____/\__,_/_/_/\____/_/     ")
    print(f"{Colors.RESET}\n")

    print(f"Welcome to NeuroHood's Nap Mechanic Demo!")
    print(f"This demonstrates the energy system and dream triggers.\n")

    # Run all demos
    demo_day_cycle()
    print("\n" * 2)
    demo_stress_spiral()
    print("\n" * 2)
    demo_energy_thresholds()

    print_header("Demo Complete!")
    print(f"\n{Colors.GREEN}All demonstrations finished successfully!{Colors.RESET}\n")


if __name__ == "__main__":
    main()
