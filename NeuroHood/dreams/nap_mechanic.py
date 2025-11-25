"""
Nap Mechanic: Energy System and Dream Triggers
===============================================

Version: 1.0.0
Date: 2025-11-24
Status: Production Ready

The Nap Mechanic implements NeuroHood's energy system that governs when residents
sleep, nap, and have dreams. Energy depletes through activities and stress, triggers
fragmented dreams when low, and mandates sleep when critical.

Core Philosophy:
    "Energy is a resident's biological reality - it shapes behavior and dreams."

Architecture:
    1. EnergyState: Tracks current energy (0-100%), thresholds, depletion rates
    2. NapMechanic: Manages energy updates, depletion, restoration, dream triggers
    3. NapEvent: Records naps/sleep with energy changes and dream outcomes

Energy Model:
    - Base depletion: ~2% per hour (passive exhaustion)
    - Activity costs: Work (-10%), Conflict (-15%), Social (-5%), Exercise (-12%)
    - Stress events: -20% per event
    - Recovery: Nap (1-2h): +20-30%, Sleep (6-8h): +80-100%

Dream Trigger Logic:
    - Energy < 30%: Mandatory exhaustion dreams
    - Energy 30-60%: Fragmented dreams if resting
    - Energy 60-80%: Normal dreams if resting
    - Energy > 80%: Vivid dreams during sleep

Dream Intensity (0.0-1.0):
    - High energy (80-100%): 0.8-1.0 (vivid, clear)
    - Medium energy (60-80%): 0.6-0.8 (normal)
    - Low energy (30-60%): 0.3-0.6 (fragmented)
    - Critical energy (<30%): 0.8-1.0 (intense exhaustion)

Integration Points:
    - symbolic_encoder.py: Dream intensity → symbol selection
    - dream_matching.py: Nap times → shared dream detection
    - consciousness_slider.py: Energy → quality/stability sliders
    - personality.py: Stress multipliers → activity costs
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
from enum import Enum


logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class DreamType(Enum):
    """Type of dream based on energy and rest context."""
    NAP_DREAM = "nap"               # 1-2 hour nap
    SLEEP_DREAM = "sleep"            # 6-8 hour sleep
    EXHAUSTION_DREAM = "exhaustion"   # Mandatory rest (energy < 30%)
    FRAGMENTED = "fragmented"         # Low energy dreams (30-60%)
    NORMAL = "normal"                 # Medium energy dreams (60-80%)
    VIVID = "vivid"                   # High energy dreams (80-100%)


class ActivityType(Enum):
    """Types of activities that consume energy."""
    WORK = "work"
    CONFLICT = "conflict"
    SOCIAL = "social"
    EXERCISE = "exercise"
    STRESS_EVENT = "stress_event"
    TRAVEL = "travel"
    CREATIVE = "creative"
    REST = "rest"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class EnergyState:
    """Current energy state of a resident."""

    current_energy: float           # 0.0 to 100.0
    max_energy: float               # Usually 100.0, default energy level
    depletion_rate: float            # Base energy lost per hour (default 2.0)
    last_update: datetime            # Last time energy was recalculated

    dream_quality_threshold: float   # Below this = fragmented dreams (default 60.0)
    sleep_threshold: float           # Below this = must sleep (default 30.0)

    # Status flags
    needs_sleep: bool = False        # True if energy < sleep_threshold
    is_resting: bool = False         # True if currently napping/sleeping

    def __post_init__(self):
        """Validate energy state."""
        if not (0.0 <= self.current_energy <= self.max_energy):
            logger.warning(
                f"Energy {self.current_energy} outside valid range [0, {self.max_energy}]. "
                "Clamping to valid range."
            )
            self.current_energy = max(0.0, min(self.max_energy, self.current_energy))

        if self.current_energy < self.sleep_threshold:
            self.needs_sleep = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'current_energy': round(self.current_energy, 2),
            'max_energy': self.max_energy,
            'depletion_rate': self.depletion_rate,
            'last_update': self.last_update.isoformat(),
            'dream_quality_threshold': self.dream_quality_threshold,
            'sleep_threshold': self.sleep_threshold,
            'needs_sleep': self.needs_sleep,
            'is_resting': self.is_resting,
        }


@dataclass
class NapEvent:
    """Record of a nap or sleep session."""

    resident_id: str
    start_time: datetime
    duration_hours: float            # Duration of nap/sleep
    energy_before: float             # Energy before rest
    energy_after: float              # Energy after rest

    dream_triggered: bool            # Whether dream occurred
    dream_type: DreamType            # Type of dream (if triggered)
    dream_intensity: float           # 0.0-1.0 intensity of dream

    # Optional metadata
    forced_sleep: bool = False       # True if energy < sleep_threshold
    interruptions: int = 0           # Number of times woken during rest
    quality_rating: Optional[float] = None  # User's subjective quality (0-1)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'resident_id': self.resident_id,
            'start_time': self.start_time.isoformat(),
            'duration_hours': round(self.duration_hours, 2),
            'energy_before': round(self.energy_before, 2),
            'energy_after': round(self.energy_after, 2),
            'dream_triggered': self.dream_triggered,
            'dream_type': self.dream_type.value,
            'dream_intensity': round(self.dream_intensity, 3),
            'forced_sleep': self.forced_sleep,
            'interruptions': self.interruptions,
            'quality_rating': self.quality_rating,
        }


# ============================================================================
# Activity Costs
# ============================================================================


ACTIVITY_COSTS: Dict[ActivityType, float] = {
    ActivityType.WORK: -10.0,
    ActivityType.CONFLICT: -15.0,
    ActivityType.SOCIAL: -5.0,
    ActivityType.EXERCISE: -12.0,
    ActivityType.STRESS_EVENT: -20.0,
    ActivityType.TRAVEL: -8.0,
    ActivityType.CREATIVE: -6.0,
    ActivityType.REST: 0.0,
}

BASE_HOURLY_DEPLETION = 2.0  # Energy lost per hour passively


# ============================================================================
# NapMechanic Class
# ============================================================================


class NapMechanic:
    """Manages energy system and dream triggers for a resident."""

    def __init__(
        self,
        resident_id: str,
        initial_energy: float = 100.0,
        depletion_rate: float = BASE_HOURLY_DEPLETION,
        dream_quality_threshold: float = 60.0,
        sleep_threshold: float = 30.0,
    ):
        """
        Initialize nap mechanic for a resident.

        Args:
            resident_id: Unique identifier for resident
            initial_energy: Starting energy level (0-100)
            depletion_rate: Energy lost per hour (typically 2.0)
            dream_quality_threshold: Below this = fragmented dreams
            sleep_threshold: Below this = must sleep
        """
        self.resident_id = resident_id
        self.energy_state = EnergyState(
            current_energy=initial_energy,
            max_energy=100.0,
            depletion_rate=depletion_rate,
            last_update=datetime.now(),
            dream_quality_threshold=dream_quality_threshold,
            sleep_threshold=sleep_threshold,
        )

        # History tracking
        self.nap_history: List[NapEvent] = []
        self.activity_history: List[Tuple[datetime, ActivityType, float]] = []

        logger.info(
            f"NapMechanic initialized for {resident_id} "
            f"(energy={initial_energy}, depletion_rate={depletion_rate}/hr)"
        )

    def update_energy(self, hours_elapsed: float) -> EnergyState:
        """
        Update energy based on time passage.

        Depletes energy at the configured depletion_rate per hour.
        Updates needs_sleep flag if energy falls below threshold.

        Args:
            hours_elapsed: Time elapsed since last update

        Returns:
            Updated EnergyState
        """
        if hours_elapsed < 0:
            logger.warning(f"Negative time elapsed: {hours_elapsed}. Ignoring.")
            return self.energy_state

        # Calculate depletion
        depletion = self.energy_state.depletion_rate * hours_elapsed
        self.energy_state.current_energy = max(0.0, self.energy_state.current_energy - depletion)

        # Update timestamp
        self.energy_state.last_update = datetime.now()

        # Check thresholds
        if self.energy_state.current_energy < self.energy_state.sleep_threshold:
            self.energy_state.needs_sleep = True
        else:
            self.energy_state.needs_sleep = False

        logger.debug(
            f"{self.resident_id}: Energy updated after {hours_elapsed}h: "
            f"{self.energy_state.current_energy:.1f}% "
            f"(needs_sleep={self.energy_state.needs_sleep})"
        )

        return self.energy_state

    def deplete_energy(self, activity: ActivityType, multiplier: float = 1.0) -> EnergyState:
        """
        Manually deplete energy for a specific activity.

        Applies activity-specific costs, modified by multiplier (stress, personality, etc.).

        Args:
            activity: Type of activity
            multiplier: Stress/personality modifier (1.0 = normal, >1.0 = stressed)

        Returns:
            Updated EnergyState
        """
        base_cost = ACTIVITY_COSTS[activity]
        actual_cost = abs(base_cost * multiplier)  # Make cost positive for clarity

        self.energy_state.current_energy = max(0.0, self.energy_state.current_energy - actual_cost)
        self.energy_state.last_update = datetime.now()

        # Track activity
        self.activity_history.append((datetime.now(), activity, actual_cost))

        # Update needs_sleep flag
        if self.energy_state.current_energy < self.energy_state.sleep_threshold:
            self.energy_state.needs_sleep = True

        logger.debug(
            f"{self.resident_id}: {activity.value} depleted {actual_cost:.1f}% energy "
            f"(new: {self.energy_state.current_energy:.1f}%)"
        )

        return self.energy_state

    def restore_energy(
        self,
        sleep_hours: float,
        quality_modifier: float = 1.0,
    ) -> Tuple[EnergyState, NapEvent]:
        """
        Restore energy through nap/sleep.

        Nap (1-2h): Restores 20-30% energy
        Sleep (6-8h): Restores 80-100% energy

        Longer sleep = more restoration, with diminishing returns.
        Quality modifier: stress/comfort affects restoration efficiency.

        Args:
            sleep_hours: Duration of nap/sleep (0.5-8+)
            quality_modifier: Sleep quality factor (0.5 = poor, 1.0 = normal, 1.5 = excellent)

        Returns:
            Tuple of (updated_energy_state, nap_event)
        """
        if sleep_hours <= 0:
            logger.warning(f"Sleep duration {sleep_hours} is invalid. Ignoring.")
            return self.energy_state, None

        # Determine restoration amount based on duration (logarithmic curve)
        # Base restoration: 5% per hour, capped by max_energy
        base_restoration = sleep_hours * 5.0  # 5% per hour baseline

        # Add bonus for longer sleep (diminishing returns)
        if sleep_hours >= 6.0:
            bonus = min(50.0, sleep_hours * 2.0)  # Additional 2% per hour after 6h
            base_restoration += bonus

        # Apply quality modifier
        actual_restoration = base_restoration * quality_modifier

        # Clamp to max energy
        energy_before = self.energy_state.current_energy
        self.energy_state.current_energy = min(
            self.energy_state.max_energy,
            self.energy_state.current_energy + actual_restoration
        )

        # Determine dream outcome
        dream_triggered, dream_type, dream_intensity = self._calculate_dream_outcome(
            sleep_hours, energy_before, self.energy_state.current_energy
        )

        # Create nap event
        nap_event = NapEvent(
            resident_id=self.resident_id,
            start_time=datetime.now() - timedelta(hours=sleep_hours),
            duration_hours=sleep_hours,
            energy_before=energy_before,
            energy_after=self.energy_state.current_energy,
            dream_triggered=dream_triggered,
            dream_type=dream_type,
            dream_intensity=dream_intensity,
            forced_sleep=(energy_before < self.energy_state.sleep_threshold),
        )

        # Update state
        self.energy_state.last_update = datetime.now()
        # Clear needs_sleep flag if energy above threshold
        if self.energy_state.current_energy >= self.energy_state.sleep_threshold:
            self.energy_state.needs_sleep = False
        self.nap_history.append(nap_event)

        logger.info(
            f"{self.resident_id}: Sleep for {sleep_hours:.1f}h restored "
            f"{actual_restoration:.1f}% energy (now {self.energy_state.current_energy:.1f}%). "
            f"Dream: {dream_type.value} (intensity={dream_intensity:.2f})"
        )

        return self.energy_state, nap_event

    def _calculate_dream_outcome(
        self,
        sleep_hours: float,
        energy_before: float,
        energy_after: float,
    ) -> Tuple[bool, DreamType, float]:
        """
        Calculate whether dream should trigger and its intensity.

        Rules:
        - Energy < 30% + sleep: Mandatory exhaustion dream
        - Energy recovering + sleep >= 1h: Dream likely
        - Energy > 80% + sleep: Vivid dream guaranteed
        - Otherwise: Low chance

        Args:
            sleep_hours: Duration of rest period
            energy_before: Energy level before rest
            energy_after: Energy level after rest

        Returns:
            Tuple of (should_trigger, dream_type, intensity)
        """
        # Minimum rest for dream is 1 hour
        if sleep_hours < 1.0:
            return False, DreamType.NORMAL, 0.0

        # Force dream if energy critical before sleep
        if energy_before < self.energy_state.sleep_threshold:
            intensity = self._calculate_dream_intensity(energy_before)
            dream_type = DreamType.EXHAUSTION_DREAM
            return True, dream_type, intensity

        # Guaranteed dream for long sleep (use energy_before to classify, not after)
        if sleep_hours >= 6.0:
            intensity = self._calculate_dream_intensity(energy_before)
            dream_type = self._classify_dream_by_energy(energy_before)
            return True, dream_type, intensity

        # Nap (1-2h): Dream likely if recovering energy
        if 1.0 <= sleep_hours < 6.0:
            # Only trigger if energy is low or recovering
            if energy_before < self.energy_state.dream_quality_threshold or energy_after > energy_before + 10:
                intensity = self._calculate_dream_intensity(energy_before)
                dream_type = DreamType.NAP_DREAM
                return True, dream_type, intensity

        # Short rest: No dream
        return False, DreamType.NORMAL, 0.0

    def _calculate_dream_intensity(self, energy_level: float) -> float:
        """
        Calculate dream intensity (0.0-1.0) based on energy level.

        High energy: vivid, clear dreams (0.8-1.0)
        Medium energy: normal dreams (0.6-0.8)
        Low energy: fragmented dreams (0.3-0.6)
        Critical energy: intense exhaustion dreams (0.7-1.0)

        Args:
            energy_level: Current energy percentage

        Returns:
            Dream intensity 0.0-1.0
        """
        if energy_level < self.energy_state.sleep_threshold:
            # Critical: Intense, fragmented dreams
            # Map 0-30 to 0.7-1.0
            return 0.7 + (energy_level / self.energy_state.sleep_threshold) * 0.3

        elif energy_level < self.energy_state.dream_quality_threshold:
            # Low: Fragmented dreams
            # Map 30-60 to 0.3-0.6
            normalized = (energy_level - self.energy_state.sleep_threshold) / \
                        (self.energy_state.dream_quality_threshold - self.energy_state.sleep_threshold)
            return 0.3 + normalized * 0.3

        else:
            # High: Vivid dreams
            # Map 60-100 to 0.6-1.0
            normalized = (energy_level - self.energy_state.dream_quality_threshold) / \
                        (100.0 - self.energy_state.dream_quality_threshold)
            return 0.6 + normalized * 0.4

    def _classify_dream_by_energy(self, energy_level: float) -> DreamType:
        """Classify dream type based on energy level."""
        if energy_level < self.energy_state.sleep_threshold:
            return DreamType.EXHAUSTION_DREAM
        elif energy_level < self.energy_state.dream_quality_threshold:
            return DreamType.FRAGMENTED
        elif energy_level < 80.0:
            return DreamType.NORMAL
        else:
            return DreamType.VIVID

    def get_dream_quality_metrics(self) -> Dict:
        """
        Get metrics about dream quality based on current energy.

        Returns:
            Dict with clarity, vividness, emotional_intensity, complexity
        """
        energy = self.energy_state.current_energy

        # Clarity: how sharp/coherent are dream images
        clarity = max(0.0, (energy - 30.0) / 70.0)  # 30-100 → 0.0-1.0

        # Vividness: how intense and colorful
        vividness = max(0.0, (energy - 20.0) / 80.0)  # 20-100 → 0.0-1.0

        # Emotional intensity: how emotionally charged
        emotional = 1.0 - (energy / 100.0)  # Inverse: low energy = high emotional charge

        # Complexity: how many scenes/symbols in dream
        complexity = 0.3 + ((energy - 30.0) / 70.0) * 0.4  # Capped at 0.3-0.7

        return {
            'clarity': round(max(0.0, min(1.0, clarity)), 3),
            'vividness': round(max(0.0, min(1.0, vividness)), 3),
            'emotional_intensity': round(max(0.0, min(1.0, emotional)), 3),
            'complexity': round(max(0.0, min(1.0, complexity)), 3),
            'overall_quality': round(max(0.0, min(1.0, (clarity + vividness) / 2.0)), 3),
        }

    def get_status(self) -> Dict:
        """Get comprehensive status of resident's energy and sleep state."""
        return {
            'resident_id': self.resident_id,
            'energy_state': self.energy_state.to_dict(),
            'dream_quality': self.get_dream_quality_metrics(),
            'nap_count': len(self.nap_history),
            'total_sleep_hours': sum(n.duration_hours for n in self.nap_history),
            'has_unmet_dream': any(
                n.dream_triggered for n in self.nap_history[-10:]  # Last 10 naps
            ),
        }


# ============================================================================
# Utility Functions
# ============================================================================


def create_nap_mechanic(resident_id: str) -> NapMechanic:
    """Factory function to create NapMechanic with defaults."""
    return NapMechanic(
        resident_id=resident_id,
        initial_energy=100.0,
        depletion_rate=BASE_HOURLY_DEPLETION,
        dream_quality_threshold=60.0,
        sleep_threshold=30.0,
    )
