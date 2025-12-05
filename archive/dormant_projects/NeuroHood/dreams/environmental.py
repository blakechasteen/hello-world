"""
NeuroHood v8.6: Environmental System (Life-Stage Aware Weather)

Life-stage aware weather and environment system where the same weather affects
different age groups differently - kids love snow days, elders struggle with ice.

Key Features:
- 12 Weather Types with Markov chain transitions
- 8 Special Weather Events
- Life-stage specific reactions (kids vs parents vs elders)
- Seasonal Affective Disorder (SAD) modeling
- Personality-weather interactions
- 30+ weather moodlets

Author: Claude
Date: November 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import random
import math

# Import life stages for type hints
try:
    from life_stages import LifeStage
except ImportError:
    # Define locally if not available
    class LifeStage(Enum):
        CHILDHOOD = "childhood"
        ADOLESCENCE = "adolescence"
        EMERGING_ADULT = "emerging_adult"
        YOUNG_ADULT = "young_adult"
        ESTABLISHED = "established"
        MIDLIFE = "midlife"
        ACTIVE_ELDER = "active_elder"
        LATE_ELDER = "late_elder"


# =============================================================================
# ENUMS
# =============================================================================

class Weather(Enum):
    """12 Weather Types."""
    SUNNY = "sunny"
    PARTLY_CLOUDY = "partly_cloudy"
    OVERCAST = "overcast"
    FOGGY = "foggy"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    LIGHT_SNOW = "light_snow"
    HEAVY_SNOW = "heavy_snow"
    BLIZZARD = "blizzard"
    HOT_HUMID = "hot_humid"
    COLD_DRY = "cold_dry"


class SpecialWeatherEvent(Enum):
    """8 Special Weather Events."""
    SNOW_DAY = "snow_day"              # School/work closure
    HEAT_WAVE = "heat_wave"            # 3+ days hot
    COLD_SNAP = "cold_snap"            # Sudden cold
    PERFECT_DAY = "perfect_day"        # Optimal conditions
    FIRST_SNOW = "first_snow"          # Season's first
    SPRING_ARRIVAL = "spring_arrival"  # First warm day
    FALL_PEAK = "fall_peak"            # Autumn colors
    RAINBOW = "rainbow"                # After storm


class Season(Enum):
    """Four seasons."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class WeatherReaction:
    """Age-specific reaction to weather."""
    life_stage: LifeStage
    weather: Weather
    mood_modifier: float        # -1 to +1
    energy_modifier: float      # 0.5 to 1.5
    activity_preference: str    # "indoor", "outdoor", "either"
    moodlet: Optional[str] = None


@dataclass
class WeatherState:
    """Current weather state."""
    current: Weather = Weather.SUNNY
    temperature: float = 70.0  # Fahrenheit
    humidity: float = 0.5
    consecutive_days: int = 1  # Days of same weather type
    season: Season = Season.SPRING
    day_of_year: int = 100

    # Season tracking
    has_snowed_this_season: bool = False
    has_had_warm_spring_day: bool = False
    days_since_sun: int = 0
    hot_streak: int = 0
    cold_streak: int = 0


@dataclass
class AgentWeatherState:
    """Per-agent weather tracking."""
    agent_id: str
    days_of_winter: int = 0
    sad_level: float = 0.0
    indoor_days_streak: int = 0
    outdoor_preference: float = 0.5  # 0=prefers indoor, 1=outdoor


# =============================================================================
# WEATHER TRANSITIONS (Markov Chain)
# =============================================================================

# Transition probabilities by season (current -> next)
WEATHER_TRANSITIONS: Dict[Season, Dict[Weather, Dict[Weather, float]]] = {
    Season.SPRING: {
        Weather.SUNNY: {
            Weather.SUNNY: 0.4, Weather.PARTLY_CLOUDY: 0.3,
            Weather.LIGHT_RAIN: 0.2, Weather.OVERCAST: 0.1
        },
        Weather.PARTLY_CLOUDY: {
            Weather.SUNNY: 0.3, Weather.PARTLY_CLOUDY: 0.3,
            Weather.LIGHT_RAIN: 0.2, Weather.OVERCAST: 0.2
        },
        Weather.LIGHT_RAIN: {
            Weather.SUNNY: 0.2, Weather.PARTLY_CLOUDY: 0.3,
            Weather.LIGHT_RAIN: 0.2, Weather.HEAVY_RAIN: 0.2,
            Weather.THUNDERSTORM: 0.1
        },
        Weather.HEAVY_RAIN: {
            Weather.LIGHT_RAIN: 0.4, Weather.PARTLY_CLOUDY: 0.3,
            Weather.THUNDERSTORM: 0.2, Weather.SUNNY: 0.1
        },
    },
    Season.SUMMER: {
        Weather.SUNNY: {
            Weather.SUNNY: 0.5, Weather.HOT_HUMID: 0.25,
            Weather.PARTLY_CLOUDY: 0.15, Weather.THUNDERSTORM: 0.1
        },
        Weather.HOT_HUMID: {
            Weather.HOT_HUMID: 0.4, Weather.SUNNY: 0.25,
            Weather.THUNDERSTORM: 0.25, Weather.PARTLY_CLOUDY: 0.1
        },
        Weather.THUNDERSTORM: {
            Weather.SUNNY: 0.4, Weather.PARTLY_CLOUDY: 0.3,
            Weather.HOT_HUMID: 0.2, Weather.THUNDERSTORM: 0.1
        },
    },
    Season.FALL: {
        Weather.SUNNY: {
            Weather.SUNNY: 0.3, Weather.PARTLY_CLOUDY: 0.35,
            Weather.OVERCAST: 0.2, Weather.LIGHT_RAIN: 0.15
        },
        Weather.OVERCAST: {
            Weather.OVERCAST: 0.3, Weather.PARTLY_CLOUDY: 0.25,
            Weather.LIGHT_RAIN: 0.25, Weather.FOGGY: 0.2
        },
        Weather.FOGGY: {
            Weather.OVERCAST: 0.3, Weather.FOGGY: 0.25,
            Weather.PARTLY_CLOUDY: 0.25, Weather.SUNNY: 0.2
        },
    },
    Season.WINTER: {
        Weather.COLD_DRY: {
            Weather.COLD_DRY: 0.3, Weather.OVERCAST: 0.25,
            Weather.LIGHT_SNOW: 0.25, Weather.SUNNY: 0.2
        },
        Weather.LIGHT_SNOW: {
            Weather.LIGHT_SNOW: 0.25, Weather.HEAVY_SNOW: 0.25,
            Weather.COLD_DRY: 0.25, Weather.OVERCAST: 0.25
        },
        Weather.HEAVY_SNOW: {
            Weather.HEAVY_SNOW: 0.2, Weather.LIGHT_SNOW: 0.3,
            Weather.BLIZZARD: 0.1, Weather.OVERCAST: 0.4
        },
        Weather.BLIZZARD: {
            Weather.HEAVY_SNOW: 0.5, Weather.OVERCAST: 0.3,
            Weather.COLD_DRY: 0.2
        },
    },
}


# =============================================================================
# LIFE-STAGE WEATHER REACTIONS
# =============================================================================

WEATHER_REACTIONS: Dict[Tuple[LifeStage, Weather], WeatherReaction] = {
    # === SNOW DAY REACTIONS ===
    (LifeStage.CHILDHOOD, Weather.HEAVY_SNOW): WeatherReaction(
        LifeStage.CHILDHOOD, Weather.HEAVY_SNOW,
        mood_modifier=0.8, energy_modifier=1.3,
        activity_preference="outdoor", moodlet="snow_day_joy"
    ),
    (LifeStage.ADOLESCENCE, Weather.HEAVY_SNOW): WeatherReaction(
        LifeStage.ADOLESCENCE, Weather.HEAVY_SNOW,
        mood_modifier=0.6, energy_modifier=1.1,
        activity_preference="either", moodlet="snow_day_freedom"
    ),
    (LifeStage.YOUNG_ADULT, Weather.HEAVY_SNOW): WeatherReaction(
        LifeStage.YOUNG_ADULT, Weather.HEAVY_SNOW,
        mood_modifier=0.2, energy_modifier=1.0,
        activity_preference="indoor", moodlet="snow_day_cozy"
    ),
    (LifeStage.ESTABLISHED, Weather.HEAVY_SNOW): WeatherReaction(
        LifeStage.ESTABLISHED, Weather.HEAVY_SNOW,
        mood_modifier=-0.2, energy_modifier=0.9,
        activity_preference="indoor", moodlet="snow_day_stress"
    ),
    (LifeStage.ACTIVE_ELDER, Weather.HEAVY_SNOW): WeatherReaction(
        LifeStage.ACTIVE_ELDER, Weather.HEAVY_SNOW,
        mood_modifier=-0.3, energy_modifier=0.7,
        activity_preference="indoor", moodlet="snow_day_caution"
    ),
    (LifeStage.LATE_ELDER, Weather.HEAVY_SNOW): WeatherReaction(
        LifeStage.LATE_ELDER, Weather.HEAVY_SNOW,
        mood_modifier=-0.4, energy_modifier=0.6,
        activity_preference="indoor", moodlet="winter_isolation"
    ),

    # === HOT SUMMER REACTIONS ===
    (LifeStage.CHILDHOOD, Weather.HOT_HUMID): WeatherReaction(
        LifeStage.CHILDHOOD, Weather.HOT_HUMID,
        mood_modifier=0.3, energy_modifier=1.2,
        activity_preference="outdoor", moodlet="summer_fun"
    ),
    (LifeStage.ADOLESCENCE, Weather.HOT_HUMID): WeatherReaction(
        LifeStage.ADOLESCENCE, Weather.HOT_HUMID,
        mood_modifier=0.4, energy_modifier=1.1,
        activity_preference="outdoor", moodlet="summer_vibes"
    ),
    (LifeStage.YOUNG_ADULT, Weather.HOT_HUMID): WeatherReaction(
        LifeStage.YOUNG_ADULT, Weather.HOT_HUMID,
        mood_modifier=-0.1, energy_modifier=0.8,
        activity_preference="indoor", moodlet="summer_heat"
    ),
    (LifeStage.ESTABLISHED, Weather.HOT_HUMID): WeatherReaction(
        LifeStage.ESTABLISHED, Weather.HOT_HUMID,
        mood_modifier=-0.2, energy_modifier=0.75,
        activity_preference="indoor", moodlet="heat_fatigue"
    ),
    (LifeStage.ACTIVE_ELDER, Weather.HOT_HUMID): WeatherReaction(
        LifeStage.ACTIVE_ELDER, Weather.HOT_HUMID,
        mood_modifier=-0.4, energy_modifier=0.6,
        activity_preference="indoor", moodlet="heat_caution"
    ),
    (LifeStage.LATE_ELDER, Weather.HOT_HUMID): WeatherReaction(
        LifeStage.LATE_ELDER, Weather.HOT_HUMID,
        mood_modifier=-0.5, energy_modifier=0.5,
        activity_preference="indoor", moodlet="heat_danger"
    ),

    # === SUNNY DAY REACTIONS ===
    (LifeStage.CHILDHOOD, Weather.SUNNY): WeatherReaction(
        LifeStage.CHILDHOOD, Weather.SUNNY,
        mood_modifier=0.4, energy_modifier=1.2,
        activity_preference="outdoor", moodlet="sunny_playtime"
    ),
    (LifeStage.YOUNG_ADULT, Weather.SUNNY): WeatherReaction(
        LifeStage.YOUNG_ADULT, Weather.SUNNY,
        mood_modifier=0.3, energy_modifier=1.1,
        activity_preference="outdoor", moodlet="beautiful_day"
    ),
    (LifeStage.ACTIVE_ELDER, Weather.SUNNY): WeatherReaction(
        LifeStage.ACTIVE_ELDER, Weather.SUNNY,
        mood_modifier=0.4, energy_modifier=1.0,
        activity_preference="outdoor", moodlet="lovely_weather"
    ),

    # === RAINY DAY REACTIONS ===
    (LifeStage.CHILDHOOD, Weather.LIGHT_RAIN): WeatherReaction(
        LifeStage.CHILDHOOD, Weather.LIGHT_RAIN,
        mood_modifier=-0.1, energy_modifier=0.9,
        activity_preference="indoor", moodlet="rainy_boredom"
    ),
    (LifeStage.YOUNG_ADULT, Weather.LIGHT_RAIN): WeatherReaction(
        LifeStage.YOUNG_ADULT, Weather.LIGHT_RAIN,
        mood_modifier=0.1, energy_modifier=0.95,
        activity_preference="indoor", moodlet="cozy_rain"
    ),

    # === THUNDERSTORM REACTIONS ===
    (LifeStage.CHILDHOOD, Weather.THUNDERSTORM): WeatherReaction(
        LifeStage.CHILDHOOD, Weather.THUNDERSTORM,
        mood_modifier=-0.3, energy_modifier=0.8,
        activity_preference="indoor", moodlet="storm_scary"
    ),
    (LifeStage.ADOLESCENCE, Weather.THUNDERSTORM): WeatherReaction(
        LifeStage.ADOLESCENCE, Weather.THUNDERSTORM,
        mood_modifier=0.1, energy_modifier=0.95,
        activity_preference="indoor", moodlet="storm_drama"
    ),
    (LifeStage.YOUNG_ADULT, Weather.THUNDERSTORM): WeatherReaction(
        LifeStage.YOUNG_ADULT, Weather.THUNDERSTORM,
        mood_modifier=0.0, energy_modifier=0.9,
        activity_preference="indoor", moodlet="storm_watching"
    ),
}


# =============================================================================
# WEATHER MOODLETS (30+)
# =============================================================================

WEATHER_MOODLETS: Dict[str, Dict[str, Any]] = {
    # Snow-related
    "snow_day_joy": {
        "name": "Snow Day Joy",
        "mood_impact": 25,
        "duration_hours": 24,
        "description": "No school! Time to play in the snow!"
    },
    "snow_day_freedom": {
        "name": "Snow Day Freedom",
        "mood_impact": 18,
        "duration_hours": 24,
        "description": "Unexpected day off!"
    },
    "snow_day_cozy": {
        "name": "Cozy Snow Day",
        "mood_impact": 10,
        "duration_hours": 24,
        "description": "Working from home by the window"
    },
    "snow_day_stress": {
        "name": "Snow Day Stress",
        "mood_impact": -12,
        "duration_hours": 24,
        "description": "Kids are home, schedule disrupted"
    },
    "snow_day_caution": {
        "name": "Winter Caution",
        "mood_impact": -15,
        "duration_hours": 24,
        "description": "Worried about ice and falls"
    },
    "winter_isolation": {
        "name": "Winter Isolation",
        "mood_impact": -20,
        "duration_hours": 48,
        "description": "Stuck inside, can't go anywhere"
    },
    "first_snow_magic": {
        "name": "First Snow Magic",
        "mood_impact": 20,
        "duration_hours": 24,
        "description": "The first snow of the season!"
    },

    # Summer-related
    "summer_fun": {
        "name": "Summer Fun",
        "mood_impact": 15,
        "duration_hours": 12,
        "description": "Perfect day for swimming!"
    },
    "summer_vibes": {
        "name": "Summer Vibes",
        "mood_impact": 18,
        "duration_hours": 12,
        "description": "Beach weather, good times"
    },
    "summer_heat": {
        "name": "Summer Heat",
        "mood_impact": -8,
        "duration_hours": 12,
        "description": "Too hot to enjoy anything"
    },
    "heat_fatigue": {
        "name": "Heat Fatigue",
        "mood_impact": -12,
        "duration_hours": 12,
        "description": "Drained by the heat"
    },
    "heat_caution": {
        "name": "Heat Caution",
        "mood_impact": -18,
        "duration_hours": 24,
        "description": "Dangerous heat, staying inside"
    },
    "heat_danger": {
        "name": "Heat Advisory",
        "mood_impact": -25,
        "duration_hours": 24,
        "description": "Health risk from extreme heat"
    },

    # Spring-related
    "spring_liberation": {
        "name": "Spring Liberation",
        "mood_impact": 25,
        "duration_hours": 72,
        "description": "Winter is finally over!"
    },
    "spring_energy": {
        "name": "Spring Energy",
        "mood_impact": 20,
        "duration_hours": 48,
        "description": "Renewed vitality with spring"
    },
    "spring_revival": {
        "name": "Spring Revival",
        "mood_impact": 18,
        "duration_hours": 48,
        "description": "Ready to get outside again"
    },
    "spring_gratitude": {
        "name": "Spring Gratitude",
        "mood_impact": 22,
        "duration_hours": 72,
        "description": "Grateful to have survived another winter"
    },

    # Fall-related
    "fall_beauty": {
        "name": "Fall Beauty",
        "mood_impact": 15,
        "duration_hours": 24,
        "description": "The leaves are gorgeous"
    },
    "fall_melancholy": {
        "name": "Fall Melancholy",
        "mood_impact": -8,
        "duration_hours": 24,
        "description": "Wistful as days grow shorter"
    },

    # General weather
    "beautiful_day": {
        "name": "Beautiful Day",
        "mood_impact": 15,
        "duration_hours": 12,
        "description": "Perfect weather"
    },
    "lovely_weather": {
        "name": "Lovely Weather",
        "mood_impact": 15,
        "duration_hours": 12,
        "description": "Wonderful day to be outside"
    },
    "sunny_playtime": {
        "name": "Sunny Playtime",
        "mood_impact": 15,
        "duration_hours": 12,
        "description": "Great day to play outside"
    },
    "cozy_rain": {
        "name": "Cozy Rainy Day",
        "mood_impact": 8,
        "duration_hours": 12,
        "description": "Perfect day to stay in"
    },
    "rainy_boredom": {
        "name": "Rainy Day Boredom",
        "mood_impact": -8,
        "duration_hours": 12,
        "description": "Can't play outside today"
    },
    "storm_scary": {
        "name": "Scary Storm",
        "mood_impact": -15,
        "duration_hours": 6,
        "description": "Thunder is frightening"
    },
    "storm_drama": {
        "name": "Storm Drama",
        "mood_impact": 5,
        "duration_hours": 6,
        "description": "Exciting weather"
    },
    "storm_watching": {
        "name": "Storm Watching",
        "mood_impact": 3,
        "duration_hours": 6,
        "description": "Watching the storm roll in"
    },
    "perfect_weather": {
        "name": "Perfect Weather",
        "mood_impact": 22,
        "duration_hours": 24,
        "description": "Couldn't ask for better weather"
    },
    "rainbow_wonder": {
        "name": "Rainbow Wonder",
        "mood_impact": 15,
        "duration_hours": 4,
        "description": "A beautiful rainbow appeared!"
    },

    # SAD and cabin fever
    "winter_blues": {
        "name": "Winter Blues",
        "mood_impact": -15,
        "duration_hours": 168,  # Week-long
        "description": "The darkness is getting to me"
    },
    "cabin_fever": {
        "name": "Cabin Fever",
        "mood_impact": -18,
        "duration_hours": 24,
        "description": "Desperate to get outside"
    },
    "vitamin_d_deficiency": {
        "name": "Missing the Sun",
        "mood_impact": -12,
        "duration_hours": 168,
        "description": "Haven't seen sun in days"
    },
}


# =============================================================================
# ENVIRONMENTAL MANAGER
# =============================================================================

class EnvironmentalManager:
    """Manages weather and environmental effects on agents."""

    def __init__(
        self,
        life_stage_manager: Optional[Any] = None,
        moodlet_manager: Optional[Any] = None,
        seed: int = 42
    ):
        self.life_stage_manager = life_stage_manager
        self.moodlet_manager = moodlet_manager
        self.random = random.Random(seed)

        self.weather_state = WeatherState()
        self.agent_states: Dict[str, AgentWeatherState] = {}
        self.special_events_today: List[SpecialWeatherEvent] = []

        self.statistics = {
            "days_simulated": 0,
            "special_events": 0,
            "snow_days": 0,
        }

    def initialize_agent(self, agent_id: str) -> AgentWeatherState:
        """Initialize weather state for an agent."""
        state = AgentWeatherState(agent_id=agent_id)
        self.agent_states[agent_id] = state
        return state

    def advance_day(self) -> Dict[str, Any]:
        """Advance weather by one day."""
        self.statistics["days_simulated"] += 1

        # Update day of year and season
        self.weather_state.day_of_year = (self.weather_state.day_of_year + 1) % 365
        self._update_season()

        # Transition weather
        old_weather = self.weather_state.current
        self._transition_weather()

        # Track consecutive days
        if self.weather_state.current == old_weather:
            self.weather_state.consecutive_days += 1
        else:
            self.weather_state.consecutive_days = 1

        # Track sun/streaks
        if self.weather_state.current in [Weather.SUNNY, Weather.PARTLY_CLOUDY]:
            self.weather_state.days_since_sun = 0
        else:
            self.weather_state.days_since_sun += 1

        if self.weather_state.current == Weather.HOT_HUMID:
            self.weather_state.hot_streak += 1
        else:
            self.weather_state.hot_streak = 0

        if self.weather_state.current in [Weather.COLD_DRY, Weather.LIGHT_SNOW,
                                           Weather.HEAVY_SNOW, Weather.BLIZZARD]:
            self.weather_state.cold_streak += 1
        else:
            self.weather_state.cold_streak = 0

        # Check for special events
        self.special_events_today = []
        self._check_special_events()

        return {
            "weather": self.weather_state.current.value,
            "season": self.weather_state.season.value,
            "special_events": [e.value for e in self.special_events_today],
            "temperature": self.weather_state.temperature,
        }

    def _update_season(self):
        """Update season based on day of year."""
        day = self.weather_state.day_of_year

        if 79 <= day < 172:  # ~Mar 20 - Jun 20
            new_season = Season.SPRING
        elif 172 <= day < 266:  # ~Jun 21 - Sep 22
            new_season = Season.SUMMER
        elif 266 <= day < 355:  # ~Sep 23 - Dec 20
            new_season = Season.FALL
        else:  # ~Dec 21 - Mar 19
            new_season = Season.WINTER

        if new_season != self.weather_state.season:
            # Reset seasonal tracking
            if new_season == Season.WINTER:
                self.weather_state.has_snowed_this_season = False
            elif new_season == Season.SPRING:
                self.weather_state.has_had_warm_spring_day = False

            self.weather_state.season = new_season

    def _transition_weather(self):
        """Transition to next weather using Markov chain."""
        current = self.weather_state.current
        season = self.weather_state.season

        # Get transition probabilities for this season
        season_transitions = WEATHER_TRANSITIONS.get(season, {})
        weather_transitions = season_transitions.get(current)

        if not weather_transitions:
            # Default: stay the same or go to common weather
            weather_transitions = {
                current: 0.5,
                Weather.PARTLY_CLOUDY: 0.3,
                Weather.SUNNY: 0.2
            }

        # Sample next weather
        choices = list(weather_transitions.keys())
        weights = list(weather_transitions.values())
        self.weather_state.current = self.random.choices(choices, weights)[0]

        # Update temperature
        self._update_temperature()

    def _update_temperature(self):
        """Update temperature based on season and weather."""
        base_temps = {
            Season.SPRING: 55,
            Season.SUMMER: 80,
            Season.FALL: 55,
            Season.WINTER: 35,
        }

        weather_mods = {
            Weather.SUNNY: 5,
            Weather.HOT_HUMID: 15,
            Weather.COLD_DRY: -10,
            Weather.BLIZZARD: -15,
            Weather.HEAVY_SNOW: -5,
        }

        base = base_temps[self.weather_state.season]
        mod = weather_mods.get(self.weather_state.current, 0)
        variance = self.random.gauss(0, 5)

        self.weather_state.temperature = base + mod + variance

    def _check_special_events(self):
        """Detect special weather events."""
        current = self.weather_state.current
        season = self.weather_state.season

        # Snow Day: Heavy snow/blizzard (assume school closure)
        if current in [Weather.HEAVY_SNOW, Weather.BLIZZARD]:
            self.special_events_today.append(SpecialWeatherEvent.SNOW_DAY)
            self.statistics["snow_days"] += 1

        # First Snow of Season
        if current in [Weather.LIGHT_SNOW, Weather.HEAVY_SNOW]:
            if not self.weather_state.has_snowed_this_season:
                self.special_events_today.append(SpecialWeatherEvent.FIRST_SNOW)
                self.weather_state.has_snowed_this_season = True

        # Heat Wave: 3+ consecutive hot days
        if self.weather_state.hot_streak >= 3:
            self.special_events_today.append(SpecialWeatherEvent.HEAT_WAVE)

        # Cold Snap: Temperature dropped significantly
        if self.weather_state.cold_streak == 1 and season != Season.WINTER:
            self.special_events_today.append(SpecialWeatherEvent.COLD_SNAP)

        # Spring Arrival: First warm day after winter
        if season == Season.SPRING and self.weather_state.temperature > 65:
            if not self.weather_state.has_had_warm_spring_day:
                self.special_events_today.append(SpecialWeatherEvent.SPRING_ARRIVAL)
                self.weather_state.has_had_warm_spring_day = True

        # Perfect Weather Day (rare)
        if (current == Weather.SUNNY and
            65 <= self.weather_state.temperature <= 75 and
            self.weather_state.humidity < 0.5):
            if self.random.random() < 0.3:  # Not every nice day
                self.special_events_today.append(SpecialWeatherEvent.PERFECT_DAY)

        # Rainbow: After thunderstorm with sun
        if current == Weather.SUNNY and self.weather_state.consecutive_days == 1:
            if self.random.random() < 0.2:
                self.special_events_today.append(SpecialWeatherEvent.RAINBOW)

        # Fall Peak: Mid-fall
        if (season == Season.FALL and
            290 <= self.weather_state.day_of_year <= 310):
            if self.random.random() < 0.1:
                self.special_events_today.append(SpecialWeatherEvent.FALL_PEAK)

        if self.special_events_today:
            self.statistics["special_events"] += len(self.special_events_today)

    def get_weather_impact(
        self,
        agent_id: str,
        life_stage: LifeStage,
        personality: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Calculate weather impact for specific agent."""
        reaction = WEATHER_REACTIONS.get(
            (life_stage, self.weather_state.current)
        )

        if not reaction:
            # Default neutral reaction
            return {
                "mood_modifier": 0.0,
                "energy_modifier": 1.0,
                "activity_preference": "either",
                "moodlet": None
            }

        # Modify by personality
        mood_mod = reaction.mood_modifier
        energy_mod = reaction.energy_modifier

        if personality:
            extraversion = personality.get("extraversion", 0.5)
            neuroticism = personality.get("neuroticism", 0.5)

            # Extroverts hate being stuck inside
            if reaction.activity_preference == "indoor" and extraversion > 0.7:
                mood_mod -= 0.1

            # Introverts enjoy indoor weather
            if reaction.activity_preference == "indoor" and extraversion < 0.3:
                mood_mod += 0.1

            # High neuroticism amplifies negative weather effects
            if mood_mod < 0 and neuroticism > 0.6:
                mood_mod *= 1.2

        return {
            "mood_modifier": mood_mod,
            "energy_modifier": energy_mod,
            "activity_preference": reaction.activity_preference,
            "moodlet": reaction.moodlet
        }

    def calculate_sad_impact(self, agent_id: str) -> float:
        """Calculate Seasonal Affective Disorder impact."""
        if agent_id not in self.agent_states:
            return 0.0

        state = self.agent_states[agent_id]

        # Track winter days
        if self.weather_state.season == Season.WINTER:
            state.days_of_winter += 1
        else:
            state.days_of_winter = max(0, state.days_of_winter - 2)  # Recover

        days = state.days_of_winter

        # SAD builds over time
        if days < 30:
            base_sad = 0.0
        elif days < 60:
            base_sad = 0.3
        elif days < 90:
            base_sad = 0.6
        else:
            base_sad = 1.0

        # Factor in days without sun
        sun_factor = min(self.weather_state.days_since_sun / 10, 1.0)

        state.sad_level = base_sad * 0.7 + sun_factor * 0.3
        return state.sad_level

    def apply_moodlets(self, agent_id: str, life_stage: LifeStage):
        """Apply weather-related moodlets to agent."""
        if not self.moodlet_manager:
            return

        # Get base weather moodlet
        reaction = WEATHER_REACTIONS.get(
            (life_stage, self.weather_state.current)
        )
        if reaction and reaction.moodlet:
            self.moodlet_manager.add_moodlet(
                agent_id, reaction.moodlet, "weather"
            )

        # Special event moodlets
        for event in self.special_events_today:
            if event == SpecialWeatherEvent.FIRST_SNOW:
                self.moodlet_manager.add_moodlet(
                    agent_id, "first_snow_magic", "weather"
                )
            elif event == SpecialWeatherEvent.SPRING_ARRIVAL:
                if life_stage in [LifeStage.ACTIVE_ELDER, LifeStage.LATE_ELDER]:
                    self.moodlet_manager.add_moodlet(
                        agent_id, "spring_gratitude", "weather"
                    )
                else:
                    self.moodlet_manager.add_moodlet(
                        agent_id, "spring_liberation", "weather"
                    )
            elif event == SpecialWeatherEvent.RAINBOW:
                if life_stage == LifeStage.CHILDHOOD:
                    self.moodlet_manager.add_moodlet(
                        agent_id, "rainbow_wonder", "weather"
                    )
            elif event == SpecialWeatherEvent.PERFECT_DAY:
                self.moodlet_manager.add_moodlet(
                    agent_id, "perfect_weather", "weather"
                )

        # SAD moodlet
        sad_level = self.calculate_sad_impact(agent_id)
        if sad_level > 0.5:
            self.moodlet_manager.add_moodlet(
                agent_id, "winter_blues", "weather"
            )

    def format_weather_panel(self) -> str:
        """Format weather info for display."""
        lines = [
            f"══════ WEATHER ══════",
            f"Condition: {self.weather_state.current.value.replace('_', ' ').title()}",
            f"Temperature: {self.weather_state.temperature:.0f}°F",
            f"Season: {self.weather_state.season.value.title()}",
            f"Day: {self.weather_state.day_of_year}/365",
        ]

        if self.special_events_today:
            events = ", ".join(e.value.replace("_", " ").title()
                              for e in self.special_events_today)
            lines.append(f"Special: {events}")

        if self.weather_state.days_since_sun > 3:
            lines.append(f"Days without sun: {self.weather_state.days_since_sun}")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "current_weather": self.weather_state.current.value,
            "current_season": self.weather_state.season.value,
            "temperature": self.weather_state.temperature,
            **self.statistics
        }


# =============================================================================
# DEMO
# =============================================================================

def demo_environmental():
    """Demonstrate environmental system."""
    print("=" * 60)
    print("NEUROHOOD v8.6: ENVIRONMENTAL SYSTEM DEMO")
    print("=" * 60)

    manager = EnvironmentalManager()

    # Initialize some agents
    agents = ["child_001", "teen_001", "adult_001", "elder_001"]
    for agent_id in agents:
        manager.initialize_agent(agent_id)

    print("\n1. SIMULATING 30 DAYS OF WEATHER")
    print("-" * 50)

    for day in range(30):
        result = manager.advance_day()

        if result["special_events"]:
            print(f"Day {day + 1}: {result['weather']} ({result['temperature']:.0f}°F) "
                  f"- SPECIAL: {result['special_events']}")
        elif day % 7 == 0:  # Weekly summary
            print(f"Day {day + 1}: {result['weather']} ({result['temperature']:.0f}°F)")

    print("\n2. LIFE-STAGE WEATHER REACTIONS")
    print("-" * 50)

    # Set weather to heavy snow for testing
    manager.weather_state.current = Weather.HEAVY_SNOW

    life_stages = [
        (LifeStage.CHILDHOOD, "Child"),
        (LifeStage.ESTABLISHED, "Parent"),
        (LifeStage.ACTIVE_ELDER, "Elder"),
    ]

    for stage, label in life_stages:
        impact = manager.get_weather_impact("test", stage)
        print(f"{label} (Snow Day):")
        print(f"  Mood: {impact['mood_modifier']:+.0%}")
        print(f"  Energy: {impact['energy_modifier']:.0%}")
        print(f"  Moodlet: {impact['moodlet']}")

    print("\n3. SEASONAL AFFECTIVE DISORDER")
    print("-" * 50)

    # Simulate winter for SAD
    manager.weather_state.season = Season.WINTER
    manager.weather_state.current = Weather.OVERCAST

    for week in range(12):  # 12 weeks of winter
        for _ in range(7):
            manager.advance_day()

        sad_level = manager.calculate_sad_impact("adult_001")
        if sad_level > 0:
            print(f"Week {week + 1}: SAD level = {sad_level:.0%}")

    print("\n4. CURRENT WEATHER PANEL")
    print("-" * 50)
    print(manager.format_weather_panel())

    print("\n5. STATISTICS")
    print("-" * 50)
    stats = manager.get_statistics()
    print(f"Days simulated: {stats['days_simulated']}")
    print(f"Special events: {stats['special_events']}")
    print(f"Snow days: {stats['snow_days']}")

    print("\n" + "=" * 60)
    print("ENVIRONMENTAL SYSTEM DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_environmental()
