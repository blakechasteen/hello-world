"""
v8.3 Career Progression System

Dual-track career system: Traditional Employment AND Entrepreneurial Paths.
Implements 12 distinct entrepreneur venture types with realistic progression.

Author: Claude
Date: November 2025
Version: 8.3
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import random


# =============================================================================
# ENUMS
# =============================================================================

class CareerTrack(Enum):
    """Which track the agent is on."""
    EMPLOYEE = "employee"
    ENTREPRENEUR = "entrepreneur"


class VentureType(Enum):
    """12 distinct entrepreneur paths."""
    INVENTOR = "inventor"           # Patents, prototypes, licensing
    FARMER = "farmer"               # Land, crops, farm-to-table
    CONTENT_CREATOR = "content_creator"  # Audience, sponsorships
    CONSULTANT = "consultant"       # Expertise monetization
    ARBITRAGE = "arbitrage"         # Buy low, sell high
    RESTAURATEUR = "restaurateur"   # Restaurant ownership
    ARTIST = "artist"               # Gallery shows, commissions
    MUSICIAN = "musician"           # Gigs, albums, touring
    HANDYMAN = "handyman"           # Service business, referrals
    SCIENTIST = "scientist"         # Research grants, publications
    WRITER = "writer"               # Books, articles, royalties
    CRAFTSPERSON = "craftsperson"   # Artisan goods, markets


class VenturePhase(Enum):
    """Venture lifecycle phases."""
    STARTUP = "startup"             # 0-6 months, 20% income, high risk
    GROWTH = "growth"               # 6-18 months, 60% income, medium risk
    ESTABLISHED = "established"     # 18-36 months, 100% income, low risk
    SCALING = "scaling"             # 36+ months, 150% income, minimal risk
    FAILED = "failed"               # Venture has failed


class EmploymentLevel(Enum):
    """Employee career levels."""
    ENTRY = 1
    MID = 2
    SENIOR = 3
    LEAD = 4
    EXECUTIVE = 5


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EntrepreneurVenture:
    """Tracks a specific entrepreneurial venture."""
    venture_type: VentureType
    phase: VenturePhase = VenturePhase.STARTUP
    capital_invested: float = 0.0
    revenue_this_month: float = 0.0
    total_revenue: float = 0.0
    months_active: int = 0
    reputation: float = 0.0          # 0-100, affects clients/income
    failure_risk: float = 0.30       # Base risk, decreases with experience
    client_count: int = 0
    last_month_successful: bool = True
    consecutive_failures: int = 0

    def get_phase_multiplier(self) -> float:
        """Income multiplier based on phase."""
        return {
            VenturePhase.STARTUP: 0.2,
            VenturePhase.GROWTH: 0.6,
            VenturePhase.ESTABLISHED: 1.0,
            VenturePhase.SCALING: 1.5,
            VenturePhase.FAILED: 0.0,
        }[self.phase]

    def get_failure_risk(self) -> float:
        """Get current failure risk based on phase and history."""
        base_risks = {
            VenturePhase.STARTUP: 0.30,
            VenturePhase.GROWTH: 0.15,
            VenturePhase.ESTABLISHED: 0.05,
            VenturePhase.SCALING: 0.02,
            VenturePhase.FAILED: 0.0,
        }
        risk = base_risks[self.phase]

        # Reputation reduces risk
        risk *= (1 - self.reputation / 200)  # Max 50% reduction

        # Consecutive failures increase risk
        risk += self.consecutive_failures * 0.05

        return min(0.5, max(0.01, risk))


@dataclass
class EmployeeState:
    """Tracks employee career progression."""
    level: EmploymentLevel = EmploymentLevel.ENTRY
    industry: str = "general"
    job_title: str = "Worker"
    performance_score: float = 50.0    # 0-100
    satisfaction: float = 70.0         # 0-100
    years_in_role: int = 0
    years_at_company: int = 0
    times_promoted: int = 0
    times_passed_over: int = 0
    salary: float = 40000.0

    def get_promotion_probability(self) -> float:
        """Calculate promotion probability."""
        # Base probability by level (harder to promote higher)
        base_probs = {
            EmploymentLevel.ENTRY: 0.15,
            EmploymentLevel.MID: 0.10,
            EmploymentLevel.SENIOR: 0.07,
            EmploymentLevel.LEAD: 0.03,
            EmploymentLevel.EXECUTIVE: 0.0,  # Can't promote higher
        }
        prob = base_probs[self.level]

        # Performance bonus
        if self.performance_score > 80:
            prob *= 1.5
        elif self.performance_score > 60:
            prob *= 1.2
        elif self.performance_score < 40:
            prob *= 0.5

        # Years in role bonus (up to 3 years)
        prob *= (1 + min(self.years_in_role, 3) * 0.1)

        return min(0.3, prob)


@dataclass
class CareerState:
    """Complete career state for an agent."""
    track: CareerTrack = CareerTrack.EMPLOYEE
    occupation: str = "worker"

    # Employee track
    employee: Optional[EmployeeState] = None

    # Entrepreneur track
    venture: Optional[EntrepreneurVenture] = None
    past_ventures: List[Tuple[VentureType, str]] = field(default_factory=list)
    # List of (VentureType, outcome: "success"/"failed")

    # Common fields
    skills: Dict[str, float] = field(default_factory=dict)
    total_career_earnings: float = 0.0
    career_start_day: int = 0
    pivot_cooldown_days: int = 0      # Must wait 30 days between pivots

    def is_entrepreneur(self) -> bool:
        return self.track == CareerTrack.ENTREPRENEUR

    def get_current_income(self) -> float:
        """Get current monthly income."""
        if self.track == CareerTrack.EMPLOYEE:
            return self.employee.salary / 12 if self.employee else 0
        elif self.track == CareerTrack.ENTREPRENEUR:
            return self.venture.revenue_this_month if self.venture else 0
        return 0


# =============================================================================
# VENTURE PROFILES
# =============================================================================

@dataclass
class VentureProfile:
    """Configuration for a venture type."""
    startup_capital: float
    base_monthly_income: float
    risk_level: str              # "low", "medium", "high", "very_high"
    required_skills: List[str]
    best_personality_traits: Dict[str, Tuple[float, float]]  # trait: (min, max)
    description: str


VENTURE_PROFILES: Dict[VentureType, VentureProfile] = {
    VentureType.INVENTOR: VentureProfile(
        startup_capital=20000,
        base_monthly_income=8000,
        risk_level="high",
        required_skills=["creativity", "technical"],
        best_personality_traits={
            "openness": (0.7, 1.0),
            "conscientiousness": (0.5, 1.0),
        },
        description="Patents, prototypes, and licensing deals",
    ),
    VentureType.FARMER: VentureProfile(
        startup_capital=50000,
        base_monthly_income=4000,
        risk_level="medium",
        required_skills=["physical", "planning"],
        best_personality_traits={
            "conscientiousness": (0.6, 1.0),
            "agreeableness": (0.5, 1.0),
        },
        description="Land acquisition, crop management, farm-to-table",
    ),
    VentureType.CONTENT_CREATOR: VentureProfile(
        startup_capital=2000,
        base_monthly_income=3000,
        risk_level="high",
        required_skills=["charisma", "creativity"],
        best_personality_traits={
            "extraversion": (0.6, 1.0),
            "openness": (0.6, 1.0),
        },
        description="Audience building, sponsorships, viral moments",
    ),
    VentureType.CONSULTANT: VentureProfile(
        startup_capital=1000,
        base_monthly_income=10000,
        risk_level="low",
        required_skills=["expertise", "communication"],
        best_personality_traits={
            "conscientiousness": (0.7, 1.0),
            "extraversion": (0.4, 0.8),
        },
        description="Expertise monetization, client acquisition",
    ),
    VentureType.ARBITRAGE: VentureProfile(
        startup_capital=10000,
        base_monthly_income=5000,
        risk_level="very_high",
        required_skills=["analysis", "timing"],
        best_personality_traits={
            "neuroticism": (0.0, 0.4),  # Need low neuroticism
            "openness": (0.5, 1.0),
        },
        description="Buy low, sell high, market timing",
    ),
    VentureType.RESTAURATEUR: VentureProfile(
        startup_capital=100000,
        base_monthly_income=6000,
        risk_level="high",
        required_skills=["management", "cooking"],
        best_personality_traits={
            "extraversion": (0.5, 1.0),
            "conscientiousness": (0.6, 1.0),
        },
        description="Restaurant ownership, menu development, reviews",
    ),
    VentureType.ARTIST: VentureProfile(
        startup_capital=5000,
        base_monthly_income=2000,
        risk_level="medium",
        required_skills=["creativity", "reputation"],
        best_personality_traits={
            "openness": (0.8, 1.0),
            "neuroticism": (0.3, 0.7),  # Some angst helps
        },
        description="Gallery shows, commissions, artistic reputation",
    ),
    VentureType.MUSICIAN: VentureProfile(
        startup_capital=3000,
        base_monthly_income=2500,
        risk_level="high",
        required_skills=["music", "charisma"],
        best_personality_traits={
            "extraversion": (0.5, 1.0),
            "openness": (0.7, 1.0),
        },
        description="Gigs, albums, fan base, touring",
    ),
    VentureType.HANDYMAN: VentureProfile(
        startup_capital=5000,
        base_monthly_income=5000,
        risk_level="low",
        required_skills=["technical", "physical"],
        best_personality_traits={
            "agreeableness": (0.5, 1.0),
            "conscientiousness": (0.5, 1.0),
        },
        description="Service business, referrals, tool investment",
    ),
    VentureType.SCIENTIST: VentureProfile(
        startup_capital=0,  # Grant-funded
        base_monthly_income=7000,
        risk_level="medium",
        required_skills=["research", "writing"],
        best_personality_traits={
            "openness": (0.7, 1.0),
            "conscientiousness": (0.7, 1.0),
        },
        description="Research grants, publications, discoveries",
    ),
    VentureType.WRITER: VentureProfile(
        startup_capital=500,
        base_monthly_income=2000,
        risk_level="high",
        required_skills=["writing", "marketing"],
        best_personality_traits={
            "openness": (0.7, 1.0),
            "neuroticism": (0.3, 0.8),  # Writers often introspective
        },
        description="Books, articles, royalties, literary reputation",
    ),
    VentureType.CRAFTSPERSON: VentureProfile(
        startup_capital=3000,
        base_monthly_income=3500,
        risk_level="low",
        required_skills=["crafting", "marketing"],
        best_personality_traits={
            "conscientiousness": (0.6, 1.0),
            "openness": (0.5, 1.0),
        },
        description="Artisan goods, farmers markets, online sales",
    ),
}


# =============================================================================
# CAREER MOODLETS
# =============================================================================

CAREER_MOODLETS: Dict[str, Dict[str, Any]] = {
    # Positive Employee Moodlets
    "just_promoted": {
        "mood": 25, "energy": 10, "duration": 72,
        "description": "Got a well-deserved promotion!",
    },
    "big_raise": {
        "mood": 20, "energy": 5, "duration": 48,
        "description": "Received a significant raise.",
    },
    "work_recognition": {
        "mood": 15, "energy": 5, "duration": 24,
        "description": "Work was recognized by management.",
    },
    "good_performance_review": {
        "mood": 10, "energy": 5, "duration": 24,
        "description": "Got positive feedback in review.",
    },
    "work_from_home": {
        "mood": 10, "energy": 10, "duration": 8,
        "description": "Enjoying working from home.",
    },

    # Negative Employee Moodlets
    "passed_over": {
        "mood": -20, "energy": -10, "duration": 72,
        "description": "Passed over for a promotion.",
    },
    "bad_review": {
        "mood": -15, "energy": -5, "duration": 48,
        "description": "Received negative performance review.",
    },
    "job_insecurity": {
        "mood": -15, "energy": -10, "duration": 168,
        "description": "Worried about job security.",
    },
    "office_politics": {
        "mood": -10, "energy": -5, "duration": 24,
        "description": "Dealing with workplace drama.",
    },
    "boring_work": {
        "mood": -5, "energy": -5, "duration": 8,
        "description": "Work feels tedious and unfulfilling.",
    },

    # Entrepreneur Positive Moodlets
    "venture_launched": {
        "mood": 30, "energy": 20, "duration": 72,
        "description": "Launched a new business venture!",
    },
    "first_customer": {
        "mood": 25, "energy": 15, "duration": 48,
        "description": "Got the first paying customer!",
    },
    "big_client": {
        "mood": 20, "energy": 10, "duration": 48,
        "description": "Landed a major client.",
    },
    "viral_moment": {
        "mood": 30, "energy": 20, "duration": 24,
        "description": "Content went viral!",
    },
    "patent_granted": {
        "mood": 25, "energy": 10, "duration": 72,
        "description": "Patent application was approved!",
    },
    "book_published": {
        "mood": 30, "energy": 10, "duration": 120,
        "description": "Published a book!",
    },
    "phase_advanced": {
        "mood": 20, "energy": 10, "duration": 48,
        "description": "Business moved to next phase!",
    },
    "profitable_month": {
        "mood": 15, "energy": 5, "duration": 24,
        "description": "Had a profitable month.",
    },
    "great_review": {
        "mood": 15, "energy": 5, "duration": 24,
        "description": "Received a glowing customer review.",
    },
    "sold_out": {
        "mood": 20, "energy": 10, "duration": 24,
        "description": "Products completely sold out!",
    },

    # Entrepreneur Negative Moodlets
    "venture_failed": {
        "mood": -40, "energy": -30, "duration": 168,
        "description": "Business venture failed.",
    },
    "bad_month": {
        "mood": -15, "energy": -10, "duration": 24,
        "description": "Had a rough month financially.",
    },
    "client_lost": {
        "mood": -15, "energy": -5, "duration": 24,
        "description": "Lost an important client.",
    },
    "bad_review_business": {
        "mood": -10, "energy": -5, "duration": 48,
        "description": "Received a negative business review.",
    },
    "startup_stress": {
        "mood": -10, "energy": -15, "duration": 24,
        "description": "Startup phase is stressful.",
    },
    "cash_flow_worry": {
        "mood": -15, "energy": -10, "duration": 48,
        "description": "Worried about cash flow.",
    },
    "competition_threat": {
        "mood": -10, "energy": -5, "duration": 24,
        "description": "Facing new competition.",
    },

    # Career Transition Moodlets
    "career_pivot": {
        "mood": 5, "energy": -5, "duration": 48,
        "description": "Navigating a career change.",
    },
    "new_chapter": {
        "mood": 15, "energy": 10, "duration": 72,
        "description": "Excited about a new career chapter.",
    },
    "leaving_comfort": {
        "mood": -10, "energy": -5, "duration": 24,
        "description": "Nervous about leaving comfort zone.",
    },

    # Purpose-Related
    "meaningful_work": {
        "mood": 15, "energy": 10, "duration": 24,
        "description": "Work feels deeply meaningful.",
    },
    "making_difference": {
        "mood": 20, "energy": 10, "duration": 48,
        "description": "Making a real difference through work.",
    },
}


# =============================================================================
# CAREER EVENTS
# =============================================================================

CAREER_EVENTS: Dict[str, Dict[str, Any]] = {
    # Employee Events
    "annual_review": {
        "type": "employee",
        "frequency": "yearly",
        "description": "Annual performance review",
    },
    "promotion_opportunity": {
        "type": "employee",
        "frequency": "variable",
        "description": "Opportunity for promotion",
    },
    "layoff_scare": {
        "type": "employee",
        "frequency": "rare",
        "description": "Company announces layoffs",
    },
    "company_acquisition": {
        "type": "employee",
        "frequency": "rare",
        "description": "Company is being acquired",
    },

    # Entrepreneur Events
    "investor_interest": {
        "type": "entrepreneur",
        "frequency": "rare",
        "description": "An investor shows interest",
    },
    "market_shift": {
        "type": "entrepreneur",
        "frequency": "variable",
        "description": "Market conditions change significantly",
    },
    "seasonal_boom": {
        "type": "entrepreneur",
        "frequency": "seasonal",
        "description": "Seasonal increase in business",
    },
    "seasonal_slump": {
        "type": "entrepreneur",
        "frequency": "seasonal",
        "description": "Seasonal decrease in business",
    },
}


# =============================================================================
# CAREER MANAGER
# =============================================================================

class CareerManager:
    """
    Manages career progression for all agents.

    Supports both employee track (traditional jobs) and
    entrepreneur track (12 venture types).
    """

    def __init__(
        self,
        economic_manager: Optional[Any] = None,
        moodlet_manager: Optional[Any] = None,
        seed: int = 42,
    ):
        self.careers: Dict[str, CareerState] = {}
        self.economic = economic_manager
        self.moodlets = moodlet_manager
        self.rng = random.Random(seed)
        self.current_day: int = 0

        # Statistics
        self.stats = {
            "total_promotions": 0,
            "total_ventures_started": 0,
            "total_ventures_failed": 0,
            "total_ventures_succeeded": 0,
            "total_career_pivots": 0,
        }

    def initialize_career(
        self,
        agent_id: str,
        track: CareerTrack = CareerTrack.EMPLOYEE,
        occupation: str = "worker",
        starting_salary: float = 40000.0,
        day: int = 0,
    ) -> CareerState:
        """Initialize career state for an agent."""
        career = CareerState(
            track=track,
            occupation=occupation,
            career_start_day=day,
        )

        if track == CareerTrack.EMPLOYEE:
            career.employee = EmployeeState(
                job_title=occupation,
                salary=starting_salary,
            )

        self.careers[agent_id] = career
        return career

    def get_career(self, agent_id: str) -> Optional[CareerState]:
        """Get career state for agent."""
        return self.careers.get(agent_id)

    # =========================================================================
    # ENTREPRENEUR OPERATIONS
    # =========================================================================

    def can_start_venture(
        self,
        agent_id: str,
        venture_type: VentureType,
        available_capital: float,
    ) -> Tuple[bool, str]:
        """Check if agent can start a venture."""
        career = self.careers.get(agent_id)
        if not career:
            return False, "No career state initialized"

        if career.pivot_cooldown_days > 0:
            return False, f"Must wait {career.pivot_cooldown_days} days before career change"

        profile = VENTURE_PROFILES[venture_type]
        if available_capital < profile.startup_capital:
            return False, f"Need ${profile.startup_capital:,.0f} capital (have ${available_capital:,.0f})"

        return True, "Ready to start venture"

    def start_venture(
        self,
        agent_id: str,
        venture_type: VentureType,
        initial_capital: float,
    ) -> Dict[str, Any]:
        """Start a new entrepreneurial venture."""
        career = self.careers.get(agent_id)
        if not career:
            return {"success": False, "error": "No career state"}

        profile = VENTURE_PROFILES[venture_type]

        # Create venture
        venture = EntrepreneurVenture(
            venture_type=venture_type,
            capital_invested=initial_capital,
        )

        # Update career state
        career.track = CareerTrack.ENTREPRENEUR
        career.venture = venture
        career.occupation = venture_type.value
        career.pivot_cooldown_days = 30

        self.stats["total_ventures_started"] += 1

        # Add moodlet
        if self.moodlets:
            self.moodlets.add_moodlet(agent_id, "venture_launched", "career")

        return {
            "success": True,
            "venture_type": venture_type.value,
            "capital_invested": initial_capital,
            "description": profile.description,
        }

    def process_venture_month(self, agent_id: str) -> Dict[str, Any]:
        """Process monthly venture income and progression."""
        career = self.careers.get(agent_id)
        if not career or not career.venture:
            return {"success": False}

        venture = career.venture
        profile = VENTURE_PROFILES[venture.venture_type]

        # Check for failure
        failure_risk = venture.get_failure_risk()
        if self.rng.random() < failure_risk:
            venture.consecutive_failures += 1
            venture.last_month_successful = False

            # Three consecutive failures = venture fails
            if venture.consecutive_failures >= 3:
                return self._fail_venture(agent_id)

            # Add bad month moodlet
            if self.moodlets:
                self.moodlets.add_moodlet(agent_id, "bad_month", "career")

            return {
                "success": True,
                "income": 0,
                "message": "Difficult month",
            }

        # Calculate income
        base_income = profile.base_monthly_income
        phase_mult = venture.get_phase_multiplier()
        reputation_bonus = 1 + (venture.reputation / 100)
        variance = self.rng.uniform(0.6, 1.4)  # High volatility

        income = base_income * phase_mult * reputation_bonus * variance

        # Update venture stats
        venture.revenue_this_month = income
        venture.total_revenue += income
        venture.months_active += 1
        venture.consecutive_failures = 0
        venture.last_month_successful = True

        # Build reputation on successful months
        venture.reputation = min(100, venture.reputation + self.rng.uniform(1, 3))

        # Check for phase advancement
        phase_result = self._check_phase_advancement(agent_id)

        # Update career earnings
        career.total_career_earnings += income

        # Update economic system if available
        if self.economic and hasattr(self.economic, "add_income"):
            self.economic.add_income(agent_id, income, "venture")

        # Add moodlet
        if self.moodlets and income > profile.base_monthly_income:
            self.moodlets.add_moodlet(agent_id, "profitable_month", "career")

        return {
            "success": True,
            "income": income,
            "phase": venture.phase.value,
            "reputation": venture.reputation,
            "phase_advanced": phase_result.get("advanced", False),
        }

    def _check_phase_advancement(self, agent_id: str) -> Dict[str, Any]:
        """Check if venture should advance to next phase."""
        career = self.careers.get(agent_id)
        venture = career.venture

        old_phase = venture.phase
        advanced = False

        # Phase thresholds (in months)
        if venture.phase == VenturePhase.STARTUP and venture.months_active >= 6:
            if venture.reputation >= 20:
                venture.phase = VenturePhase.GROWTH
                advanced = True
        elif venture.phase == VenturePhase.GROWTH and venture.months_active >= 18:
            if venture.reputation >= 40:
                venture.phase = VenturePhase.ESTABLISHED
                advanced = True
        elif venture.phase == VenturePhase.ESTABLISHED and venture.months_active >= 36:
            if venture.reputation >= 60:
                venture.phase = VenturePhase.SCALING
                advanced = True

        if advanced:
            self.stats["total_ventures_succeeded"] += 1
            if self.moodlets:
                self.moodlets.add_moodlet(agent_id, "phase_advanced", "career")

        return {"advanced": advanced, "old_phase": old_phase.value, "new_phase": venture.phase.value}

    def _fail_venture(self, agent_id: str) -> Dict[str, Any]:
        """Handle venture failure."""
        career = self.careers.get(agent_id)
        venture = career.venture

        # Record failure
        career.past_ventures.append((venture.venture_type, "failed"))
        venture.phase = VenturePhase.FAILED

        self.stats["total_ventures_failed"] += 1

        # Add failure moodlet
        if self.moodlets:
            self.moodlets.add_moodlet(agent_id, "venture_failed", "career")

        # Add debt from failure
        if self.economic and hasattr(self.economic, "add_debt"):
            debt = venture.capital_invested * 0.5  # Lose half of capital
            self.economic.add_debt(agent_id, debt, "venture_failure")

        return {
            "success": True,
            "failed": True,
            "venture_type": venture.venture_type.value,
            "total_revenue": venture.total_revenue,
            "months_active": venture.months_active,
        }

    # =========================================================================
    # EMPLOYEE OPERATIONS
    # =========================================================================

    def process_employee_month(self, agent_id: str) -> Dict[str, Any]:
        """Process monthly employee updates."""
        career = self.careers.get(agent_id)
        if not career or not career.employee:
            return {"success": False}

        employee = career.employee

        # Monthly salary
        monthly_salary = employee.salary / 12
        career.total_career_earnings += monthly_salary

        # Update economic system
        if self.economic and hasattr(self.economic, "add_income"):
            self.economic.add_income(agent_id, monthly_salary, "salary")

        # Random performance fluctuation
        employee.performance_score += self.rng.uniform(-5, 5)
        employee.performance_score = max(0, min(100, employee.performance_score))

        # Satisfaction drift
        if employee.performance_score > 70:
            employee.satisfaction += self.rng.uniform(0, 2)
        elif employee.performance_score < 40:
            employee.satisfaction -= self.rng.uniform(0, 3)
        employee.satisfaction = max(0, min(100, employee.satisfaction))

        return {
            "success": True,
            "salary": monthly_salary,
            "performance": employee.performance_score,
            "satisfaction": employee.satisfaction,
        }

    def check_promotion(self, agent_id: str) -> Dict[str, Any]:
        """Check if employee gets promoted."""
        career = self.careers.get(agent_id)
        if not career or not career.employee:
            return {"success": False}

        employee = career.employee

        # Can't promote past executive
        if employee.level == EmploymentLevel.EXECUTIVE:
            return {"success": True, "promoted": False, "reason": "Already at top level"}

        prob = employee.get_promotion_probability()

        if self.rng.random() < prob:
            # Promoted!
            old_level = employee.level
            employee.level = EmploymentLevel(employee.level.value + 1)
            employee.times_promoted += 1
            employee.years_in_role = 0

            # Salary increase
            old_salary = employee.salary
            employee.salary *= 1.15 + (self.rng.random() * 0.1)  # 15-25% raise

            self.stats["total_promotions"] += 1

            if self.moodlets:
                self.moodlets.add_moodlet(agent_id, "just_promoted", "career")

            return {
                "success": True,
                "promoted": True,
                "old_level": old_level.name,
                "new_level": employee.level.name,
                "old_salary": old_salary,
                "new_salary": employee.salary,
            }
        else:
            employee.times_passed_over += 1

            if employee.times_passed_over >= 3:
                if self.moodlets:
                    self.moodlets.add_moodlet(agent_id, "passed_over", "career")

            return {"success": True, "promoted": False}

    # =========================================================================
    # CAREER TRANSITIONS
    # =========================================================================

    def pivot_career(
        self,
        agent_id: str,
        new_track: CareerTrack,
        venture_type: Optional[VentureType] = None,
        available_capital: float = 0,
    ) -> Dict[str, Any]:
        """Switch between employee and entrepreneur tracks."""
        career = self.careers.get(agent_id)
        if not career:
            return {"success": False, "error": "No career state"}

        if career.pivot_cooldown_days > 0:
            return {"success": False, "error": f"Cooldown: {career.pivot_cooldown_days} days left"}

        old_track = career.track

        if new_track == CareerTrack.ENTREPRENEUR:
            if not venture_type:
                return {"success": False, "error": "Must specify venture type"}

            can_start, reason = self.can_start_venture(agent_id, venture_type, available_capital)
            if not can_start:
                return {"success": False, "error": reason}

            # Record old employee state if applicable
            if career.employee:
                career.employee = None

            return self.start_venture(agent_id, venture_type, available_capital)

        elif new_track == CareerTrack.EMPLOYEE:
            # Entrepreneur -> Employee (closing business or going back to work)
            if career.venture:
                if career.venture.phase != VenturePhase.FAILED:
                    career.past_ventures.append((career.venture.venture_type, "closed"))
                career.venture = None

            career.track = CareerTrack.EMPLOYEE
            career.employee = EmployeeState(
                job_title="Worker",
                salary=40000,  # Start at entry level
            )
            career.occupation = "worker"
            career.pivot_cooldown_days = 30

            self.stats["total_career_pivots"] += 1

            if self.moodlets:
                self.moodlets.add_moodlet(agent_id, "career_pivot", "career")

            return {"success": True, "new_track": "employee"}

    # =========================================================================
    # PERSONALITY MATCHING
    # =========================================================================

    def get_personality_venture_fit(
        self,
        personality: Dict[str, float],
        venture_type: VentureType,
    ) -> float:
        """Calculate how well personality fits venture type (0-1)."""
        profile = VENTURE_PROFILES[venture_type]

        total_score = 0.0
        num_traits = len(profile.best_personality_traits)

        for trait, (min_val, max_val) in profile.best_personality_traits.items():
            trait_value = personality.get(trait, 0.5)

            if min_val <= trait_value <= max_val:
                # Perfect fit
                total_score += 1.0
            elif trait_value < min_val:
                # Below range
                total_score += max(0, 1 - (min_val - trait_value) * 2)
            else:
                # Above range
                total_score += max(0, 1 - (trait_value - max_val) * 2)

        return total_score / max(num_traits, 1)

    def recommend_ventures(
        self,
        agent_id: str,
        personality: Dict[str, float],
        available_capital: float,
        top_n: int = 3,
    ) -> List[Tuple[VentureType, float, str]]:
        """Recommend best ventures based on personality and capital."""
        recommendations = []

        for venture_type, profile in VENTURE_PROFILES.items():
            # Check capital requirement
            if available_capital < profile.startup_capital:
                continue

            fit_score = self.get_personality_venture_fit(personality, venture_type)
            recommendations.append((venture_type, fit_score, profile.description))

        # Sort by fit score
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:top_n]

    # =========================================================================
    # DAILY UPDATE
    # =========================================================================

    def daily_update(self, day: int):
        """Update all careers daily."""
        self.current_day = day

        for agent_id, career in self.careers.items():
            # Decrement pivot cooldown
            if career.pivot_cooldown_days > 0:
                career.pivot_cooldown_days -= 1

            # Update years in role (every 365 days)
            if career.employee and day > 0 and day % 365 == 0:
                career.employee.years_in_role += 1
                career.employee.years_at_company += 1

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get career system statistics."""
        employees = sum(1 for c in self.careers.values() if c.track == CareerTrack.EMPLOYEE)
        entrepreneurs = sum(1 for c in self.careers.values() if c.track == CareerTrack.ENTREPRENEUR)

        venture_phases = {}
        for c in self.careers.values():
            if c.venture:
                phase = c.venture.phase.value
                venture_phases[phase] = venture_phases.get(phase, 0) + 1

        return {
            "total_agents": len(self.careers),
            "employees": employees,
            "entrepreneurs": entrepreneurs,
            "venture_phases": venture_phases,
            **self.stats,
        }

    def get_career_panel(self, agent_id: str) -> str:
        """Get formatted career panel for display."""
        career = self.careers.get(agent_id)
        if not career:
            return "No career data"

        lines = []
        lines.append(f"{'='*40}")
        lines.append(f"CAREER: {career.occupation.upper()}")
        lines.append(f"Track: {career.track.value}")
        lines.append(f"Total Earnings: ${career.total_career_earnings:,.2f}")

        if career.track == CareerTrack.EMPLOYEE and career.employee:
            emp = career.employee
            lines.append(f"\nEMPLOYEE STATUS:")
            lines.append(f"  Level: {emp.level.name} ({emp.level.value}/5)")
            lines.append(f"  Salary: ${emp.salary:,.0f}/year")
            lines.append(f"  Performance: {emp.performance_score:.1f}/100")
            lines.append(f"  Satisfaction: {emp.satisfaction:.1f}/100")
            lines.append(f"  Promotions: {emp.times_promoted}")

        elif career.track == CareerTrack.ENTREPRENEUR and career.venture:
            ven = career.venture
            profile = VENTURE_PROFILES[ven.venture_type]
            lines.append(f"\nVENTURE STATUS:")
            lines.append(f"  Type: {ven.venture_type.value}")
            lines.append(f"  Phase: {ven.phase.value}")
            lines.append(f"  Months Active: {ven.months_active}")
            lines.append(f"  Reputation: {ven.reputation:.1f}/100")
            lines.append(f"  Last Month: ${ven.revenue_this_month:,.0f}")
            lines.append(f"  Total Revenue: ${ven.total_revenue:,.0f}")
            lines.append(f"  Capital Invested: ${ven.capital_invested:,.0f}")
            lines.append(f"  Risk Level: {profile.risk_level}")

        if career.past_ventures:
            lines.append(f"\nPAST VENTURES:")
            for vtype, outcome in career.past_ventures[-3:]:
                lines.append(f"  - {vtype.value}: {outcome}")

        lines.append(f"{'='*40}")
        return "\n".join(lines)


# =============================================================================
# DEMO FUNCTION
# =============================================================================

async def demo_career_system():
    """Demonstrate the career progression system."""
    print("=" * 60)
    print("v8.3 CAREER PROGRESSION SYSTEM DEMO")
    print("=" * 60)

    manager = CareerManager(seed=42)

    # Create some agents with different paths
    agents = [
        ("alice", "employee", 50000),
        ("bob", "employee", 35000),
        ("carol", "entrepreneur", 60000),
        ("david", "entrepreneur", 15000),
    ]

    for agent_id, track, starting_value in agents:
        if track == "employee":
            manager.initialize_career(agent_id, CareerTrack.EMPLOYEE, "analyst", starting_value)
        else:
            manager.initialize_career(agent_id, CareerTrack.EMPLOYEE, "worker", 40000)

    print("\n1. INITIAL CAREERS")
    print("-" * 40)
    for agent_id, _, _ in agents:
        career = manager.get_career(agent_id)
        print(f"{agent_id}: {career.track.value} - {career.occupation}")

    # Carol starts a content creator venture
    print("\n2. CAROL STARTS A VENTURE")
    print("-" * 40)
    personality = {"extraversion": 0.8, "openness": 0.7, "conscientiousness": 0.5}
    recommendations = manager.recommend_ventures("carol", personality, 5000)
    print("Recommended ventures for Carol:")
    for vtype, score, desc in recommendations:
        print(f"  {vtype.value}: {score:.2f} fit - {desc}")

    result = manager.start_venture("carol", VentureType.CONTENT_CREATOR, 2000)
    print(f"\nStarted: {result}")

    # David tries farming (needs more capital)
    print("\n3. DAVID ATTEMPTS FARMING (needs capital)")
    print("-" * 40)
    can_start, reason = manager.can_start_venture("david", VentureType.FARMER, 15000)
    print(f"Can start farming: {can_start}")
    print(f"Reason: {reason}")

    # David starts handyman instead
    result = manager.start_venture("david", VentureType.HANDYMAN, 5000)
    print(f"Started handyman instead: {result['success']}")

    # Simulate 12 months
    print("\n4. SIMULATING 12 MONTHS")
    print("-" * 40)

    for month in range(1, 13):
        manager.daily_update(month * 30)

        for agent_id, _, _ in agents:
            career = manager.get_career(agent_id)

            if career.track == CareerTrack.EMPLOYEE:
                manager.process_employee_month(agent_id)

                # Check for promotion annually
                if month == 12:
                    result = manager.check_promotion(agent_id)
                    if result.get("promoted"):
                        print(f"  {agent_id} promoted to {result['new_level']}!")

            elif career.track == CareerTrack.ENTREPRENEUR:
                result = manager.process_venture_month(agent_id)
                if result.get("failed"):
                    print(f"  {agent_id}'s venture FAILED!")
                elif result.get("phase_advanced"):
                    print(f"  {agent_id} advanced to {result.get('phase', 'next')} phase!")

    # Final status
    print("\n5. FINAL STATUS")
    print("-" * 40)
    for agent_id, _, _ in agents:
        print(manager.get_career_panel(agent_id))

    # Statistics
    print("\n6. SYSTEM STATISTICS")
    print("-" * 40)
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_career_system())
