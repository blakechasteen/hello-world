"""
Solo Activities System for NeuroHood

Adds personal time, journaling, and ongoing personal projects to agents.
Agents need alone time, keep journals, and work on meaningful projects.

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from autonomous_agents import (
    AutonomousNeighborhoodEngine,
    AutonomousAgent,
    AutonomousReasoner,
    DeepBackstory,
    DEEP_BACKSTORIES,
    Goal,
    GoalPriority,
    GoalStatus,
    ActionType,
    PlannedAction,
)
from nemesis_enhanced import NeedsSystem, NeedType


# =============================================================================
# PERSONAL PROJECT SYSTEM
# =============================================================================

class ProjectStatus(Enum):
    """Status of a personal project."""
    IDEA = "idea"              # Just conceived
    PLANNING = "planning"      # Figuring out how
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"        # Life got in the way
    NEAR_COMPLETION = "near_completion"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class PersonalProject:
    """An ongoing personal project an agent is working on."""
    project_id: str
    name: str
    description: str
    motivation: str  # Why this matters to them
    status: ProjectStatus = ProjectStatus.IDEA
    progress: float = 0.0  # 0.0 to 1.0

    # Work sessions
    hours_invested: float = 0.0
    last_worked_on: Optional[datetime] = None
    work_sessions: List[Dict[str, Any]] = field(default_factory=list)

    # Emotional connection
    emotional_investment: float = 0.5  # How much they care
    frustration_level: float = 0.0
    breakthroughs: List[str] = field(default_factory=list)
    obstacles: List[str] = field(default_factory=list)

    # Connection to backstory
    related_to_dream: bool = False  # Connected to secret dream?
    related_to_fear: bool = False   # Confronting a fear?
    related_to_regret: bool = False # Addressing a regret?

    def add_work_session(self, description: str, hours: float,
                         progress_made: float, mood: str):
        """Record a work session on this project."""
        self.work_sessions.append({
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "hours": hours,
            "progress_made": progress_made,
            "mood": mood
        })
        self.hours_invested += hours
        self.progress = min(1.0, self.progress + progress_made)
        self.last_worked_on = datetime.now()

        # Update status based on progress
        if self.progress >= 0.95:
            self.status = ProjectStatus.COMPLETED
        elif self.progress >= 0.8:
            self.status = ProjectStatus.NEAR_COMPLETION
        elif self.progress > 0:
            self.status = ProjectStatus.IN_PROGRESS


# Default personal projects based on backstories
DEFAULT_PROJECTS: Dict[str, List[PersonalProject]] = {
    "maya_001": [
        PersonalProject(
            project_id="maya_mural",
            name="Grandmother's Garden Mural",
            description="A large mural depicting my grandmother's garden from memory",
            motivation="To preserve her legacy in a way that others can experience",
            status=ProjectStatus.IN_PROGRESS,
            progress=0.35,
            hours_invested=47.5,
            emotional_investment=0.95,
            related_to_dream=True,
            breakthroughs=["Found the exact shade of purple she loved"],
            obstacles=["Can't quite capture the light through the tomato leaves"]
        ),
        PersonalProject(
            project_id="maya_sketchbook",
            name="365 Days of Sketches",
            description="One sketch every day for a year",
            motivation="Discipline to push through creative blocks",
            status=ProjectStatus.IN_PROGRESS,
            progress=0.23,  # Day 84 of 365
            hours_invested=42.0,
            emotional_investment=0.6,
            related_to_fear=True  # Fear of losing creativity
        )
    ],
    "jorge_002": [
        PersonalProject(
            project_id="jorge_cookbook",
            name="Mama's Kitchen: A Family Cookbook",
            description="Writing down all my mother's recipes before they're forgotten",
            motivation="So her love lives on in every kitchen that uses these recipes",
            status=ProjectStatus.PLANNING,
            progress=0.15,
            hours_invested=12.0,
            emotional_investment=0.9,
            related_to_regret=True,
            obstacles=["Some recipes she never wrote down - just 'a pinch of this'"]
        )
    ],
    "elena_003": [
        PersonalProject(
            project_id="elena_comet",
            name="Comet Discovery Project",
            description="Systematic observation program to discover a new comet",
            motivation="To contribute something lasting to human knowledge",
            status=ProjectStatus.IN_PROGRESS,
            progress=0.08,
            hours_invested=156.0,  # Many nights of observation
            emotional_investment=0.85,
            related_to_dream=True,
            obstacles=["Light pollution getting worse", "Clear nights are rare"],
            breakthroughs=["Spotted an interesting asteroid candidate"]
        ),
        PersonalProject(
            project_id="elena_letters",
            name="Letters to Professor Chen",
            description="Writing letters to my late mentor about what I've learned since",
            motivation="Processing grief and honoring his memory",
            status=ProjectStatus.IN_PROGRESS,
            progress=0.4,
            hours_invested=8.0,
            emotional_investment=0.75
        )
    ],
    "carlos_004": [
        PersonalProject(
            project_id="carlos_family_tree",
            name="Family Heritage Map",
            description="Tracing my family tree and documenting our stories",
            motivation="Understanding where I come from to know who I am",
            status=ProjectStatus.IDEA,
            progress=0.05,
            emotional_investment=0.7,
            related_to_fear=True  # Fear of rootlessness
        ),
        PersonalProject(
            project_id="carlos_novel",
            name="The Unnamed Novel",
            description="A novel about a family spanning generations",
            motivation="To process my own journey through fiction",
            status=ProjectStatus.ON_HOLD,
            progress=0.12,
            hours_invested=45.0,
            emotional_investment=0.6,
            obstacles=["Not sure how it should end", "Too close to real life"]
        )
    ],
    "sofia_005": [
        PersonalProject(
            project_id="sofia_music_nights",
            name="Open Mic Night Series",
            description="Monthly live music nights featuring local artists",
            motivation="To give emerging artists the stage I never had",
            status=ProjectStatus.PLANNING,
            progress=0.2,
            emotional_investment=0.85,
            related_to_dream=True,
            breakthroughs=["Found three interested performers!"]
        ),
        PersonalProject(
            project_id="sofia_secret_recipe",
            name="The Perfect Empanada",
            description="Developing my own signature recipe",
            motivation="Every cafe needs that one thing people come back for",
            status=ProjectStatus.IN_PROGRESS,
            progress=0.6,
            hours_invested=20.0,
            emotional_investment=0.5,
            breakthroughs=["The dough is finally right"],
            obstacles=["Still not as good as my aunt's"]
        )
    ],
    "diego_006": [
        PersonalProject(
            project_id="diego_symphony",
            name="Symphony of the Neighborhood",
            description="A musical piece incorporating sounds from our community",
            motivation="To capture this moment in time before it changes",
            status=ProjectStatus.IN_PROGRESS,
            progress=0.25,
            hours_invested=78.0,
            emotional_investment=0.95,
            related_to_fear=True,  # Fear of going deaf
            related_to_dream=True,
            breakthroughs=["The fountain theme is perfect", "Maya's brushstrokes make a beautiful rhythm"],
            obstacles=["How do you notate birdsong?", "Need more low frequencies"]
        ),
        PersonalProject(
            project_id="diego_hearing_journal",
            name="Sound Memory Archive",
            description="Recording and describing every beautiful sound before I can't hear them",
            motivation="Building a treasure chest of auditory memories",
            status=ProjectStatus.IN_PROGRESS,
            progress=0.4,
            hours_invested=30.0,
            emotional_investment=0.9,
            related_to_fear=True
        )
    ]
}


# =============================================================================
# JOURNAL SYSTEM
# =============================================================================

class JournalEntryType(Enum):
    """Types of journal entries."""
    DAILY_REFLECTION = "daily_reflection"
    DREAM_RECORD = "dream_record"
    EMOTIONAL_PROCESSING = "emotional_processing"
    GRATITUDE = "gratitude"
    PROJECT_NOTES = "project_notes"
    LETTER_UNSENT = "letter_unsent"
    MEMORY_CAPTURE = "memory_capture"
    GOAL_SETTING = "goal_setting"
    CREATIVE_WRITING = "creative_writing"


@dataclass
class JournalEntry:
    """A single journal entry."""
    entry_id: str
    entry_type: JournalEntryType
    timestamp: datetime
    content: str
    mood_before: str
    mood_after: str

    # What prompted this entry
    trigger: Optional[str] = None

    # Connections
    mentions_people: List[str] = field(default_factory=list)
    mentions_project: Optional[str] = None
    mentions_dream: bool = False
    mentions_fear: bool = False


@dataclass
class PersonalJournal:
    """An agent's journal - their private thoughts."""
    agent_id: str
    entries: List[JournalEntry] = field(default_factory=list)

    # Journaling habits
    preferred_time: str = "evening"  # morning, evening, random
    consistency_streak: int = 0
    total_entries: int = 0

    def add_entry(self, entry: JournalEntry):
        """Add a new journal entry."""
        self.entries.append(entry)
        self.total_entries += 1

    def get_recent_entries(self, count: int = 5) -> List[JournalEntry]:
        """Get most recent entries."""
        return self.entries[-count:] if self.entries else []

    def get_entries_about(self, person_id: str) -> List[JournalEntry]:
        """Get entries mentioning a specific person."""
        return [e for e in self.entries if person_id in e.mentions_people]


# =============================================================================
# SOLITUDE SYSTEM
# =============================================================================

class SolitudeActivity(Enum):
    """Activities done during alone time."""
    MEDITATION = "meditation"
    WALKING = "walking"
    READING = "reading"
    JOURNALING = "journaling"
    PROJECT_WORK = "project_work"
    DAYDREAMING = "daydreaming"
    NAPPING = "napping"
    SELF_CARE = "self_care"
    REFLECTION = "reflection"
    CRYING = "crying"  # Sometimes needed
    CREATIVE_FLOW = "creative_flow"


@dataclass
class SolitudeSession:
    """A period of intentional alone time."""
    session_id: str
    agent_id: str
    activity: SolitudeActivity
    location: str
    duration_minutes: int

    # Internal experience
    thoughts: List[str] = field(default_factory=list)
    realizations: List[str] = field(default_factory=list)
    emotional_shift: Dict[str, float] = field(default_factory=dict)

    # Outcomes
    recharged_by: float = 0.0  # How much it helped
    led_to_action: Optional[str] = None


# =============================================================================
# SOLITUDE NEED (Extends NeedsSystem)
# =============================================================================

class ExtendedNeedType(Enum):
    """Extended needs including solitude."""
    SOCIAL = "social"
    CREATIVE = "creative"
    KNOWLEDGE = "knowledge"
    COMFORT = "comfort"
    RECOGNITION = "recognition"
    ROUTINE = "routine"
    SOLITUDE = "solitude"  # NEW
    PURPOSE = "purpose"    # NEW - feeling meaningful


# =============================================================================
# LLM-POWERED SOLO ACTIVITIES
# =============================================================================

class SoloActivityReasoner:
    """LLM-powered reasoning for solo activities."""

    def __init__(self, ollama_client):
        self.client = ollama_client

    async def generate_journal_entry(
        self,
        agent,
        backstory: DeepBackstory,
        entry_type: JournalEntryType,
        recent_events: List[str],
        current_mood: str,
        projects: List[PersonalProject]
    ) -> JournalEntry:
        """Generate a journal entry using LLM."""

        # Build context about what's been happening
        events_text = "\n".join(f"- {e}" for e in recent_events[-5:]) if recent_events else "Nothing notable today."

        projects_text = ""
        if projects:
            active = [p for p in projects if p.status in [ProjectStatus.IN_PROGRESS, ProjectStatus.PLANNING]]
            if active:
                projects_text = "Personal projects:\n" + "\n".join(
                    f"- {p.name} ({p.progress*100:.0f}% done): {p.description}"
                    for p in active[:2]
                )

        system_prompt = f"""You are writing a private journal entry for {agent.name}.

CHARACTER:
- Personality: {backstory.life_philosophy}
- Speech pattern: {backstory.speech_pattern}
- Deepest fear: {backstory.deepest_fear}
- Secret dream: {backstory.secret_dream}
- Coping mechanism: {backstory.coping_mechanism}

JOURNAL ENTRY TYPE: {entry_type.value}
CURRENT MOOD: {current_mood}

RECENT EVENTS:
{events_text}

{projects_text}

Write a deeply personal, authentic journal entry (100-200 words).
This is private - they can be vulnerable, honest, messy.
Include specific details and genuine emotions.
Match their speech pattern even in writing."""

        try:
            content = await self.client.generate(
                system_prompt=system_prompt,
                user_prompt=f"Write {agent.name}'s {entry_type.value.replace('_', ' ')} journal entry:",
                max_tokens=300
            )

            return JournalEntry(
                entry_id=f"{agent.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                entry_type=entry_type,
                timestamp=datetime.now(),
                content=content.strip(),
                mood_before=current_mood,
                mood_after=current_mood,  # Could be updated by LLM
                trigger="daily_routine" if entry_type == JournalEntryType.DAILY_REFLECTION else "spontaneous"
            )

        except Exception as e:
            print(f"[Journal] Error generating entry: {e}")
            return JournalEntry(
                entry_id=f"{agent.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                entry_type=entry_type,
                timestamp=datetime.now(),
                content=f"Tried to write but the words wouldn't come today...",
                mood_before=current_mood,
                mood_after=current_mood
            )

    async def generate_project_work_session(
        self,
        agent,
        backstory: DeepBackstory,
        project: PersonalProject,
        available_hours: float,
        current_mood: str
    ) -> Dict[str, Any]:
        """Generate what happens during a project work session."""

        system_prompt = f"""You are simulating {agent.name} working on their personal project.

CHARACTER:
- Philosophy: {backstory.life_philosophy}
- Coping mechanism: {backstory.coping_mechanism}

PROJECT: {project.name}
Description: {project.description}
Motivation: {project.motivation}
Current progress: {project.progress*100:.0f}%
Hours already invested: {project.hours_invested:.1f}
Previous breakthroughs: {', '.join(project.breakthroughs) if project.breakthroughs else 'None yet'}
Current obstacles: {', '.join(project.obstacles) if project.obstacles else 'None yet'}

CURRENT MOOD: {current_mood}
TIME AVAILABLE: {available_hours:.1f} hours

Simulate what happens during this work session. Return JSON:
{{
    "description": "What they actually did during the session",
    "progress_made": 0.01 to 0.1 (small realistic increments),
    "breakthrough": "A realization or achievement (or null)",
    "obstacle_encountered": "A new challenge (or null)",
    "emotional_journey": "How they felt during the work",
    "ending_mood": "satisfied/frustrated/inspired/tired/peaceful",
    "inner_thought": "What they're thinking as they finish"
}}"""

        try:
            response = await self.client.generate(
                system_prompt=system_prompt,
                user_prompt="Simulate the project work session:",
                max_tokens=250
            )

            # Parse JSON response
            import json
            # Clean up response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            return json.loads(response)

        except Exception as e:
            print(f"[Project] Error simulating session: {e}")
            return {
                "description": f"Worked quietly on {project.name}",
                "progress_made": 0.02,
                "breakthrough": None,
                "obstacle_encountered": None,
                "emotional_journey": "Focused but tired",
                "ending_mood": "satisfied",
                "inner_thought": "Small steps forward..."
            }

    async def generate_solitude_experience(
        self,
        agent,
        backstory: DeepBackstory,
        activity: SolitudeActivity,
        location: str,
        duration_minutes: int,
        recent_stressors: List[str]
    ) -> SolitudeSession:
        """Generate what happens during alone time."""

        stressors_text = "\n".join(f"- {s}" for s in recent_stressors) if recent_stressors else "Nothing pressing."

        system_prompt = f"""You are simulating {agent.name}'s alone time.

CHARACTER:
- Philosophy: {backstory.life_philosophy}
- Deepest fear: {backstory.deepest_fear}
- Secret dream: {backstory.secret_dream}
- Coping mechanism: {backstory.coping_mechanism}

SOLITUDE ACTIVITY: {activity.value}
LOCATION: {location}
DURATION: {duration_minutes} minutes

RECENT STRESSORS:
{stressors_text}

Simulate their internal experience during this solitude. Return JSON:
{{
    "thoughts": ["List of 2-3 thoughts that pass through their mind"],
    "realizations": ["Any insights or realizations (0-2)"],
    "emotional_shift": {{"anxiety": -0.1 to 0.1, "peace": -0.2 to 0.2, "energy": -0.2 to 0.2}},
    "recharged_by": 0.1 to 0.5 (how much this helped),
    "led_to_action": "Something they decided to do (or null)"
}}"""

        try:
            response = await self.client.generate(
                system_prompt=system_prompt,
                user_prompt="Simulate their solitude experience:",
                max_tokens=200
            )

            import json
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            data = json.loads(response)

            return SolitudeSession(
                session_id=f"solitude_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                agent_id=agent.agent_id,
                activity=activity,
                location=location,
                duration_minutes=duration_minutes,
                thoughts=data.get("thoughts", []),
                realizations=data.get("realizations", []),
                emotional_shift=data.get("emotional_shift", {}),
                recharged_by=data.get("recharged_by", 0.2),
                led_to_action=data.get("led_to_action")
            )

        except Exception as e:
            print(f"[Solitude] Error: {e}")
            return SolitudeSession(
                session_id=f"solitude_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                agent_id=agent.agent_id,
                activity=activity,
                location=location,
                duration_minutes=duration_minutes,
                thoughts=["Mind wandered..."],
                recharged_by=0.2
            )


# =============================================================================
# EXTENDED AUTONOMOUS ENGINE WITH SOLO ACTIVITIES
# =============================================================================

class SoloActivityEngine(AutonomousNeighborhoodEngine):
    """Extends autonomous agents with solo activities."""

    def __init__(self, ollama_config=None):
        super().__init__(ollama_config)

        # Personal projects per agent
        self.projects: Dict[str, List[PersonalProject]] = {}

        # Journals per agent
        self.journals: Dict[str, PersonalJournal] = {}

        # Solitude tracker
        self.solitude_sessions: Dict[str, List[SolitudeSession]] = {}

        # Solo activity reasoner
        self.solo_reasoner: Optional[SoloActivityReasoner] = None

        # Extended needs (adding solitude)
        self.solitude_needs: Dict[str, float] = {}  # 0.0 = satisfied, 1.0 = desperate for alone time

    async def initialize(self, load_path=None):
        """Initialize with solo activity systems."""
        await super().initialize(load_path)

        # Create solo reasoner if Ollama available
        # Check for _use_ollama (set by parent) or ollama_client existence
        use_ollama = getattr(self, '_use_ollama', False) or hasattr(self, 'ollama_client')
        if use_ollama and self.ollama_client:
            self.solo_reasoner = SoloActivityReasoner(self.ollama_client)

        # Initialize projects from defaults
        for agent_id in self.state.agents:
            self.projects[agent_id] = list(DEFAULT_PROJECTS.get(agent_id, []))
            self.journals[agent_id] = PersonalJournal(agent_id=agent_id)
            self.solitude_sessions[agent_id] = []
            self.solitude_needs[agent_id] = random.uniform(0.2, 0.5)

    def get_solitude_need(self, agent_id: str) -> float:
        """Get how much an agent needs alone time (0-1)."""
        return self.solitude_needs.get(agent_id, 0.5)

    def satisfy_solitude(self, agent_id: str, amount: float):
        """Satisfy solitude need."""
        if agent_id in self.solitude_needs:
            self.solitude_needs[agent_id] = max(0.0, self.solitude_needs[agent_id] - amount)

    def increase_solitude_need(self, agent_id: str, amount: float):
        """Increase need for alone time (after social interactions)."""
        if agent_id in self.solitude_needs:
            self.solitude_needs[agent_id] = min(1.0, self.solitude_needs[agent_id] + amount)

    async def do_project_work(self, agent_id: str, project_id: str, hours: float) -> Optional[Dict]:
        """Agent works on a personal project."""
        if agent_id not in self.projects:
            return None

        project = None
        for p in self.projects[agent_id]:
            if p.project_id == project_id:
                project = p
                break

        if not project:
            return None

        agent = self.state.agents.get(agent_id)
        backstory = DEEP_BACKSTORIES.get(agent_id)

        if not agent or not backstory:
            return None

        # Get current mood from emotions
        mood = "neutral"
        if agent.emotions.get("happiness", 0) > 0.3:
            mood = "happy"
        elif agent.emotions.get("happiness", 0) < -0.3:
            mood = "melancholy"
        if agent.emotions.get("anxiety", 0) > 0.3:
            mood = "anxious"

        # Generate work session
        if self.solo_reasoner:
            result = await self.solo_reasoner.generate_project_work_session(
                agent, backstory, project, hours, mood
            )
        else:
            result = {
                "description": f"Worked on {project.name}",
                "progress_made": 0.02,
                "ending_mood": "satisfied"
            }

        # Update project
        project.add_work_session(
            description=result.get("description", ""),
            hours=hours,
            progress_made=result.get("progress_made", 0.02),
            mood=result.get("ending_mood", "neutral")
        )

        # Add breakthroughs/obstacles
        if result.get("breakthrough"):
            project.breakthroughs.append(result["breakthrough"])
        if result.get("obstacle_encountered"):
            project.obstacles.append(result["obstacle_encountered"])

        # Satisfy creative need
        if hasattr(self, 'needs_system'):
            self.needs_system.fulfill_need(agent_id, NeedType.CREATIVE, 0.3)

        # Satisfy solitude need (working alone)
        self.satisfy_solitude(agent_id, 0.2)

        return result

    async def write_journal_entry(
        self,
        agent_id: str,
        entry_type: JournalEntryType = JournalEntryType.DAILY_REFLECTION
    ) -> Optional[JournalEntry]:
        """Agent writes in their journal."""
        agent = self.state.agents.get(agent_id)
        backstory = DEEP_BACKSTORIES.get(agent_id)
        journal = self.journals.get(agent_id)

        if not agent or not backstory or not journal:
            return None

        # Get recent events from agent's memories
        recent_events = []
        if hasattr(agent, 'memories') and agent.memories:
            for mem in agent.memories[-5:]:
                if isinstance(mem, dict):
                    recent_events.append(mem.get("description", ""))

        # Get current mood
        mood = "neutral"
        if agent.emotions.get("happiness", 0) > 0.3:
            mood = "content"
        elif agent.emotions.get("happiness", 0) < -0.3:
            mood = "low"

        # Get projects
        projects = self.projects.get(agent_id, [])

        # Generate entry
        if self.solo_reasoner:
            entry = await self.solo_reasoner.generate_journal_entry(
                agent, backstory, entry_type, recent_events, mood, projects
            )
        else:
            entry = JournalEntry(
                entry_id=f"{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                entry_type=entry_type,
                timestamp=datetime.now(),
                content="Words didn't come easily today...",
                mood_before=mood,
                mood_after=mood
            )

        # Add to journal
        journal.add_entry(entry)

        # Satisfy solitude need
        self.satisfy_solitude(agent_id, 0.15)

        return entry

    async def take_solitude_time(
        self,
        agent_id: str,
        activity: SolitudeActivity,
        location: str,
        duration_minutes: int = 30
    ) -> Optional[SolitudeSession]:
        """Agent takes intentional alone time."""
        agent = self.state.agents.get(agent_id)
        backstory = DEEP_BACKSTORIES.get(agent_id)

        if not agent or not backstory:
            return None

        # Get recent stressors from memories
        stressors = []
        if hasattr(agent, 'memories') and agent.memories:
            for mem in agent.memories[-10:]:
                if isinstance(mem, dict):
                    impact = mem.get("emotional_impact", {})
                    if impact.get("anxiety", 0) > 0 or impact.get("happiness", 0) < -0.2:
                        stressors.append(mem.get("description", ""))

        # Generate solitude experience
        if self.solo_reasoner:
            session = await self.solo_reasoner.generate_solitude_experience(
                agent, backstory, activity, location, duration_minutes, stressors
            )
        else:
            session = SolitudeSession(
                session_id=f"solitude_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                agent_id=agent_id,
                activity=activity,
                location=location,
                duration_minutes=duration_minutes,
                recharged_by=0.2
            )

        # Store session
        if agent_id not in self.solitude_sessions:
            self.solitude_sessions[agent_id] = []
        self.solitude_sessions[agent_id].append(session)

        # Apply emotional shifts
        for emotion, delta in session.emotional_shift.items():
            if emotion in agent.emotions:
                agent.emotions[emotion] = max(-1, min(1, agent.emotions[emotion] + delta))

        # Satisfy solitude need
        self.satisfy_solitude(agent_id, session.recharged_by)

        return session

    async def simulate_solo_step(self, agent_id: str) -> List[Dict[str, Any]]:
        """Simulate one solo activity step for an agent."""
        events = []
        agent = self.state.agents.get(agent_id)

        if not agent:
            return events

        # Check solitude need
        solitude_need = self.get_solitude_need(agent_id)

        # High solitude need = more likely to do solo activity
        if random.random() < solitude_need:
            activity_type = random.choices(
                ["project", "journal", "solitude"],
                weights=[0.4, 0.3, 0.3]
            )[0]

            if activity_type == "project" and self.projects.get(agent_id):
                # Work on a project
                active_projects = [
                    p for p in self.projects[agent_id]
                    if p.status in [ProjectStatus.IN_PROGRESS, ProjectStatus.PLANNING]
                ]
                if active_projects:
                    project = random.choice(active_projects)
                    result = await self.do_project_work(agent_id, project.project_id, 1.0)
                    if result:
                        events.append({
                            "type": "project_work",
                            "agent_id": agent_id,
                            "agent_name": agent.name,
                            "project": project.name,
                            "description": result.get("description"),
                            "progress": project.progress,
                            "mood": result.get("ending_mood")
                        })

            elif activity_type == "journal":
                # Write journal entry
                entry_type = random.choice([
                    JournalEntryType.DAILY_REFLECTION,
                    JournalEntryType.EMOTIONAL_PROCESSING,
                    JournalEntryType.GRATITUDE
                ])
                entry = await self.write_journal_entry(agent_id, entry_type)
                if entry:
                    events.append({
                        "type": "journaling",
                        "agent_id": agent_id,
                        "agent_name": agent.name,
                        "entry_type": entry_type.value,
                        "excerpt": entry.content[:100] + "..." if len(entry.content) > 100 else entry.content
                    })

            else:
                # Take solitude time
                activity = random.choice([
                    SolitudeActivity.MEDITATION,
                    SolitudeActivity.WALKING,
                    SolitudeActivity.DAYDREAMING,
                    SolitudeActivity.REFLECTION
                ])
                location = self.state.agent_locations.get(agent_id, "home")
                session = await self.take_solitude_time(agent_id, activity, location, 30)
                if session:
                    events.append({
                        "type": "solitude",
                        "agent_id": agent_id,
                        "agent_name": agent.name,
                        "activity": activity.value,
                        "thoughts": session.thoughts,
                        "realizations": session.realizations,
                        "recharged": session.recharged_by
                    })

        # Increase solitude need slightly after social activities
        # (This would be called from conversation simulation)

        return events


# =============================================================================
# DEMO
# =============================================================================

async def demo_solo_activities():
    """Demonstrate solo activities system."""
    print("=" * 70)
    print("SOLO ACTIVITIES DEMO")
    print("Alone Time, Journaling, and Personal Projects")
    print("=" * 70)

    engine = SoloActivityEngine()
    await engine.initialize()

    # === Demo 1: Personal Projects ===
    print("\n" + "=" * 50)
    print("DEMO 1: PERSONAL PROJECTS")
    print("=" * 50)

    for agent_id in ["maya_001", "diego_006"]:
        agent = engine.state.agents.get(agent_id)
        projects = engine.projects.get(agent_id, [])

        print(f"\n{agent.name}'s Projects:")
        for p in projects:
            print(f"\n  {p.name}")
            print(f"    Status: {p.status.value}")
            print(f"    Progress: {p.progress*100:.0f}%")
            print(f"    Hours invested: {p.hours_invested:.1f}")
            print(f"    Motivation: \"{p.motivation}\"")
            if p.breakthroughs:
                print(f"    Breakthroughs: {p.breakthroughs[0]}")
            if p.obstacles:
                print(f"    Current obstacle: {p.obstacles[0]}")

    # === Demo 2: Project Work Session ===
    print("\n" + "=" * 50)
    print("DEMO 2: PROJECT WORK SESSION")
    print("=" * 50)

    print("\n[Maya works on her mural for 2 hours...]")
    result = await engine.do_project_work("maya_001", "maya_mural", 2.0)

    project = None
    for p in engine.projects["maya_001"]:
        if p.project_id == "maya_mural":
            project = p
            break

    if result:
        print(f"\n  What happened: {result.get('description', 'Worked quietly')}")
        print(f"  Emotional journey: {result.get('emotional_journey', 'focused')}")
        print(f"  Progress: {project.progress*100:.1f}%")
        if result.get("breakthrough"):
            print(f"  [BREAKTHROUGH] {result['breakthrough']}")
        if result.get("obstacle_encountered"):
            print(f"  [OBSTACLE] {result['obstacle_encountered']}")
        print(f"  Ending mood: {result.get('ending_mood', 'satisfied')}")

    # === Demo 3: Journaling ===
    print("\n" + "=" * 50)
    print("DEMO 3: JOURNALING")
    print("=" * 50)

    for agent_id in ["elena_003", "carlos_004"]:
        agent = engine.state.agents.get(agent_id)

        print(f"\n[{agent.name} writes in their journal...]")
        entry = await engine.write_journal_entry(agent_id, JournalEntryType.DAILY_REFLECTION)

        if entry:
            print(f"\n  --- {agent.name}'s Journal ---")
            print(f"  {entry.content}")
            print(f"  ---")

    # === Demo 4: Solitude Time ===
    print("\n" + "=" * 50)
    print("DEMO 4: SOLITUDE TIME")
    print("=" * 50)

    agent = engine.state.agents.get("sofia_005")
    print(f"\n[{agent.name} takes time to meditate at the cafe before opening...]")

    session = await engine.take_solitude_time(
        "sofia_005",
        SolitudeActivity.MEDITATION,
        "corner_cafe",
        20
    )

    if session:
        print(f"\n  Activity: {session.activity.value}")
        print(f"  Duration: {session.duration_minutes} minutes")
        print(f"  Thoughts that passed through:")
        for thought in session.thoughts:
            print(f"    - \"{thought}\"")
        if session.realizations:
            print(f"  Realizations:")
            for r in session.realizations:
                print(f"    * {r}")
        print(f"  Recharged by: {session.recharged_by*100:.0f}%")
        if session.led_to_action:
            print(f"  Decided to: {session.led_to_action}")

    # === Demo 5: Solitude Need ===
    print("\n" + "=" * 50)
    print("DEMO 5: SOLITUDE NEEDS")
    print("=" * 50)

    print("\n  Solitude needs by resident:")
    for agent_id, need in engine.solitude_needs.items():
        agent = engine.state.agents.get(agent_id)
        status = "desperate for alone time" if need > 0.7 else "comfortable" if need < 0.3 else "could use some quiet"
        print(f"    {agent.name}: {need*100:.0f}% ({status})")

    # === Demo 6: Autonomous Solo Step ===
    print("\n" + "=" * 50)
    print("DEMO 6: AUTONOMOUS SOLO ACTIVITIES")
    print("=" * 50)

    # Increase solitude need to trigger solo activity
    engine.solitude_needs["diego_006"] = 0.8

    print("\n[Diego needs alone time... simulating autonomous choice...]")
    events = await engine.simulate_solo_step("diego_006")

    for event in events:
        print(f"\n  Type: {event['type']}")
        if event['type'] == 'project_work':
            print(f"  Working on: {event['project']}")
            print(f"  Description: {event['description']}")
            print(f"  Progress: {event['progress']*100:.1f}%")
        elif event['type'] == 'journaling':
            print(f"  Entry type: {event['entry_type']}")
            print(f"  Excerpt: \"{event['excerpt']}\"")
        elif event['type'] == 'solitude':
            print(f"  Activity: {event['activity']}")
            print(f"  Thoughts: {event['thoughts']}")
            if event['realizations']:
                print(f"  Realizations: {event['realizations']}")

    # Cleanup
    await engine.close()

    print("\n" + "=" * 70)
    print("SOLO ACTIVITIES DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_solo_activities())
