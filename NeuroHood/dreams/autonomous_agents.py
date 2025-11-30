"""
Autonomous Agent System - Self-Directed NPCs

NPCs that:
1. Generate their own goals based on backstory, needs, and current state
2. Plan actions to achieve those goals
3. Execute actions autonomously
4. Reflect on outcomes and adjust

Inspired by:
- Stanford's "Generative Agents" (Park et al. 2023)
- The Sims wants/fears system
- Dwarf Fortress goals/values

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum

from nemesis_enhanced import (
    NemesisEnhancedEngine, SignificantEventMemory,
    EventSignificance, EventValence, NeedType
)
from enriched_neighborhood import RESIDENT_BACKSTORIES, ResidentBackstory
from neighborhood_engine import AgentState
from ollama_client import OllamaClient, OllamaConfig


# =============================================================================
# EXPANDED BACKSTORY SYSTEM
# =============================================================================

@dataclass
class DeepBackstory:
    """Extended backstory with motivational depth."""
    # Core identity
    occupation: str
    life_philosophy: str  # Core belief that drives them

    # History
    childhood_memory: str  # Formative experience
    biggest_regret: str
    proudest_moment: str

    # Relationships
    family_situation: str  # "widow", "estranged from kids", "close family nearby"
    lost_love: Optional[str]  # Someone they miss
    mentor_figure: Optional[str]  # Who shaped them

    # Inner world
    secret_dream: str  # What they truly want
    deepest_fear: str  # What they avoid
    coping_mechanism: str  # How they deal with stress

    # Expression
    speech_pattern: str
    favorite_expressions: List[str]
    body_language: str

    # Daily life
    morning_routine: str
    comfort_activity: str  # What they do when stressed
    guilty_pleasure: str


# Expanded backstories for each resident
DEEP_BACKSTORIES = {
    "maya_001": DeepBackstory(
        occupation="freelance artist",
        life_philosophy="Art is how we make sense of chaos",
        childhood_memory="My grandmother taught me to see colors everywhere - in shadows, in sounds, in silence",
        biggest_regret="Never told my father I forgave him before he passed",
        proudest_moment="A stranger once cried looking at my painting. Said it captured exactly how grief feels.",
        family_situation="estranged from most family, grandmother passed last year",
        lost_love="A musician who chose touring over staying",
        mentor_figure="Grandmother Elena, who painted until her hands couldn't hold a brush",
        secret_dream="To have my art hanging in a museum, not for fame, but so my grandmother's legacy lives on",
        deepest_fear="That my best work is behind me, that I'll never capture magic again",
        coping_mechanism="Sketches obsessively when anxious - fills notebooks",
        speech_pattern="speaks in metaphors and colors, pauses to find the right word",
        favorite_expressions=["It's like...", "You know that feeling when...", "The color of that is..."],
        body_language="hands always moving, often touches face when thinking",
        morning_routine="Coffee while watching light change through the window, then sketching",
        comfort_activity="Rearranging her studio and organizing paints by emotion, not color",
        guilty_pleasure="Trashy reality TV - says it's 'studying human nature'"
    ),

    "jorge_002": DeepBackstory(
        occupation="retired chef, community garden coordinator",
        life_philosophy="Food is love made visible",
        childhood_memory="Cooking with my mother in our tiny kitchen, her singing while she stirred",
        biggest_regret="Working too many nights, missed my kids growing up",
        proudest_moment="My daughter called to say she made my mother's recipe perfectly",
        family_situation="widow for 5 years, daughter lives across country, calls every Sunday",
        lost_love="Maria, his wife of 40 years, still talks to her photo",
        mentor_figure="Chef Bernard who taught him that cooking is patience made delicious",
        secret_dream="To write a cookbook of his mother's recipes before they're forgotten forever",
        deepest_fear="Being forgotten, that no one will carry on the family traditions",
        coping_mechanism="Cooks elaborate meals when stressed, often gives food away",
        speech_pattern="warm, uses food metaphors, often starts stories with 'When I was young...'",
        favorite_expressions=["That's a good egg", "Let it simmer", "You can't rush a good sauce"],
        body_language="pats people on the shoulder, gestures while telling stories",
        morning_routine="Waters the garden at dawn, talks to his tomato plants by name",
        comfort_activity="Making soup - 'Soup fixes everything'",
        guilty_pleasure="Instant ramen, which he'd never admit to anyone"
    ),

    "elena_003": DeepBackstory(
        occupation="software engineer, works remotely",
        life_philosophy="The universe doesn't owe us answers, but it's beautiful that we keep asking",
        childhood_memory="Seeing Saturn through my father's telescope, crying because it was so beautiful",
        biggest_regret="Choosing the safe job over the astronomy PhD",
        proudest_moment="Figured out a bug that had stumped the team for months - at 3am, alone",
        family_situation="parents passed, no siblings, has a cat named Kepler",
        lost_love="Never had one - 'Haven't found someone who understands comfortable silence'",
        mentor_figure="Dr. Chen, astronomy professor who said 'Never stop looking up'",
        secret_dream="To discover something - a comet, a pattern, anything that adds to human knowledge",
        deepest_fear="Wasting her life on code that doesn't matter, leaving no mark",
        coping_mechanism="Organizes things - her bookshelf is sorted by publication date",
        speech_pattern="precise, thoughtful pauses, occasionally uses technical analogies",
        favorite_expressions=["Statistically speaking...", "That's interesting because...", "I wonder if..."],
        body_language="still, attentive, makes strong eye contact when interested",
        morning_routine="Coffee, news, 30 minutes of stargazing notes even if it was cloudy",
        comfort_activity="Cataloging her book collection or reading papers on arxiv",
        guilty_pleasure="Romance novels - the predictability is soothing"
    ),

    "carlos_004": DeepBackstory(
        occupation="retired history teacher",
        life_philosophy="We are all stories in the end - make yours worth telling",
        childhood_memory="My grandfather telling stories of the old country, making history alive",
        biggest_regret="Never wrote the novel that's been in my head for 30 years",
        proudest_moment="A former student became a history professor, dedicated her first book to me",
        family_situation="wife passed 3 years ago, two adult children who visit holidays",
        lost_love="Still grieves his wife Rosa, keeps her reading glasses on the nightstand",
        mentor_figure="His grandfather, the family storyteller",
        secret_dream="To trace his family tree back to the old country and stand where his ancestors stood",
        deepest_fear="Dementia - losing his memories, his stories, himself",
        coping_mechanism="Tells stories, even to himself, keeps journals obsessively",
        speech_pattern="storytelling rhythm, connects everything to history",
        favorite_expressions=["This reminds me of...", "History teaches us...", "Let me tell you about..."],
        body_language="leans in when telling stories, gestures dramatically",
        morning_routine="Writes in his journal, reads the newspaper cover to cover",
        comfort_activity="Walking the neighborhood, imagining its history",
        guilty_pleasure="Detective podcasts - 'It's historical documentation of our era'"
    ),

    "sofia_005": DeepBackstory(
        occupation="cafe owner",
        life_philosophy="Everyone needs a place where they belong",
        childhood_memory="My parents' restaurant, watching people become family over shared meals",
        biggest_regret="Taking over the cafe instead of pursuing music",
        proudest_moment="When Mrs. Henderson said the cafe was the only place she didn't feel lonely",
        family_situation="never married, considers regulars her family, parents in assisted living",
        lost_love="Almost married once - he wanted her to sell the cafe and move away",
        mentor_figure="Her mother, who said 'A good host makes everyone feel like the guest of honor'",
        secret_dream="To host live music nights and discover the next big artist",
        deepest_fear="The cafe failing, losing the community she's built",
        coping_mechanism="Cleans and organizes the cafe, makes comfort food",
        speech_pattern="warm, remembers details about people, uses their names",
        favorite_expressions=["The usual?", "Tell me everything", "You look like you need..."],
        body_language="touches people's arms, leans in to listen, always busy but never rushed",
        morning_routine="Arrives at 5am, bakes fresh pastries, arranges flowers",
        comfort_activity="Making new coffee blends, giving them names based on regulars",
        guilty_pleasure="Eating standing up in the kitchen - 'A chef never sits'"
    ),

    "diego_006": DeepBackstory(
        occupation="music teacher",
        life_philosophy="Music is the silence between the notes",
        childhood_memory="First time I made my mother cry with a song - good tears",
        biggest_regret="Gave up on the band when things got hard",
        proudest_moment="A student with no musical background performed at Carnegie Hall",
        family_situation="divorced, no kids, close to his sister and her family",
        lost_love="His ex-wife - 'We loved each other, just wanted different lives'",
        mentor_figure="Old jazz musician named Earl who played in a subway station",
        secret_dream="To compose a symphony that captures the sound of this neighborhood",
        deepest_fear="Going deaf - losing music would be losing himself",
        coping_mechanism="Plays music, any instrument, often improvises his feelings",
        speech_pattern="rhythmic speech, hums between sentences, enthusiastic",
        favorite_expressions=["Feel that rhythm?", "It's like a song that...", "Let me play you something"],
        body_language="taps fingers constantly, sways when talking, closes eyes when thinking",
        morning_routine="Wakes up and plays something - lets his mood choose the instrument",
        comfort_activity="Walking and collecting sounds - traffic, birds, conversations",
        guilty_pleasure="Pop music - 'It's musically simple but I can't stop humming it'"
    )
}


# =============================================================================
# GOAL SYSTEM
# =============================================================================

class GoalPriority(Enum):
    """How important/urgent a goal is."""
    BACKGROUND = 0    # Nice to have
    NORMAL = 1        # Would like to do
    IMPORTANT = 2     # Should do soon
    URGENT = 3        # Must do now
    CRITICAL = 4      # Emergency


class GoalStatus(Enum):
    """Current status of a goal."""
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """A goal an agent wants to achieve."""
    goal_id: str
    description: str
    motivation: str  # Why they want this
    priority: GoalPriority
    status: GoalStatus = GoalStatus.ACTIVE

    # Progress tracking
    progress: float = 0.0  # 0-1
    steps_planned: List[str] = field(default_factory=list)
    steps_completed: List[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Relationships
    related_to_agent: Optional[str] = None  # If goal involves another person
    spawned_from: Optional[str] = None  # Parent goal

    # Outcomes
    success_criteria: str = ""
    failure_conditions: List[str] = field(default_factory=list)


# =============================================================================
# ACTION SYSTEM
# =============================================================================

class ActionType(Enum):
    """Types of actions an agent can take."""
    # Movement
    GO_TO = "go_to"

    # Social
    TALK_TO = "talk_to"
    HELP = "help"
    ASK_FOR_HELP = "ask_for_help"
    GIVE_GIFT = "give_gift"
    INVITE = "invite"
    APOLOGIZE = "apologize"
    THANK = "thank"

    # Activities
    WORK_ON = "work_on"  # Work on a project
    PRACTICE = "practice"  # Practice a skill
    CREATE = "create"  # Create something

    # Self-care
    REST = "rest"
    EAT = "eat"
    REFLECT = "reflect"

    # Information
    OBSERVE = "observe"
    RESEARCH = "research"
    REMEMBER = "remember"


@dataclass
class PlannedAction:
    """An action an agent plans to take."""
    action_type: ActionType
    description: str
    target: Optional[str] = None  # Agent ID or location
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Why
    reasoning: str = ""
    supports_goal: Optional[str] = None

    # When
    priority: int = 0  # Higher = sooner
    estimated_duration_minutes: int = 15


# =============================================================================
# LLM-POWERED AUTONOMOUS REASONING
# =============================================================================

class AutonomousReasoner:
    """
    LLM-powered reasoning for autonomous agents.

    Generates:
    1. Goals from backstory + current state
    2. Action plans to achieve goals
    3. Dialogue that reflects goals
    4. Reflections on outcomes
    """

    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client

    async def generate_goals(
        self,
        agent: AgentState,
        backstory: DeepBackstory,
        current_needs: Dict[NeedType, float],
        recent_events: List[str],
        current_location: str,
        nearby_agents: List[str],
        time_of_day: str
    ) -> List[Goal]:
        """
        Generate goals based on current state.

        The LLM considers:
        - Personality and backstory
        - Unfulfilled needs
        - Recent events
        - Current situation
        """

        # Build context for LLM
        needs_str = ", ".join([f"{n.value}: {v:.0%}" for n, v in current_needs.items()])
        events_str = "\n- ".join(recent_events[-5:]) if recent_events else "Nothing notable recently"
        nearby_str = ", ".join(nearby_agents) if nearby_agents else "No one nearby"

        prompt = f"""You are {agent.name}, thinking about what you want to do.

YOUR IDENTITY:
- Philosophy: {backstory.life_philosophy}
- Secret dream: {backstory.secret_dream}
- Deepest fear: {backstory.deepest_fear}
- Current coping: {backstory.coping_mechanism}

CURRENT STATE:
- Location: {current_location}
- Time: {time_of_day}
- Nearby: {nearby_str}
- Needs (low = unfulfilled): {needs_str}

RECENT EVENTS:
- {events_str}

Based on who you are and your current situation, what do you want to do?
Think about:
1. What unfulfilled need is pressing?
2. What does your personality drive you toward?
3. Is there an opportunity here?
4. What would make you feel fulfilled right now?

Respond with 1-3 goals in this JSON format:
[
  {{
    "description": "What you want to achieve",
    "motivation": "Why this matters to you personally",
    "priority": "BACKGROUND/NORMAL/IMPORTANT/URGENT",
    "involves_person": "Name or null",
    "success_looks_like": "How you'll know you succeeded"
  }}
]

Only output valid JSON, no other text."""

        try:
            response = await self.client.generate(
                system_prompt="You are a goal-generation system for an autonomous agent. Output only valid JSON.",
                user_prompt=prompt,
                max_tokens=500
            )

            # Parse JSON
            goals_data = json.loads(response)

            goals = []
            for i, gd in enumerate(goals_data[:3]):  # Max 3 goals
                priority = GoalPriority[gd.get('priority', 'NORMAL').upper()]
                goal = Goal(
                    goal_id=f"{agent.agent_id}_goal_{int(datetime.now().timestamp())}_{i}",
                    description=gd['description'],
                    motivation=gd['motivation'],
                    priority=priority,
                    related_to_agent=gd.get('involves_person'),
                    success_criteria=gd.get('success_looks_like', '')
                )
                goals.append(goal)

            return goals

        except Exception as e:
            print(f"[Reasoner] Goal generation error: {e}")
            # Fallback: generate simple needs-based goal
            return self._fallback_goals(agent, current_needs)

    def _fallback_goals(self, agent: AgentState, needs: Dict[NeedType, float]) -> List[Goal]:
        """Simple fallback goal generation."""
        goals = []

        # Find lowest need
        if needs:
            lowest_need = min(needs.items(), key=lambda x: x[1])
            if lowest_need[1] < 0.3:
                goal = Goal(
                    goal_id=f"{agent.agent_id}_need_{int(datetime.now().timestamp())}",
                    description=f"Satisfy {lowest_need[0].value} need",
                    motivation="Feeling unfulfilled",
                    priority=GoalPriority.IMPORTANT
                )
                goals.append(goal)

        return goals

    async def plan_actions(
        self,
        agent: AgentState,
        backstory: DeepBackstory,
        goal: Goal,
        available_locations: List[str],
        available_agents: Dict[str, str],  # id -> name
        current_location: str
    ) -> List[PlannedAction]:
        """
        Generate action plan for a goal.
        """

        agents_str = ", ".join([f"{name}" for name in available_agents.values()])
        locations_str = ", ".join(available_locations)

        prompt = f"""You are {agent.name}, planning how to achieve your goal.

YOUR GOAL: {goal.description}
WHY IT MATTERS: {goal.motivation}
SUCCESS LOOKS LIKE: {goal.success_criteria}

YOUR PERSONALITY:
- Philosophy: {backstory.life_philosophy}
- You cope by: {backstory.coping_mechanism}

CURRENT SITUATION:
- You are at: {current_location}
- Available locations: {locations_str}
- People in neighborhood: {agents_str}

Plan 2-4 specific actions to work toward your goal.
Consider: Where should you go? Who should you talk to? What should you do?

Respond with actions in this JSON format:
[
  {{
    "action": "GO_TO/TALK_TO/WORK_ON/CREATE/HELP/OBSERVE/REST/REFLECT",
    "description": "Specific action description",
    "target": "Location name or person name or null",
    "reasoning": "Why this helps achieve your goal",
    "priority": 1-5 (higher = do first)
  }}
]

Only output valid JSON."""

        try:
            response = await self.client.generate(
                system_prompt="You are an action planning system. Output only valid JSON.",
                user_prompt=prompt,
                max_tokens=400
            )

            actions_data = json.loads(response)

            actions = []
            for ad in actions_data[:4]:  # Max 4 actions
                try:
                    action_type = ActionType[ad['action'].upper()]
                except KeyError:
                    action_type = ActionType.WORK_ON

                action = PlannedAction(
                    action_type=action_type,
                    description=ad['description'],
                    target=ad.get('target'),
                    reasoning=ad.get('reasoning', ''),
                    supports_goal=goal.goal_id,
                    priority=ad.get('priority', 1)
                )
                actions.append(action)

            # Sort by priority
            actions.sort(key=lambda a: -a.priority)
            return actions

        except Exception as e:
            print(f"[Reasoner] Action planning error: {e}")
            return []

    async def generate_action_dialogue(
        self,
        agent: AgentState,
        backstory: DeepBackstory,
        action: PlannedAction,
        target_agent: Optional[AgentState] = None,
        context: str = ""
    ) -> str:
        """
        Generate dialogue for an action, especially social actions.
        """

        if action.action_type not in [ActionType.TALK_TO, ActionType.HELP,
                                       ActionType.INVITE, ActionType.APOLOGIZE,
                                       ActionType.THANK, ActionType.ASK_FOR_HELP]:
            return ""  # No dialogue needed

        target_name = target_agent.name if target_agent else action.target

        prompt = f"""You are {agent.name}, about to {action.description}.

YOUR PERSONALITY:
- Philosophy: {backstory.life_philosophy}
- Speech style: {backstory.speech_pattern}
- Favorite expressions: {', '.join(backstory.favorite_expressions[:2])}
- Body language: {backstory.body_language}

WHAT YOU'RE DOING: {action.description}
WHY: {action.reasoning}
TALKING TO: {target_name}
{f"CONTEXT: {context}" if context else ""}

Write what you say and do. Include:
- What you say (1-3 sentences, natural to your speech style)
- Optional *action* in asterisks

Stay in character. Be natural, not robotic."""

        try:
            response = await self.client.generate(
                system_prompt=f"You are {agent.name}. Speak naturally in character.",
                user_prompt=prompt,
                max_tokens=150
            )
            return response.strip()
        except Exception as e:
            return f"*approaches {target_name}*"

    async def reflect_on_outcome(
        self,
        agent: AgentState,
        backstory: DeepBackstory,
        goal: Goal,
        actions_taken: List[str],
        outcome: str,
        emotional_impact: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Reflect on the outcome of pursuing a goal.
        Returns insights and potential new goals.
        """

        actions_str = "\n- ".join(actions_taken)
        emotions_str = ", ".join([f"{k}: {v:+.2f}" for k, v in emotional_impact.items()])

        prompt = f"""You are {agent.name}, reflecting on what just happened.

YOUR GOAL WAS: {goal.description}
BECAUSE: {goal.motivation}

WHAT YOU DID:
- {actions_str}

OUTCOME: {outcome}

HOW YOU FEEL: {emotions_str}

Reflect on this experience. Consider:
1. Did you achieve what you wanted?
2. What did you learn about yourself?
3. What would you do differently?
4. Does this change what you want next?

Respond in JSON format:
{{
  "achieved": true/false,
  "insight": "What you learned about yourself",
  "regret": "What you'd do differently (or null)",
  "new_desire": "What you want now (or null)",
  "emotional_shift": "How this changed your feelings"
}}

Only output valid JSON."""

        try:
            response = await self.client.generate(
                system_prompt="You are a reflection system. Output only valid JSON.",
                user_prompt=prompt,
                max_tokens=200
            )

            return json.loads(response)

        except Exception as e:
            print(f"[Reasoner] Reflection error: {e}")
            return {
                "achieved": goal.progress >= 0.8,
                "insight": "I need to think about this more",
                "regret": None,
                "new_desire": None,
                "emotional_shift": "Processing my feelings"
            }


# =============================================================================
# AUTONOMOUS AGENT
# =============================================================================

class AutonomousAgent:
    """
    A fully autonomous NPC that generates its own goals and acts on them.
    """

    def __init__(
        self,
        agent_state: AgentState,
        deep_backstory: DeepBackstory,
        reasoner: AutonomousReasoner
    ):
        self.state = agent_state
        self.backstory = deep_backstory
        self.reasoner = reasoner

        # Goals
        self.active_goals: List[Goal] = []
        self.completed_goals: List[Goal] = []
        self.goal_history: List[Dict] = []

        # Action queue
        self.action_queue: List[PlannedAction] = []
        self.actions_taken_today: List[str] = []

        # Internal state
        self.current_focus: Optional[Goal] = None
        self.last_reflection: Optional[datetime] = None
        self.insights: List[str] = []

    async def think(
        self,
        current_needs: Dict[NeedType, float],
        recent_events: List[str],
        current_location: str,
        nearby_agents: List[str],
        time_of_day: str
    ):
        """
        Main thinking cycle - generate goals if needed.
        """

        # Check if we need new goals
        urgent_goals = [g for g in self.active_goals if g.priority.value >= GoalPriority.IMPORTANT.value]

        if not urgent_goals or len(self.active_goals) < 2:
            print(f"[{self.state.name}] Thinking about what I want...")

            new_goals = await self.reasoner.generate_goals(
                agent=self.state,
                backstory=self.backstory,
                current_needs=current_needs,
                recent_events=recent_events,
                current_location=current_location,
                nearby_agents=nearby_agents,
                time_of_day=time_of_day
            )

            for goal in new_goals:
                if not any(g.description == goal.description for g in self.active_goals):
                    self.active_goals.append(goal)
                    print(f"[{self.state.name}] New goal: {goal.description}")

    async def plan(
        self,
        available_locations: List[str],
        available_agents: Dict[str, str],
        current_location: str
    ):
        """
        Plan actions for highest priority goal.
        """

        if not self.active_goals:
            return

        # Focus on highest priority goal
        self.current_focus = max(self.active_goals, key=lambda g: g.priority.value)

        if not self.action_queue:
            print(f"[{self.state.name}] Planning for: {self.current_focus.description}")

            actions = await self.reasoner.plan_actions(
                agent=self.state,
                backstory=self.backstory,
                goal=self.current_focus,
                available_locations=available_locations,
                available_agents=available_agents,
                current_location=current_location
            )

            self.action_queue = actions
            self.current_focus.steps_planned = [a.description for a in actions]

            for action in actions:
                print(f"[{self.state.name}] Planned: {action.description}")

    async def act(self, engine: 'AutonomousNeighborhoodEngine') -> Optional[Dict]:
        """
        Execute the next action in the queue.
        Returns action result.
        """

        if not self.action_queue:
            return None

        action = self.action_queue.pop(0)
        print(f"[{self.state.name}] Acting: {action.description}")

        result = await self._execute_action(action, engine)

        self.actions_taken_today.append(action.description)

        if self.current_focus:
            self.current_focus.steps_completed.append(action.description)
            # Update progress
            if self.current_focus.steps_planned:
                self.current_focus.progress = len(self.current_focus.steps_completed) / len(self.current_focus.steps_planned)

        return result

    async def _execute_action(self, action: PlannedAction, engine: 'AutonomousNeighborhoodEngine') -> Dict:
        """Execute a specific action."""

        result = {"action": action.description, "success": False, "outcome": ""}

        if action.action_type == ActionType.GO_TO:
            # Find matching location
            for loc_id, loc in engine.state.locations.items():
                if action.target and action.target.lower() in loc['name'].lower():
                    engine.state.agent_locations[self.state.agent_id] = loc_id
                    result["success"] = True
                    result["outcome"] = f"Arrived at {loc['name']}"
                    break

        elif action.action_type in [ActionType.TALK_TO, ActionType.HELP, ActionType.INVITE]:
            # Find target agent
            target_agent = None
            for aid, agent in engine.state.agents.items():
                if action.target and action.target.lower() in agent.name.lower():
                    target_agent = agent
                    break

            if target_agent:
                # Generate dialogue
                dialogue = await self.reasoner.generate_action_dialogue(
                    agent=self.state,
                    backstory=self.backstory,
                    action=action,
                    target_agent=target_agent
                )
                result["success"] = True
                result["outcome"] = f"Spoke with {target_agent.name}"
                result["dialogue"] = dialogue

        elif action.action_type == ActionType.WORK_ON:
            result["success"] = True
            result["outcome"] = f"Worked on: {action.description}"

        elif action.action_type == ActionType.CREATE:
            result["success"] = True
            result["outcome"] = f"Created: {action.description}"

        elif action.action_type == ActionType.REFLECT:
            result["success"] = True
            result["outcome"] = "Spent time in reflection"

        elif action.action_type == ActionType.REST:
            result["success"] = True
            result["outcome"] = "Rested"

        return result

    async def reflect(self, outcome: str, emotional_impact: Dict[str, float]):
        """
        Reflect on goal progress.
        """

        if not self.current_focus:
            return

        if self.current_focus.progress >= 0.8:
            # Goal likely complete
            reflection = await self.reasoner.reflect_on_outcome(
                agent=self.state,
                backstory=self.backstory,
                goal=self.current_focus,
                actions_taken=self.actions_taken_today[-5:],
                outcome=outcome,
                emotional_impact=emotional_impact
            )

            print(f"[{self.state.name}] Reflection: {reflection.get('insight', 'No insight')}")

            if reflection.get('insight'):
                self.insights.append(reflection['insight'])

            if reflection.get('achieved'):
                self.current_focus.status = GoalStatus.COMPLETED
                self.current_focus.completed_at = datetime.now()
                self.completed_goals.append(self.current_focus)
                self.active_goals.remove(self.current_focus)
                print(f"[{self.state.name}] Goal completed: {self.current_focus.description}")

            if reflection.get('new_desire'):
                # Generate new goal from reflection
                new_goal = Goal(
                    goal_id=f"{self.state.agent_id}_reflected_{int(datetime.now().timestamp())}",
                    description=reflection['new_desire'],
                    motivation="Emerged from reflection",
                    priority=GoalPriority.NORMAL,
                    spawned_from=self.current_focus.goal_id
                )
                self.active_goals.append(new_goal)
                print(f"[{self.state.name}] New desire: {new_goal.description}")

            self.last_reflection = datetime.now()
            self.actions_taken_today = []


# =============================================================================
# AUTONOMOUS NEIGHBORHOOD ENGINE
# =============================================================================

class AutonomousNeighborhoodEngine(NemesisEnhancedEngine):
    """
    Neighborhood where NPCs autonomously set and pursue goals.
    """

    def __init__(self, ollama_config: OllamaConfig = None):
        super().__init__(ollama_config)

        self.reasoner = AutonomousReasoner(self.ollama_client)
        self.autonomous_agents: Dict[str, AutonomousAgent] = {}

    async def initialize(self, load_path=None):
        """Initialize with autonomous agents."""
        await super().initialize(load_path)

        # Create autonomous agents
        for agent_id, agent in self.state.agents.items():
            if agent_id in DEEP_BACKSTORIES:
                self.autonomous_agents[agent_id] = AutonomousAgent(
                    agent_state=agent,
                    deep_backstory=DEEP_BACKSTORIES[agent_id],
                    reasoner=self.reasoner
                )

    async def simulate_autonomous_step(self) -> List[Dict]:
        """
        Simulate one step where all agents think, plan, and act autonomously.
        """

        results = []

        # Get current state
        available_locations = list(self.state.locations.keys())
        available_agents = {aid: a.name for aid, a in self.state.agents.items()}

        for agent_id, auto_agent in self.autonomous_agents.items():
            agent = self.state.agents[agent_id]
            current_loc = self.state.agent_locations.get(agent_id, "unknown")

            # Get nearby agents
            nearby = [
                self.state.agents[aid].name
                for aid, loc in self.state.agent_locations.items()
                if loc == current_loc and aid != agent_id
            ]

            # Get needs
            needs = {}
            if agent_id in self.needs_system.agent_needs:
                needs = {n.need_type: n.current_level
                        for n in self.needs_system.agent_needs[agent_id].values()}

            # Get recent events
            recent_events = []
            if agent_id in self.significant_memory.events:
                recent_events = [e.description for e in self.significant_memory.events[agent_id][-5:]]

            # Think
            await auto_agent.think(
                current_needs=needs,
                recent_events=recent_events,
                current_location=self.state.locations.get(current_loc, {}).get('name', 'somewhere'),
                nearby_agents=nearby,
                time_of_day=self.state.time_of_day
            )

            # Plan
            await auto_agent.plan(
                available_locations=[self.state.locations[l]['name'] for l in available_locations],
                available_agents=available_agents,
                current_location=self.state.locations.get(current_loc, {}).get('name', 'somewhere')
            )

            # Act
            action_result = await auto_agent.act(self)
            if action_result:
                results.append({
                    "agent": agent.name,
                    "action": action_result
                })

        return results


# =============================================================================
# DEMO
# =============================================================================

async def demo_autonomous_agents():
    """Demonstrate autonomous agent behavior."""
    print("=" * 70)
    print("AUTONOMOUS AGENTS DEMO")
    print("NPCs with Self-Generated Goals and Actions")
    print("=" * 70)

    config = OllamaConfig(
        model="llama3.2:3b",
        temperature=0.85,
        max_tokens=200
    )

    engine = AutonomousNeighborhoodEngine(config)
    await engine.initialize()

    # === Demo 1: Show Deep Backstories ===
    print("\n" + "=" * 50)
    print("DEMO 1: DEEP BACKSTORIES")
    print("=" * 50)

    for name, backstory in [("Maya", DEEP_BACKSTORIES["maya_001"]),
                            ("Sofia", DEEP_BACKSTORIES["sofia_005"])]:
        print(f"\n{name}:")
        print(f"  Philosophy: \"{backstory.life_philosophy}\"")
        print(f"  Secret dream: {backstory.secret_dream}")
        print(f"  Deepest fear: {backstory.deepest_fear}")
        print(f"  Copes by: {backstory.coping_mechanism}")

    # === Demo 2: Autonomous Goal Generation ===
    print("\n" + "=" * 50)
    print("DEMO 2: AUTONOMOUS GOAL GENERATION")
    print("=" * 50)

    # Let Maya think
    maya_auto = engine.autonomous_agents["maya_001"]

    # Set up low creative need
    if "maya_001" in engine.needs_system.agent_needs:
        engine.needs_system.agent_needs["maya_001"][NeedType.CREATIVE].current_level = 0.2

    await maya_auto.think(
        current_needs={NeedType.CREATIVE: 0.2, NeedType.SOCIAL: 0.5},
        recent_events=["Watched a beautiful sunset", "Felt uninspired all day"],
        current_location="Corner Cafe",
        nearby_agents=["Sofia", "Diego"],
        time_of_day="afternoon"
    )

    print("\nMaya's generated goals:")
    for goal in maya_auto.active_goals:
        print(f"  - {goal.description}")
        print(f"    Motivation: {goal.motivation}")
        print(f"    Priority: {goal.priority.name}")

    # === Demo 3: Action Planning ===
    print("\n" + "=" * 50)
    print("DEMO 3: ACTION PLANNING")
    print("=" * 50)

    if maya_auto.active_goals:
        await maya_auto.plan(
            available_locations=["Corner Cafe", "Community Garden", "Oak Park", "Reading Corner"],
            available_agents={"sofia_005": "Sofia", "diego_006": "Diego", "jorge_002": "Jorge"},
            current_location="Corner Cafe"
        )

        print("\nMaya's action plan:")
        for i, action in enumerate(maya_auto.action_queue, 1):
            print(f"  {i}. {action.action_type.value}: {action.description}")
            print(f"     Reasoning: {action.reasoning}")

    # === Demo 4: Execute Actions with Dialogue ===
    print("\n" + "=" * 50)
    print("DEMO 4: AUTONOMOUS ACTIONS WITH DIALOGUE")
    print("=" * 50)

    for i in range(2):  # Execute 2 actions
        result = await maya_auto.act(engine)
        if result:
            print(f"\n[Maya] {result['outcome']}")
            if 'dialogue' in result:
                print(f"  \"{result['dialogue']}\"")

    # === Demo 5: Multi-Agent Autonomous Simulation ===
    print("\n" + "=" * 50)
    print("DEMO 5: MULTI-AGENT AUTONOMOUS STEP")
    print("=" * 50)

    print("\nAll agents thinking and acting autonomously...")
    results = await engine.simulate_autonomous_step()

    for result in results:
        print(f"\n{result['agent']}:")
        print(f"  Action: {result['action'].get('outcome', result['action'].get('action', 'Unknown'))}")
        if result['action'].get('dialogue'):
            print(f"  Says: \"{result['action']['dialogue']}\"")

    # === Demo 6: Reflection ===
    print("\n" + "=" * 50)
    print("DEMO 6: REFLECTION ON OUTCOMES")
    print("=" * 50)

    if maya_auto.current_focus:
        maya_auto.current_focus.progress = 0.9  # Simulate near completion

        await maya_auto.reflect(
            outcome="Had an inspiring conversation with Diego about music and art",
            emotional_impact={"happiness": 0.3, "energy": 0.2}
        )

        print("\nMaya's insights:")
        for insight in maya_auto.insights:
            print(f"  - {insight}")

    await engine.close()

    print("\n" + "=" * 70)
    print("AUTONOMOUS AGENTS DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_autonomous_agents())
