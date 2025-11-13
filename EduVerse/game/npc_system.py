"""
Educational NPC System

Creates intelligent NPCs (teachers, mentors, peers) that guide students through
learning quests with adaptive dialogue based on performance and teaching style.

Created: 2025-11-13 (Week 3 - DreamWeaver Phase 1)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from EduVerse.education.curriculum import Subject
from EduVerse.game.quest_engine import Quest, QuestType


class NPCRole(Enum):
    """Role of the NPC in the learning environment"""
    TEACHER = "teacher"      # Primary instructor
    MENTOR = "mentor"        # Experienced guide
    PEER = "peer"           # Fellow student/helper
    GUIDE = "guide"         # World navigation helper
    SPECIALIST = "specialist"  # Subject matter expert


class TeachingStyle(Enum):
    """Teaching approach of the NPC"""
    SUPPORTIVE = "supportive"          # Encouraging, patient
    CHALLENGING = "challenging"        # Pushes for excellence
    SOCRATIC = "socratic"             # Question-based learning
    DEMONSTRATIVE = "demonstrative"   # Show and tell
    COLLABORATIVE = "collaborative"   # Learn together approach


class DialogueContext(Enum):
    """Context for dialogue generation"""
    GREETING = "greeting"
    QUEST_OFFER = "quest_offer"
    QUEST_COMPLETE = "quest_complete"
    HINT_REQUEST = "hint_request"
    STRUGGLE = "struggle"
    SUCCESS = "success"
    ENCOURAGEMENT = "encouragement"
    EXPLANATION = "explanation"
    FEEDBACK = "feedback"


@dataclass
class NPCProfile:
    """Complete profile for an educational NPC"""
    npc_id: str
    name: str
    role: NPCRole
    subjects: List[Subject]
    teaching_style: TeachingStyle
    personality_traits: List[str]
    backstory: str
    appearance_description: str
    voice_style: str
    location_id: str  # Where the NPC is found

    # Dialogue templates by context
    dialogue_templates: Dict[DialogueContext, List[str]] = field(default_factory=dict)

    # Quest-giving capability
    can_give_quests: bool = True
    quest_types: List[QuestType] = field(default_factory=list)

    # Performance thresholds for adaptive dialogue
    struggle_threshold: float = 0.4  # Below this = struggling
    success_threshold: float = 0.75  # Above this = succeeding

    def __repr__(self) -> str:
        return f"<NPC {self.name} ({self.role.value}) - {', '.join(s.value for s in self.subjects[:2])}>"


@dataclass
class DialogueResponse:
    """Response from NPC dialogue system"""
    npc_id: str
    npc_name: str
    text: str
    context: DialogueContext
    quest_id: Optional[str] = None
    hint: Optional[str] = None
    emotional_tone: str = "neutral"  # supportive, challenging, excited, concerned
    subject: Optional[Subject] = None

    def __repr__(self) -> str:
        return f'[{self.npc_name}] ({self.emotional_tone}): "{self.text[:50]}..."'


class NPCDialogueManager:
    """Manages NPC dialogue generation based on context and player state"""

    def __init__(self, npc: NPCProfile):
        self.npc = npc
        self._init_default_templates()

    def _init_default_templates(self):
        """Initialize default dialogue templates based on teaching style"""
        style = self.npc.teaching_style

        # GREETING templates
        greetings = {
            TeachingStyle.SUPPORTIVE: [
                f"Welcome! I'm {self.npc.name}. I'm here to help you learn!",
                f"Hello there! Ready to explore {self.npc.subjects[0].value if self.npc.subjects else 'new topics'}?",
                f"Great to see you! Let's make learning fun together.",
            ],
            TeachingStyle.CHALLENGING: [
                f"I'm {self.npc.name}. I expect excellence and I know you can achieve it.",
                f"Ready to push yourself? {self.npc.subjects[0].value if self.npc.subjects else 'Learning'} requires dedication.",
                f"Welcome. I don't accept mediocrity, and neither should you.",
            ],
            TeachingStyle.SOCRATIC: [
                f"Hello! I'm {self.npc.name}. What brings you here today?",
                f"Welcome. Tell me, what do you already know about {self.npc.subjects[0].value if self.npc.subjects else 'this topic'}?",
                f"Greetings! Before we begin, what questions do you have?",
            ],
            TeachingStyle.DEMONSTRATIVE: [
                f"I'm {self.npc.name}. Watch and learn - that's my motto!",
                f"Hello! Let me show you how {self.npc.subjects[0].value if self.npc.subjects else 'this'} works.",
                f"Welcome! I'll demonstrate the concepts and you'll master them in no time.",
            ],
            TeachingStyle.COLLABORATIVE: [
                f"Hey! I'm {self.npc.name}. We'll figure this out together!",
                f"Welcome, partner! Let's explore {self.npc.subjects[0].value if self.npc.subjects else 'learning'} as a team.",
                f"Hi there! Two minds are better than one - let's collaborate!",
            ],
        }

        # STRUGGLE templates (player performing poorly)
        struggle = {
            TeachingStyle.SUPPORTIVE: [
                "Don't worry, everyone finds this challenging at first. Let's break it down together.",
                "You're doing better than you think! Let's try a different approach.",
                "It's okay to struggle - that's how we grow. Want a hint?",
            ],
            TeachingStyle.CHALLENGING: [
                "This is difficult, but I know you can handle it. Push through.",
                "Struggle means you're learning. Don't give up now.",
                "You're capable of more than this. Refocus and try again.",
            ],
            TeachingStyle.SOCRATIC: [
                "What part is confusing you? Let's identify the core issue.",
                "If you could solve just one piece of this, what would it be?",
                "What do you already understand? Build from there.",
            ],
            TeachingStyle.DEMONSTRATIVE: [
                "Let me show you the method step by step again.",
                "Watch closely - I'll demonstrate the tricky part.",
                "Here's how I approach this problem. See the pattern?",
            ],
            TeachingStyle.COLLABORATIVE: [
                "This is tough for everyone. Let's work through it together.",
                "I struggled with this too. Here's what helped me...",
                "Let's put our heads together and figure this out.",
            ],
        }

        # SUCCESS templates (player performing well)
        success = {
            TeachingStyle.SUPPORTIVE: [
                "Excellent work! I knew you could do it!",
                "You're really getting the hang of this! Keep it up!",
                "Wonderful! Your hard work is paying off!",
            ],
            TeachingStyle.CHALLENGING: [
                "Good. Now let's see if you can handle something harder.",
                "Acceptable. But don't get complacent - there's more to learn.",
                "You've proven yourself. Ready for the next level?",
            ],
            TeachingStyle.SOCRATIC: [
                "Interesting solution! Why did you choose that approach?",
                "You solved it! What did you learn from this?",
                "Excellent! Can you explain your reasoning?",
            ],
            TeachingStyle.DEMONSTRATIVE: [
                "Perfect! You followed the method exactly!",
                "You've got the technique down! Well demonstrated!",
                "Great execution! You clearly understood the demonstration.",
            ],
            TeachingStyle.COLLABORATIVE: [
                "We did it! Great teamwork!",
                "High five! That was a great collaboration!",
                "Awesome! Together we can tackle anything!",
            ],
        }

        # QUEST_OFFER templates
        quest_offer = {
            TeachingStyle.SUPPORTIVE: [
                "I have a fun challenge that I think you'll enjoy. Want to try?",
                "Ready for your next learning adventure?",
                "I've prepared something special for you. Interested?",
            ],
            TeachingStyle.CHALLENGING: [
                "I have a difficult quest. Think you're ready?",
                "This next challenge will test your skills. Are you prepared?",
                "Only the dedicated tackle this quest. Will you?",
            ],
            TeachingStyle.SOCRATIC: [
                "I have a problem that might interest you. What do you think?",
                "Here's a challenge. How would you approach it?",
                "I'm curious if you can solve this. Want to try?",
            ],
            TeachingStyle.DEMONSTRATIVE: [
                "I'll show you what needs to be done, then you try. Ready?",
                "Watch this demonstration, then replicate it. Sound good?",
                "Let me walk you through this quest first.",
            ],
            TeachingStyle.COLLABORATIVE: [
                "I have a quest we can work on together. In?",
                "Let's team up for this challenge!",
                "Ready to collaborate on something interesting?",
            ],
        }

        # HINT_REQUEST templates
        hints = {
            TeachingStyle.SUPPORTIVE: [
                "Of course! Here's a helpful hint: {hint}",
                "Great question! Think about this: {hint}",
                "Let me help you out: {hint}",
            ],
            TeachingStyle.CHALLENGING: [
                "You should figure this out yourself, but here's a nudge: {hint}",
                "One hint only: {hint}. Now push through.",
                "I'll give you this, but think harder: {hint}",
            ],
            TeachingStyle.SOCRATIC: [
                "Instead of giving you the answer, consider: {hint}",
                "What if I told you: {hint}? What would that mean?",
                "Here's a question to guide you: {hint}",
            ],
            TeachingStyle.DEMONSTRATIVE: [
                "Let me show you the key step: {hint}",
                "Watch this part carefully: {hint}",
                "The critical technique is: {hint}",
            ],
            TeachingStyle.COLLABORATIVE: [
                "Good thinking! Together we know: {hint}",
                "I remember something useful: {hint}",
                "Let's use this insight: {hint}",
            ],
        }

        # QUEST_COMPLETE templates
        complete = {
            TeachingStyle.SUPPORTIVE: [
                "You did it! I'm so proud of your progress!",
                "Fantastic work! You've mastered this concept!",
                "Excellent completion! You should feel proud!",
            ],
            TeachingStyle.CHALLENGING: [
                "Completed. Are you ready for something harder now?",
                "Good. You met my expectations. Next challenge?",
                "Satisfactory. Let's raise the difficulty.",
            ],
            TeachingStyle.SOCRATIC: [
                "Completed! What was your key insight?",
                "Well done! What would you do differently next time?",
                "Finished! How does this connect to what you already knew?",
            ],
            TeachingStyle.DEMONSTRATIVE: [
                "Perfect execution! You replicated the method flawlessly!",
                "Complete! You followed the demonstration well!",
                "Excellent! The technique is now yours!",
            ],
            TeachingStyle.COLLABORATIVE: [
                "We crushed it! Great partnership!",
                "Quest complete! I knew we could do it together!",
                "Team success! Let's celebrate!",
            ],
        }

        # Store all templates
        self.npc.dialogue_templates[DialogueContext.GREETING] = greetings.get(style, greetings[TeachingStyle.SUPPORTIVE])
        self.npc.dialogue_templates[DialogueContext.STRUGGLE] = struggle.get(style, struggle[TeachingStyle.SUPPORTIVE])
        self.npc.dialogue_templates[DialogueContext.SUCCESS] = success.get(style, success[TeachingStyle.SUPPORTIVE])
        self.npc.dialogue_templates[DialogueContext.QUEST_OFFER] = quest_offer.get(style, quest_offer[TeachingStyle.SUPPORTIVE])
        self.npc.dialogue_templates[DialogueContext.HINT_REQUEST] = hints.get(style, hints[TeachingStyle.SUPPORTIVE])
        self.npc.dialogue_templates[DialogueContext.QUEST_COMPLETE] = complete.get(style, complete[TeachingStyle.SUPPORTIVE])

    def generate_dialogue(
        self,
        context: DialogueContext,
        player_performance: float = 0.5,  # 0.0-1.0
        quest: Optional[Quest] = None,
        hint_text: Optional[str] = None,
        **kwargs
    ) -> DialogueResponse:
        """Generate contextual dialogue based on player state and context"""

        # Get appropriate templates for context
        templates = self.npc.dialogue_templates.get(context, [])

        if not templates:
            # Fallback generic dialogue
            templates = [f"[{self.npc.name}]: I'm here to help with {self.npc.subjects[0].value if self.npc.subjects else 'learning'}."]

        # Select random template
        template = random.choice(templates)

        # Replace placeholders
        if hint_text and '{hint}' in template:
            template = template.replace('{hint}', hint_text)

        # Determine emotional tone based on context and performance
        tone = self._determine_emotional_tone(context, player_performance)

        # Create response
        response = DialogueResponse(
            npc_id=self.npc.npc_id,
            npc_name=self.npc.name,
            text=template,
            context=context,
            quest_id=quest.quest_id if quest else None,
            hint=hint_text,
            emotional_tone=tone,
            subject=self.npc.subjects[0] if self.npc.subjects else None
        )

        return response

    def _determine_emotional_tone(self, context: DialogueContext, performance: float) -> str:
        """Determine emotional tone based on context and performance"""

        if context == DialogueContext.SUCCESS:
            return "excited"
        elif context == DialogueContext.STRUGGLE:
            return "concerned" if self.npc.teaching_style == TeachingStyle.SUPPORTIVE else "challenging"
        elif context == DialogueContext.QUEST_COMPLETE:
            if performance >= self.npc.success_threshold:
                return "proud"
            else:
                return "encouraging"
        elif context == DialogueContext.GREETING:
            return "welcoming"
        else:
            return "neutral"

    def generate_hint(self, quest: Quest, difficulty: float = 0.5) -> str:
        """Generate a contextual hint for a quest"""

        # Simple hint generation based on quest type and difficulty
        hints = {
            QuestType.TUTORIAL: [
                "Start with the basics and follow the steps.",
                "Read the instructions carefully.",
                "Take it one step at a time.",
            ],
            QuestType.PRACTICE: [
                "Remember what you learned in the tutorial.",
                "Use the same method, just different numbers.",
                "Practice makes perfect - keep trying!",
            ],
            QuestType.CHALLENGE: [
                "Think about multiple approaches.",
                "Break the problem into smaller pieces.",
                "What patterns do you notice?",
            ],
            QuestType.ASSESSMENT: [
                "Show what you've learned.",
                "Work through it methodically.",
                "Trust your knowledge.",
            ],
            QuestType.EXPLORATION: [
                "Be curious and experiment.",
                "There's no single right answer here.",
                "Discover connections between concepts.",
            ],
        }

        hint_list = hints.get(quest.quest_type, ["Think carefully about the problem."])
        return random.choice(hint_list)


class NPCFactory:
    """Factory for creating standard educational NPCs"""

    @staticmethod
    def create_math_teacher(location_id: str = "school_math_classroom") -> NPCProfile:
        """Create a supportive math teacher"""
        return NPCProfile(
            npc_id="npc_math_teacher_001",
            name="Ms. Rodriguez",
            role=NPCRole.TEACHER,
            subjects=[Subject.MATH],
            teaching_style=TeachingStyle.SUPPORTIVE,
            personality_traits=["patient", "encouraging", "methodical"],
            backstory="Ms. Rodriguez has been teaching mathematics for 15 years. She believes every student can master math with the right support and approach.",
            appearance_description="A warm, middle-aged woman with glasses and a friendly smile. Often seen with a whiteboard marker in hand.",
            voice_style="warm and patient, speaks clearly and encouragingly",
            location_id=location_id,
            can_give_quests=True,
            quest_types=[QuestType.TUTORIAL, QuestType.PRACTICE, QuestType.CHALLENGE]
        )

    @staticmethod
    def create_science_mentor(location_id: str = "school_science_lab") -> NPCProfile:
        """Create a challenging science mentor"""
        return NPCProfile(
            npc_id="npc_science_mentor_001",
            name="Dr. Chen",
            role=NPCRole.MENTOR,
            subjects=[Subject.SCIENCE],
            teaching_style=TeachingStyle.CHALLENGING,
            personality_traits=["rigorous", "passionate", "demanding"],
            backstory="Dr. Chen is a research scientist who mentors talented students. She pushes students to think like scientists and question everything.",
            appearance_description="A sharp-eyed woman in a lab coat, always carrying a tablet with research data.",
            voice_style="direct and intense, speaks with scientific precision",
            location_id=location_id,
            can_give_quests=True,
            quest_types=[QuestType.CHALLENGE, QuestType.ASSESSMENT, QuestType.EXPLORATION]
        )

    @staticmethod
    def create_history_guide(location_id: str = "school_library") -> NPCProfile:
        """Create a socratic history guide"""
        return NPCProfile(
            npc_id="npc_history_guide_001",
            name="Mr. Thompson",
            role=NPCRole.GUIDE,
            subjects=[Subject.SOCIAL_STUDIES],
            teaching_style=TeachingStyle.SOCRATIC,
            personality_traits=["thoughtful", "questioning", "wise"],
            backstory="Mr. Thompson believes history is about asking the right questions, not memorizing dates. He guides students to discover historical truths themselves.",
            appearance_description="An older man with a thoughtful expression, surrounded by books and historical documents.",
            voice_style="calm and questioning, speaks in thought-provoking ways",
            location_id=location_id,
            can_give_quests=True,
            quest_types=[QuestType.EXPLORATION, QuestType.PRACTICE]
        )

    @staticmethod
    def create_ela_specialist(location_id: str = "school_library") -> NPCProfile:
        """Create a demonstrative English Language Arts specialist"""
        return NPCProfile(
            npc_id="npc_ela_specialist_001",
            name="Ms. Patel",
            role=NPCRole.SPECIALIST,
            subjects=[Subject.ELA],
            teaching_style=TeachingStyle.DEMONSTRATIVE,
            personality_traits=["articulate", "creative", "expressive"],
            backstory="Ms. Patel is a published author who teaches writing by showing students how stories are crafted, sentence by sentence.",
            appearance_description="An animated woman who gestures expressively, often reading passages aloud with dramatic flair.",
            voice_style="expressive and theatrical, demonstrates through vivid examples",
            location_id=location_id,
            can_give_quests=True,
            quest_types=[QuestType.TUTORIAL, QuestType.PRACTICE, QuestType.CHALLENGE]
        )

    @staticmethod
    def create_ai_coach(location_id: str = "school_computer_lab") -> NPCProfile:
        """Create a collaborative AI readiness coach"""
        return NPCProfile(
            npc_id="npc_ai_coach_001",
            name="Alex Kim",
            role=NPCRole.MENTOR,
            subjects=[Subject.AI_READINESS],
            teaching_style=TeachingStyle.COLLABORATIVE,
            personality_traits=["innovative", "collaborative", "tech-savvy"],
            backstory="Alex is a young AI engineer who believes in learning together. They see AI as a tool for collaboration, not replacement.",
            appearance_description="A young person with colorful tech accessories, always experimenting with new AI tools.",
            voice_style="casual and collaborative, speaks about 'we' not 'I'",
            location_id=location_id,
            can_give_quests=True,
            quest_types=[QuestType.TUTORIAL, QuestType.EXPLORATION, QuestType.CHALLENGE]
        )

    @staticmethod
    def create_peer_helper(
        name: str,
        location_id: str,
        subjects: List[Subject],
        teaching_style: TeachingStyle = TeachingStyle.COLLABORATIVE
    ) -> NPCProfile:
        """Create a peer tutor NPC"""
        return NPCProfile(
            npc_id=f"npc_peer_{name.lower().replace(' ', '_')}",
            name=name,
            role=NPCRole.PEER,
            subjects=subjects,
            teaching_style=teaching_style,
            personality_traits=["friendly", "relatable", "helpful"],
            backstory=f"{name} is a fellow student who excels in {subjects[0].value if subjects else 'various subjects'} and loves helping classmates.",
            appearance_description=f"A student around your age, wearing casual clothes and a welcoming smile.",
            voice_style="friendly and casual, speaks like a peer",
            location_id=location_id,
            can_give_quests=True,
            quest_types=[QuestType.PRACTICE, QuestType.TUTORIAL]
        )

    @staticmethod
    def create_fantasy_wizard(location_id: str = "fantasy_wizard_tower") -> NPCProfile:
        """Create a fantasy wizard for math (Fantasy World)"""
        return NPCProfile(
            npc_id="npc_wizard_numerus",
            name="Wizard Numerus",
            role=NPCRole.TEACHER,
            subjects=[Subject.MATH],
            teaching_style=TeachingStyle.DEMONSTRATIVE,
            personality_traits=["mystical", "wise", "theatrical"],
            backstory="Numerus the Wise has spent centuries mastering the magical art of numbers. He believes mathematics is the language of magic itself.",
            appearance_description="An elderly wizard in flowing robes covered with mathematical symbols. Numbers float around his staff.",
            voice_style="mystical and grand, speaks of math as if it's magic",
            location_id=location_id,
            can_give_quests=True,
            quest_types=[QuestType.TUTORIAL, QuestType.CHALLENGE, QuestType.EXPLORATION]
        )

    @staticmethod
    def create_space_engineer(location_id: str = "space_engineering_bay") -> NPCProfile:
        """Create a space engineer for math/science (Space Station)"""
        return NPCProfile(
            npc_id="npc_engineer_nova",
            name="Engineer Nova",
            role=NPCRole.MENTOR,
            subjects=[Subject.MATH, Subject.SCIENCE],
            teaching_style=TeachingStyle.CHALLENGING,
            personality_traits=["precise", "logical", "ambitious"],
            backstory="Engineer Nova designs starship systems where mistakes aren't an option. She pushes students to think with the precision space travel demands.",
            appearance_description="A sharp engineer in a sleek uniform, surrounded by holographic schematics and engineering tools.",
            voice_style="precise and technical, speaks in engineering terms",
            location_id=location_id,
            can_give_quests=True,
            quest_types=[QuestType.CHALLENGE, QuestType.ASSESSMENT, QuestType.EXPLORATION]
        )


class NPCRegistry:
    """Registry for managing all NPCs in the game"""

    def __init__(self):
        self.npcs: Dict[str, NPCProfile] = {}
        self.dialogue_managers: Dict[str, NPCDialogueManager] = {}
        self.npcs_by_location: Dict[str, List[NPCProfile]] = {}
        self.npcs_by_subject: Dict[Subject, List[NPCProfile]] = {}

    def register_npc(self, npc: NPCProfile) -> None:
        """Register an NPC in the system"""
        self.npcs[npc.npc_id] = npc
        self.dialogue_managers[npc.npc_id] = NPCDialogueManager(npc)

        # Index by location
        if npc.location_id not in self.npcs_by_location:
            self.npcs_by_location[npc.location_id] = []
        self.npcs_by_location[npc.location_id].append(npc)

        # Index by subject
        for subject in npc.subjects:
            if subject not in self.npcs_by_subject:
                self.npcs_by_subject[subject] = []
            self.npcs_by_subject[subject].append(npc)

    def get_npc(self, npc_id: str) -> Optional[NPCProfile]:
        """Get NPC by ID"""
        return self.npcs.get(npc_id)

    def get_npcs_at_location(self, location_id: str) -> List[NPCProfile]:
        """Get all NPCs at a location"""
        return self.npcs_by_location.get(location_id, [])

    def get_npcs_for_subject(self, subject: Subject) -> List[NPCProfile]:
        """Get all NPCs who teach a subject"""
        return self.npcs_by_subject.get(subject, [])

    def get_dialogue_manager(self, npc_id: str) -> Optional[NPCDialogueManager]:
        """Get dialogue manager for an NPC"""
        return self.dialogue_managers.get(npc_id)

    def generate_dialogue(
        self,
        npc_id: str,
        context: DialogueContext,
        player_performance: float = 0.5,
        **kwargs
    ) -> Optional[DialogueResponse]:
        """Generate dialogue from a specific NPC"""
        manager = self.get_dialogue_manager(npc_id)
        if manager:
            return manager.generate_dialogue(context, player_performance, **kwargs)
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            'total_npcs': len(self.npcs),
            'npcs_by_role': {
                role.value: sum(1 for npc in self.npcs.values() if npc.role == role)
                for role in NPCRole
            },
            'npcs_by_teaching_style': {
                style.value: sum(1 for npc in self.npcs.values() if npc.teaching_style == style)
                for style in TeachingStyle
            },
            'locations_covered': len(self.npcs_by_location),
            'subjects_covered': len(self.npcs_by_subject),
        }


def create_default_npc_registry() -> NPCRegistry:
    """Create a registry with all default NPCs"""
    registry = NPCRegistry()

    # School World NPCs
    registry.register_npc(NPCFactory.create_math_teacher("school_math_classroom"))
    registry.register_npc(NPCFactory.create_science_mentor("school_science_lab"))
    registry.register_npc(NPCFactory.create_history_guide("school_library"))
    registry.register_npc(NPCFactory.create_ela_specialist("school_library"))
    registry.register_npc(NPCFactory.create_ai_coach("school_computer_lab"))

    # Peer helpers
    registry.register_npc(NPCFactory.create_peer_helper(
        "Jamie", "school_cafeteria", [Subject.MATH, Subject.SCIENCE]
    ))
    registry.register_npc(NPCFactory.create_peer_helper(
        "Sam", "school_lobby", [Subject.ELA]
    ))

    # Fantasy World NPCs
    registry.register_npc(NPCFactory.create_fantasy_wizard("fantasy_wizard_tower"))

    # Space Station NPCs
    registry.register_npc(NPCFactory.create_space_engineer("space_engineering_bay"))

    return registry


# Demo function
def demo_npc_system():
    """Demonstrate the NPC system"""
    print("\n" + "="*60)
    print("EduVerse NPC System Demo")
    print("="*60 + "\n")

    # Create registry with default NPCs
    registry = create_default_npc_registry()

    # Show summary
    summary = registry.get_summary()
    print(f"Total NPCs: {summary['total_npcs']}")
    print(f"Locations covered: {summary['locations_covered']}")
    print(f"Subjects covered: {summary['subjects_covered']}")
    print()

    print("NPCs by Role:")
    for role, count in summary['npcs_by_role'].items():
        if count > 0:
            print(f"  {role}: {count}")
    print()

    print("NPCs by Teaching Style:")
    for style, count in summary['npcs_by_teaching_style'].items():
        if count > 0:
            print(f"  {style}: {count}")
    print()

    # Demonstrate dialogue generation
    print("\n" + "-"*60)
    print("Dialogue Examples")
    print("-"*60 + "\n")

    # Ms. Rodriguez (Supportive Math Teacher)
    print("1. Ms. Rodriguez (Supportive Math Teacher)")
    print("-" * 40)

    npc_id = "npc_math_teacher_001"

    # Greeting
    response = registry.generate_dialogue(npc_id, DialogueContext.GREETING)
    print(f"Greeting: {response.text}")

    # Struggle scenario
    response = registry.generate_dialogue(npc_id, DialogueContext.STRUGGLE, player_performance=0.3)
    print(f"Student Struggling (performance=0.3): {response.text}")

    # Success scenario
    response = registry.generate_dialogue(npc_id, DialogueContext.SUCCESS, player_performance=0.9)
    print(f"Student Success (performance=0.9): {response.text}")
    print()

    # Dr. Chen (Challenging Science Mentor)
    print("2. Dr. Chen (Challenging Science Mentor)")
    print("-" * 40)

    npc_id = "npc_science_mentor_001"

    response = registry.generate_dialogue(npc_id, DialogueContext.GREETING)
    print(f"Greeting: {response.text}")

    response = registry.generate_dialogue(npc_id, DialogueContext.STRUGGLE, player_performance=0.3)
    print(f"Student Struggling (performance=0.3): {response.text}")

    response = registry.generate_dialogue(npc_id, DialogueContext.SUCCESS, player_performance=0.9)
    print(f"Student Success (performance=0.9): {response.text}")
    print()

    # Alex Kim (Collaborative AI Coach)
    print("3. Alex Kim (Collaborative AI Coach)")
    print("-" * 40)

    npc_id = "npc_ai_coach_001"

    response = registry.generate_dialogue(npc_id, DialogueContext.GREETING)
    print(f"Greeting: {response.text}")

    response = registry.generate_dialogue(npc_id, DialogueContext.QUEST_OFFER)
    print(f"Quest Offer: {response.text}")

    # Hint request
    response = registry.generate_dialogue(
        npc_id,
        DialogueContext.HINT_REQUEST,
        hint_text="Think about how AI learns from data patterns"
    )
    print(f"Hint: {response.text}")
    print()

    # Wizard Numerus (Fantasy Math Teacher)
    print("4. Wizard Numerus (Fantasy World)")
    print("-" * 40)

    npc_id = "npc_wizard_numerus"

    response = registry.generate_dialogue(npc_id, DialogueContext.GREETING)
    print(f"Greeting: {response.text}")

    response = registry.generate_dialogue(npc_id, DialogueContext.SUCCESS, player_performance=0.85)
    print(f"Student Success: {response.text}")
    print()

    # Show NPCs by location
    print("\n" + "-"*60)
    print("NPCs by Location")
    print("-"*60 + "\n")

    locations = [
        "school_math_classroom",
        "school_library",
        "fantasy_wizard_tower",
        "space_engineering_bay"
    ]

    for loc in locations:
        npcs = registry.get_npcs_at_location(loc)
        if npcs:
            print(f"{loc}:")
            for npc in npcs:
                print(f"  - {npc.name} ({npc.role.value}, {npc.teaching_style.value})")
            print()

    # Show NPCs by subject
    print("-"*60)
    print("NPCs by Subject")
    print("-"*60 + "\n")

    for subject in [Subject.MATH, Subject.SCIENCE, Subject.AI_READINESS]:
        npcs = registry.get_npcs_for_subject(subject)
        if npcs:
            print(f"{subject.value}:")
            for npc in npcs:
                print(f"  - {npc.name} ({npc.teaching_style.value})")
            print()

    print("\n" + "="*60)
    print("NPC system ready! NPCs can now guide students through quests.")
    print("="*60)


if __name__ == "__main__":
    demo_npc_system()
