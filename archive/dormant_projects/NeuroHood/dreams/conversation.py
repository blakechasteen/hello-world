"""
Conversation System - Natural Dialogue Generation

Generates conversations between residents based on:
- Personalities and speaking styles
- Relationship depth
- Current emotional states
- Shared interests and memories
- Context (location, time, events)

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random
from datetime import datetime

from resident import Resident, Relationship, RelationshipType


class ConversationTone(Enum):
    """Overall tone of conversation."""
    CASUAL = "casual"
    WARM = "warm"
    TENSE = "tense"
    DEEP = "deep"
    PLAYFUL = "playful"
    SUPPORTIVE = "supportive"
    AWKWARD = "awkward"


class TopicCategory(Enum):
    """Categories of conversation topics."""
    GREETING = "greeting"
    SMALL_TALK = "small_talk"
    PERSONAL = "personal"
    SHARED_INTEREST = "shared_interest"
    NEIGHBORHOOD = "neighborhood"
    EMOTIONAL = "emotional"
    DREAMS = "dreams"
    MEMORIES = "memories"
    PHILOSOPHY = "philosophy"
    PRACTICAL = "practical"
    FAREWELL = "farewell"


@dataclass
class DialogueLine:
    """A single line of dialogue."""
    speaker_id: str
    speaker_name: str
    text: str
    emotion: str = "neutral"
    action: str = ""  # Optional action description
    internal_thought: str = ""  # What they're really thinking


@dataclass
class ConversationContext:
    """Context for a conversation."""
    location: str
    time_of_day: str  # morning, afternoon, evening, night
    weather: str = "pleasant"
    recent_events: List[str] = field(default_factory=list)
    interruption_chance: float = 0.1
    user_present: bool = False


@dataclass
class Conversation:
    """A complete conversation between residents."""
    participants: List[str]
    location: str
    start_time: datetime
    lines: List[DialogueLine] = field(default_factory=list)
    tone: ConversationTone = ConversationTone.CASUAL
    topics_covered: List[str] = field(default_factory=list)
    emotional_impact: Dict[str, Dict[str, float]] = field(default_factory=dict)
    is_complete: bool = False

    def add_line(self, line: DialogueLine):
        self.lines.append(line)

    def summary(self) -> str:
        return f"Conversation at {self.location}: {len(self.lines)} exchanges about {', '.join(self.topics_covered[:3])}"


class ConversationGenerator:
    """Generates natural conversations between residents."""

    def __init__(self):
        self.topic_templates = self._load_topic_templates()
        self.response_templates = self._load_response_templates()
        self.emotional_responses = self._load_emotional_responses()

    def _load_topic_templates(self) -> Dict[str, List[str]]:
        """Load conversation topic templates."""
        return {
            "weather": [
                "Beautiful day, isn't it?",
                "Can you believe this weather?",
                "Looks like {weather} today.",
                "Perfect weather for {activity}.",
            ],
            "neighborhood": [
                "Did you hear about {event}?",
                "Have you seen the new {thing} on {street}?",
                "The neighborhood sure has changed lately.",
                "I love how quiet it is here in the {time}.",
            ],
            "personal_wellbeing": [
                "How have you been lately?",
                "You seem {mood} today. Everything okay?",
                "I've been thinking about you. How are things?",
            ],
            "shared_interest": [
                "Have you been doing any {hobby} lately?",
                "I just {activity} and thought of you!",
                "Remember when we talked about {topic}?",
            ],
            "dreams": [
                "I had the strangest dream last night...",
                "Do you ever remember your dreams?",
                "I keep dreaming about {symbol}. What do you think it means?",
            ],
            "memories": [
                "Remember when {event}?",
                "This reminds me of that time we {activity}.",
                "I was just thinking about {memory}.",
            ],
            "philosophy": [
                "Do you ever wonder about {topic}?",
                "What do you think happens when {scenario}?",
                "I've been thinking a lot about {concept}.",
            ],
            "practical": [
                "Could you help me with {task}?",
                "Do you know anyone who {need}?",
                "I'm looking for advice about {topic}.",
            ],
        }

    def _load_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load response templates by personality trait."""
        return {
            "agreeable": {
                "agreement": [
                    "Oh, absolutely!",
                    "I couldn't agree more.",
                    "You're so right about that.",
                    "Yes! I feel the same way.",
                ],
                "sympathy": [
                    "That must be really hard.",
                    "I'm so sorry to hear that.",
                    "Is there anything I can do to help?",
                    "I understand how you feel.",
                ],
                "interest": [
                    "Tell me more!",
                    "That's fascinating!",
                    "I love hearing about that.",
                    "How wonderful!",
                ],
            },
            "reserved": {
                "agreement": [
                    "I see.",
                    "That makes sense.",
                    "Hmm, interesting point.",
                ],
                "sympathy": [
                    "That's tough.",
                    "I understand.",
                    "Things will get better.",
                ],
                "interest": [
                    "Interesting.",
                    "I hadn't thought of that.",
                    "Go on.",
                ],
            },
            "enthusiastic": {
                "agreement": [
                    "YES! Exactly!",
                    "Oh my gosh, totally!",
                    "That's EXACTLY what I think!",
                ],
                "sympathy": [
                    "Oh no! That's terrible!",
                    "I can't even imagine!",
                    "Let me give you a hug!",
                ],
                "interest": [
                    "Wait, really?! Tell me everything!",
                    "No way! That's amazing!",
                    "I'm dying to know more!",
                ],
            },
            "thoughtful": {
                "agreement": [
                    "You raise a good point.",
                    "I've been thinking something similar.",
                    "That's a thoughtful observation.",
                ],
                "sympathy": [
                    "I can see how that affects you.",
                    "These things take time to process.",
                    "Would you like to talk about it?",
                ],
                "interest": [
                    "That's worth exploring further.",
                    "I'd like to understand better.",
                    "What led you to that?",
                ],
            },
        }

    def _load_emotional_responses(self) -> Dict[str, List[str]]:
        """Load emotional response templates."""
        return {
            "happy": [
                "*smiles warmly*",
                "*laughs*",
                "*eyes light up*",
            ],
            "sad": [
                "*sighs*",
                "*looks down*",
                "*voice softens*",
            ],
            "anxious": [
                "*fidgets*",
                "*glances around*",
                "*speaks quickly*",
            ],
            "excited": [
                "*gestures enthusiastically*",
                "*leans forward*",
                "*can barely contain excitement*",
            ],
            "thoughtful": [
                "*pauses to think*",
                "*gazes into distance*",
                "*nods slowly*",
            ],
            "surprised": [
                "*raises eyebrows*",
                "*gasps*",
                "*steps back*",
            ],
        }

    def get_speaking_style(self, resident: Resident) -> str:
        """Determine speaking style category."""
        if resident.personality.agreeableness > 0.7:
            return "agreeable"
        elif resident.personality.extraversion < 0.4:
            return "reserved"
        elif resident.personality.extraversion > 0.7:
            return "enthusiastic"
        else:
            return "thoughtful"

    def select_topic(
        self,
        speaker: Resident,
        listener: Resident,
        relationship: Relationship,
        context: ConversationContext,
        covered_topics: List[str],
    ) -> Tuple[str, str]:
        """Select appropriate conversation topic."""
        # Find shared interests
        shared_interests = set(speaker.favorite_topics) & set(listener.favorite_topics)

        # Weight topics by appropriateness
        topic_weights = {
            "weather": 0.3,
            "neighborhood": 0.4,
            "personal_wellbeing": 0.0,
            "shared_interest": 0.0,
            "dreams": 0.0,
            "memories": 0.0,
            "philosophy": 0.0,
            "practical": 0.2,
        }

        # Increase personal topics if relationship is close
        if relationship.familiarity > 0.5:
            topic_weights["personal_wellbeing"] = 0.4
            topic_weights["memories"] = 0.3

        if relationship.trust > 0.6:
            topic_weights["dreams"] = 0.3
            topic_weights["emotional"] = 0.3

        if shared_interests:
            topic_weights["shared_interest"] = 0.5

        # High openness increases philosophy
        if speaker.personality.openness > 0.7:
            topic_weights["philosophy"] = 0.4
            topic_weights["dreams"] = 0.4

        # Social need increases personal topics
        if speaker.emotional_state.social_need > 0.6:
            topic_weights["personal_wellbeing"] = 0.5

        # Don't repeat topics
        for topic in covered_topics:
            if topic in topic_weights:
                topic_weights[topic] *= 0.3

        # Weighted random selection
        total = sum(topic_weights.values())
        if total == 0:
            return "small_talk", "So, how's your day going?"

        r = random.random() * total
        cumulative = 0
        selected_category = "small_talk"

        for topic, weight in topic_weights.items():
            cumulative += weight
            if r <= cumulative:
                selected_category = topic
                break

        # Generate specific topic content
        topic_text = self._generate_topic_text(
            selected_category, speaker, listener, shared_interests, context
        )

        return selected_category, topic_text

    def _generate_topic_text(
        self,
        category: str,
        speaker: Resident,
        listener: Resident,
        shared_interests: set,
        context: ConversationContext,
    ) -> str:
        """Generate specific topic text."""
        templates = self.topic_templates.get(category, ["How are you?"])
        template = random.choice(templates)

        # Fill in template variables
        replacements = {
            "{weather}": context.weather,
            "{time}": context.time_of_day,
            "{activity}": random.choice(speaker.schedule.hobbies) if speaker.schedule.hobbies else "relaxing",
            "{hobby}": random.choice(list(shared_interests)) if shared_interests else "anything fun",
            "{topic}": random.choice(speaker.favorite_topics),
            "{mood}": speaker.emotional_state.mood_description(),
            "{event}": random.choice(["the new cafe", "the street fair", "the garden expansion"]),
            "{thing}": random.choice(["mural", "shop", "bench"]),
            "{street}": "Oak Street",
            "{symbol}": random.choice(["water", "flying", "lost paths", "old friends"]),
            "{task}": random.choice(["moving some furniture", "fixing something", "planning an event"]),
            "{need}": random.choice(["can fix things", "knows about plants", "plays music"]),
            "{concept}": random.choice(["time", "change", "community", "dreams"]),
            "{scenario}": random.choice(["we're gone", "everything changes", "we follow our dreams"]),
            "{memory}": "old times",
        }

        for key, value in replacements.items():
            template = template.replace(key, value)

        return template

    def generate_response(
        self,
        speaker: Resident,
        listener: Resident,
        previous_line: DialogueLine,
        relationship: Relationship,
        topic_category: str,
    ) -> DialogueLine:
        """Generate a response to a line of dialogue."""
        style = self.get_speaking_style(speaker)
        responses = self.response_templates.get(style, self.response_templates["thoughtful"])

        # Determine response type based on previous line content
        if "?" in previous_line.text:
            # It's a question - give an answer
            response_text = self._generate_answer(speaker, listener, previous_line, topic_category)
        elif any(word in previous_line.text.lower() for word in ["sorry", "hard", "difficult", "sad"]):
            # Sympathetic response
            base = random.choice(responses["sympathy"])
            addition = self._add_personal_touch(speaker, listener, "sympathy")
            response_text = f"{base} {addition}"
        elif any(word in previous_line.text.lower() for word in ["amazing", "wonderful", "great", "love"]):
            # Enthusiastic agreement
            base = random.choice(responses["interest"])
            addition = self._add_personal_touch(speaker, listener, "interest")
            response_text = f"{base} {addition}"
        else:
            # General response
            base = random.choice(responses["agreement"])
            addition = self._add_personal_touch(speaker, listener, "agreement")
            response_text = f"{base} {addition}"

        # Determine emotion based on speaker state
        emotion = self._determine_emotion(speaker)
        action = random.choice(self.emotional_responses.get(emotion, [""]))

        return DialogueLine(
            speaker_id=speaker.resident_id,
            speaker_name=speaker.name,
            text=response_text,
            emotion=emotion,
            action=action,
        )

    def _generate_answer(
        self,
        speaker: Resident,
        listener: Resident,
        question: DialogueLine,
        topic_category: str,
    ) -> str:
        """Generate an answer to a question."""
        question_text = question.text.lower()

        # How are you questions
        if any(phrase in question_text for phrase in ["how are you", "how have you been", "how's it going"]):
            mood = speaker.emotional_state.mood_description()
            if speaker.emotional_state.happiness > 0.6:
                return random.choice([
                    f"I'm doing really well, actually! Feeling {mood}.",
                    f"Great! Things have been {mood}.",
                    f"Can't complain! Life is good.",
                ])
            elif speaker.emotional_state.happiness < 0.4:
                return random.choice([
                    f"I've been better, to be honest. Feeling a bit {mood}.",
                    f"It's been a tough week. But hanging in there.",
                    f"So-so. You know how it is sometimes.",
                ])
            else:
                return random.choice([
                    f"Not bad, not bad. Just the usual.",
                    f"I'm alright. How about you?",
                    f"Pretty good! Keeping busy.",
                ])

        # Dreams questions
        if "dream" in question_text:
            return random.choice([
                "I had this strange dream about wandering through unfamiliar streets...",
                "Actually, yes! Last night I dreamed about flying over the neighborhood.",
                "My dreams have been vivid lately. Lots of water imagery.",
                "I keep having this recurring dream about finding a hidden door...",
            ])

        # Opinion questions
        if any(phrase in question_text for phrase in ["what do you think", "do you think", "your opinion"]):
            if speaker.personality.openness > 0.6:
                return random.choice([
                    "That's a fascinating question. I think there's more to it than meets the eye.",
                    "I've thought about that a lot. I believe everything is connected somehow.",
                    "My view is probably unconventional, but I think we should embrace change.",
                ])
            else:
                return random.choice([
                    "I think the practical approach is usually best.",
                    "In my experience, simple solutions work better.",
                    "I try not to overthink these things.",
                ])

        # Generic answer
        return random.choice([
            "That's a good question. Let me think...",
            "Well, it depends on how you look at it.",
            "I'm not sure, but I have some thoughts on that.",
        ])

    def _add_personal_touch(
        self,
        speaker: Resident,
        listener: Resident,
        response_type: str,
    ) -> str:
        """Add a personal touch based on personality and relationship."""
        rel = speaker.get_relationship(listener.resident_id)

        additions = {
            "agreement": [
                "",
                f"I was just telling someone the same thing.",
                f"That reminds me of something...",
            ],
            "sympathy": [
                "",
                f"If you ever need to talk, I'm here.",
                f"These things have a way of working out.",
            ],
            "interest": [
                "",
                f"You always have such interesting stories!",
                f"I'd love to hear more sometime.",
            ],
        }

        # Close relationships get more personal additions
        if rel.familiarity > 0.6:
            additions["agreement"].append(f"You always get it, {listener.name}.")
            additions["sympathy"].append(f"You know I'm always here for you.")
            additions["interest"].append(f"This is why I love talking to you!")

        return random.choice(additions.get(response_type, [""]))

    def _determine_emotion(self, speaker: Resident) -> str:
        """Determine current emotional expression."""
        state = speaker.emotional_state

        if state.happiness > 0.7:
            return "happy"
        elif state.happiness < 0.3:
            return "sad"
        elif state.anxiety > 0.6:
            return "anxious"
        elif state.energy > 0.7:
            return "excited"
        else:
            return "thoughtful"

    def generate_conversation(
        self,
        resident1: Resident,
        resident2: Resident,
        context: ConversationContext,
        min_exchanges: int = 3,
        max_exchanges: int = 8,
    ) -> Conversation:
        """Generate a complete conversation between two residents."""
        rel1 = resident1.get_relationship(resident2.resident_id)
        rel2 = resident2.get_relationship(resident1.resident_id)

        # Determine conversation tone
        avg_quality = (rel1.quality() + rel2.quality()) / 2
        if avg_quality > 0.7:
            tone = ConversationTone.WARM
        elif avg_quality < 0.3:
            tone = ConversationTone.AWKWARD
        else:
            tone = ConversationTone.CASUAL

        conversation = Conversation(
            participants=[resident1.resident_id, resident2.resident_id],
            location=context.location,
            start_time=datetime.now(),
            tone=tone,
        )

        # Opening greeting
        greeting1 = DialogueLine(
            speaker_id=resident1.resident_id,
            speaker_name=resident1.name,
            text=resident1.get_greeting(resident2.name, rel1),
            emotion=self._determine_emotion(resident1),
        )
        conversation.add_line(greeting1)

        greeting2 = DialogueLine(
            speaker_id=resident2.resident_id,
            speaker_name=resident2.name,
            text=resident2.get_greeting(resident1.name, rel2),
            emotion=self._determine_emotion(resident2),
        )
        conversation.add_line(greeting2)

        # Main conversation exchanges
        num_exchanges = random.randint(min_exchanges, max_exchanges)
        current_speaker = resident1
        other = resident2

        for i in range(num_exchanges):
            # Alternate speakers
            if i > 0:
                current_speaker, other = other, current_speaker

            rel = current_speaker.get_relationship(other.resident_id)

            # Select topic or continue
            if i % 2 == 0 or random.random() > 0.6:
                # Start new topic
                topic_cat, topic_text = self.select_topic(
                    current_speaker, other, rel, context, conversation.topics_covered
                )
                conversation.topics_covered.append(topic_cat)

                line = DialogueLine(
                    speaker_id=current_speaker.resident_id,
                    speaker_name=current_speaker.name,
                    text=topic_text,
                    emotion=self._determine_emotion(current_speaker),
                )
            else:
                # Respond to previous
                line = self.generate_response(
                    current_speaker, other,
                    conversation.lines[-1],
                    rel,
                    conversation.topics_covered[-1] if conversation.topics_covered else "small_talk",
                )

            conversation.add_line(line)

            # Check for interruption
            if random.random() < context.interruption_chance:
                break

        # Farewell
        farewells = [
            "Well, I should get going. Good to see you!",
            "I need to run, but let's catch up again soon!",
            "Take care! See you around!",
            "I'll let you go. Have a great day!",
        ]

        farewell = DialogueLine(
            speaker_id=current_speaker.resident_id,
            speaker_name=current_speaker.name,
            text=random.choice(farewells),
            emotion="happy",
        )
        conversation.add_line(farewell)

        conversation.is_complete = True

        # Calculate emotional impact
        for participant_id in conversation.participants:
            conversation.emotional_impact[participant_id] = {
                "happiness": 0.05 * len(conversation.lines) / 10,
                "social_need": -0.2,
                "loneliness": -0.15,
            }

        return conversation

    def generate_user_response_options(
        self,
        user_context: Dict,
        other_resident: Resident,
        last_line: DialogueLine,
        topic: str,
    ) -> List[str]:
        """Generate response options for user participation."""
        options = []

        # Friendly response
        options.append(self._generate_friendly_response(last_line, topic))

        # Curious response (ask question)
        options.append(self._generate_curious_response(last_line, topic))

        # Supportive response
        options.append(self._generate_supportive_response(last_line, topic))

        # Share personal experience
        options.append(self._generate_personal_response(last_line, topic))

        return options

    def _generate_friendly_response(self, last_line: DialogueLine, topic: str) -> str:
        """Generate a friendly response option."""
        return random.choice([
            "That's really interesting! I've been thinking about that too.",
            "I love how you see things. Tell me more!",
            "You always have such great perspectives.",
        ])

    def _generate_curious_response(self, last_line: DialogueLine, topic: str) -> str:
        """Generate a curious/questioning response option."""
        return random.choice([
            "What made you think about that?",
            "How long have you felt that way?",
            "Do you think that's common around here?",
        ])

    def _generate_supportive_response(self, last_line: DialogueLine, topic: str) -> str:
        """Generate a supportive response option."""
        return random.choice([
            "I understand completely. It's not easy.",
            "That makes total sense. I'm here if you want to talk more.",
            "You're handling it really well.",
        ])

    def _generate_personal_response(self, last_line: DialogueLine, topic: str) -> str:
        """Generate a personal sharing response option."""
        return random.choice([
            "That reminds me of something similar that happened to me...",
            "I've been experiencing something like that too lately.",
            "You know, I had a dream about something similar...",
        ])


# Demo
if __name__ == "__main__":
    from resident import create_neighborhood_residents

    print("=" * 70)
    print("CONVERSATION SYSTEM DEMO")
    print("=" * 70)

    residents = create_neighborhood_residents()
    generator = ConversationGenerator()

    maya = residents["maya_001"]
    jorge = residents["jorge_002"]

    context = ConversationContext(
        location="community_garden",
        time_of_day="morning",
        weather="sunny",
    )

    print(f"\n--- Location: {context.location} ({context.time_of_day}, {context.weather}) ---\n")

    conversation = generator.generate_conversation(maya, jorge, context)

    print(f"[Conversation Tone: {conversation.tone.value}]")
    print(f"[Topics: {', '.join(conversation.topics_covered)}]\n")

    for line in conversation.lines:
        if line.action:
            print(f"  {line.action}")
        print(f"{line.speaker_name}: \"{line.text}\"")
        print()

    print("-" * 70)
    print(conversation.summary())
