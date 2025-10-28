#!/usr/bin/env python3
"""
Universal Journey Mappings - Domain-Specific Stage Definitions
================================================================

Defines stage mappings for multiple journey types:
- Hero's Journey (Campbell - Mythology)
- Business Journey (Startup/Scale)
- Learning Journey (Skill Mastery)
- Scientific Journey (Research Process)
- Personal Journey (Growth/Transformation)
- Product Journey (Development Lifecycle)

Each journey has:
- 12 stages (aligned with Campbell's structure)
- Stage-specific keywords for detection
- Color coding for visualization
- Universal pattern alignments
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class JourneyStage:
    """Single stage in a journey."""
    name: str
    keywords: List[str]
    description: str
    universal_pattern: str  # Maps to cross-domain archetype

@dataclass
class JourneyDefinition:
    """Complete journey definition."""
    id: str
    name: str
    domain: str
    color: str  # Hex color for visualization
    stages: List[JourneyStage]

    def get_stage_names(self) -> List[str]:
        return [stage.name for stage in self.stages]

    def get_keywords(self, stage_name: str) -> List[str]:
        for stage in self.stages:
            if stage.name == stage_name:
                return stage.keywords
        return []


# ============================================================================
# JOURNEY DEFINITIONS
# ============================================================================

HERO_JOURNEY = JourneyDefinition(
    id="hero",
    name="Hero's Journey",
    domain="Mythology",
    color="#F97316",  # Orange - Fire/transformation
    stages=[
        JourneyStage(
            name="Ordinary World",
            keywords=["normal", "ordinary", "daily", "routine", "mundane", "familiar", "home", "comfort"],
            description="The hero's normal life before the adventure",
            universal_pattern="STATUS_QUO"
        ),
        JourneyStage(
            name="Call to Adventure",
            keywords=["call", "opportunity", "challenge", "invitation", "discovery", "news", "message", "destiny"],
            description="Something disrupts the ordinary world",
            universal_pattern="CATALYST"
        ),
        JourneyStage(
            name="Refusal of Call",
            keywords=["refuse", "reject", "fear", "doubt", "hesitate", "resist", "deny", "avoid"],
            description="The hero resists change out of fear",
            universal_pattern="RESISTANCE"
        ),
        JourneyStage(
            name="Meeting Mentor",
            keywords=["mentor", "guide", "teacher", "advisor", "wisdom", "gift", "training", "counsel"],
            description="A wise figure provides guidance",
            universal_pattern="GUIDANCE"
        ),
        JourneyStage(
            name="Crossing Threshold",
            keywords=["threshold", "crossing", "departure", "commit", "leave", "enter", "begin", "embark"],
            description="The hero commits to the journey",
            universal_pattern="COMMITMENT"
        ),
        JourneyStage(
            name="Tests, Allies, Enemies",
            keywords=["test", "trial", "ally", "friend", "enemy", "foe", "challenge", "obstacle"],
            description="The hero learns the rules of the new world",
            universal_pattern="EXPLORATION"
        ),
        JourneyStage(
            name="Approach Inmost Cave",
            keywords=["approach", "prepare", "inner", "danger", "lair", "fortress", "gather"],
            description="Preparation for the major challenge",
            universal_pattern="PREPARATION"
        ),
        JourneyStage(
            name="Ordeal",
            keywords=["ordeal", "crisis", "death", "defeat", "battle", "confrontation", "darkest", "lowest"],
            description="The hero faces their greatest fear",
            universal_pattern="CRISIS"
        ),
        JourneyStage(
            name="Reward",
            keywords=["reward", "prize", "treasure", "victory", "achievement", "seize", "claim", "win"],
            description="The hero claims their reward",
            universal_pattern="BREAKTHROUGH"
        ),
        JourneyStage(
            name="Road Back",
            keywords=["return", "escape", "pursue", "chase", "consequence", "road back", "journey home"],
            description="The hero begins the return journey",
            universal_pattern="REINTEGRATION"
        ),
        JourneyStage(
            name="Resurrection",
            keywords=["resurrection", "rebirth", "final test", "climax", "purification", "transformation"],
            description="The hero undergoes final transformation",
            universal_pattern="TRANSFORMATION"
        ),
        JourneyStage(
            name="Return with Elixir",
            keywords=["elixir", "gift", "wisdom", "change", "share", "new life", "complete"],
            description="The hero returns with newfound wisdom",
            universal_pattern="COMPLETION"
        )
    ]
)

BUSINESS_JOURNEY = JourneyDefinition(
    id="business",
    name="Business Journey",
    domain="Startup/Scale",
    color="#10B981",  # Green - Growth/prosperity
    stages=[
        JourneyStage(
            name="Ideation",
            keywords=["idea", "concept", "brainstorm", "vision", "problem", "opportunity", "inspiration"],
            description="The initial business idea emerges",
            universal_pattern="STATUS_QUO"
        ),
        JourneyStage(
            name="Validation",
            keywords=["validate", "research", "customer", "feedback", "interview", "survey", "pain point"],
            description="Testing if the idea solves a real problem",
            universal_pattern="CATALYST"
        ),
        JourneyStage(
            name="Doubt & Fear",
            keywords=["doubt", "risk", "uncertain", "competitor", "failure", "quit", "imposter"],
            description="Confronting fears about starting",
            universal_pattern="RESISTANCE"
        ),
        JourneyStage(
            name="Advisor/Investor",
            keywords=["advisor", "mentor", "investor", "accelerator", "coach", "funding", "guidance"],
            description="Finding experienced guidance and resources",
            universal_pattern="GUIDANCE"
        ),
        JourneyStage(
            name="MVP Launch",
            keywords=["mvp", "launch", "product", "beta", "release", "deploy", "ship", "live"],
            description="Committing with a minimum viable product",
            universal_pattern="COMMITMENT"
        ),
        JourneyStage(
            name="Early Traction",
            keywords=["traction", "users", "customers", "growth", "iterate", "pivot", "learn", "adapt"],
            description="Learning from early users and competitors",
            universal_pattern="EXPLORATION"
        ),
        JourneyStage(
            name="Preparing to Scale",
            keywords=["scale", "hire", "infrastructure", "process", "fundraise", "prepare", "optimize"],
            description="Building capacity for growth",
            universal_pattern="PREPARATION"
        ),
        JourneyStage(
            name="Cash Crunch",
            keywords=["runway", "burn", "crisis", "layoff", "cash", "survival", "emergency", "critical"],
            description="Facing existential business challenges",
            universal_pattern="CRISIS"
        ),
        JourneyStage(
            name="Product-Market Fit",
            keywords=["fit", "traction", "retention", "metrics", "revenue", "growth", "success", "momentum"],
            description="Achieving sustainable business model",
            universal_pattern="BREAKTHROUGH"
        ),
        JourneyStage(
            name="Scaling Operations",
            keywords=["expand", "hire", "market", "sales", "operations", "execute", "deliver"],
            description="Rapid growth and expansion",
            universal_pattern="REINTEGRATION"
        ),
        JourneyStage(
            name="Market Leadership",
            keywords=["leader", "market share", "dominant", "mature", "optimize", "innovate", "defend"],
            description="Becoming an industry leader",
            universal_pattern="TRANSFORMATION"
        ),
        JourneyStage(
            name="Exit or Legacy",
            keywords=["exit", "acquire", "ipo", "legacy", "succession", "impact", "next chapter"],
            description="Creating lasting value and impact",
            universal_pattern="COMPLETION"
        )
    ]
)

LEARNING_JOURNEY = JourneyDefinition(
    id="learning",
    name="Learning Journey",
    domain="Skill Mastery",
    color="#3B82F6",  # Blue - Knowledge/wisdom
    stages=[
        JourneyStage(
            name="Unconscious Incompetence",
            keywords=["unaware", "ignorant", "naive", "don't know", "beginner", "novice"],
            description="Not knowing what you don't know",
            universal_pattern="STATUS_QUO"
        ),
        JourneyStage(
            name="Awareness",
            keywords=["aware", "realize", "discover", "interest", "curiosity", "intrigued", "fascinated"],
            description="Becoming aware of the skill gap",
            universal_pattern="CATALYST"
        ),
        JourneyStage(
            name="Overwhelm",
            keywords=["overwhelm", "difficult", "complex", "too hard", "frustrate", "intimidate", "confused"],
            description="Facing the difficulty of learning",
            universal_pattern="RESISTANCE"
        ),
        JourneyStage(
            name="Finding Teacher",
            keywords=["teacher", "course", "tutorial", "book", "resource", "learn from", "study"],
            description="Finding learning resources and guidance",
            universal_pattern="GUIDANCE"
        ),
        JourneyStage(
            name="Commitment to Practice",
            keywords=["practice", "commit", "discipline", "schedule", "dedicate", "regular", "habit"],
            description="Committing to regular practice",
            universal_pattern="COMMITMENT"
        ),
        JourneyStage(
            name="Deliberate Practice",
            keywords=["practice", "exercise", "drill", "repeat", "improve", "feedback", "progress"],
            description="Active skill development and refinement",
            universal_pattern="EXPLORATION"
        ),
        JourneyStage(
            name="Plateau Preparation",
            keywords=["plateau", "stuck", "ready", "next level", "advanced", "deeper"],
            description="Preparing to break through learning plateau",
            universal_pattern="PREPARATION"
        ),
        JourneyStage(
            name="Learning Crisis",
            keywords=["crisis", "failure", "mistake", "setback", "discourage", "quit", "doubt"],
            description="Confronting major setbacks or failures",
            universal_pattern="CRISIS"
        ),
        JourneyStage(
            name="Breakthrough",
            keywords=["breakthrough", "aha", "click", "understand", "competent", "achieve", "milestone"],
            description="Achieving true competence",
            universal_pattern="BREAKTHROUGH"
        ),
        JourneyStage(
            name="Application",
            keywords=["apply", "use", "implement", "project", "real world", "practice", "integrate"],
            description="Applying skills in real contexts",
            universal_pattern="REINTEGRATION"
        ),
        JourneyStage(
            name="Mastery",
            keywords=["master", "expert", "intuitive", "effortless", "fluent", "automatic", "unconscious"],
            description="Achieving unconscious competence",
            universal_pattern="TRANSFORMATION"
        ),
        JourneyStage(
            name="Teaching Others",
            keywords=["teach", "mentor", "share", "guide", "help", "contribute", "legacy"],
            description="Passing knowledge to the next generation",
            universal_pattern="COMPLETION"
        )
    ]
)

SCIENTIFIC_JOURNEY = JourneyDefinition(
    id="scientific",
    name="Scientific Journey",
    domain="Research Process",
    color="#8B5CF6",  # Purple - Discovery/mystery
    stages=[
        JourneyStage(
            name="Observation",
            keywords=["observe", "notice", "pattern", "phenomenon", "data", "existing", "current"],
            description="Noticing patterns in the world",
            universal_pattern="STATUS_QUO"
        ),
        JourneyStage(
            name="Question",
            keywords=["question", "why", "how", "wonder", "curious", "mystery", "unknown"],
            description="Formulating the research question",
            universal_pattern="CATALYST"
        ),
        JourneyStage(
            name="Doubt & Skepticism",
            keywords=["doubt", "skeptical", "impossible", "reject", "controversial", "criticize"],
            description="Facing skepticism about the question",
            universal_pattern="RESISTANCE"
        ),
        JourneyStage(
            name="Literature Review",
            keywords=["literature", "review", "research", "prior", "study", "reference", "citation"],
            description="Learning from existing research",
            universal_pattern="GUIDANCE"
        ),
        JourneyStage(
            name="Hypothesis",
            keywords=["hypothesis", "predict", "theory", "propose", "postulate", "assume", "expect"],
            description="Committing to a testable prediction",
            universal_pattern="COMMITMENT"
        ),
        JourneyStage(
            name="Experimental Design",
            keywords=["design", "method", "protocol", "experiment", "test", "measure", "control"],
            description="Designing the investigation",
            universal_pattern="EXPLORATION"
        ),
        JourneyStage(
            name="Preparation",
            keywords=["prepare", "setup", "equipment", "materials", "plan", "ready", "before"],
            description="Preparing for data collection",
            universal_pattern="PREPARATION"
        ),
        JourneyStage(
            name="Failed Experiments",
            keywords=["fail", "wrong", "error", "mistake", "null", "insignificant", "unexpected"],
            description="Facing experimental failures",
            universal_pattern="CRISIS"
        ),
        JourneyStage(
            name="Discovery",
            keywords=["discover", "find", "breakthrough", "result", "significant", "evidence", "proof"],
            description="Making the key finding",
            universal_pattern="BREAKTHROUGH"
        ),
        JourneyStage(
            name="Analysis",
            keywords=["analyze", "interpret", "explain", "understand", "meaning", "significance"],
            description="Understanding the implications",
            universal_pattern="REINTEGRATION"
        ),
        JourneyStage(
            name="Theory Formation",
            keywords=["theory", "model", "framework", "explain", "generalize", "principle", "law"],
            description="Developing broader understanding",
            universal_pattern="TRANSFORMATION"
        ),
        JourneyStage(
            name="Publication & Impact",
            keywords=["publish", "paper", "peer review", "citation", "impact", "contribute", "advance"],
            description="Sharing knowledge with the world",
            universal_pattern="COMPLETION"
        )
    ]
)

PERSONAL_JOURNEY = JourneyDefinition(
    id="personal",
    name="Personal Journey",
    domain="Growth & Transformation",
    color="#EC4899",  # Pink - Heart/emotion
    stages=[
        JourneyStage(
            name="Comfort Zone",
            keywords=["comfortable", "safe", "familiar", "routine", "easy", "secure", "known"],
            description="Living within familiar boundaries",
            universal_pattern="STATUS_QUO"
        ),
        JourneyStage(
            name="Awakening",
            keywords=["awaken", "realize", "awareness", "dissatisfied", "unfulfilled", "yearning", "restless"],
            description="Recognizing need for change",
            universal_pattern="CATALYST"
        ),
        JourneyStage(
            name="Resistance",
            keywords=["fear", "resist", "excuse", "procrastinate", "avoid", "deny", "defend"],
            description="Resisting personal change",
            universal_pattern="RESISTANCE"
        ),
        JourneyStage(
            name="Seeking Guidance",
            keywords=["therapist", "coach", "guide", "help", "support", "counselor", "mentor"],
            description="Finding support for growth",
            universal_pattern="GUIDANCE"
        ),
        JourneyStage(
            name="Decision to Change",
            keywords=["decide", "commit", "choose", "accept", "surrender", "ready", "willing"],
            description="Committing to transformation",
            universal_pattern="COMMITMENT"
        ),
        JourneyStage(
            name="Self-Discovery",
            keywords=["explore", "discover", "learn", "understand", "reflect", "introspect", "aware"],
            description="Exploring inner landscape",
            universal_pattern="EXPLORATION"
        ),
        JourneyStage(
            name="Facing Shadows",
            keywords=["shadow", "confront", "face", "truth", "honest", "vulnerable", "expose"],
            description="Preparing to face difficult truths",
            universal_pattern="PREPARATION"
        ),
        JourneyStage(
            name="Dark Night",
            keywords=["dark", "crisis", "breakdown", "despair", "loss", "death", "rock bottom"],
            description="Facing the deepest challenges",
            universal_pattern="CRISIS"
        ),
        JourneyStage(
            name="Breakthrough",
            keywords=["breakthrough", "insight", "liberation", "release", "heal", "transform", "shift"],
            description="Achieving major insight or healing",
            universal_pattern="BREAKTHROUGH"
        ),
        JourneyStage(
            name="Integration",
            keywords=["integrate", "embody", "practice", "apply", "live", "be", "authentic"],
            description="Living the new way of being",
            universal_pattern="REINTEGRATION"
        ),
        JourneyStage(
            name="Wholeness",
            keywords=["whole", "complete", "integrated", "authentic", "actualized", "self", "being"],
            description="Becoming fully oneself",
            universal_pattern="TRANSFORMATION"
        ),
        JourneyStage(
            name="Service",
            keywords=["serve", "help", "give", "share", "teach", "inspire", "contribute"],
            description="Helping others on their journey",
            universal_pattern="COMPLETION"
        )
    ]
)

PRODUCT_JOURNEY = JourneyDefinition(
    id="product",
    name="Product Journey",
    domain="Development Lifecycle",
    color="#06B6D4",  # Cyan - Technology/innovation
    stages=[
        JourneyStage(
            name="Problem Space",
            keywords=["problem", "pain", "need", "gap", "opportunity", "user", "existing"],
            description="Understanding the problem to solve",
            universal_pattern="STATUS_QUO"
        ),
        JourneyStage(
            name="Solution Hypothesis",
            keywords=["solution", "idea", "concept", "vision", "could", "might", "imagine"],
            description="Envisioning a potential solution",
            universal_pattern="CATALYST"
        ),
        JourneyStage(
            name="Technical Doubt",
            keywords=["feasible", "technical", "difficult", "complex", "impossible", "challenge"],
            description="Questioning technical feasibility",
            universal_pattern="RESISTANCE"
        ),
        JourneyStage(
            name="Research & Discovery",
            keywords=["research", "competitive", "benchmark", "best practice", "pattern", "example"],
            description="Learning from existing solutions",
            universal_pattern="GUIDANCE"
        ),
        JourneyStage(
            name="Design Decision",
            keywords=["design", "architecture", "approach", "technology", "decide", "choose", "commit"],
            description="Committing to a design approach",
            universal_pattern="COMMITMENT"
        ),
        JourneyStage(
            name="Prototyping",
            keywords=["prototype", "mockup", "sketch", "experiment", "test", "try", "explore"],
            description="Exploring design possibilities",
            universal_pattern="EXPLORATION"
        ),
        JourneyStage(
            name="Development Sprint",
            keywords=["develop", "build", "code", "implement", "sprint", "feature", "work"],
            description="Building the product",
            universal_pattern="PREPARATION"
        ),
        JourneyStage(
            name="Critical Bug",
            keywords=["bug", "broken", "fail", "error", "crash", "issue", "problem", "blocker"],
            description="Facing major technical challenges",
            universal_pattern="CRISIS"
        ),
        JourneyStage(
            name="Working Product",
            keywords=["working", "functional", "demo", "complete", "success", "done", "ready"],
            description="Achieving a working solution",
            universal_pattern="BREAKTHROUGH"
        ),
        JourneyStage(
            name="User Testing",
            keywords=["test", "user", "feedback", "validate", "learn", "iterate", "improve"],
            description="Validating with real users",
            universal_pattern="REINTEGRATION"
        ),
        JourneyStage(
            name="Product-Market Fit",
            keywords=["adoption", "retention", "engagement", "love", "viral", "growth", "fit"],
            description="Achieving strong product-market fit",
            universal_pattern="TRANSFORMATION"
        ),
        JourneyStage(
            name="Scale & Impact",
            keywords=["scale", "millions", "impact", "change", "industry", "standard", "legacy"],
            description="Achieving widespread impact",
            universal_pattern="COMPLETION"
        )
    ]
)

# ============================================================================
# UNIVERSAL PATTERNS - The Meta-Journey
# ============================================================================

UNIVERSAL_PATTERNS = {
    "STATUS_QUO": {
        "name": "The Status Quo",
        "description": "The beginning state before transformation",
        "keywords": ["current", "existing", "normal", "before", "initial"],
        "energy": "stable",
        "stages": {
            "hero": "Ordinary World",
            "business": "Ideation",
            "learning": "Unconscious Incompetence",
            "scientific": "Observation",
            "personal": "Comfort Zone",
            "product": "Problem Space"
        }
    },
    "CATALYST": {
        "name": "The Catalyst",
        "description": "The disruption that begins the journey",
        "keywords": ["change", "trigger", "spark", "begin", "start"],
        "energy": "rising",
        "stages": {
            "hero": "Call to Adventure",
            "business": "Validation",
            "learning": "Awareness",
            "scientific": "Question",
            "personal": "Awakening",
            "product": "Solution Hypothesis"
        }
    },
    "RESISTANCE": {
        "name": "The Resistance",
        "description": "Fear and doubt before commitment",
        "keywords": ["fear", "doubt", "resist", "hesitate", "avoid"],
        "energy": "conflicted",
        "stages": {
            "hero": "Refusal of Call",
            "business": "Doubt & Fear",
            "learning": "Overwhelm",
            "scientific": "Doubt & Skepticism",
            "personal": "Resistance",
            "product": "Technical Doubt"
        }
    },
    "GUIDANCE": {
        "name": "The Guidance",
        "description": "Receiving wisdom and support",
        "keywords": ["guide", "mentor", "help", "learn", "support"],
        "energy": "supported",
        "stages": {
            "hero": "Meeting Mentor",
            "business": "Advisor/Investor",
            "learning": "Finding Teacher",
            "scientific": "Literature Review",
            "personal": "Seeking Guidance",
            "product": "Research & Discovery"
        }
    },
    "COMMITMENT": {
        "name": "The Commitment",
        "description": "The point of no return",
        "keywords": ["commit", "decide", "begin", "cross", "start"],
        "energy": "determined",
        "stages": {
            "hero": "Crossing Threshold",
            "business": "MVP Launch",
            "learning": "Commitment to Practice",
            "scientific": "Hypothesis",
            "personal": "Decision to Change",
            "product": "Design Decision"
        }
    },
    "EXPLORATION": {
        "name": "The Exploration",
        "description": "Learning the new territory",
        "keywords": ["explore", "discover", "learn", "test", "experiment"],
        "energy": "curious",
        "stages": {
            "hero": "Tests, Allies, Enemies",
            "business": "Early Traction",
            "learning": "Deliberate Practice",
            "scientific": "Experimental Design",
            "personal": "Self-Discovery",
            "product": "Prototyping"
        }
    },
    "PREPARATION": {
        "name": "The Preparation",
        "description": "Getting ready for the major challenge",
        "keywords": ["prepare", "ready", "gather", "approach", "build"],
        "energy": "focused",
        "stages": {
            "hero": "Approach Inmost Cave",
            "business": "Preparing to Scale",
            "learning": "Plateau Preparation",
            "scientific": "Preparation",
            "personal": "Facing Shadows",
            "product": "Development Sprint"
        }
    },
    "CRISIS": {
        "name": "The Crisis",
        "description": "The darkest hour and greatest challenge",
        "keywords": ["crisis", "ordeal", "death", "fail", "dark"],
        "energy": "intense",
        "stages": {
            "hero": "Ordeal",
            "business": "Cash Crunch",
            "learning": "Learning Crisis",
            "scientific": "Failed Experiments",
            "personal": "Dark Night",
            "product": "Critical Bug"
        }
    },
    "BREAKTHROUGH": {
        "name": "The Breakthrough",
        "description": "The victory and reward",
        "keywords": ["breakthrough", "success", "achieve", "win", "discover"],
        "energy": "triumphant",
        "stages": {
            "hero": "Reward",
            "business": "Product-Market Fit",
            "learning": "Breakthrough",
            "scientific": "Discovery",
            "personal": "Breakthrough",
            "product": "Working Product"
        }
    },
    "REINTEGRATION": {
        "name": "The Reintegration",
        "description": "Bringing the new back to the old",
        "keywords": ["return", "integrate", "apply", "bring back"],
        "energy": "synthesizing",
        "stages": {
            "hero": "Road Back",
            "business": "Scaling Operations",
            "learning": "Application",
            "scientific": "Analysis",
            "personal": "Integration",
            "product": "User Testing"
        }
    },
    "TRANSFORMATION": {
        "name": "The Transformation",
        "description": "Becoming something new",
        "keywords": ["transform", "rebirth", "mastery", "become"],
        "energy": "transcendent",
        "stages": {
            "hero": "Resurrection",
            "business": "Market Leadership",
            "learning": "Mastery",
            "scientific": "Theory Formation",
            "personal": "Wholeness",
            "product": "Product-Market Fit"
        }
    },
    "COMPLETION": {
        "name": "The Completion",
        "description": "The gift to the world",
        "keywords": ["complete", "share", "teach", "legacy", "impact"],
        "energy": "fulfilled",
        "stages": {
            "hero": "Return with Elixir",
            "business": "Exit or Legacy",
            "learning": "Teaching Others",
            "scientific": "Publication & Impact",
            "personal": "Service",
            "product": "Scale & Impact"
        }
    }
}

# All journey definitions
ALL_JOURNEYS = {
    "hero": HERO_JOURNEY,
    "business": BUSINESS_JOURNEY,
    "learning": LEARNING_JOURNEY,
    "scientific": SCIENTIFIC_JOURNEY,
    "personal": PERSONAL_JOURNEY,
    "product": PRODUCT_JOURNEY
}


def get_journey(journey_id: str) -> JourneyDefinition:
    """Get journey definition by ID."""
    return ALL_JOURNEYS.get(journey_id)


def get_all_journey_ids() -> List[str]:
    """Get all journey IDs."""
    return list(ALL_JOURNEYS.keys())


def find_universal_pattern(journey_id: str, stage_name: str) -> str:
    """Find the universal pattern for a given stage."""
    journey = get_journey(journey_id)
    if not journey:
        return "UNKNOWN"

    for stage in journey.stages:
        if stage.name == stage_name:
            return stage.universal_pattern

    return "UNKNOWN"


def get_aligned_stages(universal_pattern: str) -> Dict[str, str]:
    """Get all stages across journeys that map to a universal pattern."""
    pattern = UNIVERSAL_PATTERNS.get(universal_pattern, {})
    return pattern.get("stages", {})


if __name__ == "__main__":
    # Test the mappings
    print("Universal Journey Mappings")
    print("=" * 70)

    for journey_id, journey in ALL_JOURNEYS.items():
        print(f"\n{journey.name} ({journey.domain})")
        print(f"Color: {journey.color}")
        print(f"Stages: {len(journey.stages)}")
        print("  1.", journey.stages[0].name, "→", journey.stages[0].universal_pattern)
        print("  ...", "(10 more stages)")
        print(" 12.", journey.stages[-1].name, "→", journey.stages[-1].universal_pattern)

    print("\n" + "=" * 70)
    print("\nUniversal Patterns:")
    for pattern_id, pattern in UNIVERSAL_PATTERNS.items():
        aligned = list(pattern["stages"].values())
        print(f"\n{pattern['name']}: {len(aligned)} aligned stages")
        print(f"  Energy: {pattern['energy']}")
