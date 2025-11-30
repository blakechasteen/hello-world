"""
Resident Generator - Scalable Procedural Resident Creation

Generates diverse, believable residents with:
- Unique personalities (Big Five)
- Rich backstories from archetypes
- Personal projects aligned with interests
- Life goals and aspirations
- Relationship seeds

Can scale from 6 to 600+ residents while maintaining
narrative coherence and interesting character dynamics.

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

from resident import Resident, Personality, DailySchedule
from solo_activities import PersonalProject, ProjectStatus


# =============================================================================
# ARCHETYPES - Character templates that can be mixed and matched
# =============================================================================

class Archetype(Enum):
    """Base character archetypes for procedural generation."""
    # Creative types
    ARTIST = "artist"
    WRITER = "writer"
    MUSICIAN = "musician"
    CRAFTSPERSON = "craftsperson"

    # Professional types
    TEACHER = "teacher"
    ENTREPRENEUR = "entrepreneur"
    SCIENTIST = "scientist"
    HEALTHCARE = "healthcare"
    TECH_WORKER = "tech_worker"

    # Community types
    COMMUNITY_ORGANIZER = "community_organizer"
    ELDER = "elder"
    NEWCOMER = "newcomer"
    LOCAL_BUSINESS = "local_business"

    # Life stage types
    STUDENT = "student"
    RETIREE = "retiree"
    YOUNG_PARENT = "young_parent"
    EMPTY_NESTER = "empty_nester"

    # Personality types
    INTROVERT_SAGE = "introvert_sage"
    SOCIAL_BUTTERFLY = "social_butterfly"
    SKEPTIC = "skeptic"
    OPTIMIST = "optimist"
    HEALER = "healer"


@dataclass
class ArchetypeTemplate:
    """Template data for each archetype."""
    archetype: Archetype
    occupations: List[str]
    hobbies: List[str]
    personality_tendencies: Dict[str, Tuple[float, float]]  # trait: (min, max)
    backstory_elements: List[str]
    project_types: List[str]
    life_goal_types: List[str]
    typical_age_range: Tuple[int, int]
    skill_affinities: List[str]


# Archetype definitions
ARCHETYPE_TEMPLATES: Dict[Archetype, ArchetypeTemplate] = {
    Archetype.ARTIST: ArchetypeTemplate(
        archetype=Archetype.ARTIST,
        occupations=["Painter", "Sculptor", "Muralist", "Art Teacher", "Gallery Owner", "Illustrator"],
        hobbies=["painting", "sketching", "visiting galleries", "nature walks", "photography"],
        personality_tendencies={
            "openness": (0.7, 0.95),
            "conscientiousness": (0.3, 0.6),
            "extraversion": (0.3, 0.7),
            "agreeableness": (0.5, 0.8),
            "neuroticism": (0.4, 0.7),
        },
        backstory_elements=[
            "studied art in {city}",
            "inspired by {relative}'s creativity",
            "left corporate life to pursue art",
            "discovered painting after {life_event}",
            "sells work at local markets",
        ],
        project_types=["painting", "mural", "sculpture", "art_series", "exhibition"],
        life_goal_types=["creative_mastery", "artistic_recognition", "teaching_legacy"],
        typical_age_range=(25, 65),
        skill_affinities=["Painting", "Creativity", "Photography"],
    ),

    Archetype.WRITER: ArchetypeTemplate(
        archetype=Archetype.WRITER,
        occupations=["Novelist", "Journalist", "Blogger", "Poet", "Copywriter", "Editor"],
        hobbies=["reading", "journaling", "coffee shop visits", "people watching", "meditation"],
        personality_tendencies={
            "openness": (0.7, 0.95),
            "conscientiousness": (0.4, 0.8),
            "extraversion": (0.2, 0.5),
            "agreeableness": (0.4, 0.7),
            "neuroticism": (0.3, 0.7),
        },
        backstory_elements=[
            "working on first novel",
            "former journalist seeking quieter life",
            "writes poetry about {theme}",
            "published short stories in {magazine}",
            "inspired by the neighborhood's stories",
        ],
        project_types=["novel", "memoir", "poetry_collection", "blog", "screenplay"],
        life_goal_types=["publish_book", "storytelling_mastery", "capture_community_history"],
        typical_age_range=(22, 70),
        skill_affinities=["Writing", "Research", "Creativity"],
    ),

    Archetype.MUSICIAN: ArchetypeTemplate(
        archetype=Archetype.MUSICIAN,
        occupations=["Music Teacher", "Session Musician", "Band Member", "Composer", "DJ", "Sound Engineer"],
        hobbies=["practicing instruments", "going to concerts", "vinyl collecting", "jamming", "songwriting"],
        personality_tendencies={
            "openness": (0.7, 0.9),
            "conscientiousness": (0.3, 0.7),
            "extraversion": (0.5, 0.85),
            "agreeableness": (0.5, 0.8),
            "neuroticism": (0.3, 0.6),
        },
        backstory_elements=[
            "plays {instrument} since childhood",
            "toured with {band_type} band",
            "teaches music to neighborhood kids",
            "composing an album about {theme}",
            "started a community choir",
        ],
        project_types=["album", "song_cycle", "music_lessons", "community_performance", "instrument_mastery"],
        life_goal_types=["musical_recognition", "community_harmony", "teaching_music"],
        typical_age_range=(20, 60),
        skill_affinities=["Guitar", "Piano", "Singing", "Creativity"],
    ),

    Archetype.TEACHER: ArchetypeTemplate(
        archetype=Archetype.TEACHER,
        occupations=["Elementary Teacher", "High School Teacher", "Professor", "Tutor", "Special Ed Teacher", "Principal"],
        hobbies=["reading", "gardening", "cooking", "volunteering", "hiking"],
        personality_tendencies={
            "openness": (0.5, 0.8),
            "conscientiousness": (0.6, 0.9),
            "extraversion": (0.5, 0.8),
            "agreeableness": (0.7, 0.95),
            "neuroticism": (0.2, 0.5),
        },
        backstory_elements=[
            "taught for {years} years",
            "mentored countless students",
            "wrote curriculum for {subject}",
            "believes every child can learn",
            "started after-school program",
        ],
        project_types=["curriculum", "tutoring_program", "educational_book", "mentorship"],
        life_goal_types=["inspire_students", "educational_innovation", "community_education"],
        typical_age_range=(25, 65),
        skill_affinities=["Teaching", "Research", "Communication"],
    ),

    Archetype.ENTREPRENEUR: ArchetypeTemplate(
        archetype=Archetype.ENTREPRENEUR,
        occupations=["Small Business Owner", "Startup Founder", "Consultant", "Freelancer", "Investor"],
        hobbies=["networking", "reading business books", "tennis", "golf", "mentoring"],
        personality_tendencies={
            "openness": (0.6, 0.85),
            "conscientiousness": (0.6, 0.9),
            "extraversion": (0.6, 0.9),
            "agreeableness": (0.4, 0.7),
            "neuroticism": (0.2, 0.5),
        },
        backstory_elements=[
            "started {business_type} from scratch",
            "sold previous company",
            "helps local businesses grow",
            "believes in community investment",
            "failed twice before succeeding",
        ],
        project_types=["business_plan", "product_launch", "mentorship_program", "investment"],
        life_goal_types=["financial_success", "job_creation", "business_legacy"],
        typical_age_range=(28, 60),
        skill_affinities=["Business", "Charisma", "Programming"],
    ),

    Archetype.ELDER: ArchetypeTemplate(
        archetype=Archetype.ELDER,
        occupations=["Retired", "Part-time Consultant", "Volunteer", "Community Elder", "Mentor"],
        hobbies=["gardening", "woodworking", "storytelling", "chess", "walking", "cooking traditional recipes"],
        personality_tendencies={
            "openness": (0.4, 0.7),
            "conscientiousness": (0.5, 0.8),
            "extraversion": (0.3, 0.6),
            "agreeableness": (0.6, 0.9),
            "neuroticism": (0.2, 0.5),
        },
        backstory_elements=[
            "lived in neighborhood for {years} years",
            "raised family here",
            "remembers when {historical_event}",
            "keeper of neighborhood stories",
            "lost spouse {years} years ago",
        ],
        project_types=["memoir", "family_history", "garden", "mentorship", "recipe_collection"],
        life_goal_types=["legacy", "family_connection", "wisdom_sharing"],
        typical_age_range=(65, 85),
        skill_affinities=["Gardening", "Cooking", "Handiness"],
    ),

    Archetype.NEWCOMER: ArchetypeTemplate(
        archetype=Archetype.NEWCOMER,
        occupations=["Remote Worker", "Freelancer", "Job Seeker", "Student", "Transplant Professional"],
        hobbies=["exploring the area", "trying new restaurants", "social events", "online gaming", "fitness"],
        personality_tendencies={
            "openness": (0.6, 0.9),
            "conscientiousness": (0.4, 0.7),
            "extraversion": (0.4, 0.8),
            "agreeableness": (0.5, 0.8),
            "neuroticism": (0.4, 0.7),
        },
        backstory_elements=[
            "just moved from {city}",
            "seeking fresh start",
            "doesn't know anyone yet",
            "attracted by {neighborhood_feature}",
            "left behind {thing_left_behind}",
        ],
        project_types=["apartment_decoration", "finding_community", "career_change", "self_discovery"],
        life_goal_types=["belonging", "career_success", "building_friendships"],
        typical_age_range=(22, 45),
        skill_affinities=["Social", "Fitness", "Programming"],
    ),

    Archetype.HEALTHCARE: ArchetypeTemplate(
        archetype=Archetype.HEALTHCARE,
        occupations=["Nurse", "Doctor", "Therapist", "Caregiver", "EMT", "Pharmacist"],
        hobbies=["yoga", "reading", "gardening", "cooking healthy meals", "meditation"],
        personality_tendencies={
            "openness": (0.4, 0.7),
            "conscientiousness": (0.7, 0.95),
            "extraversion": (0.4, 0.7),
            "agreeableness": (0.7, 0.95),
            "neuroticism": (0.3, 0.6),
        },
        backstory_elements=[
            "cares for patients at {hospital}",
            "chose medicine to help people",
            "works long hours but loves it",
            "deals with burnout sometimes",
            "neighborhood's informal health advisor",
        ],
        project_types=["wellness_program", "health_education", "self_care_routine", "community_health"],
        life_goal_types=["save_lives", "community_health", "work_life_balance"],
        typical_age_range=(25, 60),
        skill_affinities=["Fitness", "Cooking", "Meditation"],
    ),

    Archetype.LOCAL_BUSINESS: ArchetypeTemplate(
        archetype=Archetype.LOCAL_BUSINESS,
        occupations=["Cafe Owner", "Bookshop Owner", "Baker", "Florist", "Barber", "Restaurant Owner"],
        hobbies=["cooking", "gardening", "community events", "local history", "crafts"],
        personality_tendencies={
            "openness": (0.4, 0.7),
            "conscientiousness": (0.6, 0.9),
            "extraversion": (0.6, 0.9),
            "agreeableness": (0.6, 0.9),
            "neuroticism": (0.3, 0.6),
        },
        backstory_elements=[
            "opened {business} {years} years ago",
            "knows everyone in the neighborhood",
            "business is a gathering place",
            "learned trade from {relative}",
            "third generation owner",
        ],
        project_types=["business_expansion", "community_event", "recipe_book", "mentoring_apprentice"],
        life_goal_types=["business_success", "community_pillar", "passing_on_trade"],
        typical_age_range=(30, 65),
        skill_affinities=["Cooking", "Charisma", "Business"],
    ),

    Archetype.STUDENT: ArchetypeTemplate(
        archetype=Archetype.STUDENT,
        occupations=["College Student", "Grad Student", "Intern", "Part-time Worker"],
        hobbies=["studying", "gaming", "social media", "parties", "sports", "coffee shops"],
        personality_tendencies={
            "openness": (0.6, 0.9),
            "conscientiousness": (0.3, 0.7),
            "extraversion": (0.5, 0.85),
            "agreeableness": (0.5, 0.8),
            "neuroticism": (0.4, 0.7),
        },
        backstory_elements=[
            "studying {major} at university",
            "first one in family to go to college",
            "struggling to balance work and school",
            "passionate about {cause}",
            "figuring out life direction",
        ],
        project_types=["thesis", "internship", "student_org", "side_project", "study_group"],
        life_goal_types=["graduation", "career_launch", "self_discovery"],
        typical_age_range=(18, 28),
        skill_affinities=["Research", "Programming", "Social"],
    ),

    Archetype.RETIREE: ArchetypeTemplate(
        archetype=Archetype.RETIREE,
        occupations=["Retired", "Part-time Volunteer", "Consultant"],
        hobbies=["gardening", "golf", "reading", "travel", "grandchildren", "crafts", "bridge"],
        personality_tendencies={
            "openness": (0.4, 0.7),
            "conscientiousness": (0.5, 0.8),
            "extraversion": (0.4, 0.7),
            "agreeableness": (0.6, 0.85),
            "neuroticism": (0.2, 0.5),
        },
        backstory_elements=[
            "retired from {career} after {years} years",
            "finally has time for hobbies",
            "adjusting to slower pace",
            "misses the structure sometimes",
            "volunteering to stay busy",
        ],
        project_types=["memoir", "garden_project", "travel_plans", "family_history", "hobby_mastery"],
        life_goal_types=["enjoying_retirement", "family_time", "giving_back"],
        typical_age_range=(60, 80),
        skill_affinities=["Gardening", "Cooking", "Handiness"],
    ),

    Archetype.HEALER: ArchetypeTemplate(
        archetype=Archetype.HEALER,
        occupations=["Therapist", "Yoga Instructor", "Life Coach", "Massage Therapist", "Spiritual Guide"],
        hobbies=["meditation", "yoga", "nature", "crystals", "reading spiritual texts", "journaling"],
        personality_tendencies={
            "openness": (0.7, 0.95),
            "conscientiousness": (0.4, 0.7),
            "extraversion": (0.4, 0.7),
            "agreeableness": (0.8, 0.95),
            "neuroticism": (0.2, 0.5),
        },
        backstory_elements=[
            "found calling after {transformation}",
            "helps others find peace",
            "trained in {modality}",
            "runs community wellness circles",
            "intuitive and empathetic",
        ],
        project_types=["wellness_workshop", "book_on_healing", "retreat_planning", "community_circles"],
        life_goal_types=["heal_others", "spiritual_growth", "peaceful_community"],
        typical_age_range=(30, 60),
        skill_affinities=["Meditation", "Fitness", "Charisma"],
    ),

    Archetype.CRAFTSPERSON: ArchetypeTemplate(
        archetype=Archetype.CRAFTSPERSON,
        occupations=["Woodworker", "Potter", "Jeweler", "Weaver", "Blacksmith", "Leatherworker"],
        hobbies=["woodworking", "pottery", "making things", "markets", "tool collecting"],
        personality_tendencies={
            "openness": (0.5, 0.8),
            "conscientiousness": (0.6, 0.9),
            "extraversion": (0.3, 0.6),
            "agreeableness": (0.5, 0.8),
            "neuroticism": (0.2, 0.5),
        },
        backstory_elements=[
            "learned craft from {relative}",
            "sells handmade goods at markets",
            "believes in quality over quantity",
            "teaching the next generation",
        ],
        project_types=["craft_commission", "workshop_setup", "teaching_apprentice", "artisan_collective"],
        life_goal_types=["master_craft", "pass_on_skills", "sustainable_living"],
        typical_age_range=(25, 65),
        skill_affinities=["Handiness", "Creativity", "Business"],
    ),

    Archetype.SCIENTIST: ArchetypeTemplate(
        archetype=Archetype.SCIENTIST,
        occupations=["Researcher", "Lab Technician", "Professor", "Data Scientist", "Environmental Scientist"],
        hobbies=["reading journals", "experiments", "hiking", "puzzles", "astronomy"],
        personality_tendencies={
            "openness": (0.7, 0.95),
            "conscientiousness": (0.7, 0.95),
            "extraversion": (0.3, 0.6),
            "agreeableness": (0.4, 0.7),
            "neuroticism": (0.3, 0.6),
        },
        backstory_elements=[
            "researches {major} at the university",
            "passionate about discovery",
            "curious since childhood",
            "working on breakthrough research",
        ],
        project_types=["research_paper", "citizen_science", "science_outreach", "experiment"],
        life_goal_types=["scientific_discovery", "education", "solve_big_problem"],
        typical_age_range=(25, 70),
        skill_affinities=["Research", "Programming", "Logic"],
    ),

    Archetype.TECH_WORKER: ArchetypeTemplate(
        archetype=Archetype.TECH_WORKER,
        occupations=["Software Developer", "UX Designer", "Product Manager", "DevOps Engineer", "Data Analyst"],
        hobbies=["coding", "gaming", "tech meetups", "podcasts", "building gadgets"],
        personality_tendencies={
            "openness": (0.6, 0.85),
            "conscientiousness": (0.5, 0.8),
            "extraversion": (0.3, 0.7),
            "agreeableness": (0.4, 0.7),
            "neuroticism": (0.3, 0.6),
        },
        backstory_elements=[
            "works remotely for tech company",
            "self-taught programmer",
            "building side projects",
            "trying to find work-life balance",
        ],
        project_types=["app_development", "open_source", "tech_blog", "automation"],
        life_goal_types=["startup_success", "work_life_balance", "build_something_meaningful"],
        typical_age_range=(22, 55),
        skill_affinities=["Programming", "Logic", "Research"],
    ),

    Archetype.COMMUNITY_ORGANIZER: ArchetypeTemplate(
        archetype=Archetype.COMMUNITY_ORGANIZER,
        occupations=["Nonprofit Director", "Activist", "Social Worker", "Event Planner", "Union Rep"],
        hobbies=["volunteering", "attending meetings", "reading about social issues", "networking"],
        personality_tendencies={
            "openness": (0.6, 0.85),
            "conscientiousness": (0.6, 0.85),
            "extraversion": (0.7, 0.95),
            "agreeableness": (0.6, 0.85),
            "neuroticism": (0.3, 0.6),
        },
        backstory_elements=[
            "organizing for {cause}",
            "believes in collective power",
            "brought the community together",
            "fighting for neighborhood improvements",
        ],
        project_types=["community_event", "advocacy_campaign", "mutual_aid", "neighborhood_watch"],
        life_goal_types=["social_change", "empower_community", "build_solidarity"],
        typical_age_range=(25, 65),
        skill_affinities=["Charisma", "Social", "Communication"],
    ),

    Archetype.YOUNG_PARENT: ArchetypeTemplate(
        archetype=Archetype.YOUNG_PARENT,
        occupations=["Stay-at-home Parent", "Part-time Worker", "Freelancer", "Teacher"],
        hobbies=["family activities", "playground visits", "cooking", "baby groups", "napping"],
        personality_tendencies={
            "openness": (0.4, 0.7),
            "conscientiousness": (0.5, 0.8),
            "extraversion": (0.4, 0.7),
            "agreeableness": (0.6, 0.9),
            "neuroticism": (0.4, 0.7),
        },
        backstory_elements=[
            "just had first child",
            "balancing work and family",
            "sleep-deprived but happy",
            "building connections with other parents",
        ],
        project_types=["family_photo_album", "baby_milestone_book", "parent_group", "work_from_home"],
        life_goal_types=["raise_happy_kids", "work_life_balance", "build_family_memories"],
        typical_age_range=(25, 40),
        skill_affinities=["Cooking", "Fitness", "Social"],
    ),

    Archetype.EMPTY_NESTER: ArchetypeTemplate(
        archetype=Archetype.EMPTY_NESTER,
        occupations=["Professional", "Consultant", "Part-time Worker", "Volunteer"],
        hobbies=["travel", "hobbies rediscovered", "grandchildren", "gardening", "book clubs"],
        personality_tendencies={
            "openness": (0.5, 0.8),
            "conscientiousness": (0.5, 0.8),
            "extraversion": (0.4, 0.7),
            "agreeableness": (0.6, 0.85),
            "neuroticism": (0.2, 0.5),
        },
        backstory_elements=[
            "kids just left for college",
            "rediscovering personal interests",
            "more time for themselves",
            "reconnecting as a couple",
        ],
        project_types=["travel_plans", "hobby_mastery", "home_renovation", "volunteer_work"],
        life_goal_types=["travel", "new_chapter", "reconnect_with_partner"],
        typical_age_range=(45, 65),
        skill_affinities=["Gardening", "Cooking", "Travel"],
    ),

    Archetype.INTROVERT_SAGE: ArchetypeTemplate(
        archetype=Archetype.INTROVERT_SAGE,
        occupations=["Writer", "Researcher", "Librarian", "Philosopher", "Archivist"],
        hobbies=["reading", "solitary walks", "contemplation", "writing", "stargazing"],
        personality_tendencies={
            "openness": (0.7, 0.95),
            "conscientiousness": (0.5, 0.8),
            "extraversion": (0.1, 0.35),
            "agreeableness": (0.5, 0.8),
            "neuroticism": (0.3, 0.6),
        },
        backstory_elements=[
            "finds wisdom in solitude",
            "deep thinker",
            "people come to them for advice",
            "prefers quality over quantity in relationships",
        ],
        project_types=["philosophical_writing", "research", "mentorship", "quiet_garden"],
        life_goal_types=["wisdom", "inner_peace", "leaving_knowledge"],
        typical_age_range=(30, 75),
        skill_affinities=["Writing", "Research", "Meditation"],
    ),

    Archetype.SOCIAL_BUTTERFLY: ArchetypeTemplate(
        archetype=Archetype.SOCIAL_BUTTERFLY,
        occupations=["Event Planner", "PR Specialist", "Bartender", "Sales", "Influencer"],
        hobbies=["parties", "networking events", "social media", "dining out", "hosting"],
        personality_tendencies={
            "openness": (0.6, 0.85),
            "conscientiousness": (0.3, 0.6),
            "extraversion": (0.8, 0.98),
            "agreeableness": (0.6, 0.85),
            "neuroticism": (0.3, 0.6),
        },
        backstory_elements=[
            "knows everyone in the neighborhood",
            "organizes all the gatherings",
            "can't stand being alone",
            "the connector of people",
        ],
        project_types=["big_party", "networking_group", "social_media", "community_event"],
        life_goal_types=["be_known", "connect_everyone", "never_be_bored"],
        typical_age_range=(22, 50),
        skill_affinities=["Charisma", "Social", "Communication"],
    ),

    Archetype.SKEPTIC: ArchetypeTemplate(
        archetype=Archetype.SKEPTIC,
        occupations=["Journalist", "Investigator", "Lawyer", "Critic", "Auditor"],
        hobbies=["research", "debate", "documentaries", "fact-checking", "puzzles"],
        personality_tendencies={
            "openness": (0.5, 0.8),
            "conscientiousness": (0.6, 0.85),
            "extraversion": (0.3, 0.6),
            "agreeableness": (0.2, 0.5),
            "neuroticism": (0.4, 0.7),
        },
        backstory_elements=[
            "questions everything",
            "was burned before, now cautious",
            "values truth above comfort",
            "the neighborhood's fact-checker",
        ],
        project_types=["investigation", "expose_writing", "documentary", "research_deep_dive"],
        life_goal_types=["uncover_truth", "protect_others", "maintain_integrity"],
        typical_age_range=(28, 65),
        skill_affinities=["Research", "Logic", "Writing"],
    ),

    Archetype.OPTIMIST: ArchetypeTemplate(
        archetype=Archetype.OPTIMIST,
        occupations=["Life Coach", "Teacher", "Nonprofit Worker", "Counselor", "Motivational Speaker"],
        hobbies=["volunteering", "hiking", "gratitude journaling", "positive podcasts", "community service"],
        personality_tendencies={
            "openness": (0.6, 0.85),
            "conscientiousness": (0.5, 0.8),
            "extraversion": (0.6, 0.85),
            "agreeableness": (0.7, 0.95),
            "neuroticism": (0.1, 0.35),
        },
        backstory_elements=[
            "sees the bright side always",
            "overcame adversity with positivity",
            "lifts everyone's spirits",
            "believes things work out",
        ],
        project_types=["positivity_campaign", "gratitude_project", "community_uplift", "motivational_content"],
        life_goal_types=["spread_joy", "help_others_see_good", "live_fully"],
        typical_age_range=(25, 60),
        skill_affinities=["Charisma", "Social", "Meditation"],
    ),
}


# =============================================================================
# NAME GENERATION
# =============================================================================

# Diverse name pools
FIRST_NAMES = {
    "hispanic": ["Sofia", "Diego", "Luna", "Mateo", "Valentina", "Santiago", "Camila", "Sebastian", "Isabella", "Miguel", "Alejandra", "Carlos", "Elena", "Jorge", "Maya"],
    "anglo": ["James", "Emma", "Oliver", "Ava", "William", "Sophia", "Benjamin", "Mia", "Lucas", "Charlotte", "Henry", "Amelia", "Alexander", "Harper", "Daniel"],
    "african_american": ["Jayden", "Imani", "Malik", "Zara", "Darius", "Aaliyah", "Marcus", "Nia", "Terrell", "Jasmine", "Andre", "Destiny", "Isaiah", "Keisha", "DeShawn"],
    "asian": ["Wei", "Mei", "Jin", "Yuki", "Hiroshi", "Sakura", "Kenji", "Aiko", "Chen", "Li", "Min", "Soo", "Raj", "Priya", "Arun"],
    "mixed": ["Jordan", "Alex", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery", "Phoenix", "Sage", "River", "Sky", "Ash", "Blake", "Cameron"],
}

LAST_NAMES = {
    "hispanic": ["Garcia", "Rodriguez", "Martinez", "Lopez", "Hernandez", "Gonzalez", "Ramirez", "Torres", "Flores", "Rivera"],
    "anglo": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson", "Anderson", "Thomas"],
    "african_american": ["Washington", "Jefferson", "Jackson", "Robinson", "Freeman", "Brooks", "Coleman", "Hayes", "Banks", "Ellis"],
    "asian": ["Chen", "Wang", "Kim", "Nguyen", "Patel", "Singh", "Lee", "Park", "Tanaka", "Yamamoto"],
    "mixed": ["Santos", "Cruz", "Young", "Green", "Adams", "Campbell", "Mitchell", "Turner", "Phillips", "Evans"],
}


# =============================================================================
# BACKSTORY GENERATION
# =============================================================================

CITIES = ["New York", "Los Angeles", "Chicago", "Miami", "Seattle", "Austin", "Denver", "Portland", "Boston", "Atlanta"]
RELATIVES = ["grandmother", "grandfather", "mother", "father", "aunt", "uncle", "mentor", "childhood friend"]
LIFE_EVENTS = ["a health scare", "a breakup", "losing a job", "the pandemic", "turning 40", "a spiritual awakening"]
THEMES = ["loss and renewal", "the changing seasons", "family bonds", "urban solitude", "nature's resilience"]
BAND_TYPES = ["jazz", "rock", "indie", "folk", "hip-hop", "classical"]
INSTRUMENTS = ["guitar", "piano", "violin", "drums", "saxophone", "flute", "cello"]
BUSINESS_TYPES = ["a cafe", "a consulting firm", "an online store", "a food truck", "a coworking space"]
HISTORICAL_EVENTS = ["the old bakery closed", "the park was just a empty lot", "the neighborhood was much rougher"]
THINGS_LEFT_BEHIND = ["a relationship", "a high-pressure job", "a small town", "family expectations"]
NEIGHBORHOOD_FEATURES = ["the community vibe", "the local art scene", "the affordable rent", "a job opportunity"]
CAREERS = ["teaching", "engineering", "medicine", "law", "business", "the military", "public service"]
MAJORS = ["computer science", "psychology", "art history", "biology", "business", "English literature", "engineering"]
CAUSES = ["climate change", "social justice", "mental health awareness", "education equity"]
MODALITIES = ["Reiki", "meditation", "somatic therapy", "breathwork", "sound healing"]
TRANSFORMATIONS = ["a personal crisis", "traveling abroad", "a near-death experience", "deep therapy work"]


def generate_backstory(template: ArchetypeTemplate, name: str, age: int) -> str:
    """Generate a rich backstory from template elements."""
    elements = []

    # Pick 2-3 backstory elements and fill in placeholders
    chosen = random.sample(template.backstory_elements, min(2, len(template.backstory_elements)))

    for element in chosen:
        filled = element
        filled = filled.replace("{city}", random.choice(CITIES))
        filled = filled.replace("{relative}", random.choice(RELATIVES))
        filled = filled.replace("{life_event}", random.choice(LIFE_EVENTS))
        filled = filled.replace("{theme}", random.choice(THEMES))
        filled = filled.replace("{band_type}", random.choice(BAND_TYPES))
        filled = filled.replace("{instrument}", random.choice(INSTRUMENTS))
        filled = filled.replace("{business_type}", random.choice(BUSINESS_TYPES))
        filled = filled.replace("{historical_event}", random.choice(HISTORICAL_EVENTS))
        filled = filled.replace("{thing_left_behind}", random.choice(THINGS_LEFT_BEHIND))
        filled = filled.replace("{neighborhood_feature}", random.choice(NEIGHBORHOOD_FEATURES))
        filled = filled.replace("{years}", str(random.randint(5, 40)))
        filled = filled.replace("{career}", random.choice(CAREERS))
        filled = filled.replace("{major}", random.choice(MAJORS))
        filled = filled.replace("{cause}", random.choice(CAUSES))
        filled = filled.replace("{modality}", random.choice(MODALITIES))
        filled = filled.replace("{transformation}", random.choice(TRANSFORMATIONS))
        filled = filled.replace("{subject}", random.choice(MAJORS))
        filled = filled.replace("{magazine}", f"The {random.choice(['New Yorker', 'Atlantic', 'Paris Review', 'Granta'])}")
        filled = filled.replace("{hospital}", f"{random.choice(['St. Mary', 'General', 'Memorial', 'Community'])} Hospital")
        filled = filled.replace("{business}", random.choice(["a cafe", "a bookshop", "a bakery", "a flower shop"]))
        elements.append(filled)

    # Construct backstory
    backstory = f"{name}, {age}, "
    backstory += ". ".join(elements).capitalize()
    backstory += f". {random.choice(template.backstory_elements).replace('{', '').replace('}', '').capitalize()}."

    return backstory


# =============================================================================
# PROJECT GENERATION
# =============================================================================

PROJECT_TEMPLATES = {
    "painting": [
        ("Portrait Series", "Painting portraits of neighborhood residents", "To capture the diversity of this community"),
        ("Abstract Emotions", "A series exploring emotions through color", "Processing complex feelings through art"),
        ("Urban Landscapes", "Capturing the changing cityscape", "Documenting the neighborhood's evolution"),
    ],
    "mural": [
        ("Community Mural", "A large mural for the community center", "Creating shared public art"),
        ("Memory Wall", "A mural honoring neighborhood history", "Preserving local stories"),
    ],
    "novel": [
        ("Neighborhood Stories", "A novel set in this community", "Sharing the stories I've heard"),
        ("Coming of Age", "A semi-autobiographical novel", "Processing my own journey"),
    ],
    "memoir": [
        ("Life Lessons", "Writing down my life experiences", "Leaving wisdom for the next generation"),
        ("Family History", "Documenting our family's journey", "Preserving our heritage"),
    ],
    "album": [
        ("Songs of Home", "An album about belonging", "Finding home through music"),
        ("Acoustic Sessions", "Stripped-down recordings of favorite songs", "Getting back to basics"),
    ],
    "garden": [
        ("Community Garden", "Creating a shared green space", "Building community through growing"),
        ("Medicinal Herbs", "Growing healing plants", "Connecting with natural remedies"),
    ],
    "business_plan": [
        ("Local Startup", "Planning a new local business", "Creating jobs in the community"),
        ("Social Enterprise", "A business that gives back", "Doing well by doing good"),
    ],
    "wellness_program": [
        ("Neighborhood Wellness", "Free yoga and meditation classes", "Improving community health"),
        ("Stress Relief Workshop", "Teaching coping techniques", "Helping others find peace"),
    ],
    "self_discovery": [
        ("Finding My Path", "Exploring new directions in life", "Figuring out who I really am"),
        ("Building a New Life", "Creating a life I love", "Starting fresh in a new place"),
    ],
}


def generate_project(archetype: Archetype, agent_id: str) -> PersonalProject:
    """Generate a personal project based on archetype."""
    template = ARCHETYPE_TEMPLATES[archetype]
    project_type = random.choice(template.project_types)

    if project_type in PROJECT_TEMPLATES:
        name, description, motivation = random.choice(PROJECT_TEMPLATES[project_type])
    else:
        name = f"My {project_type.replace('_', ' ').title()} Project"
        description = f"Working on a personal {project_type.replace('_', ' ')}"
        motivation = "Personal growth and fulfillment"

    return PersonalProject(
        project_id=f"{agent_id}_{project_type}",
        name=name,
        description=description,
        motivation=motivation,
        status=random.choice([ProjectStatus.IDEA, ProjectStatus.IN_PROGRESS]),
        progress=random.uniform(0, 0.5),
        hours_invested=random.uniform(0, 30),
        emotional_investment=random.uniform(0.6, 1.0),
    )


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for resident generation."""
    num_residents: int = 20
    age_distribution: Dict[str, float] = field(default_factory=lambda: {
        "young": 0.25,      # 18-30
        "middle": 0.45,     # 31-55
        "senior": 0.30,     # 56-85
    })
    archetype_weights: Optional[Dict[Archetype, float]] = None  # None = equal weights
    cultural_mix: Dict[str, float] = field(default_factory=lambda: {
        "hispanic": 0.25,
        "anglo": 0.25,
        "african_american": 0.20,
        "asian": 0.15,
        "mixed": 0.15,
    })
    ensure_variety: bool = True  # Ensure at least one of each archetype if possible
    seed: Optional[int] = None


class ResidentGenerator:
    """
    Procedural resident generator for scalable neighborhoods.

    Can generate anywhere from 6 to 600+ unique residents with:
    - Diverse personalities
    - Rich backstories
    - Personal projects
    - Varied ages and backgrounds
    """

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        if self.config.seed:
            random.seed(self.config.seed)

        self.used_names: Set[str] = set()
        self.archetype_counts: Dict[Archetype, int] = {a: 0 for a in Archetype}

    def _select_archetype(self) -> Archetype:
        """Select an archetype, ensuring variety if configured."""
        archetypes = list(Archetype)

        if self.config.archetype_weights:
            weights = [self.config.archetype_weights.get(a, 1.0) for a in archetypes]
        else:
            weights = [1.0] * len(archetypes)

        if self.config.ensure_variety:
            # Boost weight for underrepresented archetypes
            min_count = min(self.archetype_counts.values())
            for i, archetype in enumerate(archetypes):
                if self.archetype_counts[archetype] == min_count:
                    weights[i] *= 2.0

        return random.choices(archetypes, weights=weights)[0]

    def _select_culture(self) -> str:
        """Select a cultural background based on configured mix."""
        cultures = list(self.config.cultural_mix.keys())
        weights = list(self.config.cultural_mix.values())
        return random.choices(cultures, weights=weights)[0]

    def _generate_name(self, culture: str) -> Tuple[str, str]:
        """Generate a unique name."""
        for _ in range(100):  # Prevent infinite loop
            first = random.choice(FIRST_NAMES.get(culture, FIRST_NAMES["mixed"]))
            last = random.choice(LAST_NAMES.get(culture, LAST_NAMES["mixed"]))
            full_name = f"{first} {last}"

            if full_name not in self.used_names:
                self.used_names.add(full_name)
                return first, last

        # Fallback: add number suffix
        first = random.choice(FIRST_NAMES["mixed"])
        last = random.choice(LAST_NAMES["mixed"])
        suffix = random.randint(100, 999)
        return f"{first}{suffix}", last

    def _generate_age(self, archetype: Archetype) -> int:
        """Generate age based on distribution and archetype constraints."""
        template = ARCHETYPE_TEMPLATES[archetype]
        min_age, max_age = template.typical_age_range

        # Apply age distribution preferences, but constrained by archetype
        r = random.random()
        if r < self.config.age_distribution.get("young", 0.25):
            age_range = (max(18, min_age), min(30, max_age))
        elif r < self.config.age_distribution.get("young", 0.25) + self.config.age_distribution.get("middle", 0.45):
            age_range = (max(31, min_age), min(55, max_age))
        else:
            age_range = (max(56, min_age), min(85, max_age))

        # If age range is invalid (min > max), use archetype's full range
        if age_range[0] > age_range[1]:
            age_range = (min_age, max_age)

        return random.randint(age_range[0], age_range[1])

    def _generate_personality(self, archetype: Archetype) -> Personality:
        """Generate personality based on archetype tendencies."""
        template = ARCHETYPE_TEMPLATES[archetype]
        tendencies = template.personality_tendencies

        return Personality(
            openness=random.uniform(*tendencies.get("openness", (0.3, 0.7))),
            conscientiousness=random.uniform(*tendencies.get("conscientiousness", (0.3, 0.7))),
            extraversion=random.uniform(*tendencies.get("extraversion", (0.3, 0.7))),
            agreeableness=random.uniform(*tendencies.get("agreeableness", (0.3, 0.7))),
            neuroticism=random.uniform(*tendencies.get("neuroticism", (0.3, 0.7))),
        )

    def generate_resident(self, resident_number: int) -> Tuple[Resident, Archetype, PersonalProject]:
        """Generate a single resident with full details."""
        archetype = self._select_archetype()
        self.archetype_counts[archetype] += 1

        template = ARCHETYPE_TEMPLATES[archetype]
        culture = self._select_culture()
        first_name, last_name = self._generate_name(culture)
        age = self._generate_age(archetype)

        resident_id = f"{first_name.lower()}_{str(resident_number).zfill(3)}"

        resident = Resident(
            resident_id=resident_id,
            name=first_name,
            age=age,
            occupation=random.choice(template.occupations),
            personality=self._generate_personality(archetype),
            backstory=generate_backstory(template, first_name, age),
        )

        # Set schedule based on archetype
        resident.schedule.hobbies = random.sample(template.hobbies, min(3, len(template.hobbies)))

        # Generate a project
        project = generate_project(archetype, resident_id)

        return resident, archetype, project

    def generate_neighborhood(self) -> Tuple[Dict[str, Resident], Dict[str, List[PersonalProject]], Dict[str, Archetype]]:
        """
        Generate a complete neighborhood of residents.

        Returns:
            - residents: Dict of resident_id -> Resident
            - projects: Dict of resident_id -> List[PersonalProject]
            - archetypes: Dict of resident_id -> Archetype (for reference)
        """
        residents = {}
        projects = {}
        archetypes = {}

        for i in range(1, self.config.num_residents + 1):
            resident, archetype, project = self.generate_resident(i)

            residents[resident.resident_id] = resident
            projects[resident.resident_id] = [project]
            archetypes[resident.resident_id] = archetype

        return residents, projects, archetypes

    def get_generation_summary(self) -> str:
        """Get a summary of generated residents."""
        lines = [
            f"Generated {sum(self.archetype_counts.values())} residents:",
            "",
            "Archetype Distribution:",
        ]

        for archetype, count in sorted(self.archetype_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"  {archetype.value}: {count}")

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_scaled_neighborhood(
    num_residents: int = 20,
    seed: int = None
) -> Tuple[Dict[str, Resident], Dict[str, List[PersonalProject]]]:
    """
    Convenience function to generate a neighborhood of any size.

    Args:
        num_residents: Number of residents to generate (default 20)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (residents dict, projects dict)
    """
    config = GenerationConfig(num_residents=num_residents, seed=seed)
    generator = ResidentGenerator(config)
    residents, projects, _ = generator.generate_neighborhood()
    return residents, projects


def expand_existing_neighborhood(
    existing_residents: Dict[str, Resident],
    additional_count: int = 10,
) -> Tuple[Dict[str, Resident], Dict[str, List[PersonalProject]]]:
    """
    Add new residents to an existing neighborhood.

    Args:
        existing_residents: Current residents dict
        additional_count: Number of new residents to add

    Returns:
        Tuple of (new_residents dict, new_projects dict)
    """
    config = GenerationConfig(num_residents=additional_count)
    generator = ResidentGenerator(config)

    # Mark existing names as used
    for resident in existing_residents.values():
        generator.used_names.add(resident.name)

    # Generate new residents with higher starting IDs
    start_id = len(existing_residents) + 1
    new_residents = {}
    new_projects = {}

    for i in range(additional_count):
        resident, archetype, project = generator.generate_resident(start_id + i)
        new_residents[resident.resident_id] = resident
        new_projects[resident.resident_id] = [project]

    return new_residents, new_projects


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RESIDENT GENERATOR DEMO")
    print("=" * 60)

    # Generate a neighborhood of 20 residents
    config = GenerationConfig(
        num_residents=20,
        seed=42,  # Reproducible
    )

    generator = ResidentGenerator(config)
    residents, projects, archetypes = generator.generate_neighborhood()

    print(f"\nGenerated {len(residents)} residents:\n")

    for resident_id, resident in list(residents.items())[:10]:  # Show first 10
        archetype = archetypes[resident_id]
        project = projects[resident_id][0]

        print(f"  {resident.name}, {resident.age} ({resident.occupation})")
        print(f"    Archetype: {archetype.value}")
        print(f"    Personality: {resident.personality.describe()}")
        print(f"    Project: {project.name}")
        print()

    print("  ... and", len(residents) - 10, "more residents")

    print("\n" + generator.get_generation_summary())

    # Demo: Expand to 50 residents
    print("\n" + "=" * 60)
    print("EXPANDING TO 50 RESIDENTS")
    print("=" * 60)

    new_residents, new_projects = expand_existing_neighborhood(residents, 30)

    all_residents = {**residents, **new_residents}
    print(f"\nTotal residents: {len(all_residents)}")

    # Show new residents
    print("\nNew additions:")
    for resident_id, resident in list(new_residents.items())[:5]:
        print(f"  {resident.name}, {resident.age} ({resident.occupation})")
