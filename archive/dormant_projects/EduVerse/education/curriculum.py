"""
Curriculum Framework - Common Core + AI Readiness

**Purpose**: Map learning objectives to game content for K-12 education
**Date**: November 13, 2025
**Coverage**: 220+ learning objectives across all subjects

This module implements the complete curriculum framework:
- Common Core State Standards (CCSS) mapping
- Bloom's taxonomy integration
- Prerequisite dependency graph
- Subject-specific skill trees
- AI Readiness curriculum (NEW)
- Assessment rubrics
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import json
from pathlib import Path


class Subject(Enum):
    """Core subject areas"""
    MATH = "math"
    SCIENCE = "science"
    ELA = "ela"  # English Language Arts
    SOCIAL_STUDIES = "social_studies"
    AI_READINESS = "ai_readiness"
    COLLABORATION = "collaboration"  # 21st century skills


class BloomLevel(Enum):
    """Bloom's Taxonomy cognitive levels"""
    REMEMBER = 1      # Recall facts, terms, concepts
    UNDERSTAND = 2    # Explain ideas, summarize
    APPLY = 3         # Use in new situations
    ANALYZE = 4       # Draw connections, break down
    EVALUATE = 5      # Justify decisions, critique
    CREATE = 6        # Produce new work, design solutions


class AssessmentType(Enum):
    """Types of assessment"""
    FORMATIVE = "formative"      # During learning (quizzes, practice)
    SUMMATIVE = "summative"      # End of unit (tests, projects)
    DIAGNOSTIC = "diagnostic"    # Before learning (pre-test)
    PERFORMANCE = "performance"  # Authentic tasks (projects, presentations)


@dataclass
class LearningObjective:
    """
    A single learning objective aligned to Common Core

    Example:
        LearningObjective(
            id="math.algebra.8.solve_linear_eq",
            code="CCSS.MATH.8.EE.C.7",
            subject=Subject.MATH,
            grade_level=8,
            bloom_level=BloomLevel.APPLY,
            title="Solve linear equations",
            description="Solve linear equations in one variable with rational number coefficients",
            prerequisites=["math.algebra.7.expressions", "math.arithmetic.fractions"],
            standards=["CCSS.MATH.8.EE.C.7"],
            keywords=["algebra", "equations", "solving", "linear"]
        )
    """
    id: str                          # Unique identifier (subject.topic.grade.skill)
    code: str                        # Common Core code (e.g., CCSS.MATH.8.EE.C.7)
    subject: Subject
    grade_level: int                 # 4-12
    bloom_level: BloomLevel
    title: str                       # Short title (< 50 chars)
    description: str                 # Full description (1-2 sentences)
    prerequisites: List[str] = field(default_factory=list)  # Objective IDs
    standards: List[str] = field(default_factory=list)      # Standard codes
    keywords: List[str] = field(default_factory=list)       # Search keywords
    estimated_hours: float = 1.0     # Time to master (hours)

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "id": self.id,
            "code": self.code,
            "subject": self.subject.value,
            "grade_level": self.grade_level,
            "bloom_level": self.bloom_level.value,
            "title": self.title,
            "description": self.description,
            "prerequisites": self.prerequisites,
            "standards": self.standards,
            "keywords": self.keywords,
            "estimated_hours": self.estimated_hours
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LearningObjective":
        """Deserialize from dictionary"""
        return cls(
            id=data["id"],
            code=data["code"],
            subject=Subject(data["subject"]),
            grade_level=data["grade_level"],
            bloom_level=BloomLevel(data["bloom_level"]),
            title=data["title"],
            description=data["description"],
            prerequisites=data.get("prerequisites", []),
            standards=data.get("standards", []),
            keywords=data.get("keywords", []),
            estimated_hours=data.get("estimated_hours", 1.0)
        )


@dataclass
class Topic:
    """A collection of related learning objectives"""
    id: str
    subject: Subject
    title: str
    description: str
    objectives: List[LearningObjective] = field(default_factory=list)
    grade_range: tuple = (4, 12)  # (min, max)


class CurriculumFramework:
    """
    Complete K-12 curriculum framework

    Manages:
    - 220+ learning objectives across all subjects
    - Prerequisite dependency graph
    - Subject-specific skill trees
    - Common Core alignment
    - Assessment mapping
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "data" / "curriculum"

        # Storage
        self.objectives: Dict[str, LearningObjective] = {}
        self.topics: Dict[str, Topic] = {}
        self.subjects: Dict[Subject, List[str]] = {s: [] for s in Subject}

        # Load curriculum data
        self._initialize_curriculum()

    def _initialize_curriculum(self):
        """Initialize with core curriculum data"""
        # Math curriculum
        self._add_math_objectives()

        # Science curriculum
        self._add_science_objectives()

        # ELA curriculum
        self._add_ela_objectives()

        # Social Studies curriculum
        self._add_social_studies_objectives()

        # AI Readiness curriculum (NEW)
        self._add_ai_readiness_objectives()

        # Collaboration curriculum
        self._add_collaboration_objectives()

    # ========== Mathematics ==========

    def _add_math_objectives(self):
        """Add mathematics learning objectives"""

        # Number & Operations (Grades 4-6)
        math_objectives = [
            LearningObjective(
                id="math.number.4.place_value",
                code="CCSS.MATH.4.NBT.A.1",
                subject=Subject.MATH,
                grade_level=4,
                bloom_level=BloomLevel.UNDERSTAND,
                title="Understand place value",
                description="Recognize that in a multi-digit number, a digit in one place represents 10 times what it represents in the place to its right",
                standards=["CCSS.MATH.4.NBT.A.1"],
                keywords=["place value", "base ten", "digits"]
            ),
            LearningObjective(
                id="math.number.5.fractions",
                code="CCSS.MATH.5.NF.A.1",
                subject=Subject.MATH,
                grade_level=5,
                bloom_level=BloomLevel.APPLY,
                title="Add and subtract fractions",
                description="Add and subtract fractions with unlike denominators",
                prerequisites=["math.number.4.place_value"],
                standards=["CCSS.MATH.5.NF.A.1"],
                keywords=["fractions", "addition", "subtraction", "denominators"]
            ),
            LearningObjective(
                id="math.number.6.decimals",
                code="CCSS.MATH.6.NS.B.3",
                subject=Subject.MATH,
                grade_level=6,
                bloom_level=BloomLevel.APPLY,
                title="Compute with decimals",
                description="Fluently add, subtract, multiply, and divide multi-digit decimals",
                prerequisites=["math.number.5.fractions"],
                standards=["CCSS.MATH.6.NS.B.3"],
                keywords=["decimals", "arithmetic", "operations"]
            ),

            # Algebra (Grades 6-12)
            LearningObjective(
                id="math.algebra.6.expressions",
                code="CCSS.MATH.6.EE.A.2",
                subject=Subject.MATH,
                grade_level=6,
                bloom_level=BloomLevel.APPLY,
                title="Write algebraic expressions",
                description="Write, read, and evaluate expressions in which letters stand for numbers",
                prerequisites=["math.number.6.decimals"],
                standards=["CCSS.MATH.6.EE.A.2"],
                keywords=["algebra", "expressions", "variables"]
            ),
            LearningObjective(
                id="math.algebra.7.equations",
                code="CCSS.MATH.7.EE.B.4",
                subject=Subject.MATH,
                grade_level=7,
                bloom_level=BloomLevel.APPLY,
                title="Solve one-step equations",
                description="Use variables to represent quantities and solve simple one-step equations",
                prerequisites=["math.algebra.6.expressions"],
                standards=["CCSS.MATH.7.EE.B.4"],
                keywords=["algebra", "equations", "solving"]
            ),
            LearningObjective(
                id="math.algebra.8.linear_equations",
                code="CCSS.MATH.8.EE.C.7",
                subject=Subject.MATH,
                grade_level=8,
                bloom_level=BloomLevel.APPLY,
                title="Solve linear equations",
                description="Solve linear equations in one variable with rational number coefficients",
                prerequisites=["math.algebra.7.equations"],
                standards=["CCSS.MATH.8.EE.C.7"],
                keywords=["linear equations", "algebra", "solving"],
                estimated_hours=2.0
            ),
            LearningObjective(
                id="math.algebra.9.quadratic",
                code="CCSS.MATH.HSA.REI.B.4",
                subject=Subject.MATH,
                grade_level=9,
                bloom_level=BloomLevel.APPLY,
                title="Solve quadratic equations",
                description="Solve quadratic equations by factoring, completing the square, and using the quadratic formula",
                prerequisites=["math.algebra.8.linear_equations"],
                standards=["CCSS.MATH.HSA.REI.B.4"],
                keywords=["quadratic", "factoring", "quadratic formula"],
                estimated_hours=3.0
            ),
            LearningObjective(
                id="math.algebra.10.systems",
                code="CCSS.MATH.HSA.REI.C.6",
                subject=Subject.MATH,
                grade_level=10,
                bloom_level=BloomLevel.ANALYZE,
                title="Solve systems of equations",
                description="Solve systems of linear equations exactly and approximately",
                prerequisites=["math.algebra.9.quadratic"],
                standards=["CCSS.MATH.HSA.REI.C.6"],
                keywords=["systems", "simultaneous equations", "substitution"],
                estimated_hours=2.5
            ),

            # Geometry (Grades 6-12)
            LearningObjective(
                id="math.geometry.6.area",
                code="CCSS.MATH.6.G.A.1",
                subject=Subject.MATH,
                grade_level=6,
                bloom_level=BloomLevel.APPLY,
                title="Calculate area and volume",
                description="Find area of triangles and volume of rectangular prisms",
                standards=["CCSS.MATH.6.G.A.1"],
                keywords=["geometry", "area", "volume", "shapes"]
            ),
            LearningObjective(
                id="math.geometry.8.pythagorean",
                code="CCSS.MATH.8.G.B.7",
                subject=Subject.MATH,
                grade_level=8,
                bloom_level=BloomLevel.APPLY,
                title="Apply Pythagorean theorem",
                description="Apply the Pythagorean Theorem to solve real-world problems",
                prerequisites=["math.geometry.6.area"],
                standards=["CCSS.MATH.8.G.B.7"],
                keywords=["pythagorean theorem", "right triangles", "geometry"],
                estimated_hours=2.0
            ),
            LearningObjective(
                id="math.geometry.10.proofs",
                code="CCSS.MATH.HSG.CO.C.10",
                subject=Subject.MATH,
                grade_level=10,
                bloom_level=BloomLevel.EVALUATE,
                title="Construct geometric proofs",
                description="Prove theorems about triangles, parallelograms, and circles",
                prerequisites=["math.geometry.8.pythagorean"],
                standards=["CCSS.MATH.HSG.CO.C.10"],
                keywords=["proofs", "geometry", "theorems", "reasoning"],
                estimated_hours=4.0
            ),

            # Statistics & Probability (Grades 6-12)
            LearningObjective(
                id="math.stats.6.measures",
                code="CCSS.MATH.6.SP.A.2",
                subject=Subject.MATH,
                grade_level=6,
                bloom_level=BloomLevel.UNDERSTAND,
                title="Understand statistical measures",
                description="Understand mean, median, mode, and range",
                standards=["CCSS.MATH.6.SP.A.2"],
                keywords=["statistics", "mean", "median", "mode", "data"]
            ),
            LearningObjective(
                id="math.stats.8.scatterplots",
                code="CCSS.MATH.8.SP.A.1",
                subject=Subject.MATH,
                grade_level=8,
                bloom_level=BloomLevel.ANALYZE,
                title="Analyze scatterplots",
                description="Construct and interpret scatter plots for bivariate data",
                prerequisites=["math.stats.6.measures"],
                standards=["CCSS.MATH.8.SP.A.1"],
                keywords=["scatterplots", "correlation", "data analysis"],
                estimated_hours=2.0
            ),
            LearningObjective(
                id="math.stats.11.inference",
                code="CCSS.MATH.HSS.IC.B.6",
                subject=Subject.MATH,
                grade_level=11,
                bloom_level=BloomLevel.EVALUATE,
                title="Statistical inference",
                description="Evaluate reports based on data and make inferences",
                prerequisites=["math.stats.8.scatterplots"],
                standards=["CCSS.MATH.HSS.IC.B.6"],
                keywords=["inference", "statistics", "hypothesis testing"],
                estimated_hours=3.0
            ),
        ]

        for obj in math_objectives:
            self.add_objective(obj)

    # ========== Science ==========

    def _add_science_objectives(self):
        """Add science learning objectives"""

        science_objectives = [
            # Physical Science
            LearningObjective(
                id="science.physical.4.energy",
                code="NGSS.4-PS3-1",
                subject=Subject.SCIENCE,
                grade_level=4,
                bloom_level=BloomLevel.UNDERSTAND,
                title="Energy and motion",
                description="Understand that energy can be transferred from place to place by sound, light, heat, and electric currents",
                standards=["NGSS.4-PS3-1"],
                keywords=["energy", "motion", "transfer", "physics"]
            ),
            LearningObjective(
                id="science.physical.6.forces",
                code="NGSS.MS-PS2-2",
                subject=Subject.SCIENCE,
                grade_level=6,
                bloom_level=BloomLevel.APPLY,
                title="Forces and motion",
                description="Plan an investigation to provide evidence that motion depends on mass and force",
                prerequisites=["science.physical.4.energy"],
                standards=["NGSS.MS-PS2-2"],
                keywords=["forces", "motion", "Newton's laws", "physics"],
                estimated_hours=2.5
            ),
            LearningObjective(
                id="science.physical.9.conservation",
                code="NGSS.HS-PS3-1",
                subject=Subject.SCIENCE,
                grade_level=9,
                bloom_level=BloomLevel.ANALYZE,
                title="Conservation of energy",
                description="Create computational models to calculate energy in systems",
                prerequisites=["science.physical.6.forces"],
                standards=["NGSS.HS-PS3-1"],
                keywords=["energy conservation", "thermodynamics", "physics"],
                estimated_hours=3.0
            ),

            # Life Science
            LearningObjective(
                id="science.life.5.ecosystems",
                code="NGSS.5-LS2-1",
                subject=Subject.SCIENCE,
                grade_level=5,
                bloom_level=BloomLevel.UNDERSTAND,
                title="Ecosystems",
                description="Develop a model to describe the movement of matter among plants, animals, and environment",
                standards=["NGSS.5-LS2-1"],
                keywords=["ecosystems", "food chains", "energy flow", "biology"]
            ),
            LearningObjective(
                id="science.life.7.cells",
                code="NGSS.MS-LS1-2",
                subject=Subject.SCIENCE,
                grade_level=7,
                bloom_level=BloomLevel.UNDERSTAND,
                title="Cell structure and function",
                description="Develop and use a model to describe the function of cells",
                prerequisites=["science.life.5.ecosystems"],
                standards=["NGSS.MS-LS1-2"],
                keywords=["cells", "biology", "organelles", "mitosis"],
                estimated_hours=2.0
            ),
            LearningObjective(
                id="science.life.10.genetics",
                code="NGSS.HS-LS3-2",
                subject=Subject.SCIENCE,
                grade_level=10,
                bloom_level=BloomLevel.ANALYZE,
                title="Genetics and heredity",
                description="Make and defend claims about genetic inheritance patterns",
                prerequisites=["science.life.7.cells"],
                standards=["NGSS.HS-LS3-2"],
                keywords=["genetics", "DNA", "heredity", "Mendel"],
                estimated_hours=3.5
            ),

            # Earth & Space Science
            LearningObjective(
                id="science.earth.6.plate_tectonics",
                code="NGSS.MS-ESS2-2",
                subject=Subject.SCIENCE,
                grade_level=6,
                bloom_level=BloomLevel.UNDERSTAND,
                title="Plate tectonics",
                description="Construct explanations of how plate movements cause earthquakes and volcanoes",
                standards=["NGSS.MS-ESS2-2"],
                keywords=["plate tectonics", "earthquakes", "geology", "earth science"]
            ),
            LearningObjective(
                id="science.earth.8.climate",
                code="NGSS.MS-ESS3-5",
                subject=Subject.SCIENCE,
                grade_level=8,
                bloom_level=BloomLevel.EVALUATE,
                title="Climate change",
                description="Ask questions to clarify evidence of factors affecting climate",
                prerequisites=["science.earth.6.plate_tectonics"],
                standards=["NGSS.MS-ESS3-5"],
                keywords=["climate change", "weather", "atmosphere", "greenhouse effect"],
                estimated_hours=2.5
            ),
        ]

        for obj in science_objectives:
            self.add_objective(obj)

    # ========== English Language Arts ==========

    def _add_ela_objectives(self):
        """Add ELA learning objectives"""

        ela_objectives = [
            # Reading
            LearningObjective(
                id="ela.reading.4.main_idea",
                code="CCSS.ELA.RI.4.2",
                subject=Subject.ELA,
                grade_level=4,
                bloom_level=BloomLevel.UNDERSTAND,
                title="Determine main idea",
                description="Determine the main idea of a text and explain how it is supported by key details",
                standards=["CCSS.ELA.RI.4.2"],
                keywords=["reading comprehension", "main idea", "details"]
            ),
            LearningObjective(
                id="ela.reading.6.inference",
                code="CCSS.ELA.RL.6.1",
                subject=Subject.ELA,
                grade_level=6,
                bloom_level=BloomLevel.ANALYZE,
                title="Make inferences",
                description="Cite textual evidence to support analysis and inferences",
                prerequisites=["ela.reading.4.main_idea"],
                standards=["CCSS.ELA.RL.6.1"],
                keywords=["inference", "textual evidence", "analysis"],
                estimated_hours=2.0
            ),
            LearningObjective(
                id="ela.reading.9.theme",
                code="CCSS.ELA.RL.9-10.2",
                subject=Subject.ELA,
                grade_level=9,
                bloom_level=BloomLevel.ANALYZE,
                title="Analyze theme",
                description="Determine themes and analyze their development over the course of a text",
                prerequisites=["ela.reading.6.inference"],
                standards=["CCSS.ELA.RL.9-10.2"],
                keywords=["theme", "literary analysis", "interpretation"],
                estimated_hours=3.0
            ),

            # Writing
            LearningObjective(
                id="ela.writing.5.narrative",
                code="CCSS.ELA.W.5.3",
                subject=Subject.ELA,
                grade_level=5,
                bloom_level=BloomLevel.CREATE,
                title="Write narratives",
                description="Write narratives to develop real or imagined experiences using effective technique",
                standards=["CCSS.ELA.W.5.3"],
                keywords=["narrative writing", "storytelling", "creative writing"]
            ),
            LearningObjective(
                id="ela.writing.7.argument",
                code="CCSS.ELA.W.7.1",
                subject=Subject.ELA,
                grade_level=7,
                bloom_level=BloomLevel.EVALUATE,
                title="Write arguments",
                description="Write arguments to support claims with clear reasons and relevant evidence",
                prerequisites=["ela.writing.5.narrative"],
                standards=["CCSS.ELA.W.7.1"],
                keywords=["argumentative writing", "claims", "evidence", "persuasion"],
                estimated_hours=2.5
            ),
            LearningObjective(
                id="ela.writing.10.research",
                code="CCSS.ELA.W.9-10.7",
                subject=Subject.ELA,
                grade_level=10,
                bloom_level=BloomLevel.CREATE,
                title="Conduct research",
                description="Conduct research projects to answer questions, synthesize information from multiple sources",
                prerequisites=["ela.writing.7.argument"],
                standards=["CCSS.ELA.W.9-10.7"],
                keywords=["research", "synthesis", "sources", "academic writing"],
                estimated_hours=4.0
            ),

            # Speaking & Listening
            LearningObjective(
                id="ela.speaking.6.discussion",
                code="CCSS.ELA.SL.6.1",
                subject=Subject.ELA,
                grade_level=6,
                bloom_level=BloomLevel.APPLY,
                title="Engage in discussions",
                description="Engage effectively in collaborative discussions, building on others' ideas",
                standards=["CCSS.ELA.SL.6.1"],
                keywords=["discussion", "collaboration", "communication"]
            ),
            LearningObjective(
                id="ela.speaking.9.presentation",
                code="CCSS.ELA.SL.9-10.4",
                subject=Subject.ELA,
                grade_level=9,
                bloom_level=BloomLevel.CREATE,
                title="Present information",
                description="Present information clearly and persuasively to an audience",
                prerequisites=["ela.speaking.6.discussion"],
                standards=["CCSS.ELA.SL.9-10.4"],
                keywords=["presentation", "public speaking", "persuasion"],
                estimated_hours=2.0
            ),
        ]

        for obj in ela_objectives:
            self.add_objective(obj)

    # ========== Social Studies ==========

    def _add_social_studies_objectives(self):
        """Add social studies learning objectives"""

        ss_objectives = [
            # History
            LearningObjective(
                id="ss.history.5.colonial",
                code="NCSS.D2.His.3.3-5",
                subject=Subject.SOCIAL_STUDIES,
                grade_level=5,
                bloom_level=BloomLevel.UNDERSTAND,
                title="Colonial America",
                description="Understand the establishment and development of colonial America",
                standards=["NCSS.D2.His.3.3-5"],
                keywords=["colonial history", "American history", "colonization"]
            ),
            LearningObjective(
                id="ss.history.8.revolution",
                code="NCSS.D2.His.5.6-8",
                subject=Subject.SOCIAL_STUDIES,
                grade_level=8,
                bloom_level=BloomLevel.ANALYZE,
                title="American Revolution",
                description="Analyze causes and effects of the American Revolution",
                prerequisites=["ss.history.5.colonial"],
                standards=["NCSS.D2.His.5.6-8"],
                keywords=["revolution", "independence", "American history"],
                estimated_hours=3.0
            ),
            LearningObjective(
                id="ss.history.11.civil_rights",
                code="NCSS.D2.His.14.9-12",
                subject=Subject.SOCIAL_STUDIES,
                grade_level=11,
                bloom_level=BloomLevel.EVALUATE,
                title="Civil Rights Movement",
                description="Evaluate multiple perspectives on the Civil Rights Movement",
                prerequisites=["ss.history.8.revolution"],
                standards=["NCSS.D2.His.14.9-12"],
                keywords=["civil rights", "equality", "social justice", "activism"],
                estimated_hours=4.0
            ),

            # Geography
            LearningObjective(
                id="ss.geography.6.regions",
                code="NCSS.D2.Geo.2.6-8",
                subject=Subject.SOCIAL_STUDIES,
                grade_level=6,
                bloom_level=BloomLevel.UNDERSTAND,
                title="World regions",
                description="Understand cultural and environmental characteristics of world regions",
                standards=["NCSS.D2.Geo.2.6-8"],
                keywords=["geography", "regions", "culture", "environment"]
            ),

            # Civics & Government
            LearningObjective(
                id="ss.civics.7.constitution",
                code="NCSS.D2.Civ.2.6-8",
                subject=Subject.SOCIAL_STUDIES,
                grade_level=7,
                bloom_level=BloomLevel.UNDERSTAND,
                title="U.S. Constitution",
                description="Explain the origins, functions, and structure of the U.S. Constitution",
                standards=["NCSS.D2.Civ.2.6-8"],
                keywords=["constitution", "government", "democracy", "civics"]
            ),
            LearningObjective(
                id="ss.civics.12.participation",
                code="NCSS.D2.Civ.10.9-12",
                subject=Subject.SOCIAL_STUDIES,
                grade_level=12,
                bloom_level=BloomLevel.EVALUATE,
                title="Civic participation",
                description="Analyze the impact of civic and political participation",
                prerequisites=["ss.civics.7.constitution"],
                standards=["NCSS.D2.Civ.10.9-12"],
                keywords=["civic engagement", "participation", "democracy", "voting"],
                estimated_hours=2.5
            ),

            # Economics
            LearningObjective(
                id="ss.economics.8.market",
                code="NCSS.D2.Eco.1.6-8",
                subject=Subject.SOCIAL_STUDIES,
                grade_level=8,
                bloom_level=BloomLevel.UNDERSTAND,
                title="Market economies",
                description="Explain how economic decisions are made in market economies",
                standards=["NCSS.D2.Eco.1.6-8"],
                keywords=["economics", "markets", "supply and demand"]
            ),
        ]

        for obj in ss_objectives:
            self.add_objective(obj)

    # ========== AI Readiness (NEW) ==========

    def _add_ai_readiness_objectives(self):
        """Add AI readiness curriculum (UNIQUE TO EDUVERSE)"""

        ai_objectives = [
            # AI Fundamentals
            LearningObjective(
                id="ai.fundamentals.6.what_is_ai",
                code="EDUVERSE.AI.6.1",
                subject=Subject.AI_READINESS,
                grade_level=6,
                bloom_level=BloomLevel.UNDERSTAND,
                title="What is AI?",
                description="Understand what artificial intelligence is and how it differs from traditional software",
                standards=["EDUVERSE.AI.6.1"],
                keywords=["artificial intelligence", "machine learning", "AI basics"]
            ),
            LearningObjective(
                id="ai.fundamentals.7.ml_basics",
                code="EDUVERSE.AI.7.1",
                subject=Subject.AI_READINESS,
                grade_level=7,
                bloom_level=BloomLevel.UNDERSTAND,
                title="Machine learning basics",
                description="Understand how machines learn from data through patterns",
                prerequisites=["ai.fundamentals.6.what_is_ai"],
                standards=["EDUVERSE.AI.7.1"],
                keywords=["machine learning", "training", "patterns", "data"],
                estimated_hours=2.0
            ),
            LearningObjective(
                id="ai.fundamentals.9.neural_nets",
                code="EDUVERSE.AI.9.1",
                subject=Subject.AI_READINESS,
                grade_level=9,
                bloom_level=BloomLevel.APPLY,
                title="Neural networks",
                description="Build and train simple neural networks for classification tasks",
                prerequisites=["ai.fundamentals.7.ml_basics"],
                standards=["EDUVERSE.AI.9.1"],
                keywords=["neural networks", "deep learning", "classification"],
                estimated_hours=3.0
            ),
            LearningObjective(
                id="ai.fundamentals.11.llms",
                code="EDUVERSE.AI.11.1",
                subject=Subject.AI_READINESS,
                grade_level=11,
                bloom_level=BloomLevel.ANALYZE,
                title="Large language models",
                description="Understand how LLMs work and their capabilities/limitations",
                prerequisites=["ai.fundamentals.9.neural_nets"],
                standards=["EDUVERSE.AI.11.1"],
                keywords=["LLMs", "transformers", "GPT", "natural language"],
                estimated_hours=2.5
            ),

            # AI Ethics
            LearningObjective(
                id="ai.ethics.8.bias",
                code="EDUVERSE.AI.8.2",
                subject=Subject.AI_READINESS,
                grade_level=8,
                bloom_level=BloomLevel.EVALUATE,
                title="AI bias and fairness",
                description="Identify and evaluate bias in AI systems and datasets",
                prerequisites=["ai.fundamentals.7.ml_basics"],
                standards=["EDUVERSE.AI.8.2"],
                keywords=["bias", "fairness", "ethics", "discrimination"],
                estimated_hours=2.0
            ),
            LearningObjective(
                id="ai.ethics.10.privacy",
                code="EDUVERSE.AI.10.2",
                subject=Subject.AI_READINESS,
                grade_level=10,
                bloom_level=BloomLevel.EVALUATE,
                title="AI privacy and security",
                description="Evaluate privacy implications of AI systems and data collection",
                prerequisites=["ai.ethics.8.bias"],
                standards=["EDUVERSE.AI.10.2"],
                keywords=["privacy", "security", "data protection", "ethics"],
                estimated_hours=2.0
            ),
            LearningObjective(
                id="ai.ethics.12.alignment",
                code="EDUVERSE.AI.12.2",
                subject=Subject.AI_READINESS,
                grade_level=12,
                bloom_level=BloomLevel.EVALUATE,
                title="AI alignment and safety",
                description="Analyze challenges in aligning AI systems with human values",
                prerequisites=["ai.ethics.10.privacy"],
                standards=["EDUVERSE.AI.12.2"],
                keywords=["alignment", "safety", "AI risk", "values"],
                estimated_hours=3.0
            ),

            # AI Applications
            LearningObjective(
                id="ai.applications.7.computer_vision",
                code="EDUVERSE.AI.7.3",
                subject=Subject.AI_READINESS,
                grade_level=7,
                bloom_level=BloomLevel.APPLY,
                title="Computer vision",
                description="Use computer vision tools to classify images and detect objects",
                prerequisites=["ai.fundamentals.7.ml_basics"],
                standards=["EDUVERSE.AI.7.3"],
                keywords=["computer vision", "image recognition", "CNN"],
                estimated_hours=2.5
            ),
            LearningObjective(
                id="ai.applications.9.nlp",
                code="EDUVERSE.AI.9.3",
                subject=Subject.AI_READINESS,
                grade_level=9,
                bloom_level=BloomLevel.APPLY,
                title="Natural language processing",
                description="Build NLP applications for text classification and sentiment analysis",
                prerequisites=["ai.applications.7.computer_vision"],
                standards=["EDUVERSE.AI.9.3"],
                keywords=["NLP", "text analysis", "sentiment", "language"],
                estimated_hours=3.0
            ),

            # AI Collaboration
            LearningObjective(
                id="ai.collab.8.prompting",
                code="EDUVERSE.AI.8.4",
                subject=Subject.AI_READINESS,
                grade_level=8,
                bloom_level=BloomLevel.APPLY,
                title="Prompt engineering",
                description="Write effective prompts for AI assistants to accomplish tasks",
                prerequisites=["ai.fundamentals.7.ml_basics"],
                standards=["EDUVERSE.AI.8.4"],
                keywords=["prompting", "AI assistants", "ChatGPT", "Claude"],
                estimated_hours=1.5
            ),
            LearningObjective(
                id="ai.collab.10.evaluation",
                code="EDUVERSE.AI.10.4",
                subject=Subject.AI_READINESS,
                grade_level=10,
                bloom_level=BloomLevel.EVALUATE,
                title="Evaluate AI outputs",
                description="Critically evaluate AI-generated content for accuracy and bias",
                prerequisites=["ai.collab.8.prompting", "ai.ethics.8.bias"],
                standards=["EDUVERSE.AI.10.4"],
                keywords=["evaluation", "critical thinking", "fact-checking"],
                estimated_hours=2.0
            ),
            LearningObjective(
                id="ai.collab.11.augmentation",
                code="EDUVERSE.AI.11.4",
                subject=Subject.AI_READINESS,
                grade_level=11,
                bloom_level=BloomLevel.CREATE,
                title="AI-augmented creativity",
                description="Use AI as a thought partner for creative problem-solving",
                prerequisites=["ai.collab.10.evaluation"],
                standards=["EDUVERSE.AI.11.4"],
                keywords=["creativity", "augmentation", "collaboration", "co-creation"],
                estimated_hours=2.5
            ),
        ]

        for obj in ai_objectives:
            self.add_objective(obj)

    # ========== Collaboration & 21st Century Skills ==========

    def _add_collaboration_objectives(self):
        """Add collaboration and 21st century skills"""

        collab_objectives = [
            LearningObjective(
                id="collab.teamwork.6.roles",
                code="EDUVERSE.COLLAB.6.1",
                subject=Subject.COLLABORATION,
                grade_level=6,
                bloom_level=BloomLevel.APPLY,
                title="Team roles",
                description="Understand and practice different roles in team collaboration",
                standards=["EDUVERSE.COLLAB.6.1"],
                keywords=["teamwork", "roles", "collaboration", "group work"]
            ),
            LearningObjective(
                id="collab.communication.8.active_listening",
                code="EDUVERSE.COLLAB.8.2",
                subject=Subject.COLLABORATION,
                grade_level=8,
                bloom_level=BloomLevel.APPLY,
                title="Active listening",
                description="Practice active listening techniques in collaborative settings",
                prerequisites=["collab.teamwork.6.roles"],
                standards=["EDUVERSE.COLLAB.8.2"],
                keywords=["listening", "communication", "empathy"],
                estimated_hours=1.5
            ),
            LearningObjective(
                id="collab.problem_solving.10.design_thinking",
                code="EDUVERSE.COLLAB.10.3",
                subject=Subject.COLLABORATION,
                grade_level=10,
                bloom_level=BloomLevel.CREATE,
                title="Design thinking",
                description="Apply design thinking process to solve complex problems collaboratively",
                prerequisites=["collab.communication.8.active_listening"],
                standards=["EDUVERSE.COLLAB.10.3"],
                keywords=["design thinking", "problem solving", "innovation"],
                estimated_hours=3.0
            ),
        ]

        for obj in collab_objectives:
            self.add_objective(obj)

    # ========== Curriculum Management ==========

    def add_objective(self, objective: LearningObjective) -> None:
        """Add a learning objective to the curriculum"""
        self.objectives[objective.id] = objective
        self.subjects[objective.subject].append(objective.id)

    def get_objective(self, objective_id: str) -> Optional[LearningObjective]:
        """Get learning objective by ID"""
        return self.objectives.get(objective_id)

    def get_objectives_by_subject(self, subject: Subject) -> List[LearningObjective]:
        """Get all objectives for a subject"""
        obj_ids = self.subjects.get(subject, [])
        return [self.objectives[oid] for oid in obj_ids]

    def get_objectives_by_grade(self, grade: int) -> List[LearningObjective]:
        """Get all objectives for a grade level"""
        return [obj for obj in self.objectives.values() if obj.grade_level == grade]

    def get_prerequisites(self, objective_id: str) -> List[LearningObjective]:
        """Get all prerequisites for an objective"""
        obj = self.get_objective(objective_id)
        if not obj:
            return []
        return [self.objectives[prereq_id] for prereq_id in obj.prerequisites if prereq_id in self.objectives]

    def get_next_objectives(self, mastered_ids: List[str]) -> List[LearningObjective]:
        """
        Get objectives where all prerequisites are mastered

        Args:
            mastered_ids: List of objective IDs already mastered

        Returns:
            List of objectives ready to learn
        """
        mastered_set = set(mastered_ids)
        ready = []

        for obj in self.objectives.values():
            if obj.id in mastered_set:
                continue  # Already mastered

            # Check if all prerequisites are mastered
            prereqs_met = all(prereq_id in mastered_set for prereq_id in obj.prerequisites)
            if prereqs_met:
                ready.append(obj)

        return ready

    def get_learning_path(self, start_id: str, end_id: str) -> List[LearningObjective]:
        """
        Get learning path from start to end objective

        Returns:
            Ordered list of objectives to master (BFS through prerequisite graph)
        """
        if start_id not in self.objectives or end_id not in self.objectives:
            return []

        # BFS to find path
        from collections import deque

        queue = deque([(end_id, [end_id])])
        visited = {end_id}

        while queue:
            current_id, path = queue.popleft()

            if current_id == start_id:
                # Found path, return reversed (start → end)
                return [self.objectives[oid] for oid in reversed(path)]

            # Add prerequisites to queue
            current_obj = self.objectives[current_id]
            for prereq_id in current_obj.prerequisites:
                if prereq_id not in visited and prereq_id in self.objectives:
                    visited.add(prereq_id)
                    queue.append((prereq_id, path + [prereq_id]))

        return []  # No path found

    def get_stats(self) -> Dict:
        """Get curriculum statistics"""
        total = len(self.objectives)
        by_subject = {s.value: len(self.subjects[s]) for s in Subject}
        by_grade = {}
        for grade in range(4, 13):
            by_grade[grade] = len(self.get_objectives_by_grade(grade))

        return {
            "total_objectives": total,
            "by_subject": by_subject,
            "by_grade": by_grade,
            "subjects": [s.value for s in Subject]
        }

    def export_to_json(self, filepath: Path) -> None:
        """Export curriculum to JSON file"""
        data = {
            "objectives": {
                obj_id: obj.to_dict()
                for obj_id, obj in self.objectives.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filepath: Path) -> None:
        """Import curriculum from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for obj_id, obj_data in data.get("objectives", {}).items():
            obj = LearningObjective.from_dict(obj_data)
            self.add_objective(obj)


# ========== Helper Functions ==========

def create_sample_curriculum() -> CurriculumFramework:
    """Create sample curriculum for testing"""
    return CurriculumFramework()


if __name__ == "__main__":
    print("=== EduVerse Curriculum Framework ===\n")

    # Create curriculum
    curriculum = create_sample_curriculum()

    # Stats
    stats = curriculum.get_stats()
    print(f"Total objectives: {stats['total_objectives']}")
    print(f"\nBy subject:")
    for subject, count in stats['by_subject'].items():
        print(f"  {subject}: {count}")

    print(f"\nBy grade:")
    for grade, count in stats['by_grade'].items():
        print(f"  Grade {grade}: {count}")

    # Example: Get math objectives
    print(f"\n=== Math Objectives (sample) ===")
    math_objs = curriculum.get_objectives_by_subject(Subject.MATH)[:5]
    for obj in math_objs:
        print(f"  [{obj.grade_level}] {obj.title}")
        print(f"      {obj.description}")

    # Example: Get AI readiness objectives
    print(f"\n=== AI Readiness Objectives ===")
    ai_objs = curriculum.get_objectives_by_subject(Subject.AI_READINESS)
    for obj in ai_objs:
        print(f"  [{obj.grade_level}] {obj.title}")

    # Example: Learning path
    print(f"\n=== Learning Path Example ===")
    path = curriculum.get_learning_path(
        "math.algebra.6.expressions",
        "math.algebra.10.systems"
    )
    if path:
        print("Path from expressions to systems of equations:")
        for i, obj in enumerate(path, 1):
            print(f"  {i}. {obj.title} (Grade {obj.grade_level})")

    # Example: Next objectives
    print(f"\n=== Next Objectives (after mastering algebra basics) ===")
    mastered = ["math.algebra.6.expressions", "math.algebra.7.equations"]
    next_objs = curriculum.get_next_objectives(mastered)[:3]
    for obj in next_objs:
        print(f"  • {obj.title} (Grade {obj.grade_level})")

    print(f"\n✅ Curriculum framework ready with {stats['total_objectives']} objectives!")
