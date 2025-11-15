# EdWIN Quick Start Guide

**Version**: 1.0.0
**Date**: November 15, 2025
**Audience**: Developers getting started with EdWIN

---

## Overview

This guide will get you from zero to a working EdWIN AI tutor in **30 minutes**.

### What You'll Build

By the end of this guide, you'll have:
- ✅ Curriculum knowledge graph with 220+ objectives
- ✅ Working AI tutor that answers student questions
- ✅ Student progress tracking
- ✅ Adaptive difficulty recommendations
- ✅ K-12 safety guardrails

---

## Prerequisites

### System Requirements

- Python 3.10+
- 8 GB RAM minimum
- Docker (for Neo4j + Qdrant)
- Git

### Knowledge Requirements

- Basic Python
- Familiarity with async/await
- Understanding of REST APIs (for API usage)

---

## Step 1: Installation (5 minutes)

### Clone Repository

```bash
git clone https://github.com/yourusername/hello-world.git
cd hello-world
```

### Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install HoloLoom + EdWIN dependencies
pip install -r requirements.txt

# Install optional dependencies (for full features)
pip install spacy sentence-transformers anthropic
python -m spacy download en_core_web_sm
```

### Start Backend Services (Optional)

For persistent storage, start Neo4j + Qdrant:

```bash
# Start Docker services
docker-compose up -d

# Verify services are running
docker ps
# Should show neo4j and qdrant containers
```

**Note**: EdWIN works without Docker (uses in-memory storage), but Docker enables persistence.

---

## Step 2: Initialize Curriculum (5 minutes)

### Create Curriculum Knowledge Graph

Create `init_curriculum.py`:

```python
"""Initialize EdWIN curriculum knowledge graph"""

import asyncio
from EduVerse.education.curriculum import CurriculumFramework
from HoloLoom.memory.graph import KG, KGEdge

async def init_curriculum():
    """Load curriculum into knowledge graph"""

    print("🎓 Initializing EdWIN curriculum...")

    # 1. Load EduVerse curriculum (220+ objectives)
    curriculum = CurriculumFramework()

    # 2. Create HoloLoom knowledge graph
    kg = KG()

    # 3. Ingest curriculum
    print(f"📚 Ingesting {len(curriculum.objectives)} objectives...")

    for obj_id, obj in curriculum.objectives.items():
        # Add objective node
        kg.add_edges([
            KGEdge(
                src=obj_id,
                dst=f"subject_{obj.subject.value}",
                type="BELONGS_TO",
                weight=1.0,
                metadata={
                    "title": obj.title,
                    "description": obj.description,
                    "grade_level": obj.grade_level,
                    "bloom_level": obj.bloom_level.value,
                    "estimated_hours": obj.estimated_hours,
                    "keywords": obj.keywords
                }
            )
        ])

        # Add prerequisite edges
        for prereq_id in obj.prerequisites:
            kg.add_edges([
                KGEdge(
                    src=obj_id,
                    dst=prereq_id,
                    type="REQUIRES",
                    weight=1.0
                )
            ])

    # 4. Save to file (or Neo4j if available)
    kg.save("./data/curriculum_graph.json")

    print(f"✅ Curriculum initialized!")
    print(f"   - Objectives: {len(curriculum.objectives)}")
    print(f"   - Subjects: {len(curriculum.subjects)}")
    print(f"   - Edges: {len(kg.edges)}")

    return kg, curriculum

if __name__ == "__main__":
    asyncio.run(init_curriculum())
```

Run it:

```bash
python init_curriculum.py
```

Output:
```
🎓 Initializing EdWIN curriculum...
📚 Ingesting 220 objectives...
✅ Curriculum initialized!
   - Objectives: 220
   - Subjects: 6
   - Edges: 450+
```

---

## Step 3: Create Your First Tutor (10 minutes)

### Basic Tutor

Create `simple_tutor.py`:

```python
"""Simple EdWIN tutor demo"""

import asyncio
from HoloLoom.rag import SimpleRAG
from HoloLoom.config import Config
from EduVerse.education.curriculum import CurriculumFramework

async def simple_tutor_demo():
    """Demonstrate basic tutoring"""

    # 1. Load curriculum
    curriculum = CurriculumFramework()

    # 2. Create RAG tutor
    config = Config.fused()  # Full HoloLoom features

    rag = SimpleRAG(
        config=config,
        llm_provider="anthropic",  # Or "ollama" for local
        llm_model="claude-3-5-sonnet-20241022"
    )

    # 3. Ingest some curriculum content
    print("📚 Ingesting curriculum content...")

    # Get math objectives
    math_objs = curriculum.get_objectives_by_subject(
        curriculum.Subject.MATH
    )

    for obj in math_objs[:10]:  # First 10 for demo
        await rag.ingest(
            f"{obj.title}: {obj.description}"
        )

    print(f"✅ Ingested {10} math objectives")

    # 4. Ask questions
    questions = [
        "What is a linear equation?",
        "How do I solve 2x + 5 = 13?",
        "What's the Pythagorean theorem?"
    ]

    print("\n🎓 EdWIN Tutor Demo\n")

    for question in questions:
        print(f"Student: {question}")

        result = await rag.query(
            question,
            mode="verify"  # Verify answers for accuracy
        )

        print(f"EdWIN: {result.response}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Sources: {len(result.sources)}")
        print()

if __name__ == "__main__":
    asyncio.run(simple_tutor_demo())
```

Run it:

```bash
# Set API key (if using Anthropic)
export ANTHROPIC_API_KEY=your_key_here

# Run demo
python simple_tutor.py
```

Expected output:
```
📚 Ingesting curriculum content...
✅ Ingested 10 math objectives

🎓 EdWIN Tutor Demo

Student: What is a linear equation?
EdWIN: A linear equation is an equation in one variable where...
Confidence: 92.5%
Sources: 3

Student: How do I solve 2x + 5 = 13?
EdWIN: To solve 2x + 5 = 13, follow these steps...
Confidence: 95.0%
Sources: 2
```

---

## Step 4: Add Student Tracking (5 minutes)

### Track Progress

Create `student_tracking_demo.py`:

```python
"""Demonstrate student progress tracking"""

import asyncio
from EduVerse.education.player_model import PlayerModel, Skill, SkillLevel
from datetime import datetime

async def student_tracking_demo():
    """Track a student's learning journey"""

    # 1. Create student
    student = PlayerModel(
        student_id="student_001",
        name="Jane Doe",
        grade=8
    )

    print(f"🎓 Student: {student.name} (Grade {student.grade})")
    print(f"   Level: {student.stats.level}")
    print(f"   XP: {student.stats.total_xp}")

    # 2. Add skills
    algebra_skill = Skill(
        id="math.algebra.8.linear_equations",
        name="Solve linear equations",
        subject="math",
        description="Solve linear equations in one variable",
        level=SkillLevel.NOVICE,
        xp=0.0
    )

    student.add_skill(algebra_skill)
    print(f"\n📚 Added skill: {algebra_skill.name}")

    # 3. Simulate practice sessions
    print("\n🎯 Practice sessions:")

    sessions = [
        (True, 0.8, "First attempt - good effort!"),
        (True, 0.9, "Second attempt - improving!"),
        (False, 0.4, "Third attempt - struggling"),
        (True, 0.95, "Fourth attempt - mastered!"),
    ]

    for i, (success, confidence, comment) in enumerate(sessions, 1):
        leveled_up = student.gain_xp(
            skill_id=algebra_skill.id,
            xp_amount=10.0 * confidence,
            success=success
        )

        skill = student.skills[algebra_skill.id]

        print(f"\n   Session {i}: {comment}")
        print(f"   Success: {success}, Confidence: {confidence:.1%}")
        print(f"   XP gained: {10.0 * confidence:.1f}")
        print(f"   Current level: {skill.level.name}")
        print(f"   Success rate: {skill.success_rate:.1%}")

        if leveled_up:
            print(f"   🎉 LEVELED UP to {skill.level.name}!")

    # 4. Check mastery
    mastery = student.get_skill_mastery(algebra_skill.id)
    print(f"\n📊 Overall mastery: {mastery:.1%}")

    # 5. Get unlocked skills
    unlocked = student.get_unlocked_skills()
    print(f"\n🔓 Unlocked skills: {len(unlocked)}")

if __name__ == "__main__":
    asyncio.run(student_tracking_demo())
```

Run it:

```bash
python student_tracking_demo.py
```

---

## Step 5: Add Adaptive Difficulty (5 minutes)

### Thompson Sampling for Optimal Challenge

Create `adaptive_difficulty_demo.py`:

```python
"""Demonstrate adaptive difficulty selection"""

import asyncio
from EduVerse.education.curriculum import CurriculumFramework
from EduVerse.education.player_model import PlayerModel
from HoloLoom.policy.thompson_sampling import TSBandit
import random

async def adaptive_difficulty_demo():
    """Select optimal challenge using Thompson Sampling"""

    # 1. Setup
    curriculum = CurriculumFramework()
    student = PlayerModel(
        student_id="student_001",
        name="Jane Doe",
        grade=8
    )

    # 2. Get available objectives
    grade_objectives = curriculum.get_objectives_by_grade(8)

    print(f"🎓 Student: {student.name} (Grade {student.grade})")
    print(f"📚 Available objectives: {len(grade_objectives)}")

    # 3. Thompson Sampling bandit
    bandit = TSBandit(n_arms=len(grade_objectives))

    # 4. Simulate learning sessions
    print("\n🎯 Adaptive difficulty in action:\n")

    for session in range(5):
        # Thompson sample: select next objective
        selected_idx = bandit.select_arm()
        selected_obj = grade_objectives[selected_idx]

        print(f"Session {session + 1}:")
        print(f"  Selected: {selected_obj.title}")
        print(f"  Difficulty: {selected_obj.bloom_level.name}")

        # Simulate student attempt
        # Higher bloom = harder = lower success probability
        success_prob = 1.0 - (selected_obj.bloom_level.value / 7)
        success = random.random() < success_prob

        # Update bandit
        reward = 1.0 if success else 0.0
        bandit.update(selected_idx, reward)

        print(f"  Result: {'✅ Success' if success else '❌ Failed'}")
        print(f"  Reward: {reward:.1f}")
        print()

    # 5. Show learned preferences
    print("📊 Learned preferences:")
    stats = bandit.get_stats()

    # Show top 3 objectives by expected reward
    top_indices = sorted(
        range(len(grade_objectives)),
        key=lambda i: stats["alpha"][i] / (stats["alpha"][i] + stats["beta"][i]),
        reverse=True
    )[:3]

    for rank, idx in enumerate(top_indices, 1):
        obj = grade_objectives[idx]
        expected_reward = stats["alpha"][idx] / (stats["alpha"][idx] + stats["beta"][idx])
        print(f"  {rank}. {obj.title} (expected reward: {expected_reward:.2f})")

if __name__ == "__main__":
    asyncio.run(adaptive_difficulty_demo())
```

Run it:

```bash
python adaptive_difficulty_demo.py
```

Expected output:
```
🎓 Student: Jane Doe (Grade 8)
📚 Available objectives: 15

🎯 Adaptive difficulty in action:

Session 1:
  Selected: Solve linear equations
  Difficulty: APPLY
  Result: ✅ Success
  Reward: 1.0

Session 2:
  Selected: Analyze scatterplots
  Difficulty: ANALYZE
  Result: ❌ Failed
  Reward: 0.0

...

📊 Learned preferences:
  1. Solve linear equations (expected reward: 0.85)
  2. Apply Pythagorean theorem (expected reward: 0.72)
  3. Understand statistical measures (expected reward: 0.68)
```

---

## Step 6: Add Safety Guardrails (Optional)

### K-12 Content Filtering

Create `safety_demo.py`:

```python
"""Demonstrate K-12 safety guardrails"""

import asyncio
from HoloLoom.alignment import SafetyGuardrails
import textstat  # For reading level

async def safety_demo():
    """Test safety guardrails"""

    # 1. Create safety layer
    guardrails = SafetyGuardrails(enable_human_in_loop=False)

    print("🛡️ EdWIN Safety Guardrails Demo\n")

    # 2. Test cases
    test_cases = [
        {
            "response": "Photosynthesis is how plants make food using sunlight.",
            "grade": 5,
            "should_pass": True,
            "reason": "Age-appropriate science content"
        },
        {
            "response": "The mitochondrial electron transport chain couples oxidative phosphorylation...",
            "grade": 5,
            "should_pass": False,
            "reason": "Too complex (college-level)"
        },
        {
            "response": "Simple explanation at 3rd grade level.",
            "grade": 8,
            "should_pass": True,
            "reason": "Below grade level is OK"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['reason']}")
        print(f"Response: {test['response'][:60]}...")
        print(f"Grade: {test['grade']}")

        # Check reading level
        reading_level = textstat.flesch_kincaid_grade(test['response'])
        print(f"Reading level: {reading_level:.1f}")

        # Safety check
        gate_result = await guardrails.gate_action(
            action="respond_to_student",
            context={
                "response": test['response'],
                "grade": test['grade']
            }
        )

        passed = gate_result.allowed and reading_level <= test['grade'] + 2

        print(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
        if not gate_result.allowed:
            print(f"Reason: {gate_result.reason}")
        elif reading_level > test['grade'] + 2:
            print(f"Reason: Reading level too high ({reading_level:.1f} > {test['grade'] + 2})")

        print()

if __name__ == "__main__":
    # Note: Requires textstat library
    # pip install textstat
    asyncio.run(safety_demo())
```

---

## Common Workflows

### Workflow 1: Answer Student Question

```python
from HoloLoom.rag import SimpleRAG

async def answer_question(question: str, grade: int):
    """Answer a student question"""

    rag = SimpleRAG()

    # Add grade-level context
    context = f"Explain at grade {grade} level: {question}"

    result = await rag.query(context, mode="verify")

    return result.response
```

### Workflow 2: Generate Learning Path

```python
from EduVerse.education.curriculum import CurriculumFramework

def get_learning_path(student_grade: int, target_objective_id: str):
    """Get learning path to target objective"""

    curriculum = CurriculumFramework()

    # Get current grade objectives
    current_objs = curriculum.get_objectives_by_grade(student_grade)

    # Find path to target
    path = curriculum.get_learning_path(
        start_id=current_objs[0].id,
        end_id=target_objective_id
    )

    return path
```

### Workflow 3: Track Progress

```python
async def record_session(
    student: PlayerModel,
    objective_id: str,
    success: bool,
    confidence: float
):
    """Record a learning session"""

    # Update skill XP
    leveled_up = student.gain_xp(
        skill_id=objective_id,
        xp_amount=10.0 * confidence,
        success=success
    )

    # Check for mastery
    mastery = student.get_skill_mastery(objective_id)

    return {
        "leveled_up": leveled_up,
        "mastery": mastery,
        "total_xp": student.stats.total_xp
    }
```

---

## Next Steps

### Production Deployment

1. **Start API Server**:
   ```bash
   uvicorn EduVerse.edwin.api:app --reload --port 8000
   ```

2. **Test API**:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Create Student**:
   ```bash
   curl -X POST http://localhost:8000/students \
     -H "Content-Type: application/json" \
     -d '{"name": "Jane Doe", "grade": 8}'
   ```

### Advanced Features

- [ ] **Multimodal Support**: Add images, videos, interactive demos
- [ ] **Teacher Dashboard**: Progress monitoring and intervention
- [ ] **Parent Reporting**: Weekly progress reports
- [ ] **Custom Curriculum**: Add your own learning objectives
- [ ] **Integration**: Connect to LMS (Canvas, Moodle, etc.)

### Learning Resources

- **Technical Spec**: `EDWIN_TECHNICAL_SPECIFICATION.md`
- **API Reference**: `EDWIN_API_REFERENCE.md`
- **HoloLoom Docs**: `../HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md`
- **EduVerse Curriculum**: `education/curriculum.py`

---

## Troubleshooting

### Issue: "Module not found"

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=.
python your_script.py
```

### Issue: "Docker services not starting"

```bash
# Check Docker is running
docker ps

# Restart services
docker-compose down
docker-compose up -d
```

### Issue: "LLM API key not set"

```bash
# Set Anthropic key
export ANTHROPIC_API_KEY=your_key_here

# Or use Ollama (local, free)
# No API key needed!
```

### Issue: "High latency"

- Use `Config.fast()` instead of `Config.fused()` (2-3x faster)
- Enable caching: `SimpleRAG(enable_caching=True)`
- Use Ollama for local inference (no API latency)

---

## Summary

You've now built a complete AI tutor with:

✅ **Curriculum Knowledge Graph** (220+ objectives)
✅ **RAG-Powered Q&A** (context-aware answers)
✅ **Student Progress Tracking** (skills, mastery, XP)
✅ **Adaptive Difficulty** (Thompson Sampling)
✅ **Safety Guardrails** (K-12 filtering)

**Total time**: ~30 minutes
**Lines of code**: ~200
**Features unlocked**: Complete AI tutoring system

---

## What's Next?

1. **Explore**: Try different questions and grade levels
2. **Customize**: Add your own curriculum objectives
3. **Extend**: Build teacher dashboard, parent reporting
4. **Deploy**: Launch in production with FastAPI
5. **Scale**: Add more students, optimize performance

**Happy teaching! 🎓**

---

**Document Version**: 1.0.0
**Last Updated**: November 15, 2025
**Questions?** See `EDWIN_TECHNICAL_SPECIFICATION.md` for details
