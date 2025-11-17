# LMS Plugin Development Guide
**Date**: 2025-11-17
**Version**: 1.0
**Audience**: Plugin developers building extensions for the LMS orchestration ecosystem

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Plugin Architecture](#plugin-architecture)
4. [Development Setup](#development-setup)
5. [Plugin Structure](#plugin-structure)
6. [Hook System](#hook-system)
7. [API Reference](#api-reference)
8. [UI Components](#ui-components)
9. [Testing](#testing)
10. [Security](#security)
11. [Publishing](#publishing)
12. [Best Practices](#best-practices)
13. [Examples](#examples)

---

## Introduction

### What is a Plugin?

A plugin extends the LMS orchestration platform with new functionality. Like WordPress plugins, LMS plugins can:

- **Add new content types** (interactive videos, 3D models, simulations)
- **Provide assessment tools** (quizzes, peer review, portfolios)
- **Enable integrations** (Zoom, GitHub, Google Workspace)
- **Enhance analytics** (dashboards, predictive models)
- **Improve communication** (forums, chat, notifications)

### Plugin Categories

Plugins belong to one of five categories:

| Category | Purpose | Examples |
|----------|---------|----------|
| **Assessment** | Evaluate student learning | Quiz Builder, Peer Review, Portfolio |
| **Content** | Deliver learning materials | Video Player, Interactive Textbook, Code Playground |
| **Analytics** | Track and analyze data | Dashboard, Predictive Analytics, Heatmap |
| **Communication** | Enable interaction | Forum, Chat, Office Hours |
| **Integration** | Connect external services | Zoom, GitHub, Google Workspace, SIS |

### Why Build Plugins?

- **Monetization**: Sell premium plugins in the marketplace (you keep 70%)
- **Community**: Build reputation as an educational technology innovator
- **Flexibility**: Solve specific problems for your institution
- **Open Source**: Contribute to educational innovation

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for UI components)
- Git
- Basic knowledge of FastAPI and React

### Install the SDK

```bash
# Install LMS Plugin SDK
pip install lms-plugin-sdk

# Verify installation
lms-cli version
# Output: LMS Plugin SDK v1.0.0
```

### Create Your First Plugin

```bash
# Scaffold a new plugin
lms-cli create-plugin \
  --name "simple-quiz" \
  --category assessment \
  --author "Your Name"

# Output:
# Created plugin structure in ./simple-quiz/
# ├── plugin.yaml          # Plugin metadata
# ├── backend/
# │   ├── __init__.py
# │   ├── hooks.py         # Hook implementations
# │   └── api.py           # API endpoints
# ├── frontend/
# │   ├── components/      # React components
# │   └── index.tsx        # Entry point
# ├── tests/
# │   ├── test_hooks.py
# │   └── test_api.py
# └── README.md
```

### Implement Core Functionality

Edit `backend/hooks.py`:

```python
from lms_plugin_sdk import Hook, Context, HookResponse

class SimpleQuizHooks:
    @Hook.register("after_assessment_submit")
    async def on_quiz_submit(self, context: Context) -> HookResponse:
        """Called when student submits a quiz"""
        submission = context.data["submission"]
        student = context.student
        assessment = context.data["assessment"]

        # Grade the quiz
        score = self.grade_quiz(submission, assessment)

        # Update knowledge graph
        await context.knowledge_graph.record_learning_event(
            student_id=student.id,
            concept=assessment.concept,
            mastery_level=score,
            evidence=f"Quiz score: {score}"
        )

        return HookResponse(
            success=True,
            data={"score": score, "feedback": self.generate_feedback(score)}
        )

    def grade_quiz(self, submission, assessment):
        """Simple grading logic"""
        correct = 0
        total = len(assessment.questions)

        for question_id, answer in submission.answers.items():
            correct_answer = assessment.questions[question_id].correct_answer
            if answer == correct_answer:
                correct += 1

        return correct / total

    def generate_feedback(self, score):
        """Generate feedback based on score"""
        if score >= 0.9:
            return "Excellent work! You've mastered this concept."
        elif score >= 0.7:
            return "Good job! Consider reviewing the areas you missed."
        else:
            return "Keep practicing. Review the material and try again."
```

### Test Locally

```bash
# Run tests
cd simple-quiz
lms-cli test

# Start local dev server
lms-cli dev

# Output:
# Plugin server running at http://localhost:5001
# Test your plugin in the dev environment
```

### Package and Publish

```bash
# Build plugin package
lms-cli build

# Output: simple-quiz-1.0.0.lmspkg

# Publish to marketplace
lms-cli publish --file simple-quiz-1.0.0.lmspkg

# Output:
# Uploading plugin... ✓
# Security scan... ✓
# Submitted for review
# Track status at: https://marketplace.lms.edu/plugins/simple-quiz
```

---

## Plugin Architecture

### Lifecycle

Plugins go through a managed lifecycle:

```
Install → Validate → Register → Activate → Monitor → Update/Deactivate
```

#### 1. Install
- Plugin package downloaded from marketplace
- Dependencies checked and installed
- Files extracted to plugin directory

#### 2. Validate
- Security scan (malware, vulnerabilities)
- Compatibility check (LMS version, dependencies)
- Permission validation

#### 3. Register
- Hooks registered with event bus
- API routes added to FastAPI router
- UI components registered with frontend

#### 4. Activate
- Plugin initialization function called
- Database migrations run (if needed)
- Background tasks started (if needed)

#### 5. Monitor
- Health checks every 60 seconds
- Performance metrics collected
- Error tracking enabled

#### 6. Update/Deactivate
- Update: New version installed, migrations run
- Deactivate: Cleanup function called, resources released

### Execution Model

Plugins run in a **sandboxed environment** with:

- **Limited permissions**: Only access granted APIs
- **Resource limits**: CPU, memory, network quotas
- **Isolated storage**: Plugin-specific data directory
- **Monitored execution**: All actions logged

```python
# Plugins run in sandboxed context
async with PluginSandbox(plugin_id="simple-quiz") as sandbox:
    # Limited to granted permissions
    result = await sandbox.execute(hook_function, context)
```

### Communication

Plugins communicate via:

1. **Event Bus**: Publish/subscribe for plugin-to-plugin communication
2. **Shared Data Store**: Redis-backed key-value store for cross-plugin state
3. **Plugin API**: Core LMS services (database, auth, knowledge graph)

```python
# Publish event to other plugins
await context.event_bus.publish(
    "quiz_completed",
    {"student_id": student.id, "score": score}
)

# Subscribe to events from other plugins
@Hook.register("on_event:quiz_completed")
async def on_quiz_completed(self, context: Context):
    # React to quiz completion from another plugin
    pass

# Store data for other plugins
await context.shared_data.set("quiz_high_score", score)

# Read data from other plugins
high_score = await context.shared_data.get("quiz_high_score")
```

---

## Development Setup

### Environment Setup

```bash
# Clone the plugin template
git clone https://github.com/lms/plugin-template simple-quiz
cd simple-quiz

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Development Tools

#### VS Code Extensions (Recommended)

- **LMS Plugin Tools**: Syntax highlighting, debugging
- **Python**: Python language support
- **ESLint**: JavaScript/TypeScript linting
- **Prettier**: Code formatting

#### Dev Server

```bash
# Start backend dev server (with hot reload)
lms-cli dev --reload

# Start frontend dev server (separate terminal)
cd frontend
npm run dev
```

#### Debugging

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Plugin",
      "type": "python",
      "request": "launch",
      "module": "lms_plugin_sdk.dev_server",
      "args": ["--plugin-dir", "${workspaceFolder}"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "LMS_DEV_MODE": "true"
      }
    }
  ]
}
```

Set breakpoints and press F5 to debug.

---

## Plugin Structure

### File Organization

```
simple-quiz/
├── plugin.yaml              # Plugin metadata (required)
├── README.md                # Documentation (required)
├── LICENSE                  # License file (required)
├── .gitignore
│
├── backend/                 # Python backend
│   ├── __init__.py
│   ├── hooks.py             # Hook implementations
│   ├── api.py               # API endpoints
│   ├── models.py            # Data models
│   ├── services.py          # Business logic
│   └── migrations/          # Database migrations
│       └── 001_initial.sql
│
├── frontend/                # React frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── components/
│   │   ├── QuizBuilder.tsx
│   │   ├── QuizTaker.tsx
│   │   └── QuizResults.tsx
│   ├── hooks/               # React hooks
│   │   └── useQuiz.ts
│   ├── styles/              # CSS modules
│   │   └── Quiz.module.css
│   └── index.tsx            # Entry point
│
├── tests/                   # Tests
│   ├── test_hooks.py
│   ├── test_api.py
│   └── test_frontend.tsx
│
├── docs/                    # Additional documentation
│   ├── user-guide.md
│   └── api-reference.md
│
└── examples/                # Example configurations
    └── sample-quiz.json
```

### plugin.yaml

Required metadata file:

```yaml
# Plugin metadata
name: simple-quiz
display_name: Simple Quiz Builder
version: 1.0.0
author: Your Name
author_email: you@example.com
author_url: https://yoursite.com
license: MIT

# Plugin details
category: assessment
description: |
  Create and grade multiple-choice quizzes with automatic feedback.
  Integrates with knowledge graph for mastery tracking.

keywords:
  - quiz
  - assessment
  - multiple-choice
  - auto-grading

# Requirements
lms_version: ">=1.0.0"
python_version: ">=3.11"
node_version: ">=18.0"

dependencies:
  python:
    - pydantic>=2.0
    - sqlalchemy>=2.0
  npm:
    - react>=18.0
    - typescript>=5.0

# Permissions required
permissions:
  - knowledge_graph:write  # Update student knowledge graph
  - database:read          # Read lesson data
  - database:write         # Store quiz results
  - events:publish         # Publish quiz completion events
  - events:subscribe       # Subscribe to lesson events

# Hooks implemented
hooks:
  - after_assessment_submit
  - before_lesson_render
  - on_engagement_event

# API endpoints
api_routes:
  - method: POST
    path: /api/plugins/simple-quiz/create
    description: Create a new quiz
  - method: GET
    path: /api/plugins/simple-quiz/{quiz_id}
    description: Get quiz by ID
  - method: POST
    path: /api/plugins/simple-quiz/{quiz_id}/submit
    description: Submit quiz answers

# UI components
ui_components:
  - name: QuizBuilder
    type: editor
    description: Visual quiz builder for instructors
  - name: QuizTaker
    type: student
    description: Quiz-taking interface for students
  - name: QuizResults
    type: results
    description: Results and analytics display

# Configuration options
config_schema:
  time_limit:
    type: integer
    default: 3600
    description: Default time limit in seconds
  passing_score:
    type: float
    default: 0.7
    description: Minimum score to pass (0.0-1.0)
  allow_retakes:
    type: boolean
    default: true
    description: Allow students to retake quizzes
```

---

## Hook System

### Available Hooks

Hooks are events where plugins can inject custom behavior:

#### Content Hooks

```python
@Hook.register("before_lesson_render")
async def before_lesson_render(self, context: Context) -> HookResponse:
    """
    Called before lesson is rendered to student.
    Use to modify content, add supplementary materials, etc.

    Context:
    - context.data["lesson"]: Lesson object
    - context.student: Current student
    - context.knowledge_graph: Student's knowledge graph

    Return:
    - HookResponse with modified lesson data
    """
    pass

@Hook.register("after_lesson_complete")
async def after_lesson_complete(self, context: Context) -> HookResponse:
    """
    Called after student completes a lesson.
    Use to update progress, unlock next lesson, etc.
    """
    pass
```

#### Assessment Hooks

```python
@Hook.register("before_assessment_render")
async def before_assessment_render(self, context: Context) -> HookResponse:
    """
    Called before assessment is shown to student.
    Use to personalize questions, adjust difficulty, etc.
    """
    pass

@Hook.register("after_assessment_submit")
async def after_assessment_submit(self, context: Context) -> HookResponse:
    """
    Called after student submits an assessment.
    Use to grade, provide feedback, update knowledge graph.
    """
    pass

@Hook.register("on_grade_update")
async def on_grade_update(self, context: Context) -> HookResponse:
    """
    Called when grade is updated (manual grading, regrade, etc.).
    Use to notify student, update analytics, etc.
    """
    pass
```

#### Engagement Hooks

```python
@Hook.register("on_engagement_event")
async def on_engagement_event(self, context: Context) -> HookResponse:
    """
    Called on any student engagement event.
    Events: view, click, video_progress, discussion_post, etc.

    Context:
    - context.data["event_type"]: Type of engagement
    - context.data["event_data"]: Event-specific data
    - context.student: Current student
    """
    pass

@Hook.register("on_inactivity_detected")
async def on_inactivity_detected(self, context: Context) -> HookResponse:
    """
    Called when student is inactive for configured period.
    Use to send reminders, suggest interventions, etc.
    """
    pass
```

#### Communication Hooks

```python
@Hook.register("before_notification_send")
async def before_notification_send(self, context: Context) -> HookResponse:
    """
    Called before notification is sent to student.
    Use to modify message, add context, personalize, etc.
    """
    pass

@Hook.register("on_discussion_post")
async def on_discussion_post(self, context: Context) -> HookResponse:
    """
    Called when student posts in discussion forum.
    Use to moderate, award participation points, etc.
    """
    pass
```

#### Analytics Hooks

```python
@Hook.register("on_analytics_query")
async def on_analytics_query(self, context: Context) -> HookResponse:
    """
    Called when instructor runs analytics query.
    Use to add custom metrics, visualizations, etc.
    """
    pass

@Hook.register("on_at_risk_detection")
async def on_at_risk_detection(self, context: Context) -> HookResponse:
    """
    Called when AI detects at-risk student.
    Use to suggest interventions, notify instructor, etc.
    """
    pass
```

### Hook Context

Every hook receives a `Context` object with:

```python
@dataclass
class Context:
    # Request metadata
    request_id: str           # Unique request ID
    timestamp: datetime       # When hook was triggered
    hook_name: str            # Name of the hook

    # User context
    student: Optional[Student]      # Current student (if applicable)
    instructor: Optional[Instructor] # Current instructor (if applicable)
    institution: Institution   # Current institution

    # Data context
    data: Dict[str, Any]      # Hook-specific data

    # Services
    knowledge_graph: KnowledgeGraphClient  # Knowledge graph API
    database: DatabaseClient    # Database access
    event_bus: EventBusClient   # Event publishing
    shared_data: SharedDataClient # Cross-plugin data
    llm: LLMClient             # LLM integration (if enabled)

    # Plugin metadata
    plugin_id: str            # Your plugin ID
    plugin_config: Dict       # Plugin configuration
```

### Hook Response

Hooks must return a `HookResponse`:

```python
@dataclass
class HookResponse:
    success: bool               # Did hook execute successfully?
    data: Optional[Dict] = None # Modified/additional data
    error: Optional[str] = None # Error message (if success=False)
    metadata: Optional[Dict] = None # Hook execution metadata

# Example: Successful hook
return HookResponse(
    success=True,
    data={"score": 0.85, "feedback": "Great job!"},
    metadata={"execution_time_ms": 45}
)

# Example: Hook with error
return HookResponse(
    success=False,
    error="Failed to grade quiz: missing answer key"
)
```

### Hook Priority

Multiple plugins can register the same hook. Control execution order with priority:

```python
@Hook.register("after_assessment_submit", priority=10)
async def high_priority_hook(self, context: Context):
    # Runs first (higher priority)
    pass

@Hook.register("after_assessment_submit", priority=5)
async def medium_priority_hook(self, context: Context):
    # Runs second
    pass

@Hook.register("after_assessment_submit", priority=1)
async def low_priority_hook(self, context: Context):
    # Runs last (lower priority)
    pass
```

Default priority is 5. Range: 1 (lowest) to 10 (highest).

### Conditional Hooks

Execute hooks only when conditions are met:

```python
@Hook.register("on_engagement_event", condition="event_type == 'video_complete'")
async def on_video_complete(self, context: Context):
    # Only called when event_type is 'video_complete'
    pass

@Hook.register("after_assessment_submit", condition="assessment.category == 'quiz'")
async def on_quiz_submit(self, context: Context):
    # Only called for quiz submissions
    pass
```

Conditions use simple expression syntax with access to `context` variables.

---

## API Reference

### Plugin API Endpoints

Plugins can register custom API endpoints:

```python
from lms_plugin_sdk import Router, Depends, HTTPException
from pydantic import BaseModel

router = Router(prefix="/api/plugins/simple-quiz")

# Request/response models
class CreateQuizRequest(BaseModel):
    title: str
    questions: list[Question]
    time_limit: int = 3600

class QuizResponse(BaseModel):
    quiz_id: str
    title: str
    question_count: int
    created_at: datetime

# Create quiz endpoint
@router.post("/create")
async def create_quiz(
    request: CreateQuizRequest,
    context: Context = Depends(get_context)
) -> QuizResponse:
    """Create a new quiz"""

    # Validate permissions
    if not context.instructor:
        raise HTTPException(403, "Only instructors can create quizzes")

    # Store quiz in database
    quiz = await context.database.quizzes.create({
        "title": request.title,
        "questions": [q.dict() for q in request.questions],
        "time_limit": request.time_limit,
        "created_by": context.instructor.id,
        "institution_id": context.institution.id
    })

    return QuizResponse(
        quiz_id=quiz.id,
        title=quiz.title,
        question_count=len(quiz.questions),
        created_at=quiz.created_at
    )

# Get quiz endpoint
@router.get("/{quiz_id}")
async def get_quiz(
    quiz_id: str,
    context: Context = Depends(get_context)
) -> QuizResponse:
    """Get quiz by ID"""

    quiz = await context.database.quizzes.get(quiz_id)

    if not quiz:
        raise HTTPException(404, f"Quiz {quiz_id} not found")

    # Check permissions
    if quiz.institution_id != context.institution.id:
        raise HTTPException(403, "Access denied")

    return QuizResponse(
        quiz_id=quiz.id,
        title=quiz.title,
        question_count=len(quiz.questions),
        created_at=quiz.created_at
    )

# Submit quiz endpoint
@router.post("/{quiz_id}/submit")
async def submit_quiz(
    quiz_id: str,
    answers: Dict[str, str],
    context: Context = Depends(get_context)
):
    """Submit quiz answers"""

    if not context.student:
        raise HTTPException(403, "Only students can submit quizzes")

    quiz = await context.database.quizzes.get(quiz_id)

    if not quiz:
        raise HTTPException(404, f"Quiz {quiz_id} not found")

    # Grade quiz
    score = grade_quiz(answers, quiz.questions)

    # Store submission
    submission = await context.database.quiz_submissions.create({
        "quiz_id": quiz_id,
        "student_id": context.student.id,
        "answers": answers,
        "score": score,
        "submitted_at": datetime.now()
    })

    # Update knowledge graph
    await context.knowledge_graph.record_learning_event(
        student_id=context.student.id,
        concept=quiz.concept,
        mastery_level=score,
        evidence=f"Quiz: {quiz.title}"
    )

    # Publish event
    await context.event_bus.publish("quiz_completed", {
        "quiz_id": quiz_id,
        "student_id": context.student.id,
        "score": score
    })

    return {"submission_id": submission.id, "score": score}
```

### Core LMS APIs

Plugins have access to core LMS APIs via the `Context` object:

#### Knowledge Graph API

```python
# Record learning event
await context.knowledge_graph.record_learning_event(
    student_id=student.id,
    concept="machine_learning_basics",
    mastery_level=0.85,
    evidence="Quiz score: 85%",
    timestamp=datetime.now()
)

# Get student's mastered concepts
mastered = await context.knowledge_graph.get_mastered_concepts(
    student_id=student.id,
    min_confidence=0.7
)
# Returns: ["python", "statistics", "linear_algebra"]

# Get struggling concepts
struggling = await context.knowledge_graph.get_struggling_concepts(
    student_id=student.id,
    max_confidence=0.5
)
# Returns: ["backpropagation", "gradient_descent"]

# Find learning path
path = await context.knowledge_graph.find_learning_path(
    student_id=student.id,
    from_concept="linear_algebra",
    to_concept="deep_learning"
)
# Returns: ["linear_algebra", "neural_networks", "backpropagation", "deep_learning"]
```

#### Database API

```python
# Create record
quiz = await context.database.quizzes.create({
    "title": "ML Basics Quiz",
    "questions": [...],
    "created_by": instructor.id
})

# Read record
quiz = await context.database.quizzes.get(quiz_id)

# Update record
await context.database.quizzes.update(quiz_id, {
    "title": "Updated Title"
})

# Delete record
await context.database.quizzes.delete(quiz_id)

# Query records
quizzes = await context.database.quizzes.find({
    "created_by": instructor.id,
    "institution_id": institution.id
})

# Custom SQL query (use sparingly)
results = await context.database.execute(
    "SELECT * FROM quizzes WHERE created_at > :date",
    {"date": datetime.now() - timedelta(days=7)}
)
```

#### Event Bus API

```python
# Publish event
await context.event_bus.publish(
    "quiz_completed",
    data={"quiz_id": quiz_id, "score": score},
    target="analytics"  # Optional: target specific plugins
)

# Subscribe to event (in hook)
@Hook.register("on_event:quiz_completed")
async def on_quiz_completed(self, context: Context):
    event_data = context.data["event"]
    quiz_id = event_data["quiz_id"]
    score = event_data["score"]
    # React to event
```

#### Shared Data API

```python
# Set data (cross-plugin state)
await context.shared_data.set(
    key="quiz_high_scores",
    value={"quiz_123": 0.95, "quiz_456": 0.88},
    ttl=3600  # Optional: expire after 1 hour
)

# Get data
high_scores = await context.shared_data.get("quiz_high_scores")

# Delete data
await context.shared_data.delete("quiz_high_scores")

# Atomic increment
new_count = await context.shared_data.increment("quiz_completion_count")
```

#### LLM API (Optional)

```python
# Generate text
response = await context.llm.generate(
    prompt="Generate 5 multiple-choice questions about machine learning",
    max_tokens=500,
    temperature=0.7
)

# Embed text
embedding = await context.llm.embed(
    text="What is supervised learning?"
)
# Returns: array of 384 floats

# Analyze sentiment
sentiment = await context.llm.analyze_sentiment(
    text="This quiz was really hard and confusing"
)
# Returns: {"score": -0.6, "label": "negative"}
```

---

## UI Components

### React Component Structure

Plugins can provide React components for the frontend:

```typescript
// frontend/components/QuizBuilder.tsx
import React, { useState } from 'react';
import { useLMSPlugin } from '@lms/plugin-sdk';

interface Question {
  id: string;
  text: string;
  options: string[];
  correctAnswer: number;
}

export const QuizBuilder: React.FC = () => {
  const { api, context } = useLMSPlugin();
  const [title, setTitle] = useState('');
  const [questions, setQuestions] = useState<Question[]>([]);

  const addQuestion = () => {
    setQuestions([
      ...questions,
      {
        id: crypto.randomUUID(),
        text: '',
        options: ['', '', '', ''],
        correctAnswer: 0
      }
    ]);
  };

  const saveQuiz = async () => {
    try {
      const response = await api.post('/plugins/simple-quiz/create', {
        title,
        questions,
        time_limit: 3600
      });

      context.notify('Quiz created successfully!', 'success');
      context.navigate(`/quizzes/${response.quiz_id}`);
    } catch (error) {
      context.notify('Failed to create quiz', 'error');
    }
  };

  return (
    <div className="quiz-builder">
      <h2>Create Quiz</h2>

      <input
        type="text"
        placeholder="Quiz Title"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />

      {questions.map((question, index) => (
        <QuestionEditor
          key={question.id}
          question={question}
          onChange={(updated) => {
            const newQuestions = [...questions];
            newQuestions[index] = updated;
            setQuestions(newQuestions);
          }}
        />
      ))}

      <button onClick={addQuestion}>Add Question</button>
      <button onClick={saveQuiz}>Save Quiz</button>
    </div>
  );
};
```

### Plugin SDK Hooks

Frontend plugins can use React hooks from the SDK:

```typescript
import {
  useLMSPlugin,
  useKnowledgeGraph,
  useStudent,
  useCourse,
  useNotifications
} from '@lms/plugin-sdk';

// Main plugin hook
const { api, context, config } = useLMSPlugin();

// Knowledge graph access
const { concepts, loading } = useKnowledgeGraph(studentId);

// Current student
const { student, loading } = useStudent();

// Current course
const { course, lessons, loading } = useCourse();

// Notifications
const { notify, notifications } = useNotifications();
```

### Styling

Use CSS modules for scoped styling:

```typescript
// QuizBuilder.module.css
.quizBuilder {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}

.questionEditor {
  background: #f5f5f5;
  border-radius: 8px;
  padding: 1rem;
  margin: 1rem 0;
}

// Import in component
import styles from './QuizBuilder.module.css';

<div className={styles.quizBuilder}>
  ...
</div>
```

Or use Tailwind CSS (included in LMS):

```typescript
<div className="max-w-4xl mx-auto p-8">
  <h2 className="text-2xl font-bold mb-4">Create Quiz</h2>
  ...
</div>
```

---

## Testing

### Backend Tests

Use pytest for backend testing:

```python
# tests/test_hooks.py
import pytest
from lms_plugin_sdk.testing import MockContext, MockKnowledgeGraph

@pytest.fixture
def context():
    """Create mock context for testing"""
    return MockContext(
        student_id="student123",
        institution_id="inst456",
        knowledge_graph=MockKnowledgeGraph()
    )

@pytest.mark.asyncio
async def test_quiz_grading(context):
    """Test quiz grading logic"""
    from backend.hooks import SimpleQuizHooks

    hooks = SimpleQuizHooks()

    # Prepare test data
    context.data = {
        "submission": {
            "answers": {"q1": "A", "q2": "B", "q3": "C"}
        },
        "assessment": {
            "questions": {
                "q1": {"correct_answer": "A"},
                "q2": {"correct_answer": "B"},
                "q3": {"correct_answer": "D"}  # Wrong!
            },
            "concept": "machine_learning"
        }
    }

    # Call hook
    response = await hooks.on_quiz_submit(context)

    # Assert results
    assert response.success is True
    assert response.data["score"] == 2/3  # 66.7%
    assert "review" in response.data["feedback"].lower()

    # Assert knowledge graph was updated
    events = context.knowledge_graph.get_events("student123")
    assert len(events) == 1
    assert events[0]["mastery_level"] == 2/3

@pytest.mark.asyncio
async def test_quiz_creation(context):
    """Test quiz creation API"""
    from backend.api import create_quiz, CreateQuizRequest

    request = CreateQuizRequest(
        title="Test Quiz",
        questions=[
            {"text": "Q1", "options": ["A", "B"], "correct": 0},
            {"text": "Q2", "options": ["A", "B"], "correct": 1}
        ]
    )

    response = await create_quiz(request, context)

    assert response.quiz_id is not None
    assert response.title == "Test Quiz"
    assert response.question_count == 2
```

### Frontend Tests

Use React Testing Library:

```typescript
// tests/QuizBuilder.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MockLMSProvider } from '@lms/plugin-sdk/testing';
import { QuizBuilder } from '../frontend/components/QuizBuilder';

describe('QuizBuilder', () => {
  it('creates quiz with questions', async () => {
    const mockApi = {
      post: jest.fn().mockResolvedValue({ quiz_id: 'quiz123' })
    };

    render(
      <MockLMSProvider api={mockApi}>
        <QuizBuilder />
      </MockLMSProvider>
    );

    // Enter title
    const titleInput = screen.getByPlaceholderText('Quiz Title');
    fireEvent.change(titleInput, { target: { value: 'Test Quiz' } });

    // Add question
    const addButton = screen.getByText('Add Question');
    fireEvent.click(addButton);

    // Fill question
    const questionInput = screen.getByPlaceholderText('Question text');
    fireEvent.change(questionInput, { target: { value: 'What is ML?' } });

    // Save quiz
    const saveButton = screen.getByText('Save Quiz');
    fireEvent.click(saveButton);

    // Assert API was called
    await waitFor(() => {
      expect(mockApi.post).toHaveBeenCalledWith(
        '/plugins/simple-quiz/create',
        expect.objectContaining({
          title: 'Test Quiz',
          questions: expect.arrayContaining([
            expect.objectContaining({ text: 'What is ML?' })
          ])
        })
      );
    });
  });
});
```

### Integration Tests

Test end-to-end plugin behavior:

```python
# tests/test_integration.py
import pytest
from lms_plugin_sdk.testing import LMSTestClient

@pytest.fixture
async def client():
    """Create test client with plugin loaded"""
    async with LMSTestClient(plugins=["simple-quiz"]) as client:
        yield client

@pytest.mark.asyncio
async def test_complete_quiz_flow(client):
    """Test complete quiz flow: create, take, grade"""

    # Login as instructor
    await client.login(role="instructor", user_id="inst123")

    # Create quiz
    response = await client.post("/api/plugins/simple-quiz/create", json={
        "title": "ML Basics",
        "questions": [
            {"text": "What is ML?", "options": ["A", "B"], "correct": 0}
        ]
    })
    assert response.status_code == 200
    quiz_id = response.json()["quiz_id"]

    # Login as student
    await client.login(role="student", user_id="student456")

    # Take quiz
    response = await client.get(f"/api/plugins/simple-quiz/{quiz_id}")
    assert response.status_code == 200

    # Submit answers
    response = await client.post(
        f"/api/plugins/simple-quiz/{quiz_id}/submit",
        json={"answers": {"q1": "A"}}
    )
    assert response.status_code == 200
    assert response.json()["score"] == 1.0

    # Check knowledge graph was updated
    kg = await client.knowledge_graph.get_events("student456")
    assert len(kg) == 1
    assert kg[0]["mastery_level"] == 1.0
```

### Running Tests

```bash
# Run all tests
lms-cli test

# Run specific test file
lms-cli test tests/test_hooks.py

# Run with coverage
lms-cli test --coverage

# Run integration tests only
lms-cli test --integration

# Watch mode (re-run on file change)
lms-cli test --watch
```

---

## Security

### Permission Model

Plugins must declare required permissions in `plugin.yaml`:

```yaml
permissions:
  - knowledge_graph:read       # Read from knowledge graph
  - knowledge_graph:write      # Write to knowledge graph
  - database:read              # Read from database
  - database:write             # Write to database
  - events:publish             # Publish events
  - events:subscribe           # Subscribe to events
  - files:read                 # Read files
  - files:write                # Write files
  - network:external           # Make external HTTP requests
  - llm:use                    # Use LLM APIs
```

Plugins can only access APIs for granted permissions. Requests for unauthorized operations will fail with `PermissionDenied` error.

### Sandboxing

Plugins run in isolated environments:

- **Process isolation**: Separate processes, resource limits
- **Network isolation**: No direct network access (except with permission)
- **File system isolation**: Plugin-specific directory only
- **Database isolation**: Row-level security, filtered by institution

### Input Validation

Always validate user input:

```python
from pydantic import BaseModel, validator

class CreateQuizRequest(BaseModel):
    title: str
    questions: list[Question]

    @validator('title')
    def title_not_empty(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Title must be at least 3 characters')
        return v

    @validator('questions')
    def questions_not_empty(cls, v):
        if not v or len(v) < 1:
            raise ValueError('Quiz must have at least 1 question')
        return v
```

### SQL Injection Prevention

Use parameterized queries:

```python
# GOOD: Parameterized query
results = await context.database.execute(
    "SELECT * FROM quizzes WHERE id = :id",
    {"id": quiz_id}
)

# BAD: String concatenation (SQL injection!)
results = await context.database.execute(
    f"SELECT * FROM quizzes WHERE id = '{quiz_id}'"
)
```

### XSS Prevention

Sanitize user-generated content:

```typescript
import DOMPurify from 'dompurify';

// Sanitize HTML before rendering
const SafeHTML: React.FC<{ html: string }> = ({ html }) => {
  const clean = DOMPurify.sanitize(html);
  return <div dangerouslySetInnerHTML={{ __html: clean }} />;
};
```

### Secrets Management

Never hardcode secrets. Use environment variables:

```python
import os

# GOOD: Environment variable
API_KEY = os.getenv("PLUGIN_API_KEY")

# BAD: Hardcoded secret
API_KEY = "sk-1234567890abcdef"
```

In `plugin.yaml`, declare required secrets:

```yaml
secrets:
  - name: PLUGIN_API_KEY
    description: API key for external service
    required: true
```

Instructors configure secrets in the plugin settings UI.

---

## Publishing

### Pre-Publishing Checklist

Before submitting to marketplace:

- [ ] All tests pass (`lms-cli test`)
- [ ] Code follows style guide (`lms-cli lint`)
- [ ] Documentation is complete (README, API reference)
- [ ] Security scan passes (`lms-cli security-scan`)
- [ ] Performance benchmarks meet requirements
- [ ] License file included
- [ ] Screenshots/demo video prepared

### Building Package

```bash
# Build plugin package
lms-cli build

# Output: simple-quiz-1.0.0.lmspkg

# Package contents:
# - plugin.yaml (metadata)
# - backend/ (Python code)
# - frontend/ (compiled JS bundle)
# - README.md
# - LICENSE
# - package.json
```

### Marketplace Submission

```bash
# Publish to marketplace
lms-cli publish --file simple-quiz-1.0.0.lmspkg

# Or interactive mode
lms-cli publish

# Follow prompts:
# - Category: Assessment
# - Pricing: Free / Paid ($X/month)
# - Support URL: https://support.example.com
# - Demo video: https://youtube.com/...
```

### Review Process

Marketplace review typically takes 3-5 business days:

1. **Automated checks** (security, compatibility) - immediate
2. **Code review** (security, quality) - 1-2 days
3. **Functional testing** (QA team) - 1-2 days
4. **Final approval** - 1 day

You'll receive email notifications at each stage.

### Versioning

Follow semantic versioning (semver):

- **Major** (1.0.0 → 2.0.0): Breaking changes
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes, backward compatible

Update `plugin.yaml`:

```yaml
version: 1.1.0  # Increment appropriately

# Changelog
changelog:
  - version: 1.1.0
    date: 2025-11-17
    changes:
      - Added support for essay questions
      - Fixed grading bug for partial credit
  - version: 1.0.0
    date: 2025-11-01
    changes:
      - Initial release
```

### Pricing

Choose pricing model:

| Model | Description | Commission |
|-------|-------------|------------|
| **Free** | No charge | 0% |
| **One-time** | Single payment | 30% |
| **Subscription** | Monthly/yearly | 30% |
| **Freemium** | Free + paid features | 30% on paid |
| **Enterprise** | Custom pricing | 20% |

You keep 70-80% of revenue.

---

## Best Practices

### Performance

- **Cache aggressively**: Use `context.shared_data` for frequently accessed data
- **Async operations**: Use `async/await` for all I/O
- **Batch operations**: Group database queries
- **Lazy loading**: Load data only when needed

```python
# GOOD: Batch query
quizzes = await context.database.quizzes.find({
    "id": {"$in": quiz_ids}  # Single query for multiple IDs
})

# BAD: N+1 queries
quizzes = []
for quiz_id in quiz_ids:
    quiz = await context.database.quizzes.get(quiz_id)
    quizzes.append(quiz)
```

### Error Handling

- **Always handle errors**: Use try/except
- **Meaningful messages**: Help users understand what went wrong
- **Log errors**: Use `context.logger`

```python
try:
    quiz = await context.database.quizzes.get(quiz_id)
except NotFoundError:
    context.logger.warning(f"Quiz {quiz_id} not found")
    raise HTTPException(404, f"Quiz {quiz_id} not found")
except DatabaseError as e:
    context.logger.error(f"Database error: {e}")
    raise HTTPException(500, "Internal server error")
```

### Accessibility

- **Keyboard navigation**: All UI accessible without mouse
- **Screen reader support**: Proper ARIA labels
- **Color contrast**: WCAG 2.1 AA minimum
- **Focus indicators**: Visible focus states

```typescript
<button
  aria-label="Add question to quiz"
  onClick={addQuestion}
>
  Add Question
</button>
```

### Internationalization

Support multiple languages:

```typescript
import { useTranslation } from '@lms/plugin-sdk';

const { t } = useTranslation('simple-quiz');

<h2>{t('quiz_builder.title')}</h2>  // "Create Quiz" (en) or "Crear Cuestionario" (es)
```

Provide translation files:

```json
// locales/en.json
{
  "quiz_builder": {
    "title": "Create Quiz",
    "add_question": "Add Question",
    "save": "Save Quiz"
  }
}

// locales/es.json
{
  "quiz_builder": {
    "title": "Crear Cuestionario",
    "add_question": "Agregar Pregunta",
    "save": "Guardar Cuestionario"
  }
}
```

---

## Examples

### Complete Plugin: Simple Quiz

See the full example in the repository:
https://github.com/lms/plugin-examples/simple-quiz

### Other Examples

- **Peer Review Plugin**: https://github.com/lms/plugin-examples/peer-review
- **Video Player Plugin**: https://github.com/lms/plugin-examples/video-player
- **Analytics Dashboard**: https://github.com/lms/plugin-examples/analytics-dashboard
- **GitHub Integration**: https://github.com/lms/plugin-examples/github-integration

---

## Support

### Documentation

- **API Reference**: https://docs.lms.edu/plugins/api
- **SDK Reference**: https://docs.lms.edu/plugins/sdk
- **Examples**: https://github.com/lms/plugin-examples

### Community

- **Forum**: https://community.lms.edu/plugins
- **Discord**: https://discord.gg/lms-plugins
- **Stack Overflow**: Tag [lms-plugins]

### Bug Reports

- **GitHub Issues**: https://github.com/lms/plugin-sdk/issues
- **Security Issues**: security@lms.edu (private)

---

**Happy plugin building!**

If you build something cool, share it in the community forum. We love seeing what you create!

---

**Author**: Claude Code
**Date**: 2025-11-17
**Version**: 1.0
