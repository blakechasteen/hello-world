# Quiz Builder Plugin

**Version**: 1.0.0
**Category**: Assessment
**Author**: LMS Team

Simple quiz builder with multiple choice, true/false, and short answer questions.

## Features

- ✅ Multiple question types (MCQ, T/F, short answer)
- ✅ Auto-grading for objective questions
- ✅ Configurable retakes and time limits
- ✅ Question/option shuffling
- ✅ Immediate feedback
- ✅ Knowledge graph integration

## Quick Start

```python
quiz = {
    "quiz_id": "quiz_123",
    "title": "Python Basics Quiz",
    "questions": [
        {
            "question_id": "q1",
            "type": "multiple_choice",
            "question": "What is a list in Python?",
            "options": ["Array", "Dictionary", "Ordered collection", "Set"],
            "correct_answer": 2,
            "points": 10,
            "concept": "python_lists"
        },
        {
            "question_id": "q2",
            "type": "true_false",
            "question": "Python is statically typed",
            "correct_answer": False,
            "points": 5,
            "concept": "python_typing"
        }
    ],
    "total_points": 15,
    "passing_score": 10
}
```

## Configuration

```yaml
config:
  show_correct_answers: true
  allow_retakes: true
  max_attempts: 3
  shuffle_questions: false
  shuffle_options: false
  time_limit_minutes: 30
```

## API Usage

```bash
# Get quiz
GET /api/plugins/quiz-builder/quiz/{quiz_id}

# Submit answers
POST /api/plugins/quiz-builder/quiz/{quiz_id}/submit
{
  "student_id": "student_123",
  "answers": {
    "q1": 2,
    "q2": false
  }
}

# Get results
GET /api/plugins/quiz-builder/quiz/{quiz_id}/results/{student_id}
```

## License

MIT
