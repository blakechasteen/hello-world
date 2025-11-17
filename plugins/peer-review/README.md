# Peer Review Plugin

**Version**: 1.0.0
**Category**: Assessment
**Author**: LMS Team

Comprehensive peer review system with calibration training, quality scoring, blind review, and dispute resolution.

## Features

✅ **Multiple Review Types**
- Single-blind review (reviewer knows author)
- Double-blind review (neither knows the other)
- Open review (both know each other)

✅ **Calibration Training**
- Practice reviews with expert feedback
- Adaptive difficulty based on performance
- Minimum score threshold before reviewing

✅ **Quality Scoring**
- Thoroughness (length, detail)
- Specificity (concrete vs vague feedback)
- Constructiveness (actionable suggestions)
- Alignment (consistency with other reviews)

✅ **Rubric Types**
- Analytical (multi-criteria with scales)
- Holistic (single overall score)
- Checklist (binary yes/no items)
- Single-point (feedback-focused)

✅ **Dispute Resolution**
- Students can dispute unfair reviews
- AI-powered validity analysis
- Instructor oversight and resolution
- Time-limited dispute window

✅ **Smart Assignment**
- Load balancing across reviewers
- Quality-weighted selection
- Fairness algorithms
- Workload distribution

✅ **Analytics Dashboard**
- Assignment-level metrics
- Reviewer profiles
- Score distributions
- Quality trends

## Installation

```bash
# Install plugin
lms-cli install peer-review-system-1.0.0.lmspkg

# Or build from source
cd plugins/peer-review
lms-cli build
lms-cli install dist/peer-review-system-1.0.0.lmspkg
```

## Quick Start

### 1. Create Rubric

```python
from peer_review.models import Rubric, RubricCriterion, RubricType

rubric = Rubric(
    rubric_id="rubric_essay_analysis",
    name="Essay Analysis Rubric",
    rubric_type=RubricType.ANALYTICAL,
    criteria=[
        RubricCriterion(
            criterion_id="c1",
            name="Thesis",
            description="Clear, arguable thesis statement",
            max_score=20.0,
            levels=[
                {"score": 20, "description": "Excellent thesis"},
                {"score": 15, "description": "Good thesis"},
                {"score": 10, "description": "Adequate thesis"},
                {"score": 5, "description": "Weak thesis"},
                {"score": 0, "description": "No thesis"}
            ]
        ),
        RubricCriterion(
            criterion_id="c2",
            name="Evidence",
            description="Strong supporting evidence",
            max_score=30.0,
            levels=[...]
        ),
        # ... more criteria
    ],
    total_points=100.0,
    created_at=datetime.now()
)
```

### 2. Create Assignment

```python
from peer_review.models import PeerReviewAssignment, ReviewType
from datetime import datetime, timedelta

assignment = PeerReviewAssignment(
    assignment_id="essay_peer_review",
    title="Essay Peer Review",
    description="Review your peers' essays on climate change",
    instructions="Provide constructive feedback using the rubric",

    # Configuration
    review_type=ReviewType.DOUBLE_BLIND,
    rubric=rubric,
    reviews_per_submission=3,
    enable_calibration=True,
    calibration_threshold=0.8,

    # Deadlines
    submission_deadline=datetime.now() + timedelta(days=7),
    review_deadline=datetime.now() + timedelta(days=14),
    dispute_deadline=datetime.now() + timedelta(days=17),

    # Weighting
    quality_weight=0.7,       # 70% from review quality
    completion_weight=0.3,    # 30% from completing reviews

    # Metadata
    created_by="instructor_123",
    institution_id="university_456",
    created_at=datetime.now()
)
```

### 3. Student Workflow

#### A. Submit Work

```python
# Student submits essay
submission = await api.post("/plugins/peer-review/submissions", {
    "assignment_id": "essay_peer_review",
    "student_id": "student_789",
    "content": essay_text,
    "attachments": ["essay.pdf"]
})
```

#### B. Calibration (if required)

```python
# Get calibration exercise
exercise = await api.get(
    f"/plugins/peer-review/calibration/{assignment_id}"
)

# Complete calibration
calibration_attempt = await api.post(
    f"/plugins/peer-review/calibration/submit",
    {
        "exercise_id": exercise.exercise_id,
        "student_id": "student_789",
        "review": student_review  # Student's practice review
    }
)

# Check if passed
if calibration_attempt.accuracy_score >= 0.8:
    print("Calibration passed! You can now review peers.")
```

#### C. Review Peers

```python
# Get assigned review
review_assignment = await api.get(
    f"/plugins/peer-review/reviews/assigned/{student_id}"
)

# Submit review
review = await api.post("/plugins/peer-review/reviews/submit", {
    "review_id": review_assignment.review_id,
    "reviewer_id": "student_789",
    "scores": [
        {"criterion_id": "c1", "score": 18, "feedback": "Strong thesis..."},
        {"criterion_id": "c2", "score": 25, "feedback": "Good evidence..."}
    ],
    "total_score": 85.0,
    "overall_feedback": "Well-written essay with clear arguments..."
})
```

#### D. View Results

```python
# Get reviews received
results = await api.get(
    f"/plugins/peer-review/results/{submission_id}"
)

print(f"Average Score: {results.avg_score}")
print(f"Feedback: {results.consolidated_feedback}")
print(f"Strengths: {', '.join(results.key_strengths)}")
print(f"Areas to Improve: {', '.join(results.key_improvements)}")
```

#### E. Dispute Review (if needed)

```python
# Submit dispute
dispute = await api.post(
    f"/plugins/peer-review/reviews/{review_id}/dispute",
    {
        "student_id": "student_789",
        "reasons": ["unfair_scoring", "lack_of_feedback"],
        "explanation": "The reviewer gave low scores without explanation...",
        "requested_action": "reevaluate"
    }
)
```

## Configuration Options

Configure in plugin settings or `plugin.yaml`:

```yaml
config:
  reviews_per_submission: 3          # Number of peer reviews per submission
  blind_review: true                 # Hide author identity
  enable_calibration: true           # Require calibration training
  calibration_threshold: 0.8         # Min score to pass calibration
  review_deadline_days: 7            # Days after submission to review
  enable_disputes: true              # Allow review disputes
  dispute_window_days: 3             # Days to dispute
  quality_weight: 0.7                # Weight of review quality in grade
```

## API Reference

### Assignments

- `POST /api/plugins/peer-review/assignments/create` - Create assignment
- `GET /api/plugins/peer-review/assignments/{id}` - Get assignment
- `PUT /api/plugins/peer-review/assignments/{id}` - Update assignment
- `DELETE /api/plugins/peer-review/assignments/{id}` - Delete assignment

### Submissions

- `POST /api/plugins/peer-review/submissions` - Submit work
- `GET /api/plugins/peer-review/submissions/{id}` - Get submission
- `GET /api/plugins/peer-review/submissions/student/{id}` - Get student's submissions

### Reviews

- `GET /api/plugins/peer-review/reviews/assigned/{student_id}` - Get assigned reviews
- `POST /api/plugins/peer-review/reviews/submit` - Submit review
- `GET /api/plugins/peer-review/reviews/{id}` - Get review
- `GET /api/plugins/peer-review/reviews/submission/{id}` - Get reviews for submission

### Calibration

- `GET /api/plugins/peer-review/calibration/{assignment_id}` - Get calibration exercise
- `POST /api/plugins/peer-review/calibration/submit` - Submit calibration attempt
- `GET /api/plugins/peer-review/calibration/{student_id}/score` - Get calibration score

### Disputes

- `POST /api/plugins/peer-review/reviews/{id}/dispute` - File dispute
- `GET /api/plugins/peer-review/disputes/{id}` - Get dispute
- `PUT /api/plugins/peer-review/disputes/{id}/resolve` - Resolve dispute (instructor)

### Analytics

- `GET /api/plugins/peer-review/analytics/assignment/{id}` - Assignment analytics
- `GET /api/plugins/peer-review/analytics/reviewer/{id}` - Reviewer profile
- `GET /api/plugins/peer-review/analytics/quality-trends` - Quality trends

## Knowledge Graph Integration

The plugin updates the student knowledge graph with:

### For Reviewers (Collaborative Learning)
- `peer_review_skills` - Overall review quality
- `critical_analysis` - Thoroughness score
- `constructive_feedback` - Constructiveness score
- `attention_to_detail` - Specificity score

### For Authors (Performance Tracking)
- `assignment_performance` - Overall score received
- Specific skills identified in feedback (strengths/improvements)
- Learning trajectory over multiple submissions

## Background Tasks

The plugin runs background tasks every hour:

1. **Review Reminders** - Send emails for reviews due in 24 hours
2. **Overdue Handling** - Mark overdue reviews and notify instructor
3. **Analytics Updates** - Update assignment analytics
4. **Quality Monitoring** - Track review quality trends

## Testing

```bash
# Run unit tests
pytest plugins/peer-review/tests/test_plugin.py -v

# Run integration tests
pytest plugins/peer-review/tests/test_integration.py -v

# Run end-to-end tests
pytest plugins/peer-review/tests/test_e2e.py -v

# Test coverage
pytest --cov=plugins/peer-review --cov-report=html
```

## Architecture

```
PeerReviewPlugin
├── Hooks
│   ├── after_assessment_submit → Assign reviewers
│   ├── on_review_complete → Calculate results
│   ├── on_review_disputed → Handle disputes
│   └── on_calibration_required → Provide exercises
│
├── Core Services
│   ├── Reviewer Assignment (load balancing, fairness)
│   ├── Quality Scoring (4 dimensions)
│   ├── Calibration Training (adaptive difficulty)
│   ├── Dispute Resolution (AI-powered validity)
│   └── Analytics Engine (metrics, trends)
│
├── Background Tasks
│   ├── Review Reminders (hourly)
│   ├── Overdue Handling (hourly)
│   └── Analytics Updates (hourly)
│
└── Knowledge Graph
    ├── Reviewer Skills (collaborative learning)
    └── Author Performance (feedback-based)
```

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Submit work | ~200ms | Includes reviewer assignment |
| Submit review | ~150ms | Includes quality calculation |
| Get results | ~100ms | Cached if available |
| Calibration check | ~50ms | Cached per student |
| Dispute filing | ~100ms | Includes AI analysis |
| Analytics query | ~300ms | Complex aggregations |

## Security

- **Blind review**: Author identity hidden from reviewers
- **Double-blind review**: Both identities hidden
- **Sandboxed execution**: Plugin runs in isolated environment
- **Permission-gated**: All operations check permissions
- **Data encryption**: Student work encrypted at rest
- **Audit logging**: All actions logged for compliance

## Roadmap

### v1.1 (Q1 2026)
- [ ] AI-powered feedback generation
- [ ] Multi-language support
- [ ] Video submission support
- [ ] Group peer review

### v1.2 (Q2 2026)
- [ ] Advanced analytics (ML models)
- [ ] Reviewer matching algorithms
- [ ] Gamification (badges, leaderboards)
- [ ] Mobile app support

### v2.0 (Q3 2026)
- [ ] Peer review marketplace
- [ ] Cross-institution review
- [ ] Expert reviewer pool
- [ ] Blockchain certificates

## Support

- **Documentation**: https://docs.lms.edu/plugins/peer-review
- **Issues**: https://github.com/lms/plugins/peer-review/issues
- **Forum**: https://community.lms.edu/c/peer-review
- **Email**: support@lms.edu

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributors

- LMS Team - Initial development
- Community contributors - Enhancements

## Citation

If you use this plugin in research, please cite:

```bibtex
@software{lms_peer_review_2025,
  title={Peer Review Plugin for LMS Orchestration},
  author={LMS Team},
  year={2025},
  url={https://github.com/lms/plugins/peer-review}
}
```
