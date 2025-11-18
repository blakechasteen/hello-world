# Analytics Dashboard Plugin

**Version**: 1.0.0
**Category**: Analytics
**Author**: LMS Team

Comprehensive analytics dashboard with student performance tracking, course insights, and predictive analytics.

## Features

✅ **Student Analytics**
- Overall GPA and trend
- Assignment completion rate
- Time on task metrics
- Knowledge graph mastery visualization
- At-risk predictions

✅ **Course Analytics**
- Class performance distribution
- Average scores by assessment
- Engagement metrics (logins, time spent)
- Drop-off analysis
- Concept mastery heatmap

✅ **Instructor Analytics**
- Course comparison
- Student progress tracking
- Intervention recommendations
- Teaching effectiveness metrics

✅ **Institution Analytics**
- Department-wide performance
- Plugin usage statistics
- Retention rates
- Benchmark comparisons

## Quick Start

### Student Dashboard

```python
from plugins.analytics_dashboard import get_student_analytics

analytics = await get_student_analytics(student_id="student_123")

print(f"GPA: {analytics.gpa}")
print(f"Completion Rate: {analytics.completion_rate * 100}%")
print(f"At Risk: {analytics.at_risk}")
print(f"Strong Concepts: {analytics.strong_concepts}")
print(f"Weak Concepts: {analytics.weak_concepts}")
```

### Course Dashboard

```python
from plugins.analytics_dashboard import get_course_analytics

analytics = await get_course_analytics(course_id="course_456")

print(f"Average Score: {analytics.avg_score}")
print(f"Completion Rate: {analytics.completion_rate * 100}%")
print(f"At-Risk Students: {len(analytics.at_risk_students)}")
print(f"Top Performers: {analytics.top_performers}")
```

## API Usage

```bash
# Get student analytics
GET /api/plugins/analytics-dashboard/analytics/student/{student_id}

# Get course analytics
GET /api/plugins/analytics-dashboard/analytics/course/{course_id}

# Get institution analytics
GET /api/plugins/analytics-dashboard/analytics/institution/{institution_id}
```

## Response Format

```json
{
  "student_id": "student_123",
  "gpa": 3.45,
  "gpa_trend": "improving",
  "completion_rate": 0.87,
  "avg_time_per_lesson_minutes": 45.2,
  "at_risk": false,
  "strong_concepts": ["variables", "loops", "functions"],
  "weak_concepts": ["recursion", "data_structures"],
  "knowledge_graph": {
    "total_concepts": 42,
    "mastered": 28,
    "struggling": 5,
    "not_attempted": 9
  },
  "predictions": {
    "final_grade": "B+",
    "completion_probability": 0.94
  }
}
```

## Visualizations

The plugin provides React components for:
- Performance line charts
- Concept mastery radar charts
- Score distribution histograms
- Engagement heatmaps
- Knowledge graph network visualizations

## Configuration

```yaml
config:
  update_interval_minutes: 60
  show_predictions: true
  enable_benchmarking: true
```

## License

MIT
