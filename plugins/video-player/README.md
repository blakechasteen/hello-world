# Video Player Plugin

**Version**: 1.0.0
**Category**: Content
**Author**: LMS Team

Interactive video player with embedded quizzes, progress tracking, and analytics.

## Features

✅ **Progress Tracking**
- Automatic save/resume from last position
- Completion detection (90% threshold)
- Watch segment tracking (detects skips)

✅ **Embedded Quizzes**
- Pause video at specific timestamps
- Multiple choice questions
- Immediate feedback with explanations
- Updates knowledge graph on correct answers

✅ **Analytics**
- Total views and unique viewers
- Average completion rate
- Drop-off point detection
- Quiz performance metrics

✅ **Playback Controls**
- Speed control (0.5x to 2x)
- Optional skip prevention
- Auto-play configuration
- Full keyboard shortcuts

## Installation

```bash
lms-cli install video-player-1.0.0.lmspkg
```

## Quick Start

### 1. Add Video to Lesson

```python
lesson = {
    "lesson_id": "lesson_123",
    "title": "Introduction to Python",
    "videos": [
        {
            "video_id": "video_456",
            "title": "Variables and Data Types",
            "url": "https://cdn.example.com/videos/python_vars.mp4",
            "duration": 720.0,  # 12 minutes
            "thumbnail": "https://cdn.example.com/thumbs/python_vars.jpg"
        }
    ]
}
```

### 2. Add Embedded Quizzes

```python
quiz = {
    "quiz_id": "quiz_789",
    "video_id": "video_456",
    "timestamp": 360.0,  # Show at 6:00 mark
    "question": "What is a variable?",
    "options": [
        "A container for data",
        "A function",
        "A loop",
        "A class"
    ],
    "correct_answer": 0,
    "explanation": "A variable is a container for storing data values.",
    "concept": "variables"
}
```

### 3. Track Progress

```javascript
// Frontend integration
const videoPlayer = new LMSVideoPlayer({
  videoId: "video_456",
  onProgress: async (time, duration) => {
    await api.post('/plugins/video-player/progress', {
      video_id: "video_456",
      current_time: time,
      duration: duration
    });
  }
});
```

### 4. View Analytics

```python
from plugins.video_player.backend.plugin import VideoPlayerPlugin

plugin = VideoPlayerPlugin(plugin_id="video-player", config={})
await plugin.initialize()

analytics = await plugin.get_analytics(video_id="video_456")
print(f"Completion Rate: {analytics.avg_completion_rate * 100:.1f}%")
print(f"Drop-off Points: {analytics.drop_off_points}")
```

## Configuration

```yaml
config:
  autoplay: false                    # Auto-play on load
  enable_speed_control: true         # Allow speed adjustment
  enable_skip: false                 # Prevent skipping ahead
  quiz_pause_video: true             # Pause when quiz appears
  analytics_interval_seconds: 30     # Progress update frequency
```

## API Endpoints

### Track Progress
```
POST /api/plugins/video-player/video/{video_id}/progress
{
  "current_time": 360.5,
  "duration": 720.0
}
```

### Get Analytics
```
GET /api/plugins/video-player/video/{video_id}/analytics
```

Response:
```json
{
  "video_id": "video_456",
  "total_views": 125,
  "unique_viewers": 98,
  "avg_completion_rate": 0.76,
  "avg_watch_time": 548.2,
  "drop_off_points": [120.0, 360.0, 540.0],
  "quiz_scores": {
    "quiz_789": 0.82
  }
}
```

## Hooks

### before_lesson_render
Injects video player configuration and progress data.

### on_video_progress
Tracks viewing progress every 30 seconds (configurable).

### on_video_complete
Triggered when student watches 90% of video. Updates knowledge graph.

### on_embedded_quiz_submit
Handles quiz answers, provides feedback, updates knowledge graph.

## Knowledge Graph Integration

The plugin updates the student knowledge graph with:
- **Video completion**: 0.7 mastery level
- **Quiz correct**: 0.9 mastery level
- **Quiz incorrect**: 0.3 mastery level

## Frontend Components

### VideoPlayer.tsx
```typescript
import { LMSVideoPlayer } from '@lms/video-player';

<LMSVideoPlayer
  videoId="video_456"
  videoUrl="https://cdn.example.com/video.mp4"
  autoplay={false}
  enableSpeedControl={true}
  enableSkip={false}
  onProgress={(time, duration) => trackProgress(time, duration)}
  onComplete={() => handleComplete()}
/>
```

### VideoAnalytics.tsx
```typescript
import { VideoAnalytics } from '@lms/video-player';

<VideoAnalytics
  videoId="video_456"
  showDropOffPoints={true}
  showQuizScores={true}
/>
```

## Testing

```bash
pytest plugins/video-player/tests/ -v
```

## License

MIT
