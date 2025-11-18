"""Video Player Plugin

Interactive video player with embedded quizzes, progress tracking, and analytics.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from backend.plugins import PluginBase, HookContext, HookResponse


class VideoProgress(BaseModel):
    """Video viewing progress"""
    student_id: str
    video_id: str
    current_time: float  # Seconds
    duration: float
    completed: bool
    watch_segments: List[Dict[str, float]]  # [{start, end}]
    last_updated: datetime


class EmbeddedQuiz(BaseModel):
    """Quiz embedded in video"""
    quiz_id: str
    video_id: str
    timestamp: float  # When to show quiz (seconds)
    question: str
    options: List[str]
    correct_answer: int
    explanation: str


class VideoAnalytics(BaseModel):
    """Analytics for a video"""
    video_id: str
    total_views: int
    unique_viewers: int
    avg_completion_rate: float
    avg_watch_time: float
    drop_off_points: List[float]  # Timestamps where students drop off
    quiz_scores: Dict[str, float]  # quiz_id -> avg_score


class VideoPlayerPlugin(PluginBase):
    """Interactive video player plugin"""

    def __init__(self, plugin_id: str, config: Dict[str, Any] = None):
        super().__init__(plugin_id, config)
        self._progress_cache: Dict[str, VideoProgress] = {}
        self._analytics_cache: Dict[str, VideoAnalytics] = {}

    async def initialize(self) -> bool:
        """Initialize plugin"""
        # Register hooks
        self.register_hook("before_lesson_render", self.on_lesson_render)
        self.register_hook("on_video_progress", self.on_progress_update)
        self.register_hook("on_video_complete", self.on_video_complete)
        self.register_hook("on_embedded_quiz_submit", self.on_quiz_submit)

        print(f"✅ Video Player Plugin initialized")
        return True

    async def shutdown(self) -> bool:
        """Cleanup resources"""
        # Flush any pending analytics
        await self._flush_analytics()
        return True

    async def on_lesson_render(self, context: HookContext) -> HookResponse:
        """Inject video player when lesson contains videos"""
        lesson = context.data.get("lesson", {})
        student_id = context.user_id

        # Check if lesson has videos
        videos = lesson.get("videos", [])
        if not videos:
            return HookResponse(success=True)

        # Load student's progress for each video
        progress_data = []
        for video in videos:
            video_id = video.get("video_id")
            progress_key = f"{student_id}:{video_id}"

            # Get cached progress or create new
            if progress_key not in self._progress_cache:
                self._progress_cache[progress_key] = VideoProgress(
                    student_id=student_id,
                    video_id=video_id,
                    current_time=0.0,
                    duration=video.get("duration", 0.0),
                    completed=False,
                    watch_segments=[],
                    last_updated=datetime.utcnow()
                )

            progress = self._progress_cache[progress_key]
            progress_data.append({
                "video_id": video_id,
                "current_time": progress.current_time,
                "completed": progress.completed,
                "completion_percentage": (progress.current_time / progress.duration * 100)
                    if progress.duration > 0 else 0
            })

        # Load embedded quizzes
        quizzes = await self._load_embedded_quizzes(videos)

        return HookResponse(
            success=True,
            data={
                "video_player_enabled": True,
                "progress": progress_data,
                "embedded_quizzes": quizzes,
                "config": {
                    "autoplay": self.config.get("autoplay", False),
                    "enable_speed_control": self.config.get("enable_speed_control", True),
                    "enable_skip": self.config.get("enable_skip", False),
                }
            }
        )

    async def on_progress_update(self, context: HookContext) -> HookResponse:
        """Track video viewing progress"""
        student_id = context.user_id
        video_id = context.data.get("video_id")
        current_time = context.data.get("current_time", 0.0)
        duration = context.data.get("duration", 0.0)

        progress_key = f"{student_id}:{video_id}"

        # Update progress
        if progress_key in self._progress_cache:
            progress = self._progress_cache[progress_key]
        else:
            progress = VideoProgress(
                student_id=student_id,
                video_id=video_id,
                current_time=0.0,
                duration=duration,
                completed=False,
                watch_segments=[],
                last_updated=datetime.utcnow()
            )
            self._progress_cache[progress_key] = progress

        # Update current position
        progress.current_time = current_time
        progress.last_updated = datetime.utcnow()

        # Track watch segment
        if progress.watch_segments:
            last_segment = progress.watch_segments[-1]
            # Extend segment if continuous
            if abs(current_time - last_segment["end"]) < 2.0:
                last_segment["end"] = current_time
            else:
                # New segment (user skipped)
                progress.watch_segments.append({"start": current_time, "end": current_time})
        else:
            progress.watch_segments.append({"start": current_time, "end": current_time})

        # Check completion (watched at least 90%)
        completion_percentage = (current_time / duration * 100) if duration > 0 else 0
        if completion_percentage >= 90 and not progress.completed:
            progress.completed = True
            # Trigger completion hook
            await self.execute_hook("on_video_complete", context)

        return HookResponse(
            success=True,
            data={
                "completion_percentage": completion_percentage,
                "completed": progress.completed
            }
        )

    async def on_video_complete(self, context: HookContext) -> HookResponse:
        """Handle video completion"""
        student_id = context.user_id
        video_id = context.data.get("video_id")

        # Update knowledge graph (if available)
        if hasattr(context, "knowledge_graph"):
            await context.knowledge_graph.record_learning_event(
                student_id=student_id,
                resource_id=video_id,
                resource_type="video",
                mastery_level=0.7,  # Base mastery from watching
                evidence="Completed video viewing"
            )

        # Emit analytics event
        if hasattr(context, "events"):
            await context.events.publish("video.completed", {
                "student_id": student_id,
                "video_id": video_id,
                "timestamp": datetime.utcnow().isoformat()
            })

        return HookResponse(success=True, message="Video marked as complete")

    async def on_quiz_submit(self, context: HookContext) -> HookResponse:
        """Handle embedded quiz submission"""
        student_id = context.user_id
        quiz_id = context.data.get("quiz_id")
        answer = context.data.get("answer")
        correct_answer = context.data.get("correct_answer")

        is_correct = (answer == correct_answer)

        # Update knowledge graph with quiz result
        if hasattr(context, "knowledge_graph") and is_correct:
            await context.knowledge_graph.record_learning_event(
                student_id=student_id,
                concept=context.data.get("concept"),
                mastery_level=0.9 if is_correct else 0.3,
                evidence=f"Embedded quiz: {'correct' if is_correct else 'incorrect'}"
            )

        return HookResponse(
            success=True,
            data={
                "correct": is_correct,
                "explanation": context.data.get("explanation", "")
            }
        )

    async def get_analytics(self, video_id: str) -> VideoAnalytics:
        """Get analytics for a video"""
        if video_id in self._analytics_cache:
            return self._analytics_cache[video_id]

        # Calculate analytics from progress data
        video_progresses = [
            p for p in self._progress_cache.values()
            if p.video_id == video_id
        ]

        total_views = len(video_progresses)
        unique_viewers = len(set(p.student_id for p in video_progresses))
        completed = [p for p in video_progresses if p.completed]
        avg_completion_rate = len(completed) / total_views if total_views > 0 else 0
        avg_watch_time = sum(p.current_time for p in video_progresses) / total_views if total_views > 0 else 0

        # Find drop-off points (where many students stopped watching)
        drop_offs = self._calculate_drop_offs(video_progresses)

        analytics = VideoAnalytics(
            video_id=video_id,
            total_views=total_views,
            unique_viewers=unique_viewers,
            avg_completion_rate=avg_completion_rate,
            avg_watch_time=avg_watch_time,
            drop_off_points=drop_offs,
            quiz_scores={}
        )

        self._analytics_cache[video_id] = analytics
        return analytics

    def _calculate_drop_offs(self, progresses: List[VideoProgress]) -> List[float]:
        """Calculate where students typically drop off"""
        # Group by 30-second intervals and count completions
        intervals: Dict[int, int] = {}
        for progress in progresses:
            interval = int(progress.current_time / 30)
            intervals[interval] = intervals.get(interval, 0) + 1

        # Find intervals with >20% drop-off
        drop_offs = []
        total = len(progresses)
        for interval, count in sorted(intervals.items()):
            drop_percentage = (total - count) / total if total > 0 else 0
            if drop_percentage > 0.2:
                drop_offs.append(interval * 30.0)

        return drop_offs[:5]  # Top 5 drop-off points

    async def _load_embedded_quizzes(self, videos: List[Dict]) -> List[Dict]:
        """Load embedded quizzes for videos"""
        # In production, load from database
        # For now, return empty list
        return []

    async def _flush_analytics(self):
        """Flush analytics to database"""
        # In production, persist to database
        pass
