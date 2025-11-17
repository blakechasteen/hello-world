#!/usr/bin/env python3
"""
EdWIN Web Server - Flask Backend

Serves the web UI and provides REST API for lesson data and progress tracking.
"""
import asyncio
import sys
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.lesson import LessonManager, Lesson
from core.progress import ProgressTracker

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for development

# Initialize EdWIN components
content_dir = Path(__file__).parent.parent.parent / "content"
lesson_manager = LessonManager(content_dir)
progress_tracker = ProgressTracker()


@app.route('/')
def index():
    """Serve the main web UI."""
    return send_from_directory('static', 'index.html')


@app.route('/api/lessons', methods=['GET'])
def get_lessons():
    """Get all available lessons grouped by difficulty."""
    lessons_by_difficulty = {
        'beginner': [],
        'intermediate': [],
        'advanced': []
    }

    for lesson_id, lesson in lesson_manager.lessons.items():
        lesson_data = {
            'id': lesson.id,
            'title': lesson.title,
            'description': lesson.description,
            'difficulty': lesson.difficulty.value,
            'lesson_type': lesson.lesson_type.value,
            'estimated_time': lesson.estimated_time,
            'xp_reward': lesson.xp_reward,
            'prerequisites': lesson.prerequisites,
            'completed': lesson.id in progress_tracker.progress.completed_lessons,
            'locked': not all(p in progress_tracker.progress.completed_lessons
                             for p in lesson.prerequisites)
        }
        lessons_by_difficulty[lesson.difficulty.value].append(lesson_data)

    return jsonify(lessons_by_difficulty)


@app.route('/api/lessons/<lesson_id>', methods=['GET'])
def get_lesson(lesson_id):
    """Get detailed lesson content."""
    lesson = lesson_manager.get_lesson(lesson_id)
    if not lesson:
        return jsonify({'error': 'Lesson not found'}), 404

    # Convert lesson to JSON-serializable format
    lesson_data = {
        'id': lesson.id,
        'title': lesson.title,
        'description': lesson.description,
        'difficulty': lesson.difficulty.value,
        'lesson_type': lesson.lesson_type.value,
        'content': lesson.content,
        'code_examples': lesson.code_examples,
        'quiz_questions': lesson.quiz_questions,
        'challenges': [
            {
                'description': c.description,
                'starter_code': c.starter_code,
                'hints': [{'level': h.level, 'text': h.text, 'code_snippet': h.code_snippet}
                         for h in c.hints],
                'points': c.points
            }
            for c in lesson.challenges
        ],
        'estimated_time': lesson.estimated_time,
        'xp_reward': lesson.xp_reward,
        'prerequisites': lesson.prerequisites,
        'next_lessons': lesson.next_lessons,
        'tags': lesson.tags
    }

    return jsonify(lesson_data)


@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get learner's current progress."""
    stats = progress_tracker.get_stats()

    progress_data = {
        'learner_name': progress_tracker.progress.learner_name,
        'level': progress_tracker.progress.level,
        'total_xp': progress_tracker.progress.total_xp,
        'xp_to_next_level': stats['xp_to_next_level'],
        'completed_lessons': list(progress_tracker.progress.completed_lessons),
        'lessons_completed': len(progress_tracker.progress.completed_lessons),
        'challenges_completed': progress_tracker.progress.challenges_completed,
        'badges': progress_tracker.progress.badges,
        'badges_earned': len(progress_tracker.progress.badges),
        'hints_used': progress_tracker.progress.hints_used,
        'completion_rate': stats['completion_rate']
    }

    return jsonify(progress_data)


@app.route('/api/progress/start-lesson', methods=['POST'])
def start_lesson():
    """Mark a lesson as started."""
    data = request.json
    lesson_id = data.get('lesson_id')

    if not lesson_id:
        return jsonify({'error': 'lesson_id required'}), 400

    progress_tracker.start_lesson(lesson_id)
    return jsonify({'success': True})


@app.route('/api/progress/complete-lesson', methods=['POST'])
def complete_lesson():
    """Mark a lesson as complete and award XP."""
    data = request.json
    lesson_id = data.get('lesson_id')
    score = data.get('score', 100)

    if not lesson_id:
        return jsonify({'error': 'lesson_id required'}), 400

    lesson = lesson_manager.get_lesson(lesson_id)
    if not lesson:
        return jsonify({'error': 'Lesson not found'}), 404

    result = progress_tracker.mark_lesson_complete(lesson_id, lesson.xp_reward, score)

    return jsonify({
        'success': True,
        'leveled_up': result['leveled_up'],
        'new_level': result['new_level'],
        'total_xp': result['total_xp']
    })


@app.route('/api/progress/complete-challenge', methods=['POST'])
def complete_challenge():
    """Mark a challenge as complete."""
    data = request.json
    challenge_id = data.get('challenge_id')
    points = data.get('points', 10)

    if not challenge_id:
        return jsonify({'error': 'challenge_id required'}), 400

    progress_tracker.mark_challenge_complete(challenge_id, points)
    return jsonify({'success': True})


@app.route('/api/progress/use-hint', methods=['POST'])
def use_hint():
    """Record that a hint was used."""
    progress_tracker.use_hint()
    return jsonify({'success': True})


@app.route('/api/badges', methods=['GET'])
def get_badges():
    """Get all available badges and earned status."""
    badges_data = []

    for badge_id, badge in progress_tracker.available_badges.items():
        badges_data.append({
            'id': badge.id,
            'name': badge.name,
            'description': badge.description,
            'icon': badge.icon,
            'earned': badge_id in progress_tracker.progress.badges
        })

    return jsonify(badges_data)


if __name__ == '__main__':
    print("🎓 Starting EdWIN Web Server...")
    print("📚 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
