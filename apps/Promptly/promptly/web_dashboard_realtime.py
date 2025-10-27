#!/usr/bin/env python3
"""
Promptly Analytics Web Dashboard - Real-time Edition
=====================================================
Beautiful web interface with WebSocket real-time updates
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask-SocketIO not installed. Install with: pip install flask-socketio")
    FLASK_AVAILABLE = False
    sys.exit(1)

from tools.prompt_analytics import PromptAnalytics, PromptExecution
from datetime import datetime
import json
import threading
import time

app = Flask(__name__, template_folder='../templates')
app.config['SECRET_KEY'] = 'promptly-secret-key-change-in-production'
socketio = SocketIO(app, cors_allowed_origins="*")

analytics = PromptAnalytics()
last_execution_count = 0

@app.route('/')
def index():
    """Fast optimized dashboard"""
    return render_template('dashboard_fast.html')

@app.route('/full')
def full():
    """Full real-time dashboard"""
    return render_template('dashboard_realtime.html')

@app.route('/enhanced')
def enhanced():
    """Enhanced dashboard (non-realtime)"""
    return render_template('dashboard_enhanced.html')

@app.route('/api/summary')
def api_summary():
    """Get overall analytics summary"""
    try:
        summary = analytics.get_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/prompts')
def api_prompts():
    """Get list of all prompts with basic stats"""
    try:
        limit = request.args.get('limit', 50, type=int)  # Default to 50, configurable

        conn = analytics._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                prompt_name,
                COUNT(*) as executions,
                AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate,
                AVG(execution_time) as avg_time,
                AVG(quality_score) as avg_quality
            FROM executions
            GROUP BY prompt_name
            ORDER BY executions DESC
            LIMIT ?
        """, (limit,))

        prompts = []
        for row in cursor.fetchall():
            prompts.append({
                'name': row[0],
                'executions': row[1],
                'success_rate': round(row[2], 1),
                'avg_time': round(row[3], 2),
                'avg_quality': round(row[4], 2) if row[4] else None
            })

        conn.close()
        return jsonify(prompts)
    except Exception as e:
        print(f"Error in api_prompts: {e}")
        # Return empty list instead of error to prevent frontend issues
        return jsonify([])

@app.route('/api/prompt/<name>')
def api_prompt_details(name):
    """Get detailed stats for a specific prompt"""
    try:
        stats = analytics.get_prompt_stats(name)
        if not stats:
            return jsonify({'error': 'Prompt not found'}), 404

        return jsonify({
            'prompt_name': stats.prompt_name,
            'total_executions': stats.total_executions,
            'success_rate': stats.success_rate,
            'avg_execution_time': stats.avg_execution_time,
            'avg_quality_score': stats.avg_quality_score,
            'quality_trend': stats.quality_trend,
            'total_cost': stats.total_cost,
            'last_executed': stats.last_executed
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/prompt/<name>/history')
def api_prompt_history(name):
    """Get execution history for a prompt"""
    try:
        conn = analytics._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, execution_time, quality_score, success
            FROM executions
            WHERE prompt_name = ?
            ORDER BY timestamp DESC
            LIMIT 50
        """, (name,))

        history = []
        for row in cursor.fetchall():
            history.append({
                'timestamp': row[0],
                'execution_time': row[1],
                'quality_score': row[2],
                'success': bool(row[3])
            })

        conn.close()
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations')
def api_recommendations():
    """Get analytics recommendations"""
    try:
        recommendations = analytics.get_recommendations()
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/top/<metric>')
def api_top_prompts(metric):
    """Get top prompts by metric"""
    try:
        limit = request.args.get('limit', 5, type=int)
        top_prompts = analytics.get_top_prompts(metric=metric, limit=limit)

        results = []
        for stats in top_prompts:
            results.append({
                'prompt_name': stats.prompt_name,
                'total_executions': stats.total_executions,
                'success_rate': stats.success_rate,
                'avg_execution_time': stats.avg_execution_time,
                'avg_quality_score': stats.avg_quality_score,
                'total_cost': stats.total_cost
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/timeline')
def api_timeline():
    """Get execution timeline for charting"""
    try:
        conn = analytics._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as count,
                AVG(execution_time) as avg_time,
                AVG(quality_score) as avg_quality
            FROM executions
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        """)

        timeline = []
        for row in cursor.fetchall():
            timeline.append({
                'date': row[0],
                'count': row[1],
                'avg_time': round(row[2], 2),
                'avg_quality': round(row[3], 2) if row[3] else None
            })

        conn.close()
        return jsonify(timeline)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print(f'Client connected: {request.sid}')
    emit('status', {'message': 'Connected to Promptly real-time updates'})

    # Send initial data
    summary = analytics.get_summary()
    emit('summary_update', summary)

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print(f'Client disconnected: {request.sid}')

@socketio.on('request_update')
def handle_request_update():
    """Client requested manual update"""
    summary = analytics.get_summary()
    emit('summary_update', summary)

# Background thread to monitor for new executions
def background_monitor():
    """Monitor database for new executions and push updates"""
    global last_execution_count

    while True:
        try:
            summary = analytics.get_summary()
            current_count = summary['total_executions']

            if current_count > last_execution_count:
                # New execution detected!
                print(f'New execution detected: {current_count} (was {last_execution_count})')

                # Broadcast to all connected clients
                socketio.emit('summary_update', summary)
                socketio.emit('new_execution', {
                    'count': current_count - last_execution_count,
                    'total': current_count
                })

                # Also send updated prompts list
                try:
                    conn = analytics._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT
                            prompt_name,
                            COUNT(*) as executions,
                            AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate,
                            AVG(execution_time) as avg_time,
                            AVG(quality_score) as avg_quality
                        FROM executions
                        GROUP BY prompt_name
                        ORDER BY executions DESC
                    """)

                    prompts = []
                    for row in cursor.fetchall():
                        prompts.append({
                            'name': row[0],
                            'executions': row[1],
                            'success_rate': round(row[2], 1),
                            'avg_time': round(row[3], 2),
                            'avg_quality': round(row[4], 2) if row[4] else None
                        })

                    conn.close()
                    socketio.emit('prompts_update', prompts)
                except Exception as e:
                    print(f'Error updating prompts: {e}')

                last_execution_count = current_count

        except Exception as e:
            print(f'Error in background monitor: {e}')

        time.sleep(2)  # Check every 2 seconds

if __name__ == '__main__':
    print("=" * 60)
    print("Promptly Real-time Analytics Dashboard")
    print("=" * 60)
    print("\nFeatures:")
    print("  - WebSocket real-time updates")
    print("  - Live execution notifications")
    print("  - Auto-updating charts")
    print("  - Push notifications")
    print("\nStarting server...")
    print("Real-time dashboard: http://localhost:5000")
    print("Enhanced dashboard: http://localhost:5000/enhanced")
    print("\nPress Ctrl+C to stop\n")

    # Get initial count
    try:
        summary = analytics.get_summary()
        last_execution_count = summary['total_executions']
        print(f"Initial execution count: {last_execution_count}\n")
    except:
        last_execution_count = 0

    # Start background monitoring thread
    monitor_thread = threading.Thread(target=background_monitor, daemon=True)
    monitor_thread.start()

    # Run with SocketIO
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
