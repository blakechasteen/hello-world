#!/usr/bin/env python3
"""
Promptly Analytics Web Dashboard
=================================
Beautiful web interface for visualizing prompt analytics with charts
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from flask import Flask, render_template, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask not installed. Install with: pip install flask")
    FLASK_AVAILABLE = False
    sys.exit(1)

from tools.prompt_analytics import PromptAnalytics
from datetime import datetime
import json

app = Flask(__name__, template_folder='../templates')
analytics = PromptAnalytics()

@app.route('/')
def index():
    """Enhanced dashboard with all features"""
    return render_template('dashboard_enhanced.html')

@app.route('/charts')
def charts():
    """Dashboard with basic charts"""
    return render_template('dashboard_charts.html')

@app.route('/simple')
def simple():
    """Simple dashboard without charts"""
    return render_template('dashboard.html')

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
        return jsonify(prompts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

        # Get executions grouped by day
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

if __name__ == '__main__':
    print("=" * 60)
    print("Promptly Analytics Dashboard with Charts")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Dashboard: http://localhost:5000")
    print("Simple version: http://localhost:5000/simple")
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
