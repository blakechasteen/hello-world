"""
Promptly Web Dashboard - Modern Web Interface
==============================================
Real-time web dashboard with live analytics and visualization.

Features:
- Live prompt execution
- Real-time charts and metrics
- Loop flow visualization
- Cost tracking and budgets
- Skill management
- Export capabilities
"""

import sys
import os

# Graceful imports
try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask/SocketIO not installed. Install with: pip install flask flask-socketio")
    FLASK_AVAILABLE = False

import json
from datetime import datetime
from pathlib import Path

# Promptly imports
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from promptly import Promptly
    from tools.prompt_analytics import PromptAnalytics
    from tools.cost_tracker import CostTracker
    PROMPTLY_AVAILABLE = True
except ImportError as e:
    PROMPTLY_AVAILABLE = False
    print(f"Promptly not available: {e}")


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'promptly-secret-key-change-in-production'

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*") if FLASK_AVAILABLE else None

# Initialize Promptly components
promptly = Promptly() if PROMPTLY_AVAILABLE else None
analytics = PromptAnalytics() if PROMPTLY_AVAILABLE else None
cost_tracker = CostTracker() if PROMPTLY_AVAILABLE else None


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard_live.html')


@app.route('/api/execute', methods=['POST'])
def api_execute():
    """Execute a prompt"""
    if not PROMPTLY_AVAILABLE or not promptly:
        return jsonify({'error': 'Promptly not available'}), 500

    try:
        data = request.json
        prompt = data.get('prompt', '')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        # Execute prompt (simulated for now)
        result = {
            'prompt': prompt,
            'result': f"Simulated result for: {prompt}",
            'tokens': 100,
            'cost': 0.0010,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        # Broadcast to all connected clients
        if socketio:
            socketio.emit('execution_complete', result)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """Get real-time statistics"""
    try:
        if analytics:
            summary = analytics.get_summary()
        else:
            summary = {
                'total_executions': 0,
                'total_cost': 0.0,
                'total_tokens': 0,
                'average_tokens': 0
            }

        return jsonify(summary)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def api_history():
    """Get execution history"""
    try:
        if analytics:
            history = analytics.get_recent(limit=50)
        else:
            history = []

        return jsonify({'history': history})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cost/breakdown')
def api_cost_breakdown():
    """Get cost breakdown by category"""
    try:
        if cost_tracker:
            breakdown = cost_tracker.get_breakdown()
        else:
            breakdown = {
                'by_model': {},
                'by_task': {},
                'by_date': {}
            }

        return jsonify(breakdown)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/loops')
def api_loops():
    """Get active loops"""
    # Placeholder for loop tracking
    return jsonify({
        'active_loops': [],
        'completed_loops': [],
        'failed_loops': []
    })


@app.route('/api/skills')
def api_skills():
    """Get available skills"""
    skills = [
        {'name': 'Summarize', 'category': 'Analysis', 'uses': 15},
        {'name': 'Extract', 'category': 'Analysis', 'uses': 8},
        {'name': 'Classify', 'category': 'Analysis', 'uses': 5},
        {'name': 'Write', 'category': 'Generation', 'uses': 12},
        {'name': 'Rewrite', 'category': 'Generation', 'uses': 6},
        {'name': 'Reflect', 'category': 'Meta', 'uses': 3},
    ]

    return jsonify({'skills': skills})


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Promptly server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")


@socketio.on('execute_prompt')
def handle_execute_prompt(data):
    """Handle real-time prompt execution"""
    try:
        prompt = data.get('prompt', '')

        # Simulate execution
        result = {
            'prompt': prompt,
            'result': f"Live result for: {prompt}",
            'tokens': 100,
            'cost': 0.0010,
            'timestamp': datetime.now().isoformat()
        }

        # Send result back
        emit('execution_result', result)

        # Broadcast to all clients
        emit('execution_complete', result, broadcast=True)

    except Exception as e:
        emit('execution_error', {'error': str(e)})


def create_template():
    """Create HTML template for dashboard"""
    template_dir = Path(__file__).parent.parent.parent / 'templates'
    template_dir.mkdir(exist_ok=True)

    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Promptly Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { color: #667eea; font-size: 28px; }
        .subtitle { color: #888; font-size: 14px; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-card h3 { color: #888; font-size: 14px; margin-bottom: 10px; }
        .stat-card .value { font-size: 32px; font-weight: bold; color: #667eea; }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .panel h2 {
            color: #667eea;
            font-size: 20px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        input, textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            margin-bottom: 10px;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }
        button:hover { background: #5568d3; }
        #results {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            min-height: 200px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .history-item {
            padding: 10px;
            border-bottom: 1px solid #f0f0f0;
            font-size: 14px;
        }
        .history-item:last-child { border-bottom: none; }
        .timestamp { color: #888; font-size: 12px; }
        .status-online { color: #10b981; }
        .chart-container { height: 300px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🚀 Promptly Dashboard</h1>
                <div class="subtitle">Interactive Prompt Platform</div>
            </div>
            <div class="status-online" id="status">● Live</div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <h3>Total Executions</h3>
                <div class="value" id="total-executions">0</div>
            </div>
            <div class="stat-card">
                <h3>Total Cost</h3>
                <div class="value" id="total-cost">$0.00</div>
            </div>
            <div class="stat-card">
                <h3>Total Tokens</h3>
                <div class="value" id="total-tokens">0</div>
            </div>
            <div class="stat-card">
                <h3>Avg Tokens/Prompt</h3>
                <div class="value" id="avg-tokens">0</div>
            </div>
        </div>

        <div class="main-content">
            <div class="panel">
                <h2>Prompt Composer</h2>
                <textarea id="prompt-input" placeholder="Enter your prompt..." rows="4"></textarea>
                <button onclick="executePrompt()">Execute</button>
                <h3 style="margin-top: 20px; color: #888;">Results</h3>
                <div id="results">Results will appear here...</div>
            </div>

            <div class="panel">
                <h2>Execution History</h2>
                <div id="history" style="max-height: 400px; overflow-y: auto;">
                    <p style="color: #888;">No executions yet</p>
                </div>
            </div>
        </div>

        <div class="panel" style="margin-top: 20px;">
            <h2>Cost Breakdown</h2>
            <div class="chart-container">
                <canvas id="cost-chart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const socket = io();

        socket.on('connect', () => {
            console.log('Connected to server');
            document.getElementById('status').textContent = '● Live';
        });

        socket.on('disconnect', () => {
            document.getElementById('status').textContent = '● Offline';
            document.getElementById('status').style.color = '#ef4444';
        });

        socket.on('execution_complete', (data) => {
            addToHistory(data);
            updateStats();
        });

        function executePrompt() {
            const prompt = document.getElementById('prompt-input').value;
            if (!prompt) return;

            document.getElementById('results').textContent = 'Executing...';

            fetch('/api/execute', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt})
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById('results').textContent = data.result || data.error;
                if (data.error) {
                    document.getElementById('results').style.color = '#ef4444';
                } else {
                    document.getElementById('results').style.color = '#333';
                }
            })
            .catch(err => {
                document.getElementById('results').textContent = 'Error: ' + err;
                document.getElementById('results').style.color = '#ef4444';
            });
        }

        function addToHistory(data) {
            const history = document.getElementById('history');
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div><strong>${data.prompt.substring(0, 50)}...</strong></div>
                <div class="timestamp">${new Date(data.timestamp).toLocaleTimeString()} | ${data.tokens} tokens | $${data.cost}</div>
            `;
            history.insertBefore(item, history.firstChild);
        }

        function updateStats() {
            fetch('/api/stats')
            .then(res => res.json())
            .then(data => {
                document.getElementById('total-executions').textContent = data.total_executions || 0;
                document.getElementById('total-cost').textContent = '$' + (data.total_cost || 0).toFixed(4);
                document.getElementById('total-tokens').textContent = data.total_tokens || 0;
                document.getElementById('avg-tokens').textContent = data.average_tokens || 0;
            });
        }

        // Initialize chart
        const ctx = document.getElementById('cost-chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cost Over Time',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

        // Update stats on load
        updateStats();
        setInterval(updateStats, 5000);  // Update every 5 seconds
    </script>
</body>
</html>'''

    template_path = template_dir / 'dashboard_live.html'
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Created template: {template_path}")


def run_web_app(host='0.0.0.0', port=5000, debug=True):
    """Run the web dashboard"""
    if not FLASK_AVAILABLE:
        print("Flask not available. Cannot start web server.")
        return

    # Create template
    create_template()

    print(f"""
╔══════════════════════════════════════════╗
║   Promptly Web Dashboard Starting...    ║
╚══════════════════════════════════════════╝

Server: http://{host}:{port}
Dashboard: http://localhost:{port}

Press Ctrl+C to stop
""")

    if socketio:
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    else:
        app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_web_app()
