"""
Promptly Strategy Web Server - Simple and Elegant

A minimal Flask server to power the web demo.

Usage:
    python web_server.py

Then open: http://localhost:5000
"""

import asyncio
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.prompting.registry import get_registry
from HoloLoom.prompting.auto_detect import AutoDetector
from HoloLoom.prompting.strategy import StrategyContext

app = Flask(__name__, template_folder=str(Path(__file__).parent))
CORS(app)  # Enable CORS for API

# Initialize registry (done once on startup)
registry = get_registry()
detector = AutoDetector(registry=registry)


@app.route('/')
def index():
    """Serve the web demo."""
    return render_template('web_demo.html')


@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get list of available strategies."""
    strategies = []
    for name, strategy in registry.strategies.items():
        strategies.append({
            'name': name,
            'description': strategy.description,
            'category': strategy.category.value
        })
    return jsonify({
        'strategies': sorted(strategies, key=lambda x: x['name'])
    })


@app.route('/api/enhance', methods=['POST'])
def enhance_query():
    """Enhance a query with a strategy."""
    data = request.json
    query = data.get('query', '').strip()
    strategy_name = data.get('strategy', 'auto')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # Run async enhancement
        result = asyncio.run(run_enhancement(query, strategy_name))
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


async def run_enhancement(query: str, strategy_name: str):
    """Run strategy enhancement (async)."""
    context = StrategyContext(query=query)

    # Auto-detect if requested
    if strategy_name == 'auto':
        suggestions = await detector.detect(context, top_k=1)
        if suggestions:
            strategy_name = suggestions[0][0]
            confidence_score = suggestions[0][1]
        else:
            return {
                'error': 'No strategies matched'
            }
    else:
        confidence_score = None

    # Get strategy
    strategy = registry.get(strategy_name)
    if not strategy:
        return {
            'error': f"Strategy '{strategy_name}' not found"
        }

    # Enhance
    result = await strategy.enhance(context)

    # Return result
    return {
        'strategy': strategy_name,
        'confidence': result.confidence,
        'improvement': result.estimated_improvement,
        'content': result.enhanced_query,
        'metadata': result.metadata,
        'auto_detected': confidence_score is not None,
        'detection_confidence': confidence_score
    }


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Promptly Strategy Web Server")
    print("  http://localhost:5000")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
