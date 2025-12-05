"""
Dashboard API - Phase 5 Week 1 Day 2

Flask REST API + WebSocket server for real-time metrics dashboard.

Features:
- REST endpoints for metrics, trends, top strategies
- WebSocket streaming for real-time updates
- CORS enabled for frontend
- Clean error handling

Endpoints:
    GET  /api/health              - Health check
    GET  /api/stats               - Aggregated statistics
    GET  /api/trends              - Time-series trends
    GET  /api/top_strategies      - Top performing strategies
    GET  /api/cache_stats         - Cache performance
    WebSocket /socket.io          - Real-time metric updates

Usage:
    python dashboard_api.py
    # Server starts on http://localhost:5001
"""

import asyncio
import time
import logging
from typing import Dict, Any, List
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from metrics_collector import MetricsCollector, get_metrics_collector
from time_series_db import TimeSeriesDB, Query as DBQuery
from aggregator import MetricsAggregator, AggregationPeriod
from alert_engine import AlertEngine, AlertRule, WebhookChannel, SlackChannel, EmailChannel
from ab_testing import ABTestManager, ABTest, Variant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'promptly-dashboard-secret'
CORS(app)  # Enable CORS for frontend

# Create SocketIO server
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global instances
db: TimeSeriesDB = None
aggregator: MetricsAggregator = None
collector: MetricsCollector = None
alert_engine: AlertEngine = None
ab_test_manager: ABTestManager = None


def initialize():
    """Initialize database and services."""
    global db, aggregator, collector, alert_engine, ab_test_manager

    # Use test database if it exists, otherwise create new one
    db_path = Path('test_metrics.db')
    if not db_path.exists():
        db_path = Path('promptly_metrics.db')

    db = TimeSeriesDB(str(db_path))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(db.initialize())

    aggregator = MetricsAggregator(db)
    collector = get_metrics_collector()

    # Initialize alert engine (no channels by default - configured via API)
    alert_engine = AlertEngine(db_path=str(db_path), channels=[])
    loop.run_until_complete(alert_engine.start())

    # Initialize A/B test manager
    ab_test_manager = ABTestManager(db_path=str(db_path))

    logger.info(f"Dashboard API initialized with database: {db_path}")
    logger.info(f"Alert engine started")
    logger.info(f"A/B test manager initialized")


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get aggregated statistics.

    Query parameters:
        period: 1h, 24h, 7d, 30d (default: 24h)

    Returns:
        Aggregated statistics for the period
    """
    try:
        # Parse period
        period_str = request.args.get('period', '24h')
        period_map = {
            '1h': AggregationPeriod.HOUR,
            '24h': AggregationPeriod.DAY,
            '7d': AggregationPeriod.WEEK,
            '30d': AggregationPeriod.MONTH
        }
        period = period_map.get(period_str, AggregationPeriod.DAY)

        # Get stats
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stats = loop.run_until_complete(aggregator.get_stats(period))

        # Convert to JSON-serializable format
        return jsonify({
            'period': period.value,
            'start_time': stats.start_time,
            'end_time': stats.end_time,
            'total_queries': stats.total_queries,
            'avg_latency_ms': stats.avg_latency_ms,
            'p50_latency_ms': stats.p50_latency_ms,
            'p95_latency_ms': stats.p95_latency_ms,
            'p99_latency_ms': stats.p99_latency_ms,
            'avg_confidence': stats.avg_confidence,
            'median_confidence': stats.median_confidence,
            'cache_hit_rate': stats.cache_hit_rate,
            'total_cache_hits': stats.total_cache_hits,
            'total_cache_misses': stats.total_cache_misses,
            'strategy_distribution': stats.strategy_distribution,
            'strategy_performance': stats.strategy_performance
        })

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trends', methods=['GET'])
def get_trends():
    """
    Get time-series trends.

    Query parameters:
        metric: latency_ms, confidence (default: latency_ms)
        period: 1h, 24h, 7d, 30d (default: 24h)
        interval: Bucket size in minutes (default: 60)

    Returns:
        List of {timestamp, value} points
    """
    try:
        # Parse parameters
        metric = request.args.get('metric', 'latency_ms')
        period_str = request.args.get('period', '24h')
        interval_minutes = int(request.args.get('interval', '60'))

        period_map = {
            '1h': AggregationPeriod.HOUR,
            '24h': AggregationPeriod.DAY,
            '7d': AggregationPeriod.WEEK,
            '30d': AggregationPeriod.MONTH
        }
        period = period_map.get(period_str, AggregationPeriod.DAY)

        # Get trend
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        trend = loop.run_until_complete(
            aggregator.get_trend(
                metric=metric,
                period=period,
                interval_minutes=interval_minutes
            )
        )

        return jsonify({
            'metric': metric,
            'period': period.value,
            'interval_minutes': interval_minutes,
            'data': trend
        })

    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/top_strategies', methods=['GET'])
def get_top_strategies():
    """
    Get top performing strategies.

    Query parameters:
        metric: avg_confidence, avg_latency_ms, success_rate (default: avg_confidence)
        period: 1h, 24h, 7d, 30d (default: 24h)
        limit: Number of results (default: 10)

    Returns:
        List of top strategies with performance metrics
    """
    try:
        # Parse parameters
        metric = request.args.get('metric', 'avg_confidence')
        period_str = request.args.get('period', '24h')
        limit = int(request.args.get('limit', '10'))

        period_map = {
            '1h': AggregationPeriod.HOUR,
            '24h': AggregationPeriod.DAY,
            '7d': AggregationPeriod.WEEK,
            '30d': AggregationPeriod.MONTH
        }
        period = period_map.get(period_str, AggregationPeriod.DAY)

        # Get top strategies
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        top = loop.run_until_complete(
            aggregator.get_top_strategies(
                period=period,
                metric=metric,
                limit=limit
            )
        )

        return jsonify({
            'metric': metric,
            'period': period.value,
            'limit': limit,
            'strategies': top
        })

    except Exception as e:
        logger.error(f"Error getting top strategies: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache_stats', methods=['GET'])
def get_cache_stats():
    """
    Get cache performance statistics.

    Query parameters:
        period: 1h, 24h, 7d, 30d (default: 24h)

    Returns:
        Cache statistics including hit rate, speedup, etc.
    """
    try:
        # Parse period
        period_str = request.args.get('period', '24h')
        duration_map = {
            '1h': 3600,
            '24h': 86400,
            '7d': 604800,
            '30d': 2592000
        }
        duration_seconds = duration_map.get(period_str, 86400)

        # Get cache stats
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cache_stats = loop.run_until_complete(
            db.get_cache_stats(duration_seconds=duration_seconds)
        )

        return jsonify({
            'period': period_str,
            'hit_rate': cache_stats['hit_rate'],
            'total_hits': cache_stats['total_hits'],
            'total_misses': cache_stats['total_misses'],
            'total_queries': cache_stats['total_queries'],
            'avg_cached_latency_ms': cache_stats['avg_cached_latency_ms'],
            'avg_uncached_latency_ms': cache_stats['avg_uncached_latency_ms'],
            'speedup': cache_stats['speedup']
        })

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Custom Date Range API (Week 2 Days 3-5)
# ============================================================================

@app.route('/api/stats/custom', methods=['GET'])
def get_stats_custom_range():
    """
    Get statistics for a custom date range.

    Query parameters:
        start: Start timestamp (Unix seconds)
        end: End timestamp (Unix seconds)

    Returns:
        Aggregated statistics for the custom range
    """
    try:
        # Parse timestamps
        start_time = float(request.args.get('start'))
        end_time = float(request.args.get('end'))

        if start_time >= end_time:
            return jsonify({'error': 'Start time must be before end time'}), 400

        # Query database directly
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Get events in range
        query = DBQuery(
            start_time=start_time,
            end_time=end_time,
            tags={}
        )
        events = loop.run_until_complete(db.query(query))

        # Aggregate manually
        if not events:
            return jsonify({
                'start_time': start_time,
                'end_time': end_time,
                'total_queries': 0,
                'avg_latency_ms': 0,
                'avg_confidence': 0,
                'cache_hit_rate': 0
            })

        latencies = []
        confidences = []
        cache_hits = 0
        strategies = {}

        for event in events:
            if event.values.get('latency_ms'):
                latencies.append(event.values['latency_ms'])
            if event.values.get('confidence'):
                confidences.append(event.values['confidence'])
            if event.tags.get('cache_hit') == 'True':
                cache_hits += 1

            strategy = event.tags.get('strategy', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1

        return jsonify({
            'start_time': start_time,
            'end_time': end_time,
            'total_queries': len(events),
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'cache_hit_rate': cache_hits / len(events) if events else 0,
            'strategy_distribution': strategies
        })

    except ValueError as e:
        logger.error(f"Invalid timestamp: {e}")
        return jsonify({'error': 'Invalid timestamp format'}), 400
    except Exception as e:
        logger.error(f"Error getting custom range stats: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Alert Management API (Week 2 Days 3-5)
# ============================================================================

@app.route('/api/alerts/rules', methods=['GET'])
def get_alert_rules():
    """Get all configured alert rules."""
    try:
        rules = [
            {
                'id': rule.id,
                'name': rule.name,
                'metric': rule.metric,
                'condition': rule.condition,
                'threshold': rule.threshold,
                'duration_seconds': rule.duration_seconds,
                'severity': rule.severity,
                'enabled': rule.enabled,
                'channels': rule.channels,
                'cooldown_seconds': rule.cooldown_seconds
            }
            for rule in alert_engine.rules.values()
        ]

        return jsonify({'rules': rules})

    except Exception as e:
        logger.error(f"Error getting alert rules: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/rules', methods=['POST'])
def create_alert_rule():
    """
    Create a new alert rule.

    Request body:
        {
            "id": "high_latency",
            "name": "High Latency Alert",
            "metric": "avg_latency_ms",
            "condition": "greater_than",
            "threshold": 200.0,
            "duration_seconds": 300,
            "severity": "warning",
            "channels": ["webhook"],
            "cooldown_seconds": 300
        }
    """
    try:
        data = request.get_json()

        rule = AlertRule(
            id=data['id'],
            name=data['name'],
            metric=data['metric'],
            condition=data['condition'],
            threshold=float(data['threshold']),
            duration_seconds=data.get('duration_seconds', 60),
            severity=data.get('severity', 'warning'),
            enabled=data.get('enabled', True),
            channels=data.get('channels', ['webhook']),
            cooldown_seconds=data.get('cooldown_seconds', 300)
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(alert_engine.add_rule(rule))

        return jsonify({
            'status': 'success',
            'message': f'Alert rule "{rule.name}" created',
            'rule_id': rule.id
        })

    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        return jsonify({'error': f'Missing required field: {e}'}), 400
    except Exception as e:
        logger.error(f"Error creating alert rule: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/rules/<rule_id>', methods=['DELETE'])
def delete_alert_rule(rule_id: str):
    """Delete an alert rule."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(alert_engine.remove_rule(rule_id))

        return jsonify({
            'status': 'success',
            'message': f'Alert rule {rule_id} deleted'
        })

    except Exception as e:
        logger.error(f"Error deleting alert rule: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/history', methods=['GET'])
def get_alert_history():
    """
    Get alert history.

    Query parameters:
        limit: Number of results (default: 100)
        status: Filter by status (active, acknowledged, resolved)
        rule_id: Filter by rule ID
    """
    try:
        limit = int(request.args.get('limit', '100'))
        status = request.args.get('status')
        rule_id = request.args.get('rule_id')

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        alerts = loop.run_until_complete(
            alert_engine.get_alert_history(
                limit=limit,
                status=status,
                rule_id=rule_id
            )
        )

        return jsonify({
            'alerts': [
                {
                    'id': alert.id,
                    'rule_id': alert.rule_id,
                    'rule_name': alert.rule_name,
                    'severity': alert.severity,
                    'message': alert.message,
                    'metric': alert.metric,
                    'value': alert.value,
                    'threshold': alert.threshold,
                    'triggered_at': alert.triggered_at,
                    'status': alert.status,
                    'acknowledged_at': alert.acknowledged_at,
                    'resolved_at': alert.resolved_at
                }
                for alert in alerts
            ]
        })

    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(alert_engine.acknowledge_alert(alert_id))

        return jsonify({
            'status': 'success',
            'message': f'Alert {alert_id} acknowledged'
        })

    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id: str):
    """Resolve an alert."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(alert_engine.resolve_alert(alert_id))

        return jsonify({
            'status': 'success',
            'message': f'Alert {alert_id} resolved'
        })

    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Query Replay API (Week 2 Days 3-5)
# ============================================================================

@app.route('/api/replay/<query_id>', methods=['POST'])
def replay_query(query_id: str):
    """
    Replay a previous query.

    This endpoint would integrate with the Promptly orchestrator to
    re-execute a previous query and return new results.

    For now, this is a stub that returns the original query data.
    Full implementation requires orchestrator integration.
    """
    try:
        # Query database for the original event
        conn = db._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT tags_json, values_json, timestamp
            FROM events
            WHERE id = ?
        """, (query_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({'error': 'Query not found'}), 404

        tags = row[0]
        values = row[1]
        timestamp = row[2]

        # TODO: Integrate with orchestrator to re-execute query
        # For now, return original data with note
        return jsonify({
            'status': 'success',
            'message': 'Query replay not yet implemented - showing original data',
            'query_id': query_id,
            'original_timestamp': timestamp,
            'original_tags': tags,
            'original_values': values,
            'note': 'Full replay requires orchestrator integration (Week 2 Days 3-5)'
        })

    except Exception as e:
        logger.error(f"Error replaying query: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# A/B Testing API (Week 3-4)
# ============================================================================

@app.route('/api/ab-tests', methods=['GET'])
def list_ab_tests():
    """Get all A/B tests, optionally filtered by status."""
    try:
        status = request.args.get('status')
        limit = int(request.args.get('limit', '100'))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tests = loop.run_until_complete(
            ab_test_manager.list_tests(status=status, limit=limit)
        )

        return jsonify({
            'tests': [
                {
                    'id': test.id,
                    'name': test.name,
                    'description': test.description,
                    'variants': [
                        {
                            'id': v.id,
                            'strategy': v.strategy,
                            'traffic_percent': v.traffic_percent
                        }
                        for v in test.variants
                    ],
                    'metric': test.metric,
                    'status': test.status,
                    'created_at': test.created_at,
                    'started_at': test.started_at,
                    'completed_at': test.completed_at,
                    'winner': test.winner
                }
                for test in tests
            ]
        })

    except Exception as e:
        logger.error(f"Error listing A/B tests: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ab-tests', methods=['POST'])
def create_ab_test():
    """
    Create a new A/B test.

    Request body:
        {
            "id": "optimize_vs_deep",
            "name": "Optimize vs Deep Strategy",
            "description": "Compare optimize and deep strategies",
            "variants": [
                {"id": "control", "strategy": "optimize", "traffic_percent": 50},
                {"id": "treatment", "strategy": "deep", "traffic_percent": 50}
            ],
            "metric": "avg_confidence",
            "min_sample_size": 100,
            "significance_level": 0.05
        }
    """
    try:
        data = request.get_json()

        variants = [
            Variant(
                id=v['id'],
                strategy=v['strategy'],
                traffic_percent=v['traffic_percent'],
                description=v.get('description', '')
            )
            for v in data['variants']
        ]

        test = ABTest(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            variants=variants,
            metric=data.get('metric', 'avg_confidence'),
            min_sample_size=data.get('min_sample_size', 100),
            significance_level=data.get('significance_level', 0.05)
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(ab_test_manager.create_test(test))

        return jsonify({
            'status': 'success',
            'message': f'A/B test "{test.name}" created',
            'test_id': test.id
        })

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        return jsonify({'error': f'Missing required field: {e}'}), 400
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ab-tests/<test_id>', methods=['GET'])
def get_ab_test(test_id: str):
    """Get A/B test details."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        test = loop.run_until_complete(ab_test_manager.get_test(test_id))

        if not test:
            return jsonify({'error': 'Test not found'}), 404

        return jsonify({
            'id': test.id,
            'name': test.name,
            'description': test.description,
            'variants': [
                {
                    'id': v.id,
                    'strategy': v.strategy,
                    'traffic_percent': v.traffic_percent,
                    'description': v.description
                }
                for v in test.variants
            ],
            'metric': test.metric,
            'min_sample_size': test.min_sample_size,
            'significance_level': test.significance_level,
            'status': test.status,
            'created_at': test.created_at,
            'started_at': test.started_at,
            'completed_at': test.completed_at,
            'winner': test.winner
        })

    except Exception as e:
        logger.error(f"Error getting A/B test: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ab-tests/<test_id>/start', methods=['POST'])
def start_ab_test(test_id: str):
    """Start an A/B test."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(ab_test_manager.start_test(test_id))

        if not success:
            return jsonify({'error': 'Could not start test (already running or not found)'}), 400

        return jsonify({
            'status': 'success',
            'message': f'Test {test_id} started'
        })

    except Exception as e:
        logger.error(f"Error starting A/B test: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ab-tests/<test_id>/pause', methods=['POST'])
def pause_ab_test(test_id: str):
    """Pause an A/B test."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(ab_test_manager.pause_test(test_id))

        if not success:
            return jsonify({'error': 'Could not pause test'}), 400

        return jsonify({
            'status': 'success',
            'message': f'Test {test_id} paused'
        })

    except Exception as e:
        logger.error(f"Error pausing A/B test: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ab-tests/<test_id>/resume', methods=['POST'])
def resume_ab_test(test_id: str):
    """Resume a paused A/B test."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(ab_test_manager.resume_test(test_id))

        if not success:
            return jsonify({'error': 'Could not resume test'}), 400

        return jsonify({
            'status': 'success',
            'message': f'Test {test_id} resumed'
        })

    except Exception as e:
        logger.error(f"Error resuming A/B test: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ab-tests/<test_id>/complete', methods=['POST'])
def complete_ab_test(test_id: str):
    """Complete an A/B test and optionally specify winner."""
    try:
        data = request.get_json() or {}
        winner = data.get('winner')

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(ab_test_manager.complete_test(test_id, winner=winner))

        if not success:
            return jsonify({'error': 'Could not complete test'}), 400

        return jsonify({
            'status': 'success',
            'message': f'Test {test_id} completed',
            'winner': winner
        })

    except Exception as e:
        logger.error(f"Error completing A/B test: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ab-tests/<test_id>/results', methods=['GET'])
def get_ab_test_results(test_id: str):
    """Get results and statistical analysis for an A/B test."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(ab_test_manager.get_test_results(test_id))

        return jsonify({
            'test_id': results.test_id,
            'test_name': results.test_name,
            'status': results.status,
            'variants': {
                vid: {
                    'variant_id': vstats.variant_id,
                    'strategy': vstats.strategy,
                    'sample_size': vstats.sample_size,
                    'avg_latency_ms': vstats.avg_latency_ms,
                    'std_latency_ms': vstats.std_latency_ms,
                    'avg_confidence': vstats.avg_confidence,
                    'std_confidence': vstats.std_confidence,
                    'cache_hit_rate': vstats.cache_hit_rate,
                    'success_rate': vstats.success_rate,
                    'total_queries': vstats.total_queries
                }
                for vid, vstats in results.variant_stats.items()
            },
            'p_value': results.p_value,
            'statistically_significant': results.statistically_significant,
            'confidence_level': results.confidence_level,
            'effect_size': results.effect_size,
            'winner': results.winner,
            'winner_improvement': results.winner_improvement,
            'total_samples': results.total_samples,
            'test_duration_seconds': results.test_duration_seconds,
            'recommendation': results.recommendation
        })

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error getting A/B test results: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ab-tests/<test_id>/assign', methods=['POST'])
def assign_variant(test_id: str):
    """
    Assign a variant for a query.

    Request body:
        {
            "query": "What is Thompson Sampling?"
        }

    Returns:
        {
            "test_id": "optimize_vs_deep",
            "variant_id": "control",
            "strategy": "optimize"
        }
    """
    try:
        data = request.get_json()
        query = data['query']

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Get assigned variant
        variant_id = loop.run_until_complete(ab_test_manager.assign_variant(test_id, query))

        if not variant_id:
            return jsonify({'error': 'Test not running or not found'}), 404

        # Get test details to return strategy
        test = loop.run_until_complete(ab_test_manager.get_test(test_id))
        variant = next((v for v in test.variants if v.id == variant_id), None)

        return jsonify({
            'test_id': test_id,
            'variant_id': variant_id,
            'strategy': variant.strategy if variant else 'unknown'
        })

    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        return jsonify({'error': f'Missing required field: {e}'}), 400
    except Exception as e:
        logger.error(f"Error assigning variant: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ab-tests/<test_id>/record', methods=['POST'])
def record_ab_result(test_id: str):
    """
    Record a result for a variant.

    Request body:
        {
            "variant_id": "control",
            "query": "What is Thompson Sampling?",
            "strategy": "optimize",
            "latency_ms": 145.2,
            "confidence": 0.92,
            "cache_hit": true,
            "success": true
        }
    """
    try:
        data = request.get_json()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            ab_test_manager.record_result(
                test_id=test_id,
                variant_id=data['variant_id'],
                query=data['query'],
                strategy=data['strategy'],
                latency_ms=float(data['latency_ms']),
                confidence=float(data['confidence']),
                cache_hit=bool(data.get('cache_hit', False)),
                success=bool(data.get('success', True))
            )
        )

        return jsonify({
            'status': 'success',
            'message': 'Result recorded'
        })

    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        return jsonify({'error': f'Missing required field: {e}'}), 400
    except Exception as e:
        logger.error(f"Error recording result: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ab-tests/<test_id>/promote', methods=['POST'])
def promote_winner(test_id: str):
    """Promote the winning variant to production."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(ab_test_manager.promote_winner(test_id))

        if not success:
            return jsonify({'error': 'No clear winner or test not completed'}), 400

        return jsonify({
            'status': 'success',
            'message': f'Winner promoted for test {test_id}'
        })

    except Exception as e:
        logger.error(f"Error promoting winner: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# WebSocket Events (Real-time Updates)
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('subscribe_metrics')
def handle_subscribe():
    """Handle subscription to real-time metrics."""
    logger.info(f"Client subscribed to metrics: {request.sid}")
    emit('subscription_confirmed', {'status': 'subscribed'})


def broadcast_metrics():
    """
    Broadcast current metrics to all connected clients.

    This function should be called periodically (e.g., every 1-5 seconds)
    to push real-time updates to the dashboard.
    """
    try:
        # Get latest stats
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stats = loop.run_until_complete(
            aggregator.get_stats(AggregationPeriod.HOUR, force_refresh=True)
        )

        # Prepare payload
        payload = {
            'timestamp': time.time(),
            'total_queries': stats.total_queries,
            'avg_latency_ms': stats.avg_latency_ms,
            'avg_confidence': stats.avg_confidence,
            'cache_hit_rate': stats.cache_hit_rate,
            'strategy_distribution': stats.strategy_distribution
        }

        # Broadcast to all connected clients
        socketio.emit('metrics_update', payload, broadcast=True)

        logger.debug(f"Broadcasted metrics update: {payload}")

    except Exception as e:
        logger.error(f"Error broadcasting metrics: {e}")


# ============================================================================
# Background Tasks
# ============================================================================

def start_background_broadcast():
    """Start background task to broadcast metrics every 2 seconds."""
    def broadcast_loop():
        while True:
            socketio.sleep(2)  # Wait 2 seconds
            broadcast_metrics()

    # Start background thread
    socketio.start_background_task(broadcast_loop)
    logger.info("Started background metrics broadcast (every 2s)")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Promptly Performance Dashboard API")
    print("  http://localhost:5001")
    print("="*60 + "\n")

    print("Endpoints:")
    print("  GET  /api/health                  - Health check")
    print("  GET  /api/stats                   - Aggregated statistics")
    print("  GET  /api/stats/custom            - Custom date range statistics")
    print("  GET  /api/trends                  - Time-series trends")
    print("  GET  /api/top_strategies          - Top performing strategies")
    print("  GET  /api/cache_stats             - Cache performance")
    print("  GET  /api/alerts/rules            - Get all alert rules")
    print("  POST /api/alerts/rules            - Create alert rule")
    print("  DEL  /api/alerts/rules/<id>       - Delete alert rule")
    print("  GET  /api/alerts/history          - Get alert history")
    print("  POST /api/alerts/<id>/ack         - Acknowledge alert")
    print("  POST /api/alerts/<id>/resolve     - Resolve alert")
    print("  POST /api/replay/<id>             - Replay query (stub)")
    print("  GET  /api/ab-tests                - List all A/B tests")
    print("  POST /api/ab-tests                - Create A/B test")
    print("  GET  /api/ab-tests/<id>           - Get A/B test details")
    print("  POST /api/ab-tests/<id>/start     - Start A/B test")
    print("  POST /api/ab-tests/<id>/pause     - Pause A/B test")
    print("  POST /api/ab-tests/<id>/resume    - Resume A/B test")
    print("  POST /api/ab-tests/<id>/complete  - Complete A/B test")
    print("  GET  /api/ab-tests/<id>/results   - Get test results + stats")
    print("  POST /api/ab-tests/<id>/assign    - Assign variant to query")
    print("  POST /api/ab-tests/<id>/record    - Record result for variant")
    print("  POST /api/ab-tests/<id>/promote   - Promote winning variant")
    print("  WebSocket /socket.io              - Real-time updates")
    print()

    # Initialize
    initialize()

    # Start background broadcast
    start_background_broadcast()

    # Run server
    socketio.run(
        app,
        host='0.0.0.0',
        port=5001,
        debug=True,
        use_reloader=False  # Disable reloader to prevent double broadcast
    )
