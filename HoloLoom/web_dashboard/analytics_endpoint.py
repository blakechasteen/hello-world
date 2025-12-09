# Analytics Endpoint Addition to workflow_executor.py
# This file contains the new /api/workflow/analytics endpoint
# Add this code before the "if __name__ == "__main__":" line in workflow_executor.py

"""
@app.get("/api/workflow/analytics")
async def get_analytics():
    \"\"\"
    Get workflow execution analytics and performance metrics.

    Returns aggregated metrics including:
    - Total workflow executions
    - Average latency
    - Success rate and cache hit rate
    - Per-node performance statistics
    - Confidence trajectory over time
    - Detected anomalies
    \"\"\"
    try:
        from datetime import datetime, timedelta
        import random

        now = datetime.now()

        # Generate realistic mock analytics data
        workflows = [
            {
                "id": f"wf-{i:03d}",
                "name": [
                    "Multi-Query Research Pipeline",
                    "Lead Scoring Workflow",
                    "Content Analysis Pipeline",
                    "Safety Verification Check",
                    "BDR Email Sequence Generator",
                    "Knowledge Graph Expansion",
                    "Sentiment Analysis Pipeline",
                    "Document Summarization"
                ][i % 8],
                "status": ["completed", "completed", "failed", "running"][i % 4],
                "latency": random.randint(400, 2500) if i % 4 != 3 else None,
                "confidence": random.uniform(0.45, 0.95),
                "nodeCount": random.randint(3, 8),
                "edgeCount": random.randint(2, 7),
                "timestamp": (now - timedelta(hours=random.randint(0, 72))).timestamp() * 1000
            }
            for i in range(12)
        ]

        node_performance = [
            {
                "name": name,
                "avgLatency": latency,
                "calls": calls,
                "bottleneck": latency > 400,
                "variance": random.uniform(0.05, 0.25)
            }
            for name, latency, calls in [
                ("HoloLoom Query", 450, 28),
                ("Memory Search", 120, 45),
                ("Synthesizer", 280, 22),
                ("Safety Guardrails", 85, 35),
                ("Response Generator", 320, 28),
                ("Confidence Scorer", 95, 38),
                ("Context Packer", 210, 15),
            ]
        ]

        confidence_history = [
            random.uniform(0.7, 0.95) if random.random() > 0.15 else random.uniform(0.4, 0.7)
            for _ in range(12)
        ]

        anomalies = [
            {
                "type": "sudden_drop",
                "timestamp": (now - timedelta(days=2)).timestamp() * 1000,
                "query": "What are the tradeoffs?",
                "before": 0.91,
                "after": 0.72,
                "severity": "high"
            },
            {
                "type": "prolonged_low",
                "timestamp": (now - timedelta(days=1)).timestamp() * 1000,
                "description": "Confidence below 0.75 for 3 consecutive queries",
                "count": 3,
                "severity": "medium"
            },
            {
                "type": "cache_miss_cluster",
                "timestamp": (now - timedelta(hours=12)).timestamp() * 1000,
                "description": "4 cache misses in 2-minute window",
                "count": 4,
                "severity": "medium"
            }
        ]

        # Calculate aggregate metrics
        completed = sum(1 for w in workflows if w["status"] == "completed")
        with_latency = [w for w in workflows if w["latency"] is not None]

        avg_latency = int(sum(w["latency"] for w in with_latency) / len(with_latency)) if with_latency else 0
        success_rate = int((completed / len(workflows) * 100)) if workflows else 0
        cache_hit_rate = random.randint(60, 85)

        total_nodes = sum(w["nodeCount"] for w in workflows)
        total_edges = sum(w["edgeCount"] for w in workflows)

        return {
            "workflows": workflows,
            "nodePerformance": node_performance,
            "confidenceHistory": confidence_history,
            "anomalies": anomalies,
            "metrics": {
                "totalWorkflows": len(workflows),
                "avgLatency": avg_latency,
                "successRate": success_rate,
                "cacheHitRate": cache_hit_rate,
                "totalNodes": total_nodes,
                "totalEdges": total_edges,
                "timestamp": now.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Analytics query failed: {e}")
        from datetime import datetime
        return {
            "workflows": [],
            "nodePerformance": [],
            "confidenceHistory": [],
            "anomalies": [],
            "metrics": {
                "totalWorkflows": 0,
                "avgLatency": 0,
                "successRate": 0,
                "cacheHitRate": 0,
                "totalNodes": 0,
                "totalEdges": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        }
"""
