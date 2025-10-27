#!/usr/bin/env python3
"""
Health Check System
===================
Check health of all Promptly systems and backends.

Features:
- Database connectivity check
- Backend availability (Neo4j, Qdrant, Redis)
- System resource check
- Service status monitoring
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import time


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component"""
    name: str
    status: HealthStatus
    response_time_ms: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.details is None:
            self.details = {}


class HealthChecker:
    """
    Check health of all system components.

    Provides unified health monitoring.
    """

    def __init__(self):
        """Initialize health checker."""
        self.checks = []

    def check_all(self) -> Dict[str, Any]:
        """
        Check all system components.

        Returns:
            Dict with overall health and component details
        """
        results = {
            'overall_status': HealthStatus.HEALTHY.value,
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'summary': {}
        }

        # Check database
        db_health = self.check_database()
        results['components']['database'] = asdict(db_health)

        # Check HoloLoom
        hololoom_health = self.check_hololoom()
        results['components']['hololoom'] = asdict(hololoom_health)

        # Check Neo4j
        neo4j_health = self.check_neo4j()
        results['components']['neo4j'] = asdict(neo4j_health)

        # Check Qdrant
        qdrant_health = self.check_qdrant()
        results['components']['qdrant'] = asdict(qdrant_health)

        # Check Redis
        redis_health = self.check_redis()
        results['components']['redis'] = asdict(redis_health)

        # Determine overall status
        statuses = [
            db_health.status,
            hololoom_health.status,
            neo4j_health.status,
            qdrant_health.status,
            redis_health.status
        ]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            results['overall_status'] = HealthStatus.UNHEALTHY.value
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            results['overall_status'] = HealthStatus.DEGRADED.value

        # Generate summary
        results['summary'] = {
            'healthy': sum(1 for s in statuses if s == HealthStatus.HEALTHY),
            'degraded': sum(1 for s in statuses if s == HealthStatus.DEGRADED),
            'unhealthy': sum(1 for s in statuses if s == HealthStatus.UNHEALTHY),
            'total': len(statuses)
        }

        return results

    def check_database(self) -> ComponentHealth:
        """Check Promptly SQLite database."""
        try:
            import sqlite3
            from pathlib import Path

            db_path = Path.home() / ".promptly" / "analytics.db"

            start = time.time()
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM executions")
            count = cursor.fetchone()[0]
            conn.close()
            response_time = (time.time() - start) * 1000

            return ComponentHealth(
                name="Database",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                message=f"Connected ({count} executions)",
                details={'execution_count': count}
            )

        except Exception as e:
            return ComponentHealth(
                name="Database",
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}"
            )

    def check_hololoom(self) -> ComponentHealth:
        """Check HoloLoom memory system."""
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))

            from HoloLoom.memory.unified import UnifiedMemory

            start = time.time()
            mem = UnifiedMemory(user_id="health_check")

            # Try to store and retrieve
            test_id = mem.store("health check test", context={'test': True})
            results = mem.recall("health check", limit=1)

            response_time = (time.time() - start) * 1000

            return ComponentHealth(
                name="HoloLoom",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                message="Connected and functional",
                details={
                    'backend_available': mem._backend_available if hasattr(mem, '_backend_available') else False
                }
            )

        except ImportError:
            return ComponentHealth(
                name="HoloLoom",
                status=HealthStatus.DEGRADED,
                message="HoloLoom not available (not critical)"
            )
        except Exception as e:
            return ComponentHealth(
                name="HoloLoom",
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}"
            )

    def check_neo4j(self) -> ComponentHealth:
        """Check Neo4j connection."""
        try:
            from neo4j import GraphDatabase

            start = time.time()
            driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "promptly123")
            )

            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()

            driver.close()
            response_time = (time.time() - start) * 1000

            return ComponentHealth(
                name="Neo4j",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                message="Connected"
            )

        except ImportError:
            return ComponentHealth(
                name="Neo4j",
                status=HealthStatus.DEGRADED,
                message="Neo4j driver not installed (optional)"
            )
        except Exception as e:
            return ComponentHealth(
                name="Neo4j",
                status=HealthStatus.DEGRADED,
                message=f"Not available: {str(e)[:50]} (optional backend)"
            )

    def check_qdrant(self) -> ComponentHealth:
        """Check Qdrant connection."""
        try:
            from qdrant_client import QdrantClient

            start = time.time()
            client = QdrantClient(host="localhost", port=6333)
            collections = client.get_collections()
            response_time = (time.time() - start) * 1000

            return ComponentHealth(
                name="Qdrant",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                message=f"Connected ({len(collections.collections)} collections)"
            )

        except ImportError:
            return ComponentHealth(
                name="Qdrant",
                status=HealthStatus.DEGRADED,
                message="Qdrant client not installed (optional)"
            )
        except Exception as e:
            return ComponentHealth(
                name="Qdrant",
                status=HealthStatus.DEGRADED,
                message=f"Not available: {str(e)[:50]} (optional backend)"
            )

    def check_redis(self) -> ComponentHealth:
        """Check Redis connection."""
        try:
            import redis

            start = time.time()
            client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            client.ping()
            info = client.info('memory')
            response_time = (time.time() - start) * 1000

            return ComponentHealth(
                name="Redis",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                message="Connected",
                details={'memory_used': info.get('used_memory_human', 'unknown')}
            )

        except ImportError:
            return ComponentHealth(
                name="Redis",
                status=HealthStatus.DEGRADED,
                message="Redis client not installed (optional)"
            )
        except Exception as e:
            return ComponentHealth(
                name="Redis",
                status=HealthStatus.DEGRADED,
                message=f"Not available: {str(e)[:50]} (optional backend)"
            )


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_health_check() -> Dict[str, str]:
    """
    Quick health check (just statuses).

    Returns:
        Dict mapping component to status
    """
    checker = HealthChecker()
    full_check = checker.check_all()

    return {
        component: data['status']
        for component, data in full_check['components'].items()
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Promptly Health Check System\n")

    checker = HealthChecker()
    health = checker.check_all()

    print(f"=== Overall Status: {health['overall_status'].upper()} ===\n")
    print(f"Timestamp: {health['timestamp']}\n")

    print("Component Status:")
    print("-" * 60)

    for name, data in health['components'].items():
        status = data['status']
        message = data['message']
        response_time = data.get('response_time_ms')

        # Status symbol
        if status == 'healthy':
            symbol = "[+]"
        elif status == 'degraded':
            symbol = "[~]"
        elif status == 'unhealthy':
            symbol = "[!]"
        else:
            symbol = "[?]"

        print(f"{symbol} {name:15} {status:12} {message}")

        if response_time:
            print(f"   Response time: {response_time:.2f}ms")

        if data.get('details'):
            for key, value in data['details'].items():
                print(f"   {key}: {value}")

        print()

    # Summary
    summary = health['summary']
    print(f"\nSummary: {summary['healthy']}/{summary['total']} healthy, "
          f"{summary['degraded']} degraded, {summary['unhealthy']} unhealthy")
