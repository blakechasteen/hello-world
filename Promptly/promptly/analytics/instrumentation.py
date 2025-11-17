"""
Instrumentation - Wrapper to instrument Promptly with analytics

Provides a decorator-based approach to add analytics to existing Promptly code
without modifying the core implementation.
"""

import functools
from typing import Callable, Any, Optional
from datetime import datetime
from pathlib import Path

from .performance import PerformanceMonitor
from .usage import UsageAnalytics
from .quality import QualityMetrics
from .integrations import AnalyticsHub


class PromptlyAnalytics:
    """Analytics-enabled wrapper for Promptly"""

    def __init__(self, promptly_instance, config: dict = None):
        """
        Initialize analytics wrapper

        Args:
            promptly_instance: Instance of Promptly to wrap
            config: Analytics configuration
        """
        self.promptly = promptly_instance
        self.config = config or self._default_config()

        # Initialize analytics components
        analytics_dir = self.promptly.promptly_dir / 'analytics'
        analytics_dir.mkdir(exist_ok=True)

        self.performance = PerformanceMonitor(
            db_path=str(analytics_dir / 'performance.db'),
            retention_days=self.config.get('retention_days', 30),
            enable_sampling=self.config.get('enable_resource_sampling', True)
        )

        self.usage = UsageAnalytics(
            db_path=str(analytics_dir / 'usage.db'),
            retention_days=self.config.get('retention_days', 90)
        )

        self.quality = QualityMetrics(
            db_path=str(analytics_dir / 'quality.db'),
            retention_days=self.config.get('retention_days', 180)
        )

        # Initialize integrations hub
        self.hub = AnalyticsHub(self.config.get('integrations', {}))

    def _default_config(self) -> dict:
        """Get default analytics configuration"""
        return {
            'retention_days': 30,
            'enable_resource_sampling': True,
            'integrations': {
                'logging': {'enabled': True, 'path': './logs/analytics.jsonl'},
                'prometheus': {'enabled': False, 'port': 9090},
                'opentelemetry': {'enabled': False},
                'grafana': {'enabled': False, 'output_dir': './grafana'}
            }
        }

    # ========================================================================
    # Instrumented Core Methods
    # ========================================================================

    def init(self) -> str:
        """Initialize repository with analytics"""
        with self.performance.time_operation('init'):
            result = self.promptly.init()

        self.hub.record_operation('init', self.performance._operation_buffer[-1].duration_ms)
        return result

    def add(self, name: str, content: str, metadata: dict = None, user: str = None) -> str:
        """Add prompt with analytics tracking"""
        with self.performance.time_operation('add', metadata={'prompt': name}):
            result = self.promptly.add(name, content, metadata)

        # Record usage
        branch = self.promptly._get_current_branch()
        self.usage.tracker.track_prompt_access(name, branch, 'add', user, metadata)

        # Record to integrations
        self.hub.record_operation('add', self.performance._operation_buffer[-1].duration_ms)
        self.hub.record_usage(name, 'add', user, {'branch': branch})

        return result

    def get(self, name: str, version: int = None, commit_hash: str = None, user: str = None) -> Optional[dict]:
        """Get prompt with analytics tracking"""
        with self.performance.time_operation('get', metadata={'prompt': name, 'version': version}):
            result = self.promptly.get(name, version, commit_hash)

        if result:
            # Record usage
            self.usage.tracker.track_prompt_access(
                name, result['branch'], 'get', user,
                {'version': result['version'], 'commit_hash': result['commit_hash']}
            )

            # Record to integrations
            self.hub.record_usage(name, 'access', user, {'branch': result['branch']})

        duration = self.performance._operation_buffer[-1].duration_ms
        self.hub.record_operation('get', duration)

        return result

    def list_prompts(self, branch: str = None, user: str = None) -> list:
        """List prompts with analytics tracking"""
        with self.performance.time_operation('list', metadata={'branch': branch}):
            result = self.promptly.list_prompts(branch)

        # Record usage
        current_branch = branch or self.promptly._get_current_branch()
        self.usage.tracker.track_branch_operation(current_branch, 'list', user)

        duration = self.performance._operation_buffer[-1].duration_ms
        self.hub.record_operation('list', duration)

        return result

    def branch(self, branch_name: str, from_branch: str = None, user: str = None) -> str:
        """Create branch with analytics tracking"""
        with self.performance.time_operation('branch', metadata={'branch': branch_name}):
            result = self.promptly.branch(branch_name, from_branch)

        # Record usage
        self.usage.tracker.track_branch_operation(branch_name, 'create', user, {'from': from_branch})

        duration = self.performance._operation_buffer[-1].duration_ms
        self.hub.record_operation('branch', duration)

        return result

    def checkout(self, branch_name: str, user: str = None) -> str:
        """Checkout branch with analytics tracking"""
        with self.performance.time_operation('checkout', metadata={'branch': branch_name}):
            result = self.promptly.checkout(branch_name)

        # Record usage
        self.usage.tracker.track_branch_operation(branch_name, 'checkout', user)

        duration = self.performance._operation_buffer[-1].duration_ms
        self.hub.record_operation('checkout', duration)

        return result

    def eval_prompt(self, name: str, test_cases: list, model_func: Callable = None,
                   user: str = None) -> list:
        """Evaluate prompt with analytics tracking"""
        with self.performance.time_operation('eval', metadata={'prompt': name, 'tests': len(test_cases)}):
            results = self.promptly.eval_prompt(name, test_cases, model_func)

        # Calculate average score
        scores = [r['score'] for r in results if r['score'] is not None]
        avg_score = sum(scores) / len(scores) if scores else None

        # Record usage
        branch = self.promptly._get_current_branch()
        self.usage.tracker.track_evaluation(name, branch, len(test_cases), avg_score, user)

        # Record quality metrics
        if scores:
            prompt = self.promptly.get(name)
            if prompt:
                for score in scores:
                    self.quality.tracker.record_evaluation(
                        name, prompt['version'], branch,
                        score, 'default', f"eval_{datetime.now().timestamp()}"
                    )

                # Record to integrations
                self.hub.record_quality_score(
                    name, prompt['version'], avg_score, 'default',
                    {'test_count': len(test_cases)}
                )

        duration = self.performance._operation_buffer[-1].duration_ms
        self.hub.record_operation('eval', duration)

        return results

    def create_chain(self, name: str, steps: list, description: str = None, user: str = None) -> str:
        """Create chain with analytics tracking"""
        with self.performance.time_operation('chain_create', metadata={'chain': name, 'steps': len(steps)}):
            result = self.promptly.create_chain(name, steps, description)

        duration = self.performance._operation_buffer[-1].duration_ms
        self.hub.record_operation('chain_create', duration)

        return result

    def execute_chain(self, name: str, initial_input: dict, model_func: Callable = None,
                     user: str = None) -> list:
        """Execute chain with analytics tracking"""
        with self.performance.time_operation('chain_execute', metadata={'chain': name}):
            try:
                results = self.promptly.execute_chain(name, initial_input, model_func)
                success = True
            except Exception as e:
                success = False
                raise

        # Record usage
        step_count = len(results) if success else 0
        self.usage.tracker.track_chain_execution(name, step_count, success, user)

        duration = self.performance._operation_buffer[-1].duration_ms
        status = 'success' if success else 'failure'
        self.hub.record_operation('chain_execute', duration, status)

        if self.hub.prometheus:
            self.hub.prometheus.record_chain_execution(name, status)

        return results

    # ========================================================================
    # Analytics Access Methods
    # ========================================================================

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return self.performance.get_operation_stats()

    def get_usage_stats(self, days: int = 7) -> dict:
        """Get usage statistics"""
        return {
            'most_used': self.usage.get_most_used_prompts(days=days),
            'branch_activity': self.usage.get_branch_activity(days=days),
            'evaluations': self.usage.get_evaluation_stats(days=days),
            'chains': self.usage.get_chain_stats(days=days)
        }

    def get_quality_stats(self, days: int = 30) -> dict:
        """Get quality statistics"""
        return {
            'top_performing': self.quality.get_top_performing_prompts(days=days),
            'declining': self.quality.get_declining_prompts(days=days),
            'by_evaluator': self.quality.get_quality_by_evaluator(days=days),
            'alerts': self.quality.get_active_alerts()
        }

    def cleanup_old_data(self):
        """Clean up old analytics data"""
        deleted_perf = self.performance.cleanup_old_metrics()
        deleted_usage = self.usage.cleanup_old_events()
        deleted_quality = self.quality.cleanup_old_measurements()

        return {
            'performance': deleted_perf,
            'usage': deleted_usage,
            'quality': deleted_quality
        }


def enable_analytics(promptly_instance, config: dict = None) -> PromptlyAnalytics:
    """
    Enable analytics for a Promptly instance

    Args:
        promptly_instance: Instance of Promptly
        config: Analytics configuration

    Returns:
        Analytics-enabled wrapper

    Example:
        >>> from promptly import Promptly
        >>> from promptly.analytics import enable_analytics
        >>>
        >>> promptly = Promptly()
        >>> promptly = enable_analytics(promptly)
        >>>
        >>> # Now all operations are tracked
        >>> promptly.add('greeting', 'Hello {name}!')
        >>> stats = promptly.get_performance_stats()
    """
    return PromptlyAnalytics(promptly_instance, config)


# Decorator for custom analytics tracking
def track_operation(operation_name: str, analytics: PromptlyAnalytics):
    """
    Decorator to track custom operations

    Example:
        >>> @track_operation('custom_op', analytics)
        >>> def my_function():
        >>>     # do work
        >>>     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with analytics.performance.time_operation(operation_name):
                result = func(*args, **kwargs)

            duration = analytics.performance._operation_buffer[-1].duration_ms
            analytics.hub.record_operation(operation_name, duration)

            return result

        return wrapper
    return decorator
