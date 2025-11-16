"""
AutoFix Results Tracker
========================

Comprehensive tracking system for autofix applications to enable future learning.

Features:
- Per-fix tracking (before/after, confidence, outcome)
- Aggregate statistics (success rate, time saved, categories)
- Learning signals for strategy selection
- Export for analysis and model training

Author: mythRL Team
Date: November 16, 2025
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class AutoFixResult:
    """Single autofix application result."""
    # Fix metadata
    fix_id: str
    category: str
    severity: str
    message: str
    confidence: float

    # File metadata
    file_path: str
    line_number: int
    code_snippet: str

    # Fix details
    strategy: str  # ast, template, manual
    original_code: str
    fixed_code: str
    diff: str

    # Outcome
    applied: bool
    validation_passed: bool
    errors: List[str]
    warnings: List[str]

    # Git metadata
    branch: Optional[str] = None
    commit: Optional[str] = None

    # Timing
    timestamp: str = None
    duration_ms: float = 0.0

    # Learning signals
    user_feedback: Optional[str] = None  # accepted, rejected, modified
    actual_confidence: Optional[float] = None  # Measured post-application

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class AutoFixSession:
    """Complete autofix session results."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None

    # Configuration
    max_files: int = 0
    categories: List[str] = None
    confidence_threshold: float = 0.0

    # Results
    fixes: List[AutoFixResult] = None

    # Statistics (computed)
    total_fixes_attempted: int = 0
    total_fixes_applied: int = 0
    total_fixes_failed: int = 0
    total_duration_ms: float = 0.0

    # Learning metrics
    avg_confidence: float = 0.0
    success_rate: float = 0.0
    time_saved_hours: float = 0.0

    def __post_init__(self):
        if self.fixes is None:
            self.fixes = []
        if self.categories is None:
            self.categories = []


class AutoFixTracker:
    """
    Comprehensive tracking system for autofix applications.

    Tracks:
    - Individual fix results
    - Session statistics
    - Learning signals for strategy improvement
    - Export for analysis

    Usage:
        tracker = AutoFixTracker()

        # Start session
        session_id = tracker.start_session(max_files=50, confidence_threshold=0.85)

        # Track fix
        result = AutoFixResult(...)
        tracker.track_fix(result)

        # End session
        tracker.end_session()

        # Export
        tracker.export_json("autofix_results.json")
        tracker.export_learning_data("learning_data.json")
    """

    def __init__(self, output_dir: str = "./autofix_tracking"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_session: Optional[AutoFixSession] = None
        self.sessions: List[AutoFixSession] = []

        # Statistics aggregation
        self.stats_by_category = defaultdict(lambda: {
            'attempted': 0,
            'applied': 0,
            'failed': 0,
            'avg_confidence': 0.0,
            'total_confidence': 0.0
        })

        self.stats_by_strategy = defaultdict(lambda: {
            'attempted': 0,
            'applied': 0,
            'success_rate': 0.0
        })

    def start_session(
        self,
        max_files: int = 50,
        categories: List[str] = None,
        confidence_threshold: float = 0.85
    ) -> str:
        """Start new tracking session."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_session = AutoFixSession(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            max_files=max_files,
            categories=categories or [],
            confidence_threshold=confidence_threshold
        )

        print(f"✓ Started tracking session: {session_id}")
        print(f"  Max files: {max_files}")
        print(f"  Confidence threshold: {confidence_threshold:.1%}")
        print()

        return session_id

    def track_fix(self, result: AutoFixResult):
        """Track individual fix result."""
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")

        # Add to session
        self.current_session.fixes.append(result)

        # Update session stats
        self.current_session.total_fixes_attempted += 1
        if result.applied:
            self.current_session.total_fixes_applied += 1
        else:
            self.current_session.total_fixes_failed += 1
        self.current_session.total_duration_ms += result.duration_ms

        # Update category stats
        cat_stats = self.stats_by_category[result.category]
        cat_stats['attempted'] += 1
        if result.applied:
            cat_stats['applied'] += 1
        else:
            cat_stats['failed'] += 1
        cat_stats['total_confidence'] += result.confidence
        cat_stats['avg_confidence'] = (
            cat_stats['total_confidence'] / cat_stats['attempted']
        )

        # Update strategy stats
        strat_stats = self.stats_by_strategy[result.strategy]
        strat_stats['attempted'] += 1
        if result.applied:
            strat_stats['applied'] += 1
        strat_stats['success_rate'] = (
            strat_stats['applied'] / strat_stats['attempted']
        )

    def end_session(self):
        """End current session and compute final statistics."""
        if not self.current_session:
            raise ValueError("No active session.")

        # Mark end time
        self.current_session.end_time = datetime.now().isoformat()

        # Compute learning metrics
        if self.current_session.total_fixes_attempted > 0:
            # Average confidence
            total_conf = sum(f.confidence for f in self.current_session.fixes)
            self.current_session.avg_confidence = (
                total_conf / self.current_session.total_fixes_attempted
            )

            # Success rate
            self.current_session.success_rate = (
                self.current_session.total_fixes_applied /
                self.current_session.total_fixes_attempted
            )

            # Time saved (estimate: 2 min per manual fix)
            self.current_session.time_saved_hours = (
                self.current_session.total_fixes_applied * 2 / 60
            )

        # Add to sessions list
        self.sessions.append(self.current_session)

        # Print summary
        self._print_session_summary()

        # Save to disk
        self._save_session()

        # Clear current session
        self.current_session = None

    def _print_session_summary(self):
        """Print session summary."""
        session = self.current_session

        print("=" * 70)
        print("AUTOFIX SESSION SUMMARY")
        print("=" * 70)
        print()

        print(f"Session ID: {session.session_id}")
        print(f"Duration: {(datetime.fromisoformat(session.end_time) - datetime.fromisoformat(session.start_time)).total_seconds():.1f}s")
        print()

        print("Results:")
        print(f"  Attempted: {session.total_fixes_attempted}")
        print(f"  Applied: {session.total_fixes_applied}")
        print(f"  Failed: {session.total_fixes_failed}")
        print(f"  Success Rate: {session.success_rate:.1%}")
        print()

        print("Performance:")
        print(f"  Avg Confidence: {session.avg_confidence:.1%}")
        print(f"  Total Duration: {session.total_duration_ms:.0f}ms")
        if session.total_fixes_attempted > 0:
            print(f"  Avg Duration/Fix: {session.total_duration_ms / session.total_fixes_attempted:.0f}ms")
        else:
            print(f"  Avg Duration/Fix: N/A (no fixes attempted)")
        print(f"  Time Saved: {session.time_saved_hours:.1f} hours")
        print()

        print("By Category:")
        for category, stats in sorted(self.stats_by_category.items()):
            success_rate = stats['applied'] / stats['attempted'] if stats['attempted'] > 0 else 0
            print(f"  {category}:")
            print(f"    Attempted: {stats['attempted']}")
            print(f"    Applied: {stats['applied']} ({success_rate:.1%})")
            print(f"    Avg Confidence: {stats['avg_confidence']:.1%}")
        print()

        print("By Strategy:")
        for strategy, stats in sorted(self.stats_by_strategy.items()):
            print(f"  {strategy}:")
            print(f"    Attempted: {stats['attempted']}")
            print(f"    Applied: {stats['applied']} ({stats['success_rate']:.1%})")
        print()

    def _save_session(self):
        """Save session to JSON file."""
        filename = f"{self.current_session.session_id}.json"
        filepath = self.output_dir / filename

        # Convert to dict
        session_dict = asdict(self.current_session)

        with open(filepath, 'w') as f:
            json.dump(session_dict, f, indent=2)

        print(f"✓ Session saved to: {filepath}")
        print()

    def export_json(self, filename: str):
        """Export all sessions to JSON."""
        filepath = Path(filename)

        data = {
            'sessions': [asdict(s) for s in self.sessions],
            'stats_by_category': dict(self.stats_by_category),
            'stats_by_strategy': dict(self.stats_by_strategy)
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Exported to: {filepath}")

    def export_learning_data(self, filename: str):
        """
        Export data for machine learning.

        Format: CSV with columns:
        - category, severity, confidence, strategy, applied, validation_passed

        Useful for:
        - Training confidence calibration models
        - Strategy selection optimization
        - Success prediction
        """
        filepath = Path(filename)

        # Collect all fixes from all sessions
        all_fixes = []
        for session in self.sessions:
            all_fixes.extend(session.fixes)

        # Convert to learning format
        learning_data = []
        for fix in all_fixes:
            learning_data.append({
                'category': fix.category,
                'severity': fix.severity,
                'confidence': fix.confidence,
                'strategy': fix.strategy,
                'applied': 1 if fix.applied else 0,
                'validation_passed': 1 if fix.validation_passed else 0,
                'duration_ms': fix.duration_ms,
                'user_feedback': fix.user_feedback or 'none',
                'actual_confidence': fix.actual_confidence or fix.confidence
            })

        with open(filepath, 'w') as f:
            json.dump(learning_data, f, indent=2)

        print(f"✓ Learning data exported: {filepath}")
        print(f"  Total samples: {len(learning_data)}")
        print(f"  Features: 9 (category, severity, confidence, strategy, ...)")

    def get_recommendations(self) -> Dict[str, Any]:
        """
        Generate recommendations based on tracked data.

        Returns:
            {
                'adjust_confidence_threshold': bool,
                'recommended_threshold': float,
                'categories_to_disable': List[str],
                'best_strategies': Dict[str, str]
            }
        """
        recommendations = {
            'adjust_confidence_threshold': False,
            'recommended_threshold': 0.85,
            'categories_to_disable': [],
            'best_strategies': {}
        }

        # Analyze category performance
        for category, stats in self.stats_by_category.items():
            success_rate = stats['applied'] / stats['attempted'] if stats['attempted'] > 0 else 0

            # Recommend disabling low-performing categories
            if stats['attempted'] >= 10 and success_rate < 0.5:
                recommendations['categories_to_disable'].append(category)

        # Analyze strategy effectiveness
        for strategy, stats in self.stats_by_strategy.items():
            if stats['success_rate'] > 0.8 and stats['attempted'] >= 5:
                recommendations['best_strategies'][strategy] = 'excellent'
            elif stats['success_rate'] > 0.6:
                recommendations['best_strategies'][strategy] = 'good'
            else:
                recommendations['best_strategies'][strategy] = 'needs_improvement'

        # Confidence threshold adjustment
        if self.current_session and self.current_session.total_fixes_attempted >= 50:
            if self.current_session.success_rate < 0.8:
                recommendations['adjust_confidence_threshold'] = True
                recommendations['recommended_threshold'] = min(0.95, self.current_session.confidence_threshold + 0.05)
            elif self.current_session.success_rate > 0.95:
                recommendations['adjust_confidence_threshold'] = True
                recommendations['recommended_threshold'] = max(0.75, self.current_session.confidence_threshold - 0.05)

        return recommendations


# Example usage
if __name__ == '__main__':
    # Create tracker
    tracker = AutoFixTracker()

    # Start session
    session_id = tracker.start_session(
        max_files=50,
        categories=['dead_code', 'hardcoded_values', 'missing_docstrings'],
        confidence_threshold=0.85
    )

    # Simulate some fixes
    fixes = [
        AutoFixResult(
            fix_id='fix_001',
            category='dead_code',
            severity='low',
            message='Unused import',
            confidence=0.95,
            file_path='example.py',
            line_number=10,
            code_snippet='import unused',
            strategy='ast',
            original_code='import unused\nimport used',
            fixed_code='import used',
            diff='-import unused\n import used',
            applied=True,
            validation_passed=True,
            errors=[],
            warnings=[],
            branch='autofix/2025-11-16',
            commit='abc123',
            duration_ms=45.0
        ),
        AutoFixResult(
            fix_id='fix_002',
            category='hardcoded_values',
            severity='medium',
            message='Magic number',
            confidence=0.88,
            file_path='example.py',
            line_number=20,
            code_snippet='x = 42',
            strategy='template',
            original_code='x = 42',
            fixed_code='MAX_SIZE = 42\nx = MAX_SIZE',
            diff='+MAX_SIZE = 42\n-x = 42\n+x = MAX_SIZE',
            applied=True,
            validation_passed=True,
            errors=[],
            warnings=[],
            branch='autofix/2025-11-16',
            commit='def456',
            duration_ms=120.0
        )
    ]

    for fix in fixes:
        tracker.track_fix(fix)

    # End session
    tracker.end_session()

    # Export
    tracker.export_json('autofix_results.json')
    tracker.export_learning_data('learning_data.json')

    # Get recommendations
    recs = tracker.get_recommendations()
    print("Recommendations:")
    print(json.dumps(recs, indent=2))
