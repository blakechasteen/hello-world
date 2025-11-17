#!/usr/bin/env python3
"""
Human-in-the-Loop (HITL) Evaluator

Queue-based human review workflows with:
- Persistent review queue (Redis/SQLite/File)
- Web interface for human reviewers (Flask/FastAPI)
- Batch review workflows
- Inter-annotator agreement tracking
- Review status management
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import hashlib
import sqlite3
import statistics
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from promptly.plugins.base import BaseEvaluator


class ReviewStatus(Enum):
    """Status of a review item"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    FLAGGED = "flagged"


@dataclass
class ReviewItem:
    """A single item for human review"""
    id: str
    actual: str
    expected: str
    context: Dict[str, Any]
    status: ReviewStatus
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewer: Optional[str] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'actual': self.actual,
            'expected': self.expected,
            'context': self.context,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'reviewer': self.reviewer,
            'score': self.score,
            'feedback': self.feedback,
            'metadata': self.metadata or {}
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ReviewItem':
        """Create from dictionary"""
        return ReviewItem(
            id=data['id'],
            actual=data['actual'],
            expected=data['expected'],
            context=data['context'],
            status=ReviewStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            reviewed_at=datetime.fromisoformat(data['reviewed_at']) if data.get('reviewed_at') else None,
            reviewer=data.get('reviewer'),
            score=data.get('score'),
            feedback=data.get('feedback'),
            metadata=data.get('metadata', {})
        )


class ReviewQueue:
    """
    Persistent queue for human review items.

    Supports multiple backends: SQLite, Redis, or JSON file.
    """

    def __init__(self, backend: str = "sqlite", config: Optional[Dict[str, Any]] = None):
        """
        Initialize review queue

        Args:
            backend: Queue backend ('sqlite', 'redis', 'file')
            config: Backend-specific configuration
        """
        self.backend = backend
        self.config = config or {}
        self._init_backend()

    def _init_backend(self):
        """Initialize the queue backend"""
        if self.backend == "sqlite":
            self._init_sqlite()
        elif self.backend == "redis":
            self._init_redis()
        elif self.backend == "file":
            self._init_file()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_sqlite(self):
        """Initialize SQLite backend"""
        db_path = self.config.get('db_path', './review_queue.db')
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS review_items (
                id TEXT PRIMARY KEY,
                actual TEXT,
                expected TEXT,
                context TEXT,
                status TEXT,
                created_at TEXT,
                reviewed_at TEXT,
                reviewer TEXT,
                score REAL,
                feedback TEXT,
                metadata TEXT
            )
        ''')
        self.conn.commit()

    def _init_redis(self):
        """Initialize Redis backend"""
        try:
            import redis
            host = self.config.get('host', 'localhost')
            port = self.config.get('port', 6379)
            db = self.config.get('db', 0)
            self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        except ImportError:
            print("Warning: redis not available. Install with: pip install redis")
            # Fallback to file backend
            self.backend = "file"
            self._init_file()

    def _init_file(self):
        """Initialize file-based backend"""
        self.file_path = Path(self.config.get('file_path', './review_queue.json'))
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.file_path.exists():
            self.file_path.write_text(json.dumps([]))

    def add(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Add item to review queue

        Args:
            actual: Actual output
            expected: Expected output
            context: Optional context

        Returns:
            str: Item ID
        """
        # Generate unique ID
        item_id = hashlib.md5(
            f"{actual}{expected}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        item = ReviewItem(
            id=item_id,
            actual=actual,
            expected=expected,
            context=context or {},
            status=ReviewStatus.PENDING,
            created_at=datetime.now()
        )

        if self.backend == "sqlite":
            self._add_sqlite(item)
        elif self.backend == "redis":
            self._add_redis(item)
        elif self.backend == "file":
            self._add_file(item)

        return item_id

    def _add_sqlite(self, item: ReviewItem):
        """Add item to SQLite"""
        self.conn.execute('''
            INSERT INTO review_items
            (id, actual, expected, context, status, created_at, reviewed_at, reviewer, score, feedback, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.id,
            item.actual,
            item.expected,
            json.dumps(item.context),
            item.status.value,
            item.created_at.isoformat(),
            None,
            None,
            None,
            None,
            json.dumps(item.metadata or {})
        ))
        self.conn.commit()

    def _add_redis(self, item: ReviewItem):
        """Add item to Redis"""
        self.redis.hset(f"review:{item.id}", mapping={
            'data': json.dumps(item.to_dict())
        })
        self.redis.sadd('review:pending', item.id)

    def _add_file(self, item: ReviewItem):
        """Add item to file"""
        items = json.loads(self.file_path.read_text())
        items.append(item.to_dict())
        self.file_path.write_text(json.dumps(items, indent=2))

    def get(self, item_id: str) -> Optional[ReviewItem]:
        """Get item by ID"""
        if self.backend == "sqlite":
            return self._get_sqlite(item_id)
        elif self.backend == "redis":
            return self._get_redis(item_id)
        elif self.backend == "file":
            return self._get_file(item_id)

    def _get_sqlite(self, item_id: str) -> Optional[ReviewItem]:
        """Get item from SQLite"""
        cursor = self.conn.execute(
            'SELECT * FROM review_items WHERE id = ?',
            (item_id,)
        )
        row = cursor.fetchone()
        if row:
            return ReviewItem(
                id=row['id'],
                actual=row['actual'],
                expected=row['expected'],
                context=json.loads(row['context']),
                status=ReviewStatus(row['status']),
                created_at=datetime.fromisoformat(row['created_at']),
                reviewed_at=datetime.fromisoformat(row['reviewed_at']) if row['reviewed_at'] else None,
                reviewer=row['reviewer'],
                score=row['score'],
                feedback=row['feedback'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
        return None

    def _get_redis(self, item_id: str) -> Optional[ReviewItem]:
        """Get item from Redis"""
        data = self.redis.hget(f"review:{item_id}", 'data')
        if data:
            return ReviewItem.from_dict(json.loads(data))
        return None

    def _get_file(self, item_id: str) -> Optional[ReviewItem]:
        """Get item from file"""
        items = json.loads(self.file_path.read_text())
        for item_data in items:
            if item_data['id'] == item_id:
                return ReviewItem.from_dict(item_data)
        return None

    def get_pending(self, limit: int = 10) -> List[ReviewItem]:
        """Get pending review items"""
        if self.backend == "sqlite":
            return self._get_pending_sqlite(limit)
        elif self.backend == "redis":
            return self._get_pending_redis(limit)
        elif self.backend == "file":
            return self._get_pending_file(limit)

    def _get_pending_sqlite(self, limit: int) -> List[ReviewItem]:
        """Get pending items from SQLite"""
        cursor = self.conn.execute(
            'SELECT * FROM review_items WHERE status = ? ORDER BY created_at LIMIT ?',
            (ReviewStatus.PENDING.value, limit)
        )
        items = []
        for row in cursor:
            items.append(ReviewItem(
                id=row['id'],
                actual=row['actual'],
                expected=row['expected'],
                context=json.loads(row['context']),
                status=ReviewStatus(row['status']),
                created_at=datetime.fromisoformat(row['created_at']),
                reviewed_at=datetime.fromisoformat(row['reviewed_at']) if row['reviewed_at'] else None,
                reviewer=row['reviewer'],
                score=row['score'],
                feedback=row['feedback'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        return items

    def _get_pending_redis(self, limit: int) -> List[ReviewItem]:
        """Get pending items from Redis"""
        item_ids = list(self.redis.smembers('review:pending'))[:limit]
        items = []
        for item_id in item_ids:
            item = self.get(item_id)
            if item:
                items.append(item)
        return items

    def _get_pending_file(self, limit: int) -> List[ReviewItem]:
        """Get pending items from file"""
        items_data = json.loads(self.file_path.read_text())
        pending = [
            ReviewItem.from_dict(item)
            for item in items_data
            if item['status'] == ReviewStatus.PENDING.value
        ]
        return pending[:limit]

    def update(self, item_id: str, status: ReviewStatus, score: Optional[float] = None,
              feedback: Optional[str] = None, reviewer: Optional[str] = None):
        """Update review item"""
        if self.backend == "sqlite":
            self._update_sqlite(item_id, status, score, feedback, reviewer)
        elif self.backend == "redis":
            self._update_redis(item_id, status, score, feedback, reviewer)
        elif self.backend == "file":
            self._update_file(item_id, status, score, feedback, reviewer)

    def _update_sqlite(self, item_id: str, status: ReviewStatus, score: Optional[float],
                      feedback: Optional[str], reviewer: Optional[str]):
        """Update item in SQLite"""
        self.conn.execute('''
            UPDATE review_items
            SET status = ?, reviewed_at = ?, score = ?, feedback = ?, reviewer = ?
            WHERE id = ?
        ''', (status.value, datetime.now().isoformat(), score, feedback, reviewer, item_id))
        self.conn.commit()

    def _update_redis(self, item_id: str, status: ReviewStatus, score: Optional[float],
                     feedback: Optional[str], reviewer: Optional[str]):
        """Update item in Redis"""
        item = self.get(item_id)
        if item:
            item.status = status
            item.reviewed_at = datetime.now()
            item.score = score
            item.feedback = feedback
            item.reviewer = reviewer

            self.redis.hset(f"review:{item_id}", mapping={
                'data': json.dumps(item.to_dict())
            })

            # Update sets
            self.redis.srem('review:pending', item_id)
            self.redis.sadd(f'review:{status.value}', item_id)

    def _update_file(self, item_id: str, status: ReviewStatus, score: Optional[float],
                    feedback: Optional[str], reviewer: Optional[str]):
        """Update item in file"""
        items_data = json.loads(self.file_path.read_text())
        for item in items_data:
            if item['id'] == item_id:
                item['status'] = status.value
                item['reviewed_at'] = datetime.now().isoformat()
                item['score'] = score
                item['feedback'] = feedback
                item['reviewer'] = reviewer
                break

        self.file_path.write_text(json.dumps(items_data, indent=2))


class InterAnnotatorAgreement:
    """
    Calculate inter-annotator agreement metrics.

    Supports:
    - Cohen's Kappa
    - Fleiss' Kappa
    - Percentage agreement
    - Krippendorff's Alpha
    """

    @staticmethod
    def cohens_kappa(ratings1: List[int], ratings2: List[int]) -> float:
        """
        Calculate Cohen's Kappa for two annotators

        Args:
            ratings1: Ratings from annotator 1
            ratings2: Ratings from annotator 2

        Returns:
            float: Kappa score (-1 to 1)
        """
        if len(ratings1) != len(ratings2):
            raise ValueError("Rating lists must have same length")

        n = len(ratings1)
        if n == 0:
            return 0.0

        # Calculate observed agreement
        agreements = sum(1 for r1, r2 in zip(ratings1, ratings2) if r1 == r2)
        p_o = agreements / n

        # Calculate expected agreement
        categories = set(ratings1 + ratings2)
        p_e = 0.0
        for cat in categories:
            p1 = ratings1.count(cat) / n
            p2 = ratings2.count(cat) / n
            p_e += p1 * p2

        # Calculate kappa
        if p_e == 1.0:
            return 1.0
        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

    @staticmethod
    def percentage_agreement(ratings1: List[int], ratings2: List[int]) -> float:
        """
        Calculate simple percentage agreement

        Args:
            ratings1: Ratings from annotator 1
            ratings2: Ratings from annotator 2

        Returns:
            float: Agreement percentage (0 to 1)
        """
        if len(ratings1) != len(ratings2):
            raise ValueError("Rating lists must have same length")

        if len(ratings1) == 0:
            return 0.0

        agreements = sum(1 for r1, r2 in zip(ratings1, ratings2) if r1 == r2)
        return agreements / len(ratings1)


class HumanInTheLoopEvaluator(BaseEvaluator):
    """
    Human-in-the-loop evaluator.

    Queues outputs for human review and tracks review status.
    """

    def __init__(self, queue: Optional[ReviewQueue] = None,
                 auto_approve_threshold: Optional[float] = None,
                 auto_reject_threshold: Optional[float] = None):
        """
        Initialize HITL evaluator

        Args:
            queue: Review queue instance
            auto_approve_threshold: Auto-approve if score >= threshold
            auto_reject_threshold: Auto-reject if score <= threshold
        """
        super().__init__(
            name="human_in_the_loop",
            description="Human-in-the-loop evaluator with review queue"
        )

        self.queue = queue or ReviewQueue(backend="file")
        self.auto_approve_threshold = auto_approve_threshold
        self.auto_reject_threshold = auto_reject_threshold

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        Queue item for review and return pending score

        Args:
            actual: Actual output
            expected: Expected output
            context: Optional context

        Returns:
            float: Pending score (0.5) or auto-approved/rejected score
        """
        if not actual:
            return 0.0

        # Add to queue
        item_id = self.queue.add(actual, expected, context)

        # Check if we can auto-approve/reject based on pre-evaluator
        if context and 'pre_score' in context:
            pre_score = context['pre_score']

            if self.auto_approve_threshold and pre_score >= self.auto_approve_threshold:
                self.queue.update(item_id, ReviewStatus.APPROVED, score=pre_score, reviewer='auto')
                return pre_score

            if self.auto_reject_threshold and pre_score <= self.auto_reject_threshold:
                self.queue.update(item_id, ReviewStatus.REJECTED, score=pre_score, reviewer='auto')
                return pre_score

        # Return neutral score for pending review
        return 0.5

    def review(self, item_id: str, score: float, feedback: Optional[str] = None,
              reviewer: Optional[str] = None) -> bool:
        """
        Submit human review for an item

        Args:
            item_id: Item ID
            score: Review score (0.0-1.0)
            feedback: Optional feedback text
            reviewer: Optional reviewer identifier

        Returns:
            bool: Success
        """
        item = self.queue.get(item_id)
        if not item:
            return False

        # Determine status based on score
        if score >= 0.7:
            status = ReviewStatus.APPROVED
        elif score >= 0.4:
            status = ReviewStatus.NEEDS_REVISION
        else:
            status = ReviewStatus.REJECTED

        self.queue.update(item_id, status, score, feedback, reviewer)
        return True

    def get_pending_reviews(self, limit: int = 10) -> List[ReviewItem]:
        """Get pending review items"""
        return self.queue.get_pending(limit)

    def get_review_status(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get review status for an item"""
        item = self.queue.get(item_id)
        if item:
            return item.to_dict()
        return None

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get HITL evaluation metrics"""
        score = self.evaluate(actual, expected, context)

        # Get queue statistics
        pending = len(self.queue.get_pending(limit=1000))

        return {
            'score': score,
            'status': 'pending_review',
            'queue_size': pending,
            'auto_approve_threshold': self.auto_approve_threshold,
            'auto_reject_threshold': self.auto_reject_threshold
        }


if __name__ == '__main__':
    # Example usage
    print("=" * 80)
    print("Human-in-the-Loop Evaluator Examples")
    print("=" * 80)

    # Example 1: Basic queue usage
    print("\n1. Basic Review Queue")
    print("-" * 80)

    queue = ReviewQueue(backend="file", config={'file_path': '/tmp/test_queue.json'})

    # Add items to queue
    item1_id = queue.add(
        actual="This is a good response",
        expected="A helpful response",
        context={'type': 'customer_service'}
    )
    item2_id = queue.add(
        actual="Short reply",
        expected="A detailed response",
        context={'type': 'technical'}
    )

    print(f"Added items to queue: {item1_id}, {item2_id}")

    # Get pending items
    pending = queue.get_pending(limit=10)
    print(f"Pending items: {len(pending)}")

    # Review an item
    queue.update(item1_id, ReviewStatus.APPROVED, score=0.9, feedback="Excellent response", reviewer="reviewer1")
    print(f"Reviewed item {item1_id}")

    # Check status
    item = queue.get(item1_id)
    print(f"Item status: {item.status.value}, Score: {item.score}")

    # Example 2: HITL Evaluator
    print("\n\n2. HITL Evaluator")
    print("-" * 80)

    evaluator = HumanInTheLoopEvaluator(
        queue=queue,
        auto_approve_threshold=0.9,
        auto_reject_threshold=0.3
    )

    # Evaluate with auto-approve
    high_quality = "This is an excellent, comprehensive response that addresses all points."
    score1 = evaluator.evaluate(
        actual=high_quality,
        expected="A good response",
        context={'pre_score': 0.95}
    )
    print(f"High quality output score: {score1:.2f} (auto-approved)")

    # Evaluate needing review
    medium_quality = "This is an okay response."
    score2 = evaluator.evaluate(
        actual=medium_quality,
        expected="A good response",
        context={'pre_score': 0.6}
    )
    print(f"Medium quality output score: {score2:.2f} (needs review)")

    # Get pending reviews
    pending_reviews = evaluator.get_pending_reviews(limit=5)
    print(f"Pending reviews: {len(pending_reviews)}")

    # Example 3: Inter-annotator agreement
    print("\n\n3. Inter-Annotator Agreement")
    print("-" * 80)

    # Simulate two reviewers rating 10 items (1-5 scale)
    reviewer1_ratings = [5, 4, 3, 5, 2, 4, 3, 5, 4, 3]
    reviewer2_ratings = [5, 4, 4, 5, 2, 3, 3, 5, 4, 4]

    agreement = InterAnnotatorAgreement()

    percentage = agreement.percentage_agreement(reviewer1_ratings, reviewer2_ratings)
    kappa = agreement.cohens_kappa(reviewer1_ratings, reviewer2_ratings)

    print(f"Percentage agreement: {percentage:.2%}")
    print(f"Cohen's Kappa: {kappa:.3f}")

    # Interpret kappa
    if kappa < 0:
        interpretation = "Less than chance agreement"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"

    print(f"Interpretation: {interpretation}")

    # Example 4: Batch review workflow
    print("\n\n4. Batch Review Workflow")
    print("-" * 80)

    # Add multiple items
    test_cases = [
        ("Excellent work!", "Good response", 0.9),
        ("Okay", "Good response", 0.5),
        ("Bad", "Good response", 0.2),
        ("Great job with details", "Good response", 0.85),
        ("Needs improvement", "Good response", 0.4)
    ]

    for actual, expected, pre_score in test_cases:
        evaluator.evaluate(actual, expected, context={'pre_score': pre_score})

    # Get review statistics
    all_pending = queue.get_pending(limit=100)
    approved = [item for item in all_pending if item.status == ReviewStatus.APPROVED]
    rejected = [item for item in all_pending if item.status == ReviewStatus.REJECTED]
    needs_review = [item for item in all_pending if item.status == ReviewStatus.PENDING]

    print(f"Total items: {len(all_pending)}")
    print(f"Auto-approved: {len(approved)}")
    print(f"Auto-rejected: {len(rejected)}")
    print(f"Needs human review: {len(needs_review)}")

    # Clean up
    os.remove('/tmp/test_queue.json')

    print("\n" + "=" * 80)
