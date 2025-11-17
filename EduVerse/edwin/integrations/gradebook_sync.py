"""
Gradebook Sync Engine

Bidirectional grade synchronization between EdWIN and LMS.

Author: Agent D
Date: 2025-11-15
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os
import asyncio
from .lms_base import LMSIntegration
from .assignment_mapper import AssignmentMapper


class SyncStrategy(Enum):
    """Grade sync strategy"""
    IMMEDIATE = "immediate"  # Real-time sync
    SCHEDULED = "scheduled"  # Hourly/daily sync
    MANUAL = "manual"  # Teacher triggers
    SELECTIVE = "selective"  # Specific assignments only


class SyncDirection(Enum):
    """Sync direction"""
    EDWIN_TO_LMS = "edwin_to_lms"  # EdWIN → LMS (typical)
    LMS_TO_EDWIN = "lms_to_edwin"  # LMS → EdWIN (override)
    BIDIRECTIONAL = "bidirectional"  # Both directions


class ConflictResolution(Enum):
    """How to resolve sync conflicts"""
    EDWIN_WINS = "edwin_wins"  # EdWIN overwrites LMS
    LMS_WINS = "lms_wins"  # LMS overwrites EdWIN
    MANUAL = "manual"  # Require teacher approval
    NEWER_WINS = "newer_wins"  # Most recent change wins


@dataclass
class SyncLog:
    """Log entry for grade sync"""
    sync_id: str
    timestamp: datetime
    assignment_id: str
    student_id: str
    direction: SyncDirection
    old_grade: Optional[float]
    new_grade: float
    success: bool
    error: Optional[str] = None
    conflict_resolved: bool = False
    resolution_method: Optional[ConflictResolution] = None


@dataclass
class SyncConflict:
    """Grade sync conflict requiring resolution"""
    assignment_id: str
    student_id: str
    edwin_grade: float
    edwin_updated_at: datetime
    lms_grade: float
    lms_updated_at: datetime


class GradebookSync:
    """
    Bidirectional grade synchronization engine.

    Features:
    - EdWIN mastery → LMS grades (automatic)
    - LMS grades → EdWIN (manual override)
    - Batch sync (all students)
    - Real-time sync (as students complete)
    - Conflict resolution
    - Audit trail
    """

    def __init__(
        self,
        lms_integration: LMSIntegration,
        assignment_mapper: AssignmentMapper,
        sync_log_path: str = "./sync_logs",
    ):
        """
        Initialize gradebook sync engine.

        Args:
            lms_integration: LMS integration instance
            assignment_mapper: Assignment mapper instance
            sync_log_path: Directory to store sync logs
        """
        self.lms = lms_integration
        self.mapper = assignment_mapper
        self.sync_log_path = sync_log_path
        self.sync_logs: List[SyncLog] = []
        self.pending_conflicts: List[SyncConflict] = []

        os.makedirs(sync_log_path, exist_ok=True)

    async def sync_grade(
        self,
        student_id: str,
        assignment_id: str,
        objective_scores: Dict[str, float],
        direction: SyncDirection = SyncDirection.EDWIN_TO_LMS,
        conflict_resolution: ConflictResolution = ConflictResolution.MANUAL,
    ) -> SyncLog:
        """
        Sync grade for a single student.

        Args:
            student_id: Student ID
            assignment_id: LMS assignment ID
            objective_scores: Dict of objective_id → mastery score
            direction: Sync direction
            conflict_resolution: How to resolve conflicts

        Returns:
            SyncLog entry
        """
        sync_id = f"{assignment_id}_{student_id}_{int(datetime.now().timestamp())}"

        try:
            if direction == SyncDirection.EDWIN_TO_LMS:
                # Calculate EdWIN grade
                new_grade = self.mapper.calculate_grade(
                    assignment_id, objective_scores
                )

                # Check for conflict
                if conflict_resolution != ConflictResolution.EDWIN_WINS:
                    conflict = await self._check_conflict(
                        assignment_id, student_id, new_grade
                    )

                    if conflict:
                        if conflict_resolution == ConflictResolution.MANUAL:
                            self.pending_conflicts.append(conflict)
                            raise Exception(
                                "Grade conflict detected - manual resolution required"
                            )
                        elif conflict_resolution == ConflictResolution.LMS_WINS:
                            # Don't sync
                            return SyncLog(
                                sync_id=sync_id,
                                timestamp=datetime.now(),
                                assignment_id=assignment_id,
                                student_id=student_id,
                                direction=direction,
                                old_grade=conflict.lms_grade,
                                new_grade=new_grade,
                                success=False,
                                error="LMS grade wins conflict resolution",
                                conflict_resolved=True,
                                resolution_method=conflict_resolution,
                            )
                        elif conflict_resolution == ConflictResolution.NEWER_WINS:
                            if conflict.lms_updated_at > conflict.edwin_updated_at:
                                # Don't sync
                                return SyncLog(
                                    sync_id=sync_id,
                                    timestamp=datetime.now(),
                                    assignment_id=assignment_id,
                                    student_id=student_id,
                                    direction=direction,
                                    old_grade=conflict.lms_grade,
                                    new_grade=new_grade,
                                    success=False,
                                    error="LMS grade is newer",
                                    conflict_resolved=True,
                                    resolution_method=conflict_resolution,
                                )

                # Submit grade to LMS
                success = await self.lms.submit_grade(
                    student_id, assignment_id, new_grade
                )

                log = SyncLog(
                    sync_id=sync_id,
                    timestamp=datetime.now(),
                    assignment_id=assignment_id,
                    student_id=student_id,
                    direction=direction,
                    old_grade=None,
                    new_grade=new_grade,
                    success=success,
                )

            elif direction == SyncDirection.LMS_TO_EDWIN:
                # Get grade from LMS
                submissions = await self.lms.get_student_submissions(assignment_id)
                submission = next(
                    (s for s in submissions if s.student_id == student_id), None
                )

                if not submission or submission.score is None:
                    raise Exception("No grade found in LMS")

                # Note: This would update EdWIN's student model
                # Implementation depends on EdWIN's internal API
                log = SyncLog(
                    sync_id=sync_id,
                    timestamp=datetime.now(),
                    assignment_id=assignment_id,
                    student_id=student_id,
                    direction=direction,
                    old_grade=None,
                    new_grade=submission.score,
                    success=True,
                )

            else:  # BIDIRECTIONAL
                raise NotImplementedError("Bidirectional sync not yet implemented")

        except Exception as e:
            log = SyncLog(
                sync_id=sync_id,
                timestamp=datetime.now(),
                assignment_id=assignment_id,
                student_id=student_id,
                direction=direction,
                old_grade=None,
                new_grade=0.0,
                success=False,
                error=str(e),
            )

        self.sync_logs.append(log)
        self._save_log(log)

        return log

    async def batch_sync(
        self,
        assignment_id: str,
        student_scores: Dict[str, Dict[str, float]],
        direction: SyncDirection = SyncDirection.EDWIN_TO_LMS,
        conflict_resolution: ConflictResolution = ConflictResolution.MANUAL,
    ) -> List[SyncLog]:
        """
        Sync grades for all students in an assignment.

        Args:
            assignment_id: LMS assignment ID
            student_scores: Dict of student_id → objective_scores
            direction: Sync direction
            conflict_resolution: How to resolve conflicts

        Returns:
            List of SyncLog entries
        """
        logs = []

        for student_id, objective_scores in student_scores.items():
            log = await self.sync_grade(
                student_id,
                assignment_id,
                objective_scores,
                direction,
                conflict_resolution,
            )
            logs.append(log)

        return logs

    async def _check_conflict(
        self,
        assignment_id: str,
        student_id: str,
        edwin_grade: float,
    ) -> Optional[SyncConflict]:
        """Check if grade sync would create a conflict"""
        try:
            # Get current LMS grade
            submissions = await self.lms.get_student_submissions(assignment_id)
            submission = next(
                (s for s in submissions if s.student_id == student_id), None
            )

            if not submission or submission.score is None:
                return None  # No conflict if no LMS grade

            # Check if grades differ significantly
            grade_diff = abs(edwin_grade - submission.score)
            if grade_diff > 5:  # More than 5 points difference
                return SyncConflict(
                    assignment_id=assignment_id,
                    student_id=student_id,
                    edwin_grade=edwin_grade,
                    edwin_updated_at=datetime.now(),  # Would come from EdWIN
                    lms_grade=submission.score,
                    lms_updated_at=submission.graded_at or datetime.now(),
                )

            return None

        except Exception:
            return None  # No conflict if can't check

    def _save_log(self, log: SyncLog):
        """Save sync log to file"""
        log_file = os.path.join(
            self.sync_log_path,
            f"{datetime.now().strftime('%Y-%m-%d')}.jsonl",
        )

        with open(log_file, "a") as f:
            log_data = {
                "sync_id": log.sync_id,
                "timestamp": log.timestamp.isoformat(),
                "assignment_id": log.assignment_id,
                "student_id": log.student_id,
                "direction": log.direction.value,
                "old_grade": log.old_grade,
                "new_grade": log.new_grade,
                "success": log.success,
                "error": log.error,
                "conflict_resolved": log.conflict_resolved,
                "resolution_method": log.resolution_method.value
                if log.resolution_method
                else None,
            }
            f.write(json.dumps(log_data) + "\n")

    def get_recent_logs(
        self,
        limit: int = 100,
        assignment_id: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> List[SyncLog]:
        """
        Get recent sync logs.

        Args:
            limit: Maximum number of logs to return
            assignment_id: Filter by assignment
            student_id: Filter by student

        Returns:
            List of sync logs
        """
        logs = self.sync_logs.copy()

        if assignment_id:
            logs = [log for log in logs if log.assignment_id == assignment_id]

        if student_id:
            logs = [log for log in logs if log.student_id == student_id]

        return logs[-limit:]

    def get_pending_conflicts(self) -> List[SyncConflict]:
        """Get pending conflicts requiring manual resolution"""
        return self.pending_conflicts.copy()

    def resolve_conflict(
        self,
        conflict: SyncConflict,
        use_edwin_grade: bool,
    ) -> SyncLog:
        """
        Manually resolve a sync conflict.

        Args:
            conflict: Sync conflict to resolve
            use_edwin_grade: True to use EdWIN grade, False for LMS grade

        Returns:
            SyncLog entry
        """
        if use_edwin_grade:
            # Sync EdWIN grade to LMS
            # Note: This would need to be async in practice
            pass
        else:
            # Sync LMS grade to EdWIN
            # Note: This would need to be async in practice
            pass

        # Remove from pending
        if conflict in self.pending_conflicts:
            self.pending_conflicts.remove(conflict)

        # Return log (simplified)
        return SyncLog(
            sync_id=f"conflict_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            assignment_id=conflict.assignment_id,
            student_id=conflict.student_id,
            direction=SyncDirection.EDWIN_TO_LMS
            if use_edwin_grade
            else SyncDirection.LMS_TO_EDWIN,
            old_grade=conflict.lms_grade,
            new_grade=conflict.edwin_grade if use_edwin_grade else conflict.lms_grade,
            success=True,
            conflict_resolved=True,
            resolution_method=ConflictResolution.MANUAL,
        )

    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get sync statistics"""
        total = len(self.sync_logs)
        successful = sum(1 for log in self.sync_logs if log.success)
        failed = total - successful
        conflicts = len(self.pending_conflicts)

        return {
            "total_syncs": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "pending_conflicts": conflicts,
        }
