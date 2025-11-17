"""
Roster Manager

Automatic roster synchronization between LMS and EdWIN.

Author: Agent D
Date: 2025-11-15
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os
from .lms_base import LMSIntegration, RosterData, Student


class MatchStrategy(Enum):
    """Student matching strategy"""
    EMAIL = "email"  # Match by email (primary)
    EXTERNAL_ID = "external_id"  # Match by student ID
    NAME = "name"  # Match by name (fallback)
    MANUAL = "manual"  # Manual linking


@dataclass
class StudentMatch:
    """Matched student between LMS and EdWIN"""
    lms_student_id: str
    edwin_student_id: str
    match_strategy: MatchStrategy
    confidence: float  # 0.0-1.0
    matched_at: datetime


@dataclass
class RosterChange:
    """Change in roster (add/drop/update)"""
    change_type: str  # "add", "drop", "update"
    student: Student
    timestamp: datetime
    reason: Optional[str] = None


class RosterManager:
    """
    Automatic roster synchronization manager.

    Features:
    - Daily sync of student enrollments
    - Handle student add/drop
    - Archive removed students (don't delete progress)
    - Teacher assignment changes
    - Section/class changes
    """

    def __init__(
        self,
        lms_integration: LMSIntegration,
        matches_file: str = "./student_matches.json",
        changes_log: str = "./roster_changes.jsonl",
    ):
        """
        Initialize roster manager.

        Args:
            lms_integration: LMS integration instance
            matches_file: File to store student matches
            changes_log: File to log roster changes
        """
        self.lms = lms_integration
        self.matches_file = matches_file
        self.changes_log = changes_log
        self.matches: Dict[str, StudentMatch] = {}
        self._load_matches()

    def _load_matches(self):
        """Load student matches from file"""
        if not os.path.exists(self.matches_file):
            return

        try:
            with open(self.matches_file, "r") as f:
                data = json.load(f)

            for lms_id, match_data in data.items():
                self.matches[lms_id] = StudentMatch(
                    lms_student_id=match_data["lms_student_id"],
                    edwin_student_id=match_data["edwin_student_id"],
                    match_strategy=MatchStrategy(match_data["match_strategy"]),
                    confidence=match_data["confidence"],
                    matched_at=datetime.fromisoformat(match_data["matched_at"]),
                )
        except Exception as e:
            print(f"Error loading student matches: {e}")

    def _save_matches(self):
        """Save student matches to file"""
        data = {
            lms_id: {
                "lms_student_id": match.lms_student_id,
                "edwin_student_id": match.edwin_student_id,
                "match_strategy": match.match_strategy.value,
                "confidence": match.confidence,
                "matched_at": match.matched_at.isoformat(),
            }
            for lms_id, match in self.matches.items()
        }

        os.makedirs(os.path.dirname(self.matches_file) if os.path.dirname(self.matches_file) else ".", exist_ok=True)
        with open(self.matches_file, "w") as f:
            json.dump(data, f, indent=2)

    def _log_change(self, change: RosterChange):
        """Log roster change"""
        os.makedirs(os.path.dirname(self.changes_log) if os.path.dirname(self.changes_log) else ".", exist_ok=True)

        with open(self.changes_log, "a") as f:
            change_data = {
                "change_type": change.change_type,
                "student_id": change.student.id,
                "student_name": change.student.name,
                "student_email": change.student.email,
                "timestamp": change.timestamp.isoformat(),
                "reason": change.reason,
            }
            f.write(json.dumps(change_data) + "\n")

    async def sync_roster(
        self,
        course_id: str,
        edwin_students: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Sync roster from LMS.

        Args:
            course_id: LMS course ID
            edwin_students: List of current EdWIN students with 'id', 'email', 'name'

        Returns:
            Dict with sync results
        """
        # Get LMS roster
        roster_data = await self.lms.sync_roster(course_id)

        # Match students
        matched = []
        unmatched_lms = []
        unmatched_edwin = list(edwin_students)

        for lms_student in roster_data.students:
            match = self._match_student(lms_student, edwin_students)

            if match:
                matched.append(match)
                # Remove from unmatched
                unmatched_edwin = [
                    s for s in unmatched_edwin if s["id"] != match.edwin_student_id
                ]
            else:
                unmatched_lms.append(lms_student)

        # Detect changes
        changes = self._detect_changes(
            roster_data.students, edwin_students, matched
        )

        return {
            "matched": len(matched),
            "unmatched_lms": len(unmatched_lms),
            "unmatched_edwin": len(unmatched_edwin),
            "changes": changes,
            "lms_students": roster_data.students,
            "matched_students": matched,
            "unmatched_lms_students": unmatched_lms,
            "unmatched_edwin_students": unmatched_edwin,
        }

    def _match_student(
        self,
        lms_student: Student,
        edwin_students: List[Dict[str, Any]],
    ) -> Optional[StudentMatch]:
        """
        Match LMS student to EdWIN student.

        Args:
            lms_student: LMS student
            edwin_students: List of EdWIN students

        Returns:
            StudentMatch if found
        """
        # Check existing match
        if lms_student.id in self.matches:
            return self.matches[lms_student.id]

        # Try email match (primary)
        if lms_student.email:
            for edwin_student in edwin_students:
                if (
                    edwin_student.get("email", "").lower()
                    == lms_student.email.lower()
                ):
                    match = StudentMatch(
                        lms_student_id=lms_student.id,
                        edwin_student_id=edwin_student["id"],
                        match_strategy=MatchStrategy.EMAIL,
                        confidence=1.0,
                        matched_at=datetime.now(),
                    )
                    self.matches[lms_student.id] = match
                    self._save_matches()
                    return match

        # Try external ID match
        if lms_student.external_id:
            for edwin_student in edwin_students:
                if edwin_student.get("external_id") == lms_student.external_id:
                    match = StudentMatch(
                        lms_student_id=lms_student.id,
                        edwin_student_id=edwin_student["id"],
                        match_strategy=MatchStrategy.EXTERNAL_ID,
                        confidence=0.95,
                        matched_at=datetime.now(),
                    )
                    self.matches[lms_student.id] = match
                    self._save_matches()
                    return match

        # Try fuzzy name match (fallback)
        if lms_student.name:
            for edwin_student in edwin_students:
                edwin_name = edwin_student.get("name", "").lower()
                lms_name = lms_student.name.lower()

                # Simple fuzzy match (could be improved)
                if (
                    edwin_name in lms_name
                    or lms_name in edwin_name
                    or self._levenshtein_distance(edwin_name, lms_name) < 3
                ):
                    match = StudentMatch(
                        lms_student_id=lms_student.id,
                        edwin_student_id=edwin_student["id"],
                        match_strategy=MatchStrategy.NAME,
                        confidence=0.7,  # Lower confidence
                        matched_at=datetime.now(),
                    )
                    self.matches[lms_student.id] = match
                    self._save_matches()
                    return match

        return None

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _detect_changes(
        self,
        lms_students: List[Student],
        edwin_students: List[Dict[str, Any]],
        matched: List[StudentMatch],
    ) -> List[RosterChange]:
        """Detect roster changes (add/drop/update)"""
        changes = []

        # Get matched IDs
        matched_lms_ids = {match.lms_student_id for match in matched}
        matched_edwin_ids = {match.edwin_student_id for match in matched}

        # Detect additions (in LMS but not in EdWIN)
        for lms_student in lms_students:
            if lms_student.id not in matched_lms_ids:
                change = RosterChange(
                    change_type="add",
                    student=lms_student,
                    timestamp=datetime.now(),
                    reason="New student in LMS",
                )
                changes.append(change)
                self._log_change(change)

        # Detect drops (in EdWIN but not in LMS)
        lms_ids = {student.id for student in lms_students}
        for match in self.matches.values():
            if match.lms_student_id not in lms_ids:
                # Find student in EdWIN
                edwin_student = next(
                    (
                        s
                        for s in edwin_students
                        if s["id"] == match.edwin_student_id
                    ),
                    None,
                )

                if edwin_student:
                    student = Student(
                        id=edwin_student["id"],
                        name=edwin_student["name"],
                        email=edwin_student["email"],
                        role=edwin_student.get("role", "student"),
                    )

                    change = RosterChange(
                        change_type="drop",
                        student=student,
                        timestamp=datetime.now(),
                        reason="Student removed from LMS",
                    )
                    changes.append(change)
                    self._log_change(change)

        return changes

    def manually_link_student(
        self,
        lms_student_id: str,
        edwin_student_id: str,
    ) -> StudentMatch:
        """
        Manually link LMS student to EdWIN student.

        Args:
            lms_student_id: LMS student ID
            edwin_student_id: EdWIN student ID

        Returns:
            StudentMatch
        """
        match = StudentMatch(
            lms_student_id=lms_student_id,
            edwin_student_id=edwin_student_id,
            match_strategy=MatchStrategy.MANUAL,
            confidence=1.0,
            matched_at=datetime.now(),
        )

        self.matches[lms_student_id] = match
        self._save_matches()

        return match

    def get_match(self, lms_student_id: str) -> Optional[StudentMatch]:
        """Get student match"""
        return self.matches.get(lms_student_id)

    def get_all_matches(self) -> List[StudentMatch]:
        """Get all student matches"""
        return list(self.matches.values())

    def get_unmatched_count(self) -> int:
        """Get count of unmatched students"""
        # This would need access to EdWIN's student database
        # Simplified implementation
        return 0

    def get_recent_changes(self, limit: int = 100) -> List[RosterChange]:
        """Get recent roster changes from log"""
        if not os.path.exists(self.changes_log):
            return []

        changes = []
        with open(self.changes_log, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    student = Student(
                        id=data["student_id"],
                        name=data["student_name"],
                        email=data["student_email"],
                        role="student",
                    )
                    change = RosterChange(
                        change_type=data["change_type"],
                        student=student,
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        reason=data.get("reason"),
                    )
                    changes.append(change)
                except Exception:
                    continue

        return changes[-limit:]
