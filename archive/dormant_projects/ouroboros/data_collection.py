#!/usr/bin/env python3
"""
Ouroboros Data Collection System
=================================

Real-time data collection for clinical validation and continuous improvement.

Tracks:
- Every prescription checked
- Decisions (SAFE/BLOCKED/REVIEW)
- Physician overrides
- Adverse events (if occur)
- False positives/negatives
- Performance metrics
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PrescriptionCheck:
    """Single prescription check event"""

    # Identifiers (de-identified)
    check_id: str
    patient_id_hash: str  # HIPAA: Hashed patient ID
    prescriber_id_hash: str
    timestamp: str

    # Input
    medications: List[str]
    allergies: List[str]

    # Ouroboros decision
    decision: str  # SAFE, BLOCKED, REVIEW
    interactions_found: int
    critical_count: int
    high_count: int
    moderate_count: int

    # Interactions detail
    interactions: List[Dict]

    # Performance
    latency_ms: float
    cache_hit: bool

    # Clinical outcome (filled in later)
    physician_override: Optional[bool] = None
    override_reason: Optional[str] = None
    adverse_event_occurred: Optional[bool] = None
    adverse_event_description: Optional[str] = None

    # Metadata
    ehr_system: str = "Epic"
    facility_id: str = "Unknown"


@dataclass
class ClinicalValidationCase:
    """Clinical validation session case"""

    # Identifiers
    case_id: str
    session_id: str
    clinician_id: str
    specialty: str
    timestamp: str

    # Test case
    medications: List[str]
    allergies: List[str]
    case_category: str  # CRITICAL, SAFE, EDGE_CASE

    # Decisions
    clinician_decision: str  # SAFE, BLOCKED, REVIEW
    ouroboros_decision: str

    # Agreement
    agreement: bool
    confidence_rating: int  # 1-5, how confident was clinician

    # Interactions
    interactions_shown: List[Dict]

    # Qualitative feedback
    comments: str
    alert_clarity: int  # 1-5 Likert scale
    time_to_decision_seconds: float

    # Ground truth (for metrics)
    known_ground_truth: Optional[str] = None  # From literature/FDA


@dataclass
class AdverseEventReport:
    """Adverse drug event report"""

    # Identifiers
    event_id: str
    patient_id_hash: str
    timestamp: str

    # Related prescription check (if any)
    related_check_id: Optional[str]

    # Event details
    event_type: str  # BLEEDING, SEROTONIN_SYNDROME, ANAPHYLAXIS, etc.
    severity: str  # CRITICAL, HIGH, MODERATE
    outcome: str  # RESOLVED, HOSPITALIZED, DEATH

    # Medications involved
    medications_at_time: List[str]
    suspected_interaction: Optional[str]

    # System performance
    was_detected_by_ouroboros: bool
    ouroboros_decision_at_time: Optional[str]

    # Clinical notes
    description: str


# ============================================================================
# Data Collection Database
# ============================================================================

class DataCollectionDB:
    """SQLite database for data collection"""

    def __init__(self, db_path: str = "./ouroboros_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

        print(f"[DataCollection] Database: {db_path}")

    def _create_tables(self):
        """Create database schema"""

        # Prescription checks table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prescription_checks (
                check_id TEXT PRIMARY KEY,
                patient_id_hash TEXT,
                prescriber_id_hash TEXT,
                timestamp TEXT,
                medications TEXT,
                allergies TEXT,
                decision TEXT,
                interactions_found INTEGER,
                critical_count INTEGER,
                high_count INTEGER,
                moderate_count INTEGER,
                interactions TEXT,
                latency_ms REAL,
                cache_hit BOOLEAN,
                physician_override BOOLEAN,
                override_reason TEXT,
                adverse_event_occurred BOOLEAN,
                adverse_event_description TEXT,
                ehr_system TEXT,
                facility_id TEXT
            )
        """)

        # Clinical validation cases table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS validation_cases (
                case_id TEXT PRIMARY KEY,
                session_id TEXT,
                clinician_id TEXT,
                specialty TEXT,
                timestamp TEXT,
                medications TEXT,
                allergies TEXT,
                case_category TEXT,
                clinician_decision TEXT,
                ouroboros_decision TEXT,
                agreement BOOLEAN,
                confidence_rating INTEGER,
                interactions_shown TEXT,
                comments TEXT,
                alert_clarity INTEGER,
                time_to_decision_seconds REAL,
                known_ground_truth TEXT
            )
        """)

        # Adverse events table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS adverse_events (
                event_id TEXT PRIMARY KEY,
                patient_id_hash TEXT,
                timestamp TEXT,
                related_check_id TEXT,
                event_type TEXT,
                severity TEXT,
                outcome TEXT,
                medications_at_time TEXT,
                suspected_interaction TEXT,
                was_detected_by_ouroboros BOOLEAN,
                ouroboros_decision_at_time TEXT,
                description TEXT
            )
        """)

        self.conn.commit()

    def log_prescription_check(self, check: PrescriptionCheck):
        """Log a prescription check"""

        self.conn.execute("""
            INSERT INTO prescription_checks VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            check.check_id,
            check.patient_id_hash,
            check.prescriber_id_hash,
            check.timestamp,
            json.dumps(check.medications),
            json.dumps(check.allergies),
            check.decision,
            check.interactions_found,
            check.critical_count,
            check.high_count,
            check.moderate_count,
            json.dumps(check.interactions),
            check.latency_ms,
            check.cache_hit,
            check.physician_override,
            check.override_reason,
            check.adverse_event_occurred,
            check.adverse_event_description,
            check.ehr_system,
            check.facility_id
        ))

        self.conn.commit()

    def update_clinical_outcome(self, check_id: str,
                                physician_override: bool,
                                override_reason: str = None,
                                adverse_event: bool = None,
                                adverse_event_description: str = None):
        """Update check with clinical outcome (after prescription given)"""

        self.conn.execute("""
            UPDATE prescription_checks
            SET physician_override = ?,
                override_reason = ?,
                adverse_event_occurred = ?,
                adverse_event_description = ?
            WHERE check_id = ?
        """, (physician_override, override_reason, adverse_event,
              adverse_event_description, check_id))

        self.conn.commit()

    def log_validation_case(self, case: ClinicalValidationCase):
        """Log a validation case"""

        self.conn.execute("""
            INSERT INTO validation_cases VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            case.case_id,
            case.session_id,
            case.clinician_id,
            case.specialty,
            case.timestamp,
            json.dumps(case.medications),
            json.dumps(case.allergies),
            case.case_category,
            case.clinician_decision,
            case.ouroboros_decision,
            case.agreement,
            case.confidence_rating,
            json.dumps(case.interactions_shown),
            case.comments,
            case.alert_clarity,
            case.time_to_decision_seconds,
            case.known_ground_truth
        ))

        self.conn.commit()

    def log_adverse_event(self, event: AdverseEventReport):
        """Log adverse event"""

        self.conn.execute("""
            INSERT INTO adverse_events VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            event.event_id,
            event.patient_id_hash,
            event.timestamp,
            event.related_check_id,
            event.event_type,
            event.severity,
            event.outcome,
            json.dumps(event.medications_at_time),
            event.suspected_interaction,
            event.was_detected_by_ouroboros,
            event.ouroboros_decision_at_time,
            event.description
        ))

        self.conn.commit()

    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""

        # Total checks
        total_checks = self.conn.execute(
            "SELECT COUNT(*) FROM prescription_checks"
        ).fetchone()[0]

        # Decisions breakdown
        decisions = self.conn.execute("""
            SELECT decision, COUNT(*)
            FROM prescription_checks
            GROUP BY decision
        """).fetchall()

        # Override rate
        overrides = self.conn.execute("""
            SELECT COUNT(*)
            FROM prescription_checks
            WHERE physician_override = 1
        """).fetchone()[0]

        override_rate = overrides / total_checks if total_checks > 0 else 0

        # Adverse events
        adverse_events = self.conn.execute("""
            SELECT COUNT(*)
            FROM adverse_events
        """).fetchone()[0]

        # Detection rate (AEs that were detected)
        detected_aes = self.conn.execute("""
            SELECT COUNT(*)
            FROM adverse_events
            WHERE was_detected_by_ouroboros = 1
        """).fetchone()[0]

        detection_rate = detected_aes / adverse_events if adverse_events > 0 else 0

        # Validation stats
        validation_cases = self.conn.execute("""
            SELECT COUNT(*)
            FROM validation_cases
        """).fetchone()[0]

        agreements = self.conn.execute("""
            SELECT COUNT(*)
            FROM validation_cases
            WHERE agreement = 1
        """).fetchone()[0]

        agreement_rate = agreements / validation_cases if validation_cases > 0 else 0

        return {
            'total_prescription_checks': total_checks,
            'decisions': dict(decisions),
            'override_rate': override_rate,
            'adverse_events_total': adverse_events,
            'adverse_events_detected': detected_aes,
            'detection_rate': detection_rate,
            'validation_cases': validation_cases,
            'agreement_rate': agreement_rate
        }

    def export_to_csv(self, output_dir: str = "./exports"):
        """Export all data to CSV for analysis"""

        import pandas as pd

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Prescription checks
        df_checks = pd.read_sql_query(
            "SELECT * FROM prescription_checks",
            self.conn
        )
        df_checks.to_csv(output_dir / "prescription_checks.csv", index=False)

        # Validation cases
        df_validation = pd.read_sql_query(
            "SELECT * FROM validation_cases",
            self.conn
        )
        df_validation.to_csv(output_dir / "validation_cases.csv", index=False)

        # Adverse events
        df_events = pd.read_sql_query(
            "SELECT * FROM adverse_events",
            self.conn
        )
        df_events.to_csv(output_dir / "adverse_events.csv", index=False)

        print(f"[DataCollection] Exported to {output_dir}/")

    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================================================
# Helper Functions
# ============================================================================

def generate_check_id() -> str:
    """Generate unique check ID"""
    import uuid
    return str(uuid.uuid4())


def hash_patient_id(patient_id: str) -> str:
    """Hash patient ID for HIPAA compliance"""
    return hashlib.sha256(patient_id.encode()).hexdigest()[:16]


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("OUROBOROS - DATA COLLECTION DEMO")
    print("="*70)

    # Initialize database
    db = DataCollectionDB("./demo_data.db")

    # Simulate prescription check
    print("\n[1/3] Logging prescription check...")

    check = PrescriptionCheck(
        check_id=generate_check_id(),
        patient_id_hash=hash_patient_id("patient_12345"),
        prescriber_id_hash=hash_patient_id("doctor_67890"),
        timestamp=datetime.now().isoformat(),
        medications=["warfarin", "aspirin"],
        allergies=[],
        decision="BLOCKED",
        interactions_found=1,
        critical_count=1,
        high_count=0,
        moderate_count=0,
        interactions=[{
            'drug_a': 'warfarin',
            'drug_b': 'aspirin',
            'severity': 'critical',
            'effect': 'Severe bleeding risk'
        }],
        latency_ms=15.5,
        cache_hit=False,
        facility_id="Hospital_A"
    )

    db.log_prescription_check(check)
    print(f"  Logged check: {check.check_id}")

    # Simulate physician override
    print("\n[2/3] Recording physician override...")

    db.update_clinical_outcome(
        check.check_id,
        physician_override=True,
        override_reason="Patient already on aspirin for 5 years, monitored closely"
    )
    print(f"  Override recorded (rare, but happens)")

    # Simulate validation case
    print("\n[3/3] Logging validation case...")

    validation = ClinicalValidationCase(
        case_id=generate_check_id(),
        session_id="session_001",
        clinician_id="dr_smith",
        specialty="Emergency Medicine",
        timestamp=datetime.now().isoformat(),
        medications=["warfarin", "aspirin"],
        allergies=[],
        case_category="CRITICAL",
        clinician_decision="BLOCKED",
        ouroboros_decision="BLOCKED",
        agreement=True,
        confidence_rating=5,
        interactions_shown=[{
            'drug_a': 'warfarin',
            'drug_b': 'aspirin',
            'severity': 'critical'
        }],
        comments="Absolutely correct, this is a classic dangerous combo",
        alert_clarity=5,
        time_to_decision_seconds=12.3,
        known_ground_truth="BLOCKED"
    )

    db.log_validation_case(validation)
    print(f"  Logged validation case: {validation.case_id}")

    # Show summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    stats = db.get_summary_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Export
    print("\n[Export] Saving to CSV...")
    db.export_to_csv("./demo_exports")

    db.close()

    print("\n" + "="*70)
    print("[OK] Data collection demo complete")
    print("="*70)