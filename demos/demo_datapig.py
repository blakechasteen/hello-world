"""
DATAPIG Demo: Star Trek Data Quality Validation

"Computer, run level 1 diagnostic on all data systems."

This demo showcases all 10 DATAPIG detection categories with
Star Trek-themed error messages.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.datapig import DataPigDetector, Severity, engage_warp_validation
from datetime import datetime, timedelta


def demo_schema_drift():
    """Demo: The laws of physics have changed!"""
    print("\n" + "="*70)
    print("DEMO 1: SCHEMA DRIFT - 'The laws of physics are different here!'")
    print("="*70)

    data = [
        {"id": 1, "name": "Jean-Luc Picard", "rank": "Captain", "ship": "Enterprise"},
        {"id": 2, "name": "William Riker", "rank": "Commander"},  # Missing 'ship'!
        {"id": "3", "name": "Data", "rank": "Lieutenant Commander", "ship": "Enterprise"},  # ID is string!
    ]

    detector = DataPigDetector(enable_verbose=True)
    issues = detector.analyze_dataset(data)

    for issue in issues:
        print(f"\n{issue}")


def demo_data_leaks():
    """Demo: Hull breach on deck 7!"""
    print("\n" + "="*70)
    print("DEMO 2: DATA LEAKS - 'Hull breach detected!'")
    print("="*70)

    data = [
        {
            "officer": "Geordi La Forge",
            "email": "geordi@starfleet.com",  # PII leak!
            "security_code": "SK-1234567890abcdef1234567890abcdef",  # API key leak!
        },
        {
            "officer": "Beverly Crusher",
            "notes": "Patient SSN: 123-45-6789",  # SSN leak!
        }
    ]

    detector = DataPigDetector()
    issues = detector.analyze_dataset(data)

    for issue in issues:
        if issue.severity == Severity.CAPTAIN:
            print(f"\n*** RED ALERT! ***\n{issue}")


def demo_stale_data():
    """Demo: These readings are from last century!"""
    print("\n" + "="*70)
    print("DEMO 3: STALE DATA - 'These readings are from last century!'")
    print("="*70)

    old_date = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")
    recent_date = datetime.now().strftime("%Y-%m-%d")

    data = [
        {"sensor": "Warp Core", "last_calibration": old_date, "reading": 95.2},
        {"sensor": "Shields", "last_calibration": recent_date, "reading": 100.0},
    ]

    detector = DataPigDetector()
    issues = detector.analyze_dataset(data)

    for issue in issues:
        print(f"\n{issue}")


def demo_duplicates():
    """Demo: We're seeing double, Captain!"""
    print("\n" + "="*70)
    print("DEMO 4: DUPLICATES - 'We're seeing double!'")
    print("="*70)

    data = [
        {"crew_id": 1, "name": "Worf", "position": "Security"},
        {"crew_id": 2, "name": "Deanna Troi", "position": "Counselor"},
        {"crew_id": 1, "name": "Worf", "position": "Security"},  # Exact duplicate!
    ]

    detector = DataPigDetector()
    issues = detector.analyze_dataset(data)

    for issue in issues:
        print(f"\n{issue}")


def demo_outliers():
    """Demo: Captain, these readings are... impossible!"""
    print("\n" + "="*70)
    print("DEMO 5: OUTLIERS - 'Captain, these readings are impossible!'")
    print("="*70)

    data = [
        {"system": "Life Support", "power_usage": 45},
        {"system": "Sensors", "power_usage": 50},
        {"system": "Shields", "power_usage": 48},
        {"system": "Warp Core", "power_usage": 9999},  # Impossible!
        {"system": "Transporter", "power_usage": 47},
    ]

    detector = DataPigDetector()
    issues = detector.analyze_dataset(data)

    for issue in issues:
        print(f"\n{issue}")


def demo_inconsistent_formatting():
    """Demo: Universal translator malfunction!"""
    print("\n" + "="*70)
    print("DEMO 6: INCONSISTENT FORMATTING - 'Universal translator malfunction!'")
    print("="*70)

    data = [
        {"officer": "PICARD", "stardate": "2024-01-15"},
        {"officer": "riker", "stardate": "01/16/2024"},  # Different formats!
        {"officer": "Data", "stardate": "2024-01-17"},
    ]

    detector = DataPigDetector()
    issues = detector.analyze_dataset(data)

    for issue in issues:
        print(f"\n{issue}")


def demo_missing_relations():
    """Demo: Transporter lost the signal!"""
    print("\n" + "="*70)
    print("DEMO 7: MISSING RELATIONS - 'Transporter lost the signal!'")
    print("="*70)

    data = [
        {"id": 1, "mission": "Explore new worlds", "ship_id": 1701},
        {"id": 2, "mission": "Diplomatic mission", "ship_id": 9999},  # Orphaned FK!
        {"ship_id": 1701, "ship_name": "Enterprise-D"},
    ]

    detector = DataPigDetector()
    issues = detector.analyze_dataset(data)

    for issue in issues:
        print(f"\n{issue}")


def demo_sampling_bias():
    """Demo: Prime Directive violation detected!"""
    print("\n" + "="*70)
    print("DEMO 8: SAMPLING BIAS - 'Prime Directive violation!'")
    print("="*70)

    data = [
        {"species": "Human", "label": "friendly"},
        {"species": "Human", "label": "friendly"},
        {"species": "Human", "label": "friendly"},
        {"species": "Human", "label": "friendly"},
        {"species": "Human", "label": "friendly"},
        {"species": "Human", "label": "friendly"},
        {"species": "Human", "label": "friendly"},
        {"species": "Human", "label": "friendly"},
        {"species": "Human", "label": "friendly"},
        {"species": "Human", "label": "friendly"},
        {"species": "Borg", "label": "hostile"},  # Severe imbalance!
    ]

    detector = DataPigDetector()
    issues = detector.analyze_dataset(data)

    for issue in issues:
        print(f"\n{issue}")


def demo_label_noise():
    """Demo: Temporal anomaly - contradictory data!"""
    print("\n" + "="*70)
    print("DEMO 9: LABEL NOISE - 'Temporal anomaly detected!'")
    print("="*70)

    data = [
        {"scenario": "Borg attack", "crew": "Picard", "label": "hostile"},
        {"scenario": "Borg attack", "crew": "Picard", "label": "friendly"},  # Contradiction!
        {"scenario": "Romulan encounter", "crew": "Worf", "label": "neutral"},
    ]

    detector = DataPigDetector()
    issues = detector.analyze_dataset(data)

    for issue in issues:
        print(f"\n{issue}")


def demo_warp_validation():
    """Demo: Engage warp drive validation!"""
    print("\n" + "="*70)
    print("DEMO 10: WARP VALIDATION - 'Engage!'")
    print("="*70)

    # Safe data
    safe_data = [
        {"system": "Shields", "status": "online"},
        {"system": "Warp Drive", "status": "online"},
    ]

    print("\n--- Safe Data ---")
    engage_warp_validation(safe_data)

    # Unsafe data (with PII leak)
    unsafe_data = [
        {"system": "Shields", "status": "online"},
        {"system": "Database", "admin_email": "admin@enterprise.com"},  # PII!
    ]

    print("\n--- Unsafe Data ---")
    engage_warp_validation(unsafe_data)


def main():
    """Run all DATAPIG demos"""
    print("\n" + "="*70)
    print("DATAPIG: Data Analysis & Tactical Assessment Program")
    print("         for Integrity Governance")
    print("="*70)
    print("\n'Computer, run level 1 diagnostic on all data systems.'\n")

    demo_schema_drift()
    demo_data_leaks()
    demo_stale_data()
    demo_duplicates()
    demo_outliers()
    demo_inconsistent_formatting()
    demo_missing_relations()
    demo_sampling_bias()
    demo_label_noise()
    demo_warp_validation()

    print("\n" + "="*70)
    print("DATAPIG Analysis Complete. Live Long and Prosper!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
