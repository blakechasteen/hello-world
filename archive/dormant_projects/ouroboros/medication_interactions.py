#!/usr/bin/env python3
"""
Dark Trace - Medication Interaction Detection System
=====================================================

Core feature for medical AI safety: detecting dangerous drug combinations
BEFORE they're prescribed.

Key Features:
- Drug-drug interaction database
- Severity scoring (CRITICAL, HIGH, MODERATE, LOW)
- Mechanism explanations (why dangerous)
- Alternative suggestions
- Complete provenance for audit trail
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json


class InteractionSeverity(Enum):
    """Interaction risk levels (FDA classification)"""
    CRITICAL = "critical"    # Life-threatening, absolute contraindication
    HIGH = "high"           # Serious adverse effects, avoid combination
    MODERATE = "moderate"   # Monitor closely, may require dose adjustment
    LOW = "low"            # Minimal risk, monitor for side effects
    NONE = "none"          # No known interaction


@dataclass
class DrugInteraction:
    """Single drug-drug interaction"""
    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    mechanism: str          # Why it's dangerous
    effect: str            # What happens
    alternative: str       # What to use instead
    references: List[str]  # Clinical sources

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        d = asdict(self)
        d['severity'] = self.severity.value
        return d


class MedicationInteractionDetector:
    """
    Detects dangerous drug combinations.

    In production, this would query:
    - FDA drug interaction database
    - DrugBank API
    - Micromedex
    - Clinical decision support systems

    For MVP, using curated high-risk interactions.
    """

    def __init__(self):
        self.interactions = self._load_interaction_database()

    def _load_interaction_database(self) -> List[DrugInteraction]:
        """
        Load known dangerous drug interactions.

        Sources:
        - FDA Drug Safety Communications
        - American Heart Association guidelines
        - American Diabetes Association guidelines
        """

        return [
            # CRITICAL: Anticoagulant + Antiplatelet
            DrugInteraction(
                drug_a="warfarin",
                drug_b="aspirin",
                severity=InteractionSeverity.CRITICAL,
                mechanism="Both inhibit clotting - additive bleeding risk",
                effect="Severe bleeding, hemorrhage, potential death",
                alternative="Acetaminophen for pain (non-antiplatelet)",
                references=[
                    "FDA Safety Alert 2014",
                    "AHA Stroke Prevention Guidelines"
                ]
            ),

            # CRITICAL: Beta-blocker + Insulin
            DrugInteraction(
                drug_a="metoprolol",
                drug_b="insulin",
                severity=InteractionSeverity.CRITICAL,
                mechanism="Beta-blockers mask hypoglycemia symptoms (tachycardia, tremor)",
                effect="Patient can't detect low blood sugar - seizures, coma, death",
                alternative="Cardioselective beta-blockers (atenolol) with close glucose monitoring",
                references=[
                    "ADA Diabetes Management Guidelines",
                    "JAMA 2018 - Beta-blockers in Diabetes"
                ]
            ),

            # CRITICAL: MAOI + SSRI
            DrugInteraction(
                drug_a="phenelzine",
                drug_b="fluoxetine",
                severity=InteractionSeverity.CRITICAL,
                mechanism="Serotonin syndrome - excessive serotonin accumulation",
                effect="Hyperthermia, seizures, altered mental status, death",
                alternative="Wait 2-5 weeks between discontinuing MAOI and starting SSRI",
                references=[
                    "FDA Black Box Warning",
                    "Serotonin Syndrome Review - Mayo Clinic"
                ]
            ),

            # HIGH: ACE Inhibitor + Potassium Supplement
            DrugInteraction(
                drug_a="lisinopril",
                drug_b="potassium",
                severity=InteractionSeverity.HIGH,
                mechanism="ACE inhibitors reduce potassium excretion",
                effect="Hyperkalemia - cardiac arrhythmias, muscle weakness",
                alternative="Monitor potassium levels, avoid supplements unless deficient",
                references=[
                    "UpToDate - ACE Inhibitor Adverse Effects",
                    "Hyperkalemia Management Guidelines"
                ]
            ),

            # HIGH: Statin + Fibrate
            DrugInteraction(
                drug_a="atorvastatin",
                drug_b="gemfibrozil",
                severity=InteractionSeverity.HIGH,
                mechanism="Both affect muscle metabolism - additive myopathy risk",
                effect="Rhabdomyolysis - muscle breakdown, kidney failure",
                alternative="Fenofibrate (safer fibrate) or ezetimibe",
                references=[
                    "FDA Statin Safety Review",
                    "ACC/AHA Cholesterol Guidelines"
                ]
            ),

            # HIGH: Digoxin + Amiodarone
            DrugInteraction(
                drug_a="digoxin",
                drug_b="amiodarone",
                severity=InteractionSeverity.HIGH,
                mechanism="Amiodarone inhibits P-glycoprotein - increases digoxin levels",
                effect="Digoxin toxicity - arrhythmias, nausea, confusion",
                alternative="Reduce digoxin dose by 50% if amiodarone added",
                references=[
                    "Clinical Pharmacokinetics Review",
                    "Digoxin Dosing Guidelines"
                ]
            ),

            # MODERATE: Levothyroxine + Iron
            DrugInteraction(
                drug_a="levothyroxine",
                drug_b="iron",
                severity=InteractionSeverity.MODERATE,
                mechanism="Iron chelates thyroid hormone - reduces absorption",
                effect="Hypothyroidism symptoms return - fatigue, weight gain",
                alternative="Separate doses by 4+ hours",
                references=[
                    "Thyroid Medication Interactions",
                    "Endocrine Society Guidelines"
                ]
            ),

            # MODERATE: Calcium Channel Blocker + Grapefruit
            DrugInteraction(
                drug_a="amlodipine",
                drug_b="grapefruit",
                severity=InteractionSeverity.MODERATE,
                mechanism="Grapefruit inhibits CYP3A4 - increases drug levels",
                effect="Excessive blood pressure lowering - dizziness, fainting",
                alternative="Avoid grapefruit or use alternative BP medication",
                references=[
                    "FDA Grapefruit Interaction Warning",
                    "Clinical Pharmacology Review"
                ]
            ),

            # PENICILLIN ALLERGY GROUP (critical for Dark Trace demo)
            DrugInteraction(
                drug_a="penicillin_allergy",
                drug_b="amoxicillin",
                severity=InteractionSeverity.CRITICAL,
                mechanism="Amoxicillin IS a penicillin derivative",
                effect="Anaphylaxis - airway closure, shock, death",
                alternative="Azithromycin, doxycycline, or fluoroquinolone",
                references=[
                    "AAAAI Penicillin Allergy Guidelines",
                    "FDA Antibiotic Safety Review"
                ]
            ),

            DrugInteraction(
                drug_a="penicillin_allergy",
                drug_b="ampicillin",
                severity=InteractionSeverity.CRITICAL,
                mechanism="Ampicillin IS a penicillin derivative",
                effect="Anaphylaxis - airway closure, shock, death",
                alternative="Azithromycin, doxycycline, or fluoroquinolone",
                references=[
                    "AAAAI Penicillin Allergy Guidelines"
                ]
            ),

            DrugInteraction(
                drug_a="penicillin_allergy",
                drug_b="cephalexin",
                severity=InteractionSeverity.HIGH,
                mechanism="Cephalosporins have ~10% cross-reactivity with penicillin",
                effect="Possible allergic reaction (lower risk than penicillin)",
                alternative="Azithromycin (no cross-reactivity) or skin test first",
                references=[
                    "AAAAI Cephalosporin Cross-Reactivity Review"
                ]
            ),
        ]

    def check_interaction(
        self,
        drug_a: str,
        drug_b: str
    ) -> Optional[DrugInteraction]:
        """
        Check if two drugs have a known interaction.

        Args:
            drug_a: First medication (or allergy)
            drug_b: Second medication

        Returns:
            DrugInteraction if found, None otherwise
        """
        # Normalize drug names (lowercase, strip whitespace)
        drug_a = drug_a.lower().strip()
        drug_b = drug_b.lower().strip()

        # Check both orderings (A+B and B+A)
        for interaction in self.interactions:
            if ((interaction.drug_a == drug_a and interaction.drug_b == drug_b) or
                (interaction.drug_a == drug_b and interaction.drug_b == drug_a)):
                return interaction

        return None

    def check_medication_list(
        self,
        medications: List[str],
        allergies: List[str] = None
    ) -> List[DrugInteraction]:
        """
        Check all pairwise interactions in a medication list.

        Args:
            medications: List of current medications
            allergies: List of known drug allergies

        Returns:
            List of all detected interactions, sorted by severity
        """
        interactions = []

        # Check drug-drug interactions
        for i, drug_a in enumerate(medications):
            for drug_b in medications[i+1:]:
                interaction = self.check_interaction(drug_a, drug_b)
                if interaction:
                    interactions.append(interaction)

        # Check allergy-drug interactions
        if allergies:
            for allergy in allergies:
                # Convert "allergic to penicillin" → "penicillin_allergy"
                allergy_key = f"{allergy.lower().strip()}_allergy"

                for drug in medications:
                    interaction = self.check_interaction(allergy_key, drug)
                    if interaction:
                        interactions.append(interaction)

        # Sort by severity (CRITICAL first)
        severity_order = {
            InteractionSeverity.CRITICAL: 0,
            InteractionSeverity.HIGH: 1,
            InteractionSeverity.MODERATE: 2,
            InteractionSeverity.LOW: 3,
            InteractionSeverity.NONE: 4
        }
        interactions.sort(key=lambda x: severity_order[x.severity])

        return interactions

    def generate_alert(self, interaction: DrugInteraction) -> str:
        """
        Generate human-readable alert message.

        Returns:
            Formatted alert string
        """
        severity_emoji = {
            InteractionSeverity.CRITICAL: "[CRITICAL]",
            InteractionSeverity.HIGH: "[HIGH RISK]",
            InteractionSeverity.MODERATE: "[MODERATE]",
            InteractionSeverity.LOW: "[LOW RISK]",
        }

        alert = f"""
{severity_emoji[interaction.severity]} DRUG INTERACTION DETECTED

Drugs: {interaction.drug_a.upper()} + {interaction.drug_b.upper()}

Mechanism: {interaction.mechanism}

Effect: {interaction.effect}

Recommendation: {interaction.alternative}

References:
{chr(10).join(f"  - {ref}" for ref in interaction.references)}
"""
        return alert.strip()


def demo_interaction_detection():
    """Demonstrate interaction detection on real scenarios"""

    print("\n" + "="*70)
    print("DARK TRACE: Medication Interaction Detection Demo")
    print("="*70)

    detector = MedicationInteractionDetector()

    # Test scenarios (from ER practice)
    scenarios = [
        {
            "name": "Scenario 1: Warfarin + Aspirin (CRITICAL)",
            "medications": ["warfarin", "aspirin"],
            "allergies": [],
            "context": "68F with atrial fibrillation, now with headache"
        },
        {
            "name": "Scenario 2: Penicillin Allergy + Amoxicillin (CRITICAL)",
            "medications": ["amoxicillin"],
            "allergies": ["penicillin"],
            "context": "42M with pneumonia, documented penicillin allergy"
        },
        {
            "name": "Scenario 3: Multiple Medications (diabetic patient)",
            "medications": ["metoprolol", "insulin", "lisinopril"],
            "allergies": [],
            "context": "55M with diabetes, hypertension, post-MI"
        },
        {
            "name": "Scenario 4: Safe Combination",
            "medications": ["lisinopril", "metformin"],
            "allergies": [],
            "context": "60M with hypertension and diabetes (Type 2)"
        }
    ]

    all_results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {scenario['name']}")
        print("-" * 70)
        print(f"Context: {scenario['context']}")
        print(f"Medications: {', '.join(scenario['medications'])}")
        if scenario['allergies']:
            print(f"Allergies: {', '.join(scenario['allergies'])}")
        print()

        # Check interactions
        interactions = detector.check_medication_list(
            scenario['medications'],
            scenario['allergies']
        )

        if interactions:
            print(f"[ALERT] Found {len(interactions)} interaction(s):\n")
            for interaction in interactions:
                print(detector.generate_alert(interaction))
                print()
        else:
            print("[OK] No known interactions detected")

        all_results.append({
            "scenario": scenario['name'],
            "interactions": [i.to_dict() for i in interactions]
        })

    # Export results
    results_file = "interaction_detection_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, indent=2, fp=f)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_interactions = sum(len(r['interactions']) for r in all_results)
    critical = sum(1 for r in all_results for i in r['interactions'] if i['severity'] == 'critical')
    high = sum(1 for r in all_results for i in r['interactions'] if i['severity'] == 'high')

    print(f"\nScenarios tested: {len(scenarios)}")
    print(f"Total interactions found: {total_interactions}")
    print(f"  - CRITICAL: {critical}")
    print(f"  - HIGH: {high}")
    print(f"\n[OK] Results exported to: {results_file}")

    print("\n" + "="*70)
    print("PRODUCTION INTEGRATION")
    print("="*70)
    print("""
This detector integrates with HoloLoom in 3 ways:

1. Safety Guardrails:
   - Block CRITICAL interactions before prescription
   - Require human review for HIGH interactions
   - Log all checks to audit trail

2. Knowledge Graph:
   - Store interactions as typed edges
   - drug_a --[INTERACTS_WITH]--> drug_b
   - Enable multi-hop reasoning (A+B+C interactions)

3. Real-time Alerts:
   - Check every new medication against patient's current meds
   - Alert before writing prescription
   - Suggest safe alternatives

Next Steps:
  1. Integrate with HoloLoom SafetyGuardrails
  2. Add full drug database (DrugBank API)
  3. Build interaction knowledge graph
  4. Deploy to ER clinical decision support
    """)


if __name__ == "__main__":
    demo_interaction_detection()
