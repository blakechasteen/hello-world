"""
Domain Validation Tool
======================

Validates a domain registry for:
1. Schema completeness
2. Pattern correctness (regex)
3. Test phrase coverage
4. Measurement extraction
5. Entity resolution
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from mythRL_core.entity_resolution import EntityRegistry, EntityResolver
from mythRL_core.entity_resolution.extractor import EntityExtractor


class DomainValidator:
    """Validates domain registries for quality and correctness."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.errors = []
        self.warnings = []
        self.passed = []

        # Load raw JSON
        with open(registry_path, 'r') as f:
            self.raw_data = json.load(f)

        # Load registry
        try:
            self.registry = EntityRegistry.load(registry_path)
            self.resolver = EntityResolver(self.registry)
            self.extractor = EntityExtractor(self.resolver, custom_patterns=self.registry.measurement_patterns)
        except Exception as e:
            self.errors.append(f"Failed to load registry: {e}")
            self.registry = None

    def validate_all(self) -> bool:
        """Run all validation checks. Returns True if valid."""
        if not self.registry:
            return False

        self.validate_schema()
        self.validate_entities()
        self.validate_measurement_patterns()
        self.validate_test_phrases()
        self.validate_micropolicies()
        self.validate_reverse_queries()

        return len(self.errors) == 0

    def validate_schema(self):
        """Check required top-level fields."""
        required_fields = ["domain", "version", "entities"]
        for field in required_fields:
            if field not in self.raw_data:
                self.errors.append(f"Missing required field: {field}")
            else:
                self.passed.append(f"Schema: Has required field '{field}'")

        # Check recommended fields
        recommended_fields = ["description", "author", "license", "measurement_patterns"]
        for field in recommended_fields:
            if field not in self.raw_data:
                self.warnings.append(f"Recommended field missing: {field}")

    def validate_entities(self):
        """Check entity definitions for completeness."""
        if not self.registry.entities:
            self.errors.append("No entities defined in domain")
            return

        stats = self.registry.stats()
        self.passed.append(f"Entities: {stats['total_entities']} defined with {stats['total_aliases']} aliases")

        # Check minimum thresholds
        if stats['total_entities'] < 5:
            self.warnings.append(f"Only {stats['total_entities']} entities - recommend at least 5 for useful coverage")

        # Check each entity
        for entity_id, entity in self.registry.entities.items():
            # Check aliases
            if len(entity.aliases) < 2:
                self.warnings.append(f"Entity {entity_id} has only {len(entity.aliases)} aliases - recommend 3+")

            # Check for canonical ID format
            if not re.match(r'^[a-z]+-[a-z0-9-]+$', entity_id):
                self.warnings.append(f"Entity ID '{entity_id}' doesn't follow recommended format: type-descriptive-name")

            # Check entity type is defined
            if not entity.entity_type:
                self.errors.append(f"Entity {entity_id} missing entity_type")

        self.passed.append(f"Entities: All entities have required fields")

    def validate_measurement_patterns(self):
        """Validate regex patterns for measurements."""
        if not self.registry.measurement_patterns:
            self.warnings.append("No measurement patterns defined - domain may not extract structured data")
            return

        numeric_patterns = self.registry.measurement_patterns.get("numeric", {})
        categorical_patterns = self.registry.measurement_patterns.get("categorical", {})

        total_patterns = len(numeric_patterns) + len(categorical_patterns)
        self.passed.append(f"Measurements: {len(numeric_patterns)} numeric + {len(categorical_patterns)} categorical patterns")

        if total_patterns < 2:
            self.warnings.append("Only {total_patterns} measurement pattern(s) - recommend at least 3 for useful extraction")

        # Test regex compilation
        for key, spec in numeric_patterns.items():
            try:
                re.compile(spec["pattern"])
                self.passed.append(f"Pattern '{key}': Valid regex")
            except re.error as e:
                self.errors.append(f"Pattern '{key}': Invalid regex - {e}")

        for key, spec in categorical_patterns.items():
            try:
                re.compile(spec["pattern"])
                self.passed.append(f"Pattern '{key}': Valid regex")
            except re.error as e:
                self.errors.append(f"Pattern '{key}': Invalid regex - {e}")

    def validate_test_phrases(self):
        """Test that validation phrases actually work."""
        validation = self.raw_data.get("validation", {})
        test_phrases = validation.get("test_phrases", [])

        if not test_phrases:
            self.errors.append("No test phrases defined - required for validation")
            return

        if len(test_phrases) < 3:
            self.warnings.append(f"Only {len(test_phrases)} test phrases - recommend at least 5 for good coverage")

        self.passed.append(f"Test Phrases: {len(test_phrases)} defined")

        # Run each test phrase
        entity_coverage = set()
        measurement_coverage = set()

        for i, phrase in enumerate(test_phrases, 1):
            try:
                extracted = self.extractor.extract(phrase)

                # Check entity resolution
                if not extracted.entities:
                    self.warnings.append(f"Test phrase {i} extracted no entities: '{phrase[:60]}...'")
                else:
                    for entity in extracted.entities:
                        entity_coverage.add(entity["canonical_id"])

                # Check measurements
                if not extracted.measurements:
                    self.warnings.append(f"Test phrase {i} extracted no measurements: '{phrase[:60]}...'")
                else:
                    for key in extracted.measurements.keys():
                        measurement_coverage.add(key)

                self.passed.append(f"Test {i}: Extracted {len(extracted.entities)} entities, {len(extracted.measurements)} measurements")

            except Exception as e:
                self.errors.append(f"Test phrase {i} failed: {e}")

        # Check coverage
        total_entities = len(self.registry.entities)
        entity_coverage_pct = (len(entity_coverage) / total_entities * 100) if total_entities > 0 else 0
        self.passed.append(f"Coverage: {len(entity_coverage)}/{total_entities} entities ({entity_coverage_pct:.0f}%) tested")

        if entity_coverage_pct < 50:
            self.warnings.append(f"Test phrases only cover {entity_coverage_pct:.0f}% of entities - recommend 80%+")

    def validate_micropolicies(self):
        """Check micropolicy definitions."""
        micropolicies = self.raw_data.get("micropolicies", [])

        if not micropolicies:
            self.warnings.append("No micropolicies defined - consider adding expert rules")
            return

        self.passed.append(f"Micropolicies: {len(micropolicies)} defined")

        for i, policy in enumerate(micropolicies, 1):
            # Check required fields
            required = ["policy_id", "name", "trigger", "action", "priority"]
            for field in required:
                if field not in policy:
                    self.errors.append(f"Micropolicy {i} missing required field: {field}")

            # Check priority values
            if "priority" in policy and policy["priority"] not in ["high", "medium", "low"]:
                self.warnings.append(f"Micropolicy {i} has invalid priority: {policy['priority']}")

    def validate_reverse_queries(self):
        """Check reverse query definitions."""
        reverse_queries = self.raw_data.get("reverse_queries", [])

        if not reverse_queries:
            self.warnings.append("No reverse queries defined - consider adding diagnostic questions")
            return

        self.passed.append(f"Reverse Queries: {len(reverse_queries)} defined")

        for i, query in enumerate(reverse_queries, 1):
            # Check required fields
            required = ["query_id", "trigger", "question_template", "purpose"]
            for field in required:
                if field not in query:
                    self.errors.append(f"Reverse query {i} missing required field: {field}")

            # Check for placeholders in question template
            if "question_template" in query:
                template = query["question_template"]
                if "{" not in template:
                    self.warnings.append(f"Reverse query {i} question has no placeholders - may be too generic")

    def print_report(self):
        """Print validation report."""
        print("="*80)
        print(f"DOMAIN VALIDATION REPORT: {self.raw_data.get('domain', 'unknown')}")
        print("="*80 + "\n")

        if self.passed:
            print(f"[OK] PASSED ({len(self.passed)} checks):")
            for msg in self.passed:
                print(f"  [OK] {msg}")
            print()

        if self.warnings:
            print(f"[WARN] WARNINGS ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  [WARN] {msg}")
            print()

        if self.errors:
            print(f"[ERROR] ERRORS ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  [ERROR] {msg}")
            print()

        print("="*80)
        if not self.errors:
            print("VALIDATION PASSED [OK]")
            print("This domain is ready for contribution!")
        else:
            print("VALIDATION FAILED [ERROR]")
            print(f"Fix {len(self.errors)} error(s) before submitting.")
        print("="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_domain.py <path_to_registry.json>")
        print()
        print("Examples:")
        print("  python validate_domain.py mythRL_core/domains/beekeeping/registry.json")
        print("  python validate_domain.py mythRL_core/domains/automotive/registry.json")
        sys.exit(1)

    registry_path = Path(sys.argv[1])

    if not registry_path.exists():
        print(f"Error: Registry file not found: {registry_path}")
        sys.exit(1)

    validator = DomainValidator(registry_path)
    is_valid = validator.validate_all()
    validator.print_report()

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
