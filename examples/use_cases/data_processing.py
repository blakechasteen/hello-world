#!/usr/bin/env python3
"""
Data Processing Use Case
========================
ETL pipeline with validation and quality checks.

This demo shows:
- Data extraction prompts for various sources
- Transformation and normalization
- Validation and quality checks
- Error handling and recovery
- Pipeline monitoring

Requirements:
    pip install promptly

Usage:
    python examples/use_cases/data_processing.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from Promptly.promptly.promptly import Promptly


class DataProcessingDemo:
    """Data processing pipeline demo"""

    def __init__(self):
        self.demo_dir = Path(__file__).parent / 'data_processing_output'
        self.demo_dir.mkdir(exist_ok=True)
        self.promptly = None
        self.processing_stats = {
            'records_processed': 0,
            'records_failed': 0,
            'validations_passed': 0,
            'validations_failed': 0
        }

    def setup(self):
        """Initialize repository"""
        print("=" * 80)
        print("DATA PROCESSING PIPELINE DEMO".center(80))
        print("=" * 80)

        repo_path = self.demo_dir / 'etl_repo'
        repo_path.mkdir(exist_ok=True)
        self.promptly = Promptly(root_dir=str(repo_path))

        if not (repo_path / '.promptly').exists():
            self.promptly.init()
            print("✓ Initialized repository")

    def create_etl_prompts(self):
        """Create ETL pipeline prompts"""
        print("\n[1/6] Creating ETL Pipeline Prompts")
        print("-" * 80)

        prompts = {
            'extract_csv': '''Extract and parse data from CSV content:

{csv_content}

Requirements:
- Parse headers and data rows
- Handle quoted fields
- Detect data types
- Identify missing values
- Report malformed rows

Output structured JSON with:
- headers (list)
- rows (list of dicts)
- metadata (row_count, column_count, data_types)
- issues (list of problems found)''',

            'extract_json': '''Extract data from JSON structure:

{json_content}

Schema: {expected_schema}

Tasks:
- Validate against schema
- Extract nested values
- Flatten structure if needed
- Handle missing keys
- Type coercion

Output normalized data with validation report.''',

            'transform_normalize': '''Normalize this data:

{raw_data}

Normalization rules:
- {normalization_rules}

Apply:
1. Standardize date formats to ISO 8601
2. Convert currencies to {base_currency}
3. Normalize phone numbers to {phone_format}
4. Standardize country codes
5. Clean whitespace and special characters
6. Handle null/empty values per: {null_handling}

Output transformed data with change log.''',

            'validate_data': '''Validate data quality:

{data}

Validation rules:
{validation_rules}

Check:
1. **Completeness**
   - Required fields present
   - No empty values in critical columns
   - Reference integrity

2. **Accuracy**
   - Value ranges
   - Format compliance
   - Type correctness

3. **Consistency**
   - Cross-field validation
   - Business rule compliance
   - Duplicate detection

4. **Timeliness**
   - Date ranges
   - Staleness checks

Output:
- Pass/fail status
- Error details with record IDs
- Data quality score (0-100)
- Remediation suggestions''',

            'deduplicate': '''Identify and handle duplicates:

{data}

Deduplication strategy: {strategy}
Match threshold: {threshold}

Analyze:
1. Exact matches
2. Fuzzy matches (names, addresses)
3. Partial matches

For each duplicate group:
- Confidence score
- Recommended master record
- Fields to merge
- Conflicting values

Strategy options:
- keep_first: Keep first occurrence
- keep_last: Keep most recent
- merge: Combine all fields
- manual: Flag for review''',

            'enrich_data': '''Enrich data with additional information:

{data}

Enrichment sources:
{enrichment_sources}

Add:
1. **Geographic Data**
   - Coordinates from addresses
   - Timezone
   - Region/country info

2. **Demographic Data**
   - Population statistics
   - Economic indicators

3. **Derived Fields**
   - Age from birthdate
   - Tenure from start_date
   - Category from description

4. **External Data**
   - API lookups
   - Reference data joins

Output enriched records with provenance tracking.'''
        }

        for name, content in prompts.items():
            self.promptly.save(
                name=f'etl_{name}',
                content=content,
                metadata={'pipeline_stage': name.split('_')[0]}
            )
            print(f"✓ Created: etl_{name}")

    def extract_phase(self):
        """Extract data from various sources"""
        print("\n[2/6] Extraction Phase")
        print("-" * 80)

        # Simulate CSV extraction
        csv_data = '''name,email,age,country,signup_date
John Doe,john@example.com,32,USA,2024-01-15
Jane Smith,jane@example.com,28,UK,2024-01-16
Bob Johnson,bob@example.com,45,Canada,2024-01-17
'''

        print("Extracting CSV data...")
        result = {
            'headers': ['name', 'email', 'age', 'country', 'signup_date'],
            'row_count': 3,
            'issues': []
        }

        self.promptly.eval_prompt(
            name='etl_extract_csv',
            test_case=json.dumps({'csv_content': csv_data}),
            score=0.95
        )

        self.processing_stats['records_processed'] += result['row_count']
        print(f"✓ Extracted {result['row_count']} records from CSV")

        # Simulate JSON extraction
        json_data = {
            'users': [
                {'id': 1, 'name': 'Alice', 'metadata': {'role': 'admin'}},
                {'id': 2, 'name': 'Bob', 'metadata': {'role': 'user'}}
            ]
        }

        print("\nExtracting JSON data...")
        self.promptly.eval_prompt(
            name='etl_extract_json',
            test_case=json.dumps({'json_content': json.dumps(json_data)}),
            score=0.92
        )

        self.processing_stats['records_processed'] += 2
        print(f"✓ Extracted 2 records from JSON")

    def transform_phase(self):
        """Transform and normalize data"""
        print("\n[3/6] Transformation Phase")
        print("-" * 80)

        sample_data = [
            {'name': '  John Doe  ', 'phone': '555-1234', 'date': '01/15/2024'},
            {'name': 'JANE SMITH', 'phone': '(555) 5678', 'date': '2024-01-16'},
        ]

        normalization_rules = {
            'names': 'title_case',
            'phones': 'international_format',
            'dates': 'iso_8601'
        }

        print("Normalizing data...")
        self.promptly.eval_prompt(
            name='etl_transform_normalize',
            test_case=json.dumps({
                'raw_data': sample_data,
                'normalization_rules': normalization_rules
            }),
            score=0.88
        )

        print("✓ Normalized 2 records")
        print("  - Standardized names to title case")
        print("  - Converted phones to international format")
        print("  - Standardized dates to ISO 8601")

    def validate_phase(self):
        """Validate data quality"""
        print("\n[4/6] Validation Phase")
        print("-" * 80)

        validation_rules = {
            'email': {'required': True, 'format': 'email'},
            'age': {'required': True, 'min': 0, 'max': 150},
            'country': {'required': True, 'enum': ['USA', 'UK', 'Canada']},
            'signup_date': {'required': True, 'format': 'iso_date'}
        }

        test_data = [
            {'email': 'valid@example.com', 'age': 25, 'country': 'USA', 'signup_date': '2024-01-15'},
            {'email': 'invalid', 'age': 200, 'country': 'XYZ', 'signup_date': 'bad-date'},  # Invalid
            {'email': 'good@example.com', 'age': 30, 'country': 'UK', 'signup_date': '2024-01-16'},
        ]

        print("Running validation...")
        validation_results = []

        for i, record in enumerate(test_data):
            issues = []

            # Email validation
            if '@' not in record.get('email', ''):
                issues.append(f"Invalid email format")

            # Age validation
            age = record.get('age', 0)
            if age < 0 or age > 150:
                issues.append(f"Age out of range: {age}")

            # Country validation
            if record.get('country') not in ['USA', 'UK', 'Canada']:
                issues.append(f"Invalid country: {record.get('country')}")

            passed = len(issues) == 0

            if passed:
                self.processing_stats['validations_passed'] += 1
            else:
                self.processing_stats['validations_failed'] += 1

            validation_results.append({
                'record_id': i,
                'passed': passed,
                'issues': issues
            })

            status = "✓" if passed else "✗"
            print(f"  Record {i}: {status} ({'PASS' if passed else 'FAIL'})")
            for issue in issues:
                print(f"    - {issue}")

        self.promptly.eval_prompt(
            name='etl_validate_data',
            test_case=json.dumps({'data': test_data, 'validation_rules': validation_rules}),
            score=0.67  # 2 out of 3 passed
        )

    def deduplication_phase(self):
        """Deduplicate records"""
        print("\n[5/6] Deduplication Phase")
        print("-" * 80)

        sample_data = [
            {'id': 1, 'name': 'John Doe', 'email': 'john@example.com'},
            {'id': 2, 'name': 'John Doe', 'email': 'john@example.com'},  # Exact duplicate
            {'id': 3, 'name': 'J. Doe', 'email': 'john@example.com'},     # Fuzzy match
            {'id': 4, 'name': 'Jane Smith', 'email': 'jane@example.com'},
        ]

        print("Detecting duplicates...")

        duplicates_found = [
            {
                'group': [1, 2],
                'match_type': 'exact',
                'confidence': 1.0,
                'master': 1
            },
            {
                'group': [1, 3],
                'match_type': 'fuzzy',
                'confidence': 0.85,
                'master': 1
            }
        ]

        for dup in duplicates_found:
            print(f"\n  Duplicate Group: {dup['group']}")
            print(f"    Match Type: {dup['match_type']}")
            print(f"    Confidence: {dup['confidence']:.2f}")
            print(f"    Master Record: {dup['master']}")

        self.promptly.eval_prompt(
            name='etl_deduplicate',
            test_case=json.dumps({'data': sample_data, 'strategy': 'keep_first'}),
            score=0.90
        )

        print(f"\n✓ Found {len(duplicates_found)} duplicate groups")
        print(f"  - {len([d for d in duplicates_found if d['match_type'] == 'exact'])} exact matches")
        print(f"  - {len([d for d in duplicates_found if d['match_type'] == 'fuzzy'])} fuzzy matches")

    def enrichment_phase(self):
        """Enrich data"""
        print("\n[6/6] Enrichment Phase")
        print("-" * 80)

        sample_data = [
            {'name': 'John Doe', 'city': 'New York', 'birthdate': '1990-05-15'},
            {'name': 'Jane Smith', 'city': 'London', 'birthdate': '1985-08-22'},
        ]

        enrichment_sources = {
            'geocoding': 'enabled',
            'demographics': 'enabled',
            'derived_fields': 'enabled'
        }

        print("Enriching records...")

        enriched_data = [
            {
                **sample_data[0],
                'coordinates': {'lat': 40.7128, 'lon': -74.0060},
                'timezone': 'America/New_York',
                'country': 'USA',
                'age': 34
            },
            {
                **sample_data[1],
                'coordinates': {'lat': 51.5074, 'lon': -0.1278},
                'timezone': 'Europe/London',
                'country': 'UK',
                'age': 39
            }
        ]

        self.promptly.eval_prompt(
            name='etl_enrich_data',
            test_case=json.dumps({
                'data': sample_data,
                'enrichment_sources': enrichment_sources
            }),
            score=0.93
        )

        print(f"✓ Enriched {len(sample_data)} records")
        print("  - Added geographic coordinates")
        print("  - Added timezone information")
        print("  - Calculated age from birthdate")

    def generate_pipeline_report(self):
        """Generate pipeline execution report"""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION REPORT")
        print("=" * 80)

        report = {
            'execution_time': datetime.now().isoformat(),
            'stages_completed': 6,
            'statistics': self.processing_stats,
            'quality_metrics': {
                'extraction_success_rate': 1.0,
                'transformation_success_rate': 1.0,
                'validation_success_rate': 0.67,
                'deduplication_rate': 0.33,
                'enrichment_coverage': 1.0
            },
            'recommendations': [
                'Implement automated data quality monitoring',
                'Add real-time validation alerts',
                'Create data lineage tracking',
                'Set up duplicate detection rules',
                'Establish data quality SLAs'
            ]
        }

        report_path = self.demo_dir / 'etl_pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nStatistics:")
        for key, value in self.processing_stats.items():
            print(f"  {key.replace('_', ' ').title():<25}: {value}")

        print(f"\nQuality Metrics:")
        for key, value in report['quality_metrics'].items():
            print(f"  {key.replace('_', ' ').title():<30}: {value:.1%}")

        print(f"\n✓ Report saved: {report_path}")

    def run(self):
        """Run complete demo"""
        try:
            self.setup()
            self.create_etl_prompts()
            self.extract_phase()
            self.transform_phase()
            self.validate_phase()
            self.deduplication_phase()
            self.enrichment_phase()
            self.generate_pipeline_report()

            print("\n" + "=" * 80)
            print("✓ DATA PROCESSING DEMO COMPLETED")
            print("=" * 80)
            return True
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    demo = DataProcessingDemo()
    success = demo.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
