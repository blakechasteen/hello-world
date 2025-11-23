# DATAPIG: Data Analysis & Tactical Assessment Program for Integrity Governance

**"Computer, run level 1 diagnostic on all data systems."**

DATAPIG is a Star Trek-themed data quality validation system that detects 10 categories of data issues with memorable error messages inspired by Star Trek: The Next Generation.

## Quick Start

```python
from HoloLoom.datapig import DataPigDetector, engage_warp_validation

# Simple validation (one-liner)
safe_for_warp = engage_warp_validation(your_data)

# Detailed analysis
detector = DataPigDetector(enable_verbose=True)
issues = detector.analyze_dataset(your_data)

for issue in issues:
    if issue.severity == Severity.CAPTAIN:
        print(f"RED ALERT: {issue.message}")
```

## 10 Detection Categories

| Category | Star Trek Quote | What It Detects |
|----------|-----------------|-----------------|
| **SCHEMA_DRIFT** | "The laws of physics are different here!" | Type mismatches, missing fields |
| **DATA_LEAK** | "Hull breach on deck 7!" | PII, secrets (emails, SSNs, API keys) |
| **STALE_DATA** | "These readings are from last century!" | Outdated timestamps, zombie records |
| **DUPLICATES** | "We're seeing double, Captain!" | Exact duplicate detection |
| **OUTLIERS** | "Captain, these readings are... impossible!" | Statistical anomalies (IQR method) |
| **INCONSISTENT_FORMAT** | "Universal translator malfunction!" | Mixed date formats, inconsistent casing |
| **MISSING_RELATIONS** | "Transporter lost the signal!" | Broken foreign keys, orphaned records |
| **DISTRIBUTION_SHIFT** | "We've entered an alternate reality!" | Dataset drift, rare values |
| **SAMPLING_BIAS** | "Prime Directive violation detected!" | Class imbalance (>10:1 ratio) |
| **LABEL_NOISE** | "Temporal anomaly - contradictory data!" | Same input, different labels |

## Severity Levels

Issues are classified by Starfleet rank:

- **CAPTAIN** 🚨 - Critical (Red Alert) - needs immediate attention
- **COMMANDER** ⚠️  - High (Yellow Alert) - significant issue
- **LIEUTENANT** ℹ️  - Medium (Shields up) - should be addressed
- **ENSIGN** 📡 - Low (Sensors detect anomaly) - minor issue

## Usage Examples

### Basic Validation

```python
from HoloLoom.datapig import analyze_dataset

data = [
    {"id": 1, "name": "Picard", "email": "picard@enterprise.com"},
    {"id": 2, "name": "Riker"},  # Missing email field
]

issues = analyze_dataset(data)
print(f"Detected {len(issues)} issues")
```

### Warp Drive Pre-Flight Check

```python
from HoloLoom.datapig import engage_warp_validation

# Quick go/no-go check
safe = engage_warp_validation(my_dataset)

if safe:
    # Proceed with ML training, API calls, etc.
    train_model(my_dataset)
else:
    # Fix critical issues first
    print("Warp engines offline! Fix RED ALERTS.")
```

### Detailed Analysis

```python
from HoloLoom.datapig import DataPigDetector, Severity

detector = DataPigDetector(enable_verbose=True)
issues = detector.analyze_dataset(data)

# Filter by severity
red_alerts = [i for i in issues if i.severity == Severity.CAPTAIN]
yellow_alerts = [i for i in issues if i.severity == Severity.COMMANDER]

# Group by issue type
from collections import defaultdict
by_type = defaultdict(list)
for issue in issues:
    by_type[issue.issue_type].append(issue)

print(f"Schema drift issues: {len(by_type[IssueType.SCHEMA_DRIFT])}")
print(f"Data leak issues: {len(by_type[IssueType.DATA_LEAK])}")
```

## Detection Details

### 1. Schema Drift

**What it detects:**
- Missing required fields
- Type mismatches (expected `int`, got `str`)
- Unexpected new fields

**Example:**
```python
data = [
    {"id": 1, "name": "Data"},
    {"id": "2", "name": "Worf"},  # ID should be int!
]
```

**Output:**
```
[LIEUTENANT] SCHEMA_DRIFT
Message: Type mismatch in field 'id': expected int, got str
Location: row_1.id
```

### 2. Data Leaks (PII/Secrets)

**What it detects:**
- Email addresses
- Social Security Numbers (SSNs)
- Credit card numbers
- API keys (32+ character strings)
- Password fields

**Example:**
```python
data = [
    {"user": "Geordi", "api_key": "sk_live_abc123def456..."}  # Leaked!
]
```

**Output:**
```
[CAPTAIN] DATA_LEAK
Message: *** Hull breach detected! Possible api_key exposure in field 'api_key' ***
```

### 3. Stale Data

**What it detects:**
- Timestamps older than 1 year
- Outdated `created_at`, `updated_at`, `last_modified` fields

**Example:**
```python
data = [
    {"sensor": "Warp Core", "last_calibration": "2020-01-01"}  # 4+ years old!
]
```

### 4. Duplicates

**What it detects:**
- Exact row duplicates (hash-based detection)
- Fuzzy duplicates (coming soon)

### 5. Outliers

**What it detects:**
- Statistical anomalies using IQR (Interquartile Range) method
- Values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`

**Example:**
```python
data = [
    {"power": 45},
    {"power": 48},
    {"power": 9999},  # Outlier!
]
```

### 6. Inconsistent Formatting

**What it detects:**
- Mixed date formats (`YYYY-MM-DD` vs `MM/DD/YYYY`)
- Inconsistent casing (`UPPERCASE` vs `lowercase`)
- Phone number formats (US vs International)

**Detected patterns:**
- `ISO_DATE`: `2024-01-15`
- `US_DATE`: `01/15/2024`
- `US_PHONE`: `555-123-4567`
- `INTL_PHONE`: `+1-555-123-4567`
- `UPPERCASE`, `lowercase`, `TitleCase`, `mixed`

### 7. Missing Relations (Broken Foreign Keys)

**What it detects:**
- Foreign key columns (`*_id`, `*_ref`, `*_fk`) pointing to non-existent IDs
- Orphaned records

**Example:**
```python
data = [
    {"mission_id": 1, "ship_id": 1701},
    {"mission_id": 2, "ship_id": 9999},  # ship_id 9999 doesn't exist!
]
```

### 8. Distribution Shift

**What it detects:**
- Rare categorical values (<1% frequency)
- Unexpected values in established categories

### 9. Sampling Bias

**What it detects:**
- Severe class imbalance (>10:1 ratio)
- Minority class representation <10%

**Example:**
```python
data = [
    {"species": "Human", "label": "friendly"} * 100,  # 100 humans
    {"species": "Borg", "label": "hostile"},  # 1 Borg (severe bias!)
]
```

### 10. Label Noise (Contradictions)

**What it detects:**
- Same features, different labels
- Temporal anomalies in ground truth

**Example:**
```python
data = [
    {"scenario": "Borg attack", "crew": "Picard", "label": "hostile"},
    {"scenario": "Borg attack", "crew": "Picard", "label": "friendly"},  # Contradiction!
]
```

## Data Format Support

DATAPIG accepts multiple formats:

```python
# List of dicts
data = [{"id": 1, "name": "Picard"}, {"id": 2, "name": "Riker"}]

# Single dict (auto-wrapped)
data = {"id": 1, "name": "Data"}

# Pandas DataFrame
import pandas as pd
df = pd.DataFrame([{"id": 1, "name": "Worf"}])
```

All formats are normalized to `List[Dict]` internally.

## Star Trek Easter Eggs

### Stardate Calculation

Issues include a "stardate" timestamp using the TNG formula:

```python
stardate = (year - 2323) * 1000 + day_of_year
```

Example: `Stardate: -297674.00` (negative = before 2323)

### Error Messages

All error messages use Star Trek quotes:
- **Schema Drift**: "The laws of physics are different here!" (Spock on parallel universes)
- **Duplicates**: "We're seeing double, Captain!" (Sulu detecting mirror ship)
- **Outliers**: "Captain, these readings are... impossible!" (Spock's scientific skepticism)
- **Data Leaks**: "Hull breach on deck 7!" (Security officer)
- **Sampling Bias**: "Prime Directive violation detected!" (Picard on interference)

## Performance

- **Schema Drift**: O(n) per row
- **Data Leaks**: O(n × m) where m = number of regex patterns
- **Duplicates**: O(n) with hash-based deduplication
- **Outliers**: O(n log n) for IQR calculation
- **Overall**: ~100ms per 1,000 rows (typical dataset)

## Integration with HoloLoom

DATAPIG integrates with the Quality Assurance Department:

```python
from HoloLoom.departments import get_department

qa = get_department("quality_assurance")
result = await qa.process({
    "action": "validate_data",
    "data": my_dataset
})

if result["status"] == "red_alert":
    print(f"Critical issues: {result['issues']}")
```

## Running the Demo

```bash
python demos/demo_datapig.py
```

This demonstrates all 10 detection categories with Star Trek-themed examples.

## Testing

```bash
pytest HoloLoom/datapig/tests/ -v
```

## Roadmap

**Phase 2 (Future)**:
- Fuzzy duplicate detection (Levenshtein distance)
- Custom validation rules
- Auto-fixing for common issues
- Integration with xTerminator for data cleaning
- SQL database validation
- Time-series specific checks

## Credits

**Inspired by:**
- Star Trek: The Next Generation (1987-1994)
- Captain Jean-Luc Picard's commitment to data integrity
- Spock's logical approach to anomaly detection
- Chief O'Brien's "diagnostic scans" philosophy

**Built by:** The HoloLoom Quality Assurance Department

**Version:** 1.0.0-NCC-1701 (USS Enterprise registry)

---

**"Live Long and Prosper!"** 🖖

*Captain's Log, Stardate -297674: DATAPIG has successfully detected all data quality anomalies. Warp engines are cleared for full power. Picard out.*
