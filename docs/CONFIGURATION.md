# BossPig Configuration Guide

**Complete reference for customizing BossPig detection rules**

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration Files](#configuration-files)
3. [Jargon Dictionary](#jargon-dictionary)
4. [Brand Guidelines](#brand-guidelines)
5. [Governance Requirements](#governance-requirements)
6. [Document Types](#document-types)
7. [Detection Thresholds](#detection-thresholds)
8. [Advanced Configuration](#advanced-configuration)
9. [Environment Variables](#environment-variables)
10. [Configuration Examples](#configuration-examples)

---

## Overview

BossPig supports extensive customization through JSON configuration files. This guide covers all configuration options and best practices.

### Configuration Hierarchy

Configuration is loaded in this order (later overrides earlier):

1. **Built-in defaults** - Shipped with BossPig
2. **Global config** - `~/.config/bosspig/config.json`
3. **Project config** - `.bosspig/config.json` in project root
4. **Command-line flags** - Highest priority

### Quick Start

Create a project-specific configuration:

```bash
mkdir .bosspig
touch .bosspig/jargon_dict.json
touch .bosspig/brand_config.json
touch .bosspig/governance_config.json
```

---

## Configuration Files

### 1. Jargon Dictionary

**File**: `jargon_dict.json`
**Purpose**: Define corporate jargon and preferred alternatives

**Location Priority**:
1. `.bosspig/jargon_dict.json` (project-specific)
2. `~/.config/bosspig/jargon_dict.json` (global)
3. Built-in dictionary (default)

**Structure**:

```json
{
  "version": "1.0",
  "patterns": [
    {
      "pattern": "synergize",
      "severity": "warning",
      "replacement": "combine",
      "category": "buzzword",
      "context": "general"
    },
    {
      "pattern": "low-hanging fruit",
      "severity": "warning",
      "replacement": "easy wins",
      "category": "cliche",
      "context": "general"
    },
    {
      "pattern": "move the needle",
      "severity": "warning",
      "replacement": "make progress",
      "category": "cliche",
      "context": "general"
    }
  ],
  "exclusions": [
    "synergy (when referring to muscle synergy in medical docs)",
    "leverage (financial term in finance docs)"
  ]
}
```

**Fields**:

- `pattern` (string, required): Regex pattern to match
- `severity` (string, required): "info", "warning", "error", "critical"
- `replacement` (string, required): Suggested alternative
- `category` (string, optional): "buzzword", "cliche", "filler", "redundant"
- `context` (string, optional): "general", "technical", "marketing", "legal"

**Example: Custom Jargon**:

```json
{
  "version": "1.0",
  "patterns": [
    {
      "pattern": "utilize",
      "severity": "warning",
      "replacement": "use",
      "category": "unnecessary_formality"
    },
    {
      "pattern": "at this point in time",
      "severity": "warning",
      "replacement": "now",
      "category": "redundant"
    },
    {
      "pattern": "due to the fact that",
      "severity": "warning",
      "replacement": "because",
      "category": "redundant"
    }
  ]
}
```

---

### 2. Brand Guidelines

**File**: `brand_config.json`
**Purpose**: Enforce brand-specific language rules

**Structure**:

```json
{
  "version": "1.0",
  "brand_name": "TechCorp",
  "capitalization_rules": [
    {
      "term": "TechCorp",
      "correct": "TechCorp",
      "incorrect": ["Techcorp", "techcorp", "TECHCORP"],
      "severity": "error"
    },
    {
      "term": "CloudSync",
      "correct": "CloudSync",
      "incorrect": ["Cloudsync", "cloudsync", "Cloud Sync"],
      "severity": "error"
    }
  ],
  "prohibited_terms": [
    {
      "term": "best-in-class",
      "reason": "Overused marketing claim without evidence",
      "severity": "warning"
    },
    {
      "term": "world-class",
      "reason": "Vague superlative",
      "severity": "warning"
    },
    {
      "term": "cheap",
      "reason": "Brand guideline: Use 'affordable' or 'cost-effective'",
      "replacement": "affordable",
      "severity": "error"
    }
  ],
  "preferred_terminology": [
    {
      "deprecated": "customer",
      "preferred": "client",
      "reason": "Brand voice: We use 'client' in all materials",
      "severity": "warning"
    },
    {
      "deprecated": "software",
      "preferred": "platform",
      "reason": "Marketing guideline: Emphasize platform capabilities",
      "severity": "info"
    }
  ],
  "tone_guidelines": {
    "voice": "professional_friendly",
    "avoid": ["overly_casual", "slang", "emojis"],
    "encourage": ["clarity", "specificity", "actionable_language"]
  }
}
```

**Fields**:

**Capitalization Rules**:
- `term`: Official brand name
- `correct`: Correct capitalization
- `incorrect`: List of common misspellings/miscapitalizations
- `severity`: "info", "warning", "error", "critical"

**Prohibited Terms**:
- `term`: Word/phrase to flag
- `reason`: Why it's prohibited
- `replacement` (optional): Suggested alternative
- `severity`: Issue severity

**Preferred Terminology**:
- `deprecated`: Old term to avoid
- `preferred`: New term to use
- `reason`: Explanation
- `severity`: Issue severity

**Example: Healthcare Brand**:

```json
{
  "version": "1.0",
  "brand_name": "HealthPlus",
  "capitalization_rules": [
    {
      "term": "HealthPlus",
      "correct": "HealthPlus",
      "incorrect": ["Healthplus", "healthplus", "Health Plus"],
      "severity": "error"
    }
  ],
  "prohibited_terms": [
    {
      "term": "cure",
      "reason": "FDA compliance: Cannot claim to cure diseases",
      "replacement": "treat",
      "severity": "critical"
    },
    {
      "term": "guaranteed",
      "reason": "Legal: Cannot guarantee medical outcomes",
      "severity": "critical"
    }
  ],
  "preferred_terminology": [
    {
      "deprecated": "patient",
      "preferred": "member",
      "reason": "Brand voice: Use 'member' to emphasize community",
      "severity": "warning"
    }
  ]
}
```

---

### 3. Governance Requirements

**File**: `governance_config.json`
**Purpose**: Define required sections, disclaimers, and compliance rules

**Structure**:

```json
{
  "version": "1.0",
  "document_types": {
    "technical_documentation": {
      "required_sections": [
        {
          "name": "Overview",
          "pattern": "^#+\\s*(Overview|Introduction|Summary)",
          "severity": "error"
        },
        {
          "name": "Prerequisites",
          "pattern": "^#+\\s*Prerequisites",
          "severity": "warning"
        }
      ],
      "required_disclaimers": [],
      "approval_workflow": {
        "required": false
      }
    },
    "data_policies": {
      "required_sections": [
        {
          "name": "Data Classification",
          "pattern": "^#+\\s*Data Classification",
          "severity": "critical"
        },
        {
          "name": "Access Controls",
          "pattern": "^#+\\s*Access Control",
          "severity": "critical"
        }
      ],
      "required_disclaimers": [
        {
          "pattern": "(?i)(SOC\\s*2|SOC2)",
          "name": "SOC2 Compliance Statement",
          "severity": "error"
        },
        {
          "pattern": "(?i)(GDPR|General Data Protection Regulation)",
          "name": "GDPR Compliance Statement",
          "severity": "error"
        }
      ],
      "approval_workflow": {
        "required": true,
        "approvers": ["security_team", "legal"],
        "metadata_required": ["version", "last_reviewed", "next_review"]
      },
      "compliance_frameworks": ["SOC2", "GDPR", "ISO27001"]
    },
    "healthcare": {
      "required_sections": [
        {
          "name": "HIPAA Compliance",
          "pattern": "^#+\\s*HIPAA",
          "severity": "critical"
        }
      ],
      "required_disclaimers": [
        {
          "pattern": "(?i)HIPAA",
          "name": "HIPAA Disclaimer",
          "severity": "critical"
        }
      ],
      "approval_workflow": {
        "required": true,
        "approvers": ["compliance_officer", "legal"],
        "metadata_required": ["version", "effective_date", "author"]
      },
      "compliance_frameworks": ["HIPAA"]
    }
  },
  "version_control": {
    "required": true,
    "format": "semantic",
    "metadata_fields": ["version", "date", "author", "changelog"]
  }
}
```

**Fields**:

**Document Types**: Key = document type name

**Required Sections**:
- `name`: Section name (for user display)
- `pattern`: Regex to match section heading
- `severity`: Issue severity if missing

**Required Disclaimers**:
- `pattern`: Regex to match disclaimer text
- `name`: Disclaimer name (for user display)
- `severity`: Issue severity if missing

**Approval Workflow**:
- `required`: Whether approval workflow is mandatory
- `approvers`: List of required approver roles
- `metadata_required`: List of required metadata fields

**Compliance Frameworks**: List of applicable frameworks (SOC2, GDPR, HIPAA, ISO27001, etc.)

**Example: Finance Industry**:

```json
{
  "version": "1.0",
  "document_types": {
    "financial_policies": {
      "required_sections": [
        {
          "name": "Risk Assessment",
          "pattern": "^#+\\s*Risk",
          "severity": "critical"
        },
        {
          "name": "Audit Trail",
          "pattern": "^#+\\s*Audit",
          "severity": "critical"
        }
      ],
      "required_disclaimers": [
        {
          "pattern": "(?i)(SOX|Sarbanes-Oxley)",
          "name": "SOX Compliance Statement",
          "severity": "critical"
        }
      ],
      "approval_workflow": {
        "required": true,
        "approvers": ["cfo", "legal", "compliance"],
        "metadata_required": ["version", "effective_date", "expiration_date"]
      },
      "compliance_frameworks": ["SOX", "SOC2"]
    }
  }
}
```

---

## Document Types

BossPig supports different document types with customized governance rules.

### Built-in Types

1. **technical_documentation** (default)
   - Minimal governance requirements
   - Focus on clarity and completeness
   - No required disclaimers

2. **data_policies**
   - SOC2, GDPR compliance checks
   - Required sections: Data Classification, Access Controls
   - Approval workflow required

3. **healthcare**
   - HIPAA compliance checks
   - Required HIPAA disclaimer
   - Strict approval workflow

### Creating Custom Types

Add to `governance_config.json`:

```json
{
  "document_types": {
    "engineering_rfc": {
      "required_sections": [
        {"name": "Problem Statement", "pattern": "^#+\\s*Problem", "severity": "error"},
        {"name": "Proposed Solution", "pattern": "^#+\\s*(Solution|Proposal)", "severity": "error"},
        {"name": "Alternatives Considered", "pattern": "^#+\\s*Alternatives", "severity": "warning"},
        {"name": "Testing Plan", "pattern": "^#+\\s*(Testing|Test Plan)", "severity": "error"}
      ],
      "approval_workflow": {
        "required": true,
        "approvers": ["tech_lead", "architect"],
        "metadata_required": ["author", "date", "version"]
      }
    }
  }
}
```

**Usage**:

```python
from bosspig.detector import BossPigDetector

detector = BossPigDetector(document_type="engineering_rfc")
results = detector.analyze("rfc_001.md")
```

---

## Detection Thresholds

Configure detection sensitivity for each category.

### Threshold Configuration

**File**: `.bosspig/thresholds.json`

```json
{
  "version": "1.0",
  "categories": {
    "jargon": {
      "threshold": 0.02,
      "description": "Flag if jargon > 2% of words"
    },
    "passive_voice": {
      "threshold": 0.10,
      "description": "Flag if passive voice > 10% of sentences"
    },
    "vague_quantifiers": {
      "threshold": 0.05,
      "description": "Flag if vague quantifiers > 5% of words"
    }
  },
  "quality_score": {
    "clarity_weight": 0.25,
    "specificity_weight": 0.25,
    "actionability_weight": 0.20,
    "professionalism_weight": 0.15,
    "completeness_weight": 0.15
  },
  "severity_mapping": {
    "info": ["passive_voice", "vague_quantifiers"],
    "warning": ["jargon", "unmeasurable_claims"],
    "error": ["vague_commitments", "missing_dates"],
    "critical": ["ai_hallucinations", "missing_required_sections"]
  }
}
```

---

## Advanced Configuration

### NLP Configuration

Enable advanced NLP features (requires spaCy):

```python
from bosspig.detector import BossPigDetector

detector = BossPigDetector(
    enable_nlp=True,
    nlp_model="en_core_web_sm"  # or "en_core_web_md", "en_core_web_lg"
)
```

**NLP Features**:
- Part-of-speech tagging
- Dependency parsing
- Named entity recognition
- Passive voice detection (improved accuracy)

### Custom Rules

Add custom detection rules:

**File**: `.bosspig/custom_rules.json`

```json
{
  "version": "1.0",
  "rules": [
    {
      "name": "Avoid First Person",
      "pattern": "\\b(I|we|our|my)\\b",
      "category": "tone_violation",
      "severity": "warning",
      "message": "Avoid first-person pronouns in technical documentation",
      "suggestion": "Use imperative or third-person voice"
    },
    {
      "name": "Require Specific Metrics",
      "pattern": "improve\\s+performance(?!.*\\d+%)",
      "category": "unmeasurable_claim",
      "severity": "error",
      "message": "Performance claims must include specific metrics",
      "suggestion": "Add percentage improvement (e.g., 'improve performance by 25%')"
    }
  ]
}
```

**Usage**:

```python
detector = BossPigDetector(
    custom_rules_path=Path(".bosspig/custom_rules.json")
)
```

---

## Environment Variables

Configure BossPig behavior via environment variables:

```bash
# Set log level
export BOSSPIG_LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Set config directory
export BOSSPIG_CONFIG_DIR=/path/to/custom/config

# Enable/disable specific detectors
export BOSSPIG_DISABLE_JARGON=false
export BOSSPIG_DISABLE_PASSIVE_VOICE=false
export BOSSPIG_DISABLE_BRAND_CHECK=false

# Performance tuning
export BOSSPIG_MAX_DOCUMENT_SIZE=100000  # words
export BOSSPIG_TIMEOUT=30  # seconds

# Output formatting
export BOSSPIG_OUTPUT_FORMAT=json  # text, json, html
export BOSSPIG_COLOR_OUTPUT=true
```

---

## Configuration Examples

### Example 1: Marketing Team

**`.bosspig/jargon_dict.json`**:
```json
{
  "version": "1.0",
  "patterns": [
    {"pattern": "game changer", "severity": "error", "replacement": "significant improvement"},
    {"pattern": "best-in-class", "severity": "error", "replacement": "leading"},
    {"pattern": "revolutionary", "severity": "warning", "replacement": "innovative"}
  ]
}
```

**`.bosspig/brand_config.json`**:
```json
{
  "version": "1.0",
  "brand_name": "BrandName",
  "prohibited_terms": [
    {"term": "cheap", "replacement": "affordable", "severity": "error"},
    {"term": "expensive", "replacement": "premium", "severity": "warning"}
  ],
  "tone_guidelines": {
    "voice": "professional_friendly",
    "avoid": ["overly_casual", "slang"]
  }
}
```

### Example 2: Legal Team

**`.bosspig/governance_config.json`**:
```json
{
  "version": "1.0",
  "document_types": {
    "legal_contract": {
      "required_sections": [
        {"name": "Definitions", "pattern": "^#+\\s*Definitions", "severity": "critical"},
        {"name": "Liability", "pattern": "^#+\\s*Liability", "severity": "critical"},
        {"name": "Termination", "pattern": "^#+\\s*Termination", "severity": "critical"}
      ],
      "approval_workflow": {
        "required": true,
        "approvers": ["legal_counsel"],
        "metadata_required": ["version", "effective_date", "parties"]
      }
    }
  }
}
```

### Example 3: Engineering Team

**`.bosspig/custom_rules.json`**:
```json
{
  "version": "1.0",
  "rules": [
    {
      "name": "Require Code Examples",
      "pattern": "(?i)(function|method|class)(?!.*```)",
      "category": "completeness",
      "severity": "warning",
      "message": "Technical documentation should include code examples",
      "suggestion": "Add code block with example"
    }
  ]
}
```

### Example 4: Healthcare Compliance

**`.bosspig/governance_config.json`**:
```json
{
  "version": "1.0",
  "document_types": {
    "hipaa_policy": {
      "required_sections": [
        {"name": "HIPAA Compliance", "pattern": "(?i)HIPAA", "severity": "critical"},
        {"name": "Data Encryption", "pattern": "(?i)encrypt", "severity": "critical"},
        {"name": "Access Controls", "pattern": "(?i)access control", "severity": "critical"}
      ],
      "required_disclaimers": [
        {"pattern": "(?i)HIPAA", "name": "HIPAA Disclaimer", "severity": "critical"}
      ],
      "approval_workflow": {
        "required": true,
        "approvers": ["compliance_officer", "privacy_officer", "legal"],
        "metadata_required": ["version", "effective_date", "review_date", "author"]
      },
      "compliance_frameworks": ["HIPAA"]
    }
  }
}
```

---

## Configuration Best Practices

### 1. Start Simple

Begin with a minimal configuration and add rules as needed:

```json
{
  "version": "1.0",
  "patterns": [
    {"pattern": "TODO", "severity": "warning", "replacement": "Complete implementation"}
  ]
}
```

### 2. Use Severity Appropriately

- **INFO**: Style suggestions, minor improvements
- **WARNING**: Non-blocking issues, should be fixed
- **ERROR**: Blocks publication, must be fixed
- **CRITICAL**: Compliance violations, legal issues

### 3. Provide Clear Suggestions

Always include actionable suggestions:

```json
{
  "pattern": "due to the fact that",
  "severity": "warning",
  "replacement": "because",
  "message": "Use simpler phrasing for clarity"
}
```

### 4. Version Your Configuration

Include version field for backwards compatibility:

```json
{
  "version": "1.0",
  "patterns": [...]
}
```

### 5. Test Configuration Changes

After modifying configuration, test on sample documents:

```bash
python -m bosspig.detector analyze sample.md
```

### 6. Document Custom Rules

Add comments in JSON (use `description` fields):

```json
{
  "pattern": "utilize",
  "severity": "warning",
  "replacement": "use",
  "description": "Marketing guideline: Use simple language"
}
```

---

## Troubleshooting Configuration

### Issue: Configuration Not Loading

**Symptoms**: Changes to config files have no effect

**Solutions**:
1. Check file path priority (project > global > built-in)
2. Verify JSON syntax (use `jsonlint` or Python `json.load()`)
3. Check file permissions (readable by BossPig process)
4. Enable debug logging: `export BOSSPIG_LOG_LEVEL=DEBUG`

### Issue: Regex Patterns Not Matching

**Symptoms**: Expected findings not detected

**Solutions**:
1. Test regex with Python `re.compile(pattern).search(text)`
2. Check for case sensitivity (use `(?i)` for case-insensitive)
3. Escape special characters: `.`, `*`, `+`, `?`, `[`, `]`, `(`, `)`, `{`, `}`, `^`, `$`, `|`, `\`
4. Use raw strings in Python: `r"pattern"`

### Issue: Too Many False Positives

**Symptoms**: Legitimate terms flagged as jargon

**Solutions**:
1. Add exclusions to jargon dictionary
2. Adjust severity thresholds
3. Use context-specific rules
4. Disable specific detectors for certain document types

---

## Next Steps

- **[User Manual](USER_MANUAL.md)** - Complete feature reference
- **[Quick Start Guide](QUICK_START.md)** - Get started in 5 minutes
- **[API Documentation](API_REFERENCE.md)** - Programmatic API reference
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions

---

*Version: 1.0.0 (Beta) | Last Updated: 2025-11-22*
