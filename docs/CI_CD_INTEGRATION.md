# BossPig CI/CD Integration Guide

**Automate BossPig quality checks in your development workflow**

---

## Table of Contents

1. [Overview](#overview)
2. [GitHub Actions](#github-actions)
3. [GitLab CI](#gitlab-ci)
4. [Pre-Commit Hooks](#pre-commit-hooks)
5. [Jenkins](#jenkins)
6. [CircleCI](#circleci)
7. [Exit Codes](#exit-codes)
8. [Configuration Tips](#configuration-tips)
9. [Troubleshooting](#troubleshooting)

---

## Overview

BossPig integrates seamlessly with CI/CD pipelines to provide automated document quality checks. This guide covers integration with popular CI/CD platforms.

### Exit Codes

BossPig CLI uses standard exit codes for CI/CD integration:

- **0**: Success (no errors or critical issues)
- **1**: Errors detected (should be fixed but not blocking)
- **2**: Critical issues detected (must be fixed, blocks merge/deploy)

### Recommended Workflow

1. **Pre-Commit Hooks**: Catch issues locally before pushing
2. **Pull Request Checks**: Run on all PRs, comment with results
3. **Main Branch Protection**: Block merge if critical issues exist
4. **Release Validation**: Final check before package publication

---

## GitHub Actions

### Basic Workflow

Create `.github/workflows/bosspig.yml`:

```yaml
name: BossPig Quality Check

on:
  pull_request:
    paths:
      - '**.md'
      - '**.txt'
      - 'docs/**'

jobs:
  quality-check:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Install BossPig
        run: |
          pip install bosspig

      - name: Run BossPig
        run: |
          python -m bosspig.cli analyze README.md
```

### Advanced Workflow with Reports

```yaml
name: BossPig Quality Check

on:
  pull_request:
    paths:
      - '**.md'
      - '**.txt'

  push:
    branches:
      - main

jobs:
  quality-check:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Install BossPig
        run: |
          pip install bosspig
          # Optional: Install spaCy for advanced NLP
          pip install spacy
          python -m spacy download en_core_web_sm

      - name: Find files to analyze
        id: find-files
        run: |
          find . -type f \( -name "*.md" -o -name "*.txt" \) > files.txt
          echo "Found $(wc -l < files.txt) files"

      - name: Run BossPig analysis
        id: bosspig
        run: |
          mkdir -p reports
          exit_code=0

          while IFS= read -r file; do
            echo "Analyzing: $file"
            if python -m bosspig.cli analyze "$file" \
                --format json \
                --output "reports/$(basename "$file").json"; then
              echo "  ✓ Passed"
            else
              code=$?
              echo "  ✗ Failed (exit: $code)"
              if [ $code -gt $exit_code ]; then
                exit_code=$code
              fi
            fi
          done < files.txt

          echo "exit_code=$exit_code" >> $GITHUB_OUTPUT
          exit $exit_code

      - name: Upload reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: bosspig-reports
          path: reports/
          retention-days: 30

      - name: Comment on PR
        uses: actions/github-script@v6
        if: github.event_name == 'pull_request' && always()
        with:
          script: |
            const exitCode = '${{ steps.bosspig.outputs.exit_code }}';

            let status = '✅ All checks passed';
            let emoji = '✅';

            if (exitCode === '1') {
              status = '⚠️ Some errors detected';
              emoji = '⚠️';
            } else if (exitCode === '2') {
              status = '❌ Critical issues found';
              emoji = '❌';
            }

            const comment = `
            ## ${emoji} BossPig Quality Check

            ${status}

            **Exit code:** ${exitCode}
            **Reports:** Available in workflow artifacts

            ${exitCode === '2' ? '**Action required:** Fix critical issues before merging.' : ''}
            `;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });

      - name: Fail on critical issues
        if: steps.bosspig.outputs.exit_code == '2'
        run: |
          echo "Critical issues detected. Failing workflow."
          exit 1
```

### Matrix Strategy (Multiple Document Types)

```yaml
jobs:
  quality-check:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        document-type:
          - technical_documentation
          - data_policies
          - healthcare

    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install BossPig
        run: pip install bosspig

      - name: Analyze documents
        run: |
          python -m bosspig.cli analyze \
            --document-type ${{ matrix.document-type }} \
            docs/${{ matrix.document-type }}/**/*.md
```

---

## GitLab CI

### Basic Configuration

Create `.gitlab-ci.yml`:

```yaml
stages:
  - quality

bosspig:
  stage: quality
  image: python:3.10
  before_script:
    - pip install bosspig
  script:
    - python -m bosspig.cli analyze README.md
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
      changes:
        - "**/*.md"
        - "**/*.txt"
  allow_failure: false
```

### Advanced Configuration with Artifacts

```yaml
stages:
  - quality
  - report

bosspig-check:
  stage: quality
  image: python:3.10
  before_script:
    - pip install bosspig spacy
    - python -m spacy download en_core_web_sm
  script:
    - mkdir -p reports
    - |
      exit_code=0
      for file in $(find . -name "*.md" -o -name "*.txt"); do
        echo "Analyzing: $file"
        if python -m bosspig.cli analyze "$file" \
            --format json \
            --output "reports/$(basename "$file").json"; then
          echo "  ✓ Passed"
        else
          code=$?
          echo "  ✗ Failed (exit: $code)"
          if [ $code -gt $exit_code ]; then
            exit_code=$code
          fi
        fi
      done
      exit $exit_code
  artifacts:
    paths:
      - reports/
    expire_in: 30 days
    when: always
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'

bosspig-report:
  stage: report
  image: python:3.10
  before_script:
    - pip install bosspig
  script:
    - python -m bosspig.cli analyze README.md --format html --output bosspig_report.html
  artifacts:
    paths:
      - bosspig_report.html
    expire_in: 7 days
  dependencies:
    - bosspig-check
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  when: always
```

---

## Pre-Commit Hooks

### Installation

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. Install hooks:
   ```bash
   pre-commit install
   ```

3. Run manually (optional):
   ```bash
   pre-commit run --all-files
   ```

### Configuration

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      # Check for errors
      - id: bosspig-check
        name: BossPig Quality Check
        entry: python -m bosspig.cli analyze
        language: system
        files: \.(md|txt)$
        pass_filenames: true
        args:
          - --severity=error
          - --no-color
        verbose: true

      # Block commits with critical issues
      - id: bosspig-critical
        name: BossPig Critical Check
        entry: python -m bosspig.cli analyze
        language: system
        files: \.(md|txt)$
        pass_filenames: true
        args:
          - --severity=critical
          - --no-color
```

### Usage

Pre-commit hooks run automatically on `git commit`:

```bash
$ git commit -m "Update README"

BossPig Quality Check...........................Passed
BossPig Critical Check...........................Passed

[main abc1234] Update README
 1 file changed, 10 insertions(+), 5 deletions(-)
```

If issues are found:

```bash
$ git commit -m "Update docs"

BossPig Quality Check...........................Failed
- hook id: bosspig-check
- exit code: 2

ERROR: Critical issues detected in docs/policy.md
Line 42: Missing required HIPAA disclaimer

Suggestion: Add HIPAA compliance disclaimer
```

---

## Jenkins

### Pipeline Configuration

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                sh 'pip install bosspig'
            }
        }

        stage('Quality Check') {
            steps {
                script {
                    def exitCode = sh(
                        script: 'python -m bosspig.cli analyze README.md',
                        returnStatus: true
                    )

                    if (exitCode == 2) {
                        error('Critical issues detected in documentation')
                    } else if (exitCode == 1) {
                        unstable('Errors detected in documentation')
                    }
                }
            }
        }

        stage('Generate Report') {
            steps {
                sh '''
                    python -m bosspig.cli analyze README.md \
                        --format html \
                        --output bosspig_report.html
                '''

                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: '.',
                    reportFiles: 'bosspig_report.html',
                    reportName: 'BossPig Quality Report'
                ])
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'bosspig_report.html', fingerprint: true
        }
    }
}
```

---

## CircleCI

### Configuration

Create `.circleci/config.yml`:

```yaml
version: 2.1

jobs:
  quality-check:
    docker:
      - image: cimg/python:3.10

    steps:
      - checkout

      - run:
          name: Install BossPig
          command: pip install bosspig

      - run:
          name: Run Quality Check
          command: |
            mkdir -p reports
            python -m bosspig.cli analyze README.md \
              --format json \
              --output reports/bosspig.json

      - store_artifacts:
          path: reports/

      - run:
          name: Fail on Critical Issues
          command: |
            # Re-run to get exit code
            python -m bosspig.cli analyze README.md
          when: always

workflows:
  version: 2
  main:
    jobs:
      - quality-check:
          filters:
            branches:
              only:
                - main
                - develop
```

---

## Exit Codes

BossPig uses standard exit codes for CI/CD integration:

### Exit Code 0 - Success

**Meaning**: No critical issues or errors detected

**CI/CD Action**: Allow merge/deployment

**Example**:
```bash
$ python -m bosspig.cli analyze document.md
✓ Analysis complete: 3 findings (all INFO/WARNING)
Overall Quality Score: B (82%)
$ echo $?
0
```

### Exit Code 1 - Errors

**Meaning**: Errors detected (should fix but not blocking)

**CI/CD Action**: Mark as unstable, allow merge with warning

**Example**:
```bash
$ python -m bosspig.cli analyze document.md
✗ Analysis complete: 5 findings (2 ERROR, 3 WARNING)
Overall Quality Score: D (65%)
$ echo $?
1
```

### Exit Code 2 - Critical

**Meaning**: Critical issues (compliance violations, etc.)

**CI/CD Action**: Block merge/deployment

**Example**:
```bash
$ python -m bosspig.cli analyze policy.md
✗ CRITICAL: Missing required HIPAA disclaimer
Overall Quality Score: F (45%)
$ echo $?
2
```

---

## Configuration Tips

### 1. Separate Checks by Severity

Run critical checks first, then non-blocking checks:

```yaml
# GitHub Actions
- name: Critical Issues Check
  run: python -m bosspig.cli analyze docs/ --severity critical

- name: Error Check (allow failure)
  run: python -m bosspig.cli analyze docs/ --severity error
  continue-on-error: true
```

### 2. Filter by File Type

Only analyze relevant documents:

```bash
# Only check markdown files in docs/
find docs/ -name "*.md" -exec python -m bosspig.cli analyze {} \;

# Only check policy documents
python -m bosspig.cli analyze \
  --document-type data_policies \
  policies/**/*.md
```

### 3. Cache Dependencies

Speed up CI runs by caching Python packages:

```yaml
# GitHub Actions
- uses: actions/setup-python@v4
  with:
    python-version: '3.10'
    cache: 'pip'

# GitLab CI
cache:
  paths:
    - .cache/pip
```

### 4. Parallel Execution

Run checks in parallel for faster feedback:

```yaml
# GitLab CI
bosspig-docs:
  script:
    - python -m bosspig.cli analyze docs/
  parallel:
    matrix:
      - DOCUMENT_TYPE: [technical, healthcare, data_policies]
```

---

## Troubleshooting

### Issue: Pre-commit hook too slow

**Solution**: Only check changed files

```yaml
# .pre-commit-config.yaml
- id: bosspig-check
  files: \.(md|txt)$
  # This already filters to changed files
```

### Issue: Too many false positives in CI

**Solution**: Adjust severity thresholds

```bash
# Only fail on critical issues
python -m bosspig.cli analyze docs/ --severity critical

# Or create custom config
python -m bosspig.cli analyze docs/ \
  --jargon-dict .bosspig/jargon_dict.json \
  --brand-config .bosspig/brand_config.json
```

### Issue: Different results locally vs CI

**Solution**: Use same Python version and dependencies

```yaml
# Lock Python version
python-version: '3.10'

# Pin BossPig version
pip install bosspig==1.0.0
```

### Issue: spaCy model download fails in CI

**Solution**: Cache spaCy models or use fallback

```yaml
- name: Install spaCy model
  run: |
    pip install spacy
    python -m spacy download en_core_web_sm || true
  # BossPig will fall back to regex if model unavailable
```

---

## Next Steps

- **[User Manual](USER_MANUAL.md)** - Complete feature reference
- **[Configuration Guide](CONFIGURATION.md)** - Customize BossPig
- **[Quick Start Guide](QUICK_START.md)** - Get started in 5 minutes
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues

---

*Version: 1.0.0 (Beta) | Last Updated: 2025-11-22*
