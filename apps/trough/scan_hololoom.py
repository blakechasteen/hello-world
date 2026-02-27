#!/usr/bin/env python3
"""
Dogfood Trough on HoloLoom codebase
Scan all Python files and generate report
"""

import sys
import os
# Add parent directory to path so we can import trough
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import glob
from pathlib import Path
from collections import defaultdict
from trough import AISlopDetector, MLLogicDetector, Language, Severity

async def scan_file(file_path: str, slop_detector, logic_detector):
    """Scan a single file with both detectors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Run both detectors
        slop_issues = await slop_detector.detect_all(code, Language.PYTHON, file_path)
        logic_errors = await logic_detector.detect(code, Language.PYTHON, file_path)
        
        return {
            'file': file_path,
            'slop_issues': slop_issues,
            'logic_errors': logic_errors,
            'total': len(slop_issues) + len(logic_errors)
        }
    except Exception as e:
        print(f"  Error scanning {file_path}: {e}")
        return None

async def scan_directory(directory: str, max_files: int = 50):
    """Scan all Python files in directory."""
    print(f"Scanning {directory}...")
    
    # Find all Python files
    py_files = glob.glob(f"{directory}/**/*.py", recursive=True)
    
    # Exclude test files and __pycache__
    py_files = [f for f in py_files if '__pycache__' not in f and 'test_' not in os.path.basename(f)]
    
    # Limit files
    py_files = py_files[:max_files]
    
    print(f"Found {len(py_files)} Python files to scan")
    print()
    
    # Create detectors
    slop_detector = AISlopDetector()
    logic_detector = MLLogicDetector()
    
    # Scan files
    results = []
    for i, file_path in enumerate(py_files):
        print(f"[{i+1}/{len(py_files)}] Scanning {os.path.basename(file_path)}...", end=' ')
        result = await scan_file(file_path, slop_detector, logic_detector)
        if result:
            print(f"{result['total']} issues")
            results.append(result)
        else:
            print("SKIP")
    
    return results

def generate_report(results):
    """Generate markdown report."""
    # Stats
    total_files = len(results)
    total_issues = sum(r['total'] for r in results)
    files_with_issues = [r for r in results if r['total'] > 0]
    
    # By severity
    by_severity = defaultdict(int)
    by_category = defaultdict(int)
    
    for result in results:
        for issue in result['slop_issues']:
            by_severity[issue.severity.value] += 1
            by_category[issue.category.value] += 1
        for error in result['logic_errors']:
            # Logic errors use different structure - skip for now
            # TODO: Fix ML logic detector output format
            pass
    
    # Top issues
    top_files = sorted(files_with_issues, key=lambda x: x['total'], reverse=True)[:10]
    
    # Generate report
    report = []
    report.append("# Trough Scan Report - HoloLoom Codebase")
    report.append("")
    report.append(f"**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Files Scanned**: {total_files}")
    report.append(f"**Total Issues**: {total_issues}")
    report.append(f"**Files with Issues**: {len(files_with_issues)}")
    report.append("")
    
    # Severity breakdown
    report.append("## Issues by Severity")
    report.append("")
    for sev in ['critical', 'high', 'medium', 'low', 'info']:
        count = by_severity.get(sev, 0)
        if count > 0:
            report.append(f"- **{sev.upper()}**: {count}")
    report.append("")
    
    # Category breakdown
    report.append("## Issues by Category")
    report.append("")
    for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:15]:
        report.append(f"- **{cat}**: {count}")
    report.append("")
    
    # Top problematic files
    report.append("## Top 10 Files with Most Issues")
    report.append("")
    for i, result in enumerate(top_files, 1):
        rel_path = result['file'].replace('\\', '/').split('hololoom/')[-1]
        report.append(f"{i}. **{rel_path}** - {result['total']} issues")
        
        # Show top 3 issues
        all_issues = []
        for issue in result['slop_issues'][:3]:
            all_issues.append(f"   - Line {issue.line_number}: [{issue.severity.value}] {issue.category.value}")
        # Skip logic errors for now
        # for error in result['logic_errors'][:3]:
            # all_issues.append(f"   - Line {error.line_number}: [{error.severity.value}] {error.category.value}")
        
        for issue_line in all_issues[:3]:
            report.append(issue_line)
        report.append("")
    
    # Example fixes
    report.append("## Example Issues Found")
    report.append("")
    
    # Find interesting examples
    examples = []
    for result in results:
        for issue in result['slop_issues']:
            if issue.severity == Severity.HIGH or issue.severity == Severity.CRITICAL:
                rel_path = result['file'].replace('\\', '/').split('hololoom/')[-1]
                examples.append({
                    'file': rel_path,
                    'line': issue.line_number,
                    'severity': issue.severity.value,
                    'category': issue.category.value,
                    'message': issue.message,
                    'suggestion': issue.suggestion
                })
                if len(examples) >= 5:
                    break
        if len(examples) >= 5:
            break
    
    for i, ex in enumerate(examples, 1):
        report.append(f"### {i}. {ex['file']}:{ex['line']}")
        report.append(f"**Severity**: {ex['severity'].upper()}")
        report.append(f"**Category**: {ex['category']}")
        report.append(f"**Issue**: {ex['message']}")
        if ex['suggestion']:
            report.append(f"**Fix**: {ex['suggestion']}")
        report.append("")
    
    return '\n'.join(report)

async def main():
    print("="*70)
    print("TROUGH DOGFOODING - SCANNING HOLOLOOM")
    print("="*70)
    print()
    
    # Scan HoloLoom directory
    results = await scan_directory("../HoloLoom", max_files=50)
    
    # Generate report
    report = generate_report(results)
    
    # Save report
    with open('HOLOLOOM_SCAN_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print()
    print("="*70)
    print("SCAN COMPLETE")
    print("="*70)
    print(f"Report saved to: HOLOLOOM_SCAN_REPORT.md")
    print()
    
    # Print summary
    total_issues = sum(r['total'] for r in results)
    print(f"Total files scanned: {len(results)}")
    print(f"Total issues found: {total_issues}")
    print(f"Average issues per file: {total_issues/len(results):.1f}")
    print()

if __name__ == "__main__":
    asyncio.run(main())
