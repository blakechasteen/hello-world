"""
Scan TypeScript Codebases - Squad Extension & Web UI
======================================================

Scans TypeScript/JavaScript code in:
1. Squad VS Code extension (squad/)
2. Web UI components (ui/)

Tests the new TypeScript detector built in Priority 3.

Author: mythRL Team
Date: November 16, 2025
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import glob as glob_module

from trough.mcp_server import TroughMCPServer


class TypeScriptScanner:
    """Scans TypeScript/JavaScript codebases."""

    def __init__(self):
        self.trough = TroughMCPServer()
        self.results = {
            'squad_extension': {},
            'web_ui': {},
            'summary': {}
        }

    async def scan_directory(
        self,
        directory: str,
        name: str,
        patterns: List[str] = ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx']
    ) -> Dict[str, Any]:
        """Scan a directory for TypeScript/JavaScript issues."""
        print(f"\n{'=' * 70}")
        print(f"Scanning: {name}")
        print(f"Directory: {directory}")
        print(f"{'=' * 70}\n")

        # Find files
        all_files = []
        for pattern in patterns:
            full_pattern = str(Path(directory) / pattern)
            files = glob_module.glob(full_pattern, recursive=True)
            # Filter out node_modules, dist, build
            files = [
                f for f in files
                if 'node_modules' not in f
                and 'dist' not in f
                and 'build' not in f
                and '.next' not in f
            ]
            all_files.extend(files)

        # Remove duplicates
        all_files = list(set(all_files))

        if not all_files:
            print(f"⚠ No TypeScript/JavaScript files found in {directory}")
            return {
                'directory': directory,
                'files_scanned': 0,
                'total_issues': 0,
                'files': []
            }

        print(f"Found {len(all_files)} TypeScript/JavaScript files")
        print()

        # Scan files
        all_file_results = []
        total_issues = 0

        for i, file_path in enumerate(all_files, 1):
            print(f"[{i}/{len(all_files)}] {file_path}", end='\r')

            try:
                result = await self.trough.trough_scan_file(
                    file_path=file_path,
                    include_ml_logic=False
                )

                if result.get('issues'):
                    all_file_results.append({
                        'file': file_path,
                        'issues': result['issues'],
                        'issue_count': len(result['issues'])
                    })
                    total_issues += len(result['issues'])

            except Exception as e:
                print(f"\n  ✗ Error scanning {file_path}: {e}")

        print()  # New line after progress
        print(f"✓ Scanned {len(all_files)} files")
        print(f"  Total issues: {total_issues}")
        print(f"  Files with issues: {len(all_file_results)}")
        print()

        return {
            'directory': directory,
            'files_scanned': len(all_files),
            'total_issues': total_issues,
            'files': all_file_results
        }

    async def scan_all(self):
        """Scan all TypeScript codebases."""
        print("=" * 70)
        print("TYPESCRIPT CODEBASE SCANNER")
        print("=" * 70)
        print()

        # Check if directories exist
        squad_exists = Path('squad').exists()
        ui_exists = Path('ui').exists()

        if not squad_exists and not ui_exists:
            print("⚠ Neither squad/ nor ui/ directories found!")
            print("  Searched in current directory:", Path('.').resolve())
            return

        # Scan Squad VS Code extension
        if squad_exists:
            self.results['squad_extension'] = await self.scan_directory(
                directory='squad',
                name='Squad VS Code Extension',
                patterns=['**/*.ts', '**/*.tsx']
            )
        else:
            print("⚠ squad/ directory not found, skipping")

        # Scan Web UI
        if ui_exists:
            self.results['web_ui'] = await self.scan_directory(
                directory='ui',
                name='Web UI Components',
                patterns=['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx']
            )
        else:
            print("⚠ ui/ directory not found, skipping")

        # Generate summary
        self._generate_summary()

        # Print results
        self._print_results()

        # Save results
        self._save_results()

    def _generate_summary(self):
        """Generate summary statistics."""
        summary = {
            'total_files_scanned': 0,
            'total_issues': 0,
            'files_with_issues': 0,
            'by_category': {},
            'by_severity': {},
            'top_issues': []
        }

        # Aggregate from both scans
        for scan_result in [self.results.get('squad_extension', {}), self.results.get('web_ui', {})]:
            if not scan_result:
                continue

            summary['total_files_scanned'] += scan_result.get('files_scanned', 0)
            summary['total_issues'] += scan_result.get('total_issues', 0)
            summary['files_with_issues'] += len(scan_result.get('files', []))

            # Aggregate by category and severity
            for file_data in scan_result.get('files', []):
                for issue in file_data.get('issues', []):
                    category = issue.get('category', 'unknown')
                    severity = issue.get('severity', 'unknown')

                    summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
                    summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1

                    # Collect for top issues
                    summary['top_issues'].append({
                        'file': file_data['file'],
                        'category': category,
                        'severity': severity,
                        'message': issue.get('message', ''),
                        'confidence': issue.get('confidence', 0.0)
                    })

        # Sort top issues by severity and confidence
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
        summary['top_issues'].sort(
            key=lambda x: (severity_order.get(x['severity'], 5), -x['confidence'])
        )
        summary['top_issues'] = summary['top_issues'][:20]  # Top 20

        self.results['summary'] = summary

    def _print_results(self):
        """Print scan results."""
        print("\n" + "=" * 70)
        print("SCAN RESULTS SUMMARY")
        print("=" * 70)
        print()

        summary = self.results['summary']

        print(f"Total Files Scanned: {summary['total_files_scanned']}")
        print(f"Total Issues Found: {summary['total_issues']}")
        print(f"Files with Issues: {summary['files_with_issues']}")
        print()

        if summary['by_category']:
            print("Issues by Category:")
            for category, count in sorted(summary['by_category'].items(), key=lambda x: -x[1]):
                print(f"  {category}: {count}")
            print()

        if summary['by_severity']:
            print("Issues by Severity:")
            for severity, count in sorted(summary['by_severity'].items(), key=lambda x: -x[1]):
                print(f"  {severity}: {count}")
            print()

        if summary['top_issues']:
            print("Top 10 Issues:")
            for i, issue in enumerate(summary['top_issues'][:10], 1):
                print(f"\n{i}. {issue['file']}")
                print(f"   Category: {issue['category']}")
                print(f"   Severity: {issue['severity']}")
                print(f"   Message: {issue['message']}")
                print(f"   Confidence: {issue['confidence']:.2f}")

        print("\n" + "=" * 70)

    def _save_results(self):
        """Save results to JSON."""
        output_file = 'typescript_scan_results.json'

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        print()


async def main():
    """Run TypeScript scanner."""
    scanner = TypeScriptScanner()
    await scanner.scan_all()


if __name__ == '__main__':
    asyncio.run(main())
