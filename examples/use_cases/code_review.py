#!/usr/bin/env python3
"""
Code Review Use Case
====================
Automated code quality analysis with multi-language support.

This demo shows:
- Automated code reviews for multiple languages
- Security vulnerability detection
- Performance optimization suggestions
- Best practices validation
- Code complexity analysis
- Review consistency across team

Requirements:
    pip install promptly

Usage:
    python examples/use_cases/code_review.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from Promptly.promptly.promptly import Promptly


class CodeReviewDemo:
    """Automated code review demo"""

    def __init__(self):
        self.demo_dir = Path(__file__).parent / 'code_review_output'
        self.demo_dir.mkdir(exist_ok=True)
        self.promptly = None
        self.review_results = []

    def setup(self):
        """Initialize repository"""
        print("=" * 80)
        print("AUTOMATED CODE REVIEW DEMO".center(80))
        print("=" * 80)

        repo_path = self.demo_dir / 'review_repo'
        repo_path.mkdir(exist_ok=True)
        self.promptly = Promptly(root_dir=str(repo_path))

        if not (repo_path / '.promptly').exists():
            self.promptly.init()
            print("✓ Initialized repository")

    def create_review_prompts(self):
        """Create code review prompts for different aspects"""
        print("\n[1/5] Creating Review Prompts")
        print("-" * 80)

        prompts = {
            'code_quality': '''Review this {language} code for quality issues:

```{language}
{code}
```

Analyze:
1. **Readability**
   - Variable/function naming
   - Code organization
   - Comments and documentation
   - Complexity

2. **Maintainability**
   - DRY principle adherence
   - Function length
   - Coupling/cohesion
   - Error handling

3. **Code Smells**
   - Long methods
   - Large classes
   - Duplicate code
   - Dead code

Provide:
- Overall quality score (0-10)
- Specific issues with line numbers
- Refactoring suggestions
- Priority level for each issue''',

            'security_review': '''Perform security review of this {language} code:

```{language}
{code}
```

Check for:
1. **Input Validation**
   - SQL injection risks
   - XSS vulnerabilities
   - Command injection
   - Path traversal

2. **Authentication/Authorization**
   - Weak passwords
   - Insufficient access controls
   - Session management

3. **Data Protection**
   - Sensitive data exposure
   - Encryption usage
   - Logging sensitive info

4. **Dependencies**
   - Known vulnerabilities
   - Outdated libraries

Severity Levels: CRITICAL, HIGH, MEDIUM, LOW
Provide remediation steps for each finding.''',

            'performance_review': '''Analyze performance of this {language} code:

```{language}
{code}
```

Evaluate:
1. **Time Complexity**
   - Algorithm efficiency
   - Nested loops
   - Recursive calls

2. **Space Complexity**
   - Memory usage
   - Data structure choices
   - Memory leaks

3. **Database Operations**
   - N+1 queries
   - Missing indexes
   - Inefficient queries

4. **Resource Usage**
   - File handles
   - Network connections
   - Thread/process management

Provide:
- Performance score (0-10)
- Bottlenecks with impact estimation
- Optimization recommendations
- Expected improvement percentages''',

            'best_practices': '''Check if this {language} code follows best practices:

```{language}
{code}
```

Framework/Language: {framework}

Verify:
1. **Language-Specific**
   - Idioms and conventions
   - Modern syntax usage
   - Type hints/annotations

2. **Framework-Specific**
   - Pattern adherence (MVC, REST, etc.)
   - Framework features utilization
   - Configuration best practices

3. **Testing**
   - Testability
   - Edge case handling
   - Test coverage potential

4. **Documentation**
   - Docstrings/comments
   - API documentation
   - README requirements

Score each category 0-10 with specific examples.''',

            'complexity_analysis': '''Analyze code complexity:

```{language}
{code}
```

Calculate:
1. **Cyclomatic Complexity**
   - Number of decision points
   - Complexity score
   - Recommended threshold: < 10

2. **Cognitive Complexity**
   - Nested structures
   - Readability impact

3. **Halstead Metrics**
   - Volume
   - Difficulty
   - Effort

4. **Lines of Code**
   - SLOC (Source Lines)
   - Comment density
   - Code-to-comment ratio

Recommendations:
- Refactoring priorities
- Function decomposition suggestions
- Simplification opportunities'''
        }

        for name, content in prompts.items():
            self.promptly.save(
                name=f'review_{name}',
                content=content,
                metadata={'category': 'code_review', 'version': '1.0'}
            )
            print(f"✓ Created: review_{name}")

    def review_python_code(self):
        """Review sample Python code"""
        print("\n[2/5] Reviewing Python Code")
        print("-" * 80)

        sample_code = '''
def process_user_data(user_input):
    # Process user data
    data = eval(user_input)  # Security issue!

    results = []
    for item in data:
        for detail in item['details']:
            for value in detail['values']:
                results.append(value)  # N+3 complexity

    # Database query in loop
    for result in results:
        db.execute(f"SELECT * FROM users WHERE id = {result}")  # SQL injection!

    return results
'''

        reviews = {
            'quality': {'score': 3.5, 'issues': 5},
            'security': {'score': 2.0, 'critical': 2, 'high': 1},
            'performance': {'score': 4.0, 'bottlenecks': 2},
            'best_practices': {'score': 3.0, 'violations': 4},
            'complexity': {'cyclomatic': 15, 'recommendation': 'refactor'}
        }

        print(f"Code Sample: {len(sample_code)} characters")
        print("\nReview Results:")

        for review_type, results in reviews.items():
            # Simulate review
            self.promptly.eval_prompt(
                name=f'review_{review_type}',
                test_case=json.dumps({'code': sample_code, 'language': 'python'}),
                score=results.get('score', 0.5)
            )

            print(f"\n  {review_type.upper()}:")
            for key, value in results.items():
                print(f"    {key}: {value}")

            self.review_results.append({
                'type': review_type,
                'language': 'python',
                'results': results
            })

    def review_javascript_code(self):
        """Review sample JavaScript code"""
        print("\n[3/5] Reviewing JavaScript Code")
        print("-" * 80)

        sample_code = '''
async function fetchUserData(userId) {
    const response = await fetch(`/api/users/${userId}`);
    const data = await response.json();

    // No error handling
    document.getElementById('user-data').innerHTML = data.bio;  // XSS risk

    // Memory leak - event listener not removed
    window.addEventListener('resize', () => {
        updateLayout(data);
    });

    return data;
}
'''

        reviews = {
            'quality': {'score': 5.5, 'issues': 3},
            'security': {'score': 4.0, 'critical': 1, 'high': 0},
            'performance': {'score': 6.0, 'bottlenecks': 1},
            'best_practices': {'score': 5.0, 'violations': 2}
        }

        print(f"Code Sample: {len(sample_code)} characters")
        print("\nReview Results:")

        for review_type, results in reviews.items():
            self.promptly.eval_prompt(
                name=f'review_{review_type}',
                test_case=json.dumps({'code': sample_code, 'language': 'javascript'}),
                score=results.get('score', 0.5)
            )

            print(f"\n  {review_type.upper()}:")
            for key, value in results.items():
                print(f"    {key}: {value}")

    def compare_review_consistency(self):
        """Compare review consistency across different reviewers"""
        print("\n[4/5] Analyzing Review Consistency")
        print("-" * 80)

        # Simulate multiple reviews of the same code
        code_snippet = "def calculate(x, y): return x + y"

        reviewers = ['human_reviewer_1', 'human_reviewer_2', 'ai_reviewer']
        scores = {
            'human_reviewer_1': {'quality': 8.0, 'security': 9.0, 'performance': 8.5},
            'human_reviewer_2': {'quality': 7.5, 'security': 9.0, 'performance': 8.0},
            'ai_reviewer': {'quality': 7.8, 'security': 9.0, 'performance': 8.3}
        }

        print("Consistency Analysis:")
        print(f"{'Reviewer':<20} {'Quality':<12} {'Security':<12} {'Performance':<12}")
        print("-" * 60)

        for reviewer, review_scores in scores.items():
            print(f"{reviewer:<20} {review_scores['quality']:<12.1f} "
                  f"{review_scores['security']:<12.1f} {review_scores['performance']:<12.1f}")

        # Calculate variance
        quality_scores = [s['quality'] for s in scores.values()]
        variance = sum((x - sum(quality_scores)/len(quality_scores))**2
                      for x in quality_scores) / len(quality_scores)

        print(f"\nQuality Score Variance: {variance:.3f}")
        print(f"Consistency Rating: {'HIGH' if variance < 0.5 else 'MEDIUM' if variance < 1.0 else 'LOW'}")

    def generate_report(self):
        """Generate review report"""
        print("\n[5/5] Generating Report")
        print("-" * 80)

        report = {
            'generated_at': datetime.now().isoformat(),
            'reviews_completed': len(self.review_results),
            'languages_reviewed': list(set(r['language'] for r in self.review_results)),
            'critical_issues': 3,
            'high_issues': 1,
            'medium_issues': 8,
            'low_issues': 5,
            'avg_quality_score': 4.5,
            'recommendations': [
                'Fix critical security vulnerabilities immediately',
                'Refactor functions with cyclomatic complexity > 10',
                'Add comprehensive error handling',
                'Implement input validation everywhere',
                'Add unit tests for all functions'
            ]
        }

        report_path = self.demo_dir / 'code_review_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Report saved: {report_path}")
        print("\n" + "=" * 80)
        print("CODE REVIEW SUMMARY")
        print("=" * 80)
        print(f"Reviews Completed: {report['reviews_completed']}")
        print(f"Critical Issues: {report['critical_issues']}")
        print(f"High Issues: {report['high_issues']}")
        print(f"Average Quality Score: {report['avg_quality_score']:.1f}/10")

    def run(self):
        """Run complete demo"""
        try:
            self.setup()
            self.create_review_prompts()
            self.review_python_code()
            self.review_javascript_code()
            self.compare_review_consistency()
            self.generate_report()

            print("\n✓ CODE REVIEW DEMO COMPLETED\n")
            return True
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    demo = CodeReviewDemo()
    success = demo.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
