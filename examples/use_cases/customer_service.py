#!/usr/bin/env python3
"""
Customer Service Use Case
==========================
Automated response quality scoring and optimization for customer support.

This demo shows:
- Multiple response templates for different scenarios
- Quality evaluation across dimensions (empathy, clarity, accuracy)
- A/B testing of greeting templates
- Response time tracking
- Customer satisfaction correlation

Requirements:
    pip install promptly

Usage:
    python examples/use_cases/customer_service.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from Promptly.promptly.promptly import Promptly


class CustomerServiceDemo:
    """Customer service automation demo"""

    def __init__(self):
        self.demo_dir = Path(__file__).parent / 'cs_output'
        self.demo_dir.mkdir(exist_ok=True)
        self.promptly = None
        self.results = []

    def setup(self):
        """Initialize Promptly repository"""
        print("=" * 80)
        print("CUSTOMER SERVICE AUTOMATION DEMO".center(80))
        print("=" * 80)
        print()

        repo_path = self.demo_dir / 'cs_repo'
        repo_path.mkdir(exist_ok=True)

        self.promptly = Promptly(root_dir=str(repo_path))

        # Initialize if needed
        if not (repo_path / '.promptly').exists():
            self.promptly.init()
            print("✓ Initialized Promptly repository")

    def create_response_templates(self):
        """Create customer service response templates"""
        print("\n[1/6] Creating Response Templates")
        print("-" * 80)

        templates = {
            'greeting_formal': {
                'content': '''Hello {customer_name},

Thank you for contacting {company_name} support. My name is {agent_name}, and I'm here to help you today.

I understand you're experiencing {issue_category}. I'll do my best to resolve this for you as quickly as possible.

Could you please provide more details about {specific_question}?

Best regards,
{agent_name}
{company_name} Customer Support''',
                'metadata': {
                    'tone': 'formal',
                    'avg_response_time': '2-3 minutes',
                    'customer_satisfaction': 4.2,
                    'use_case': 'general_inquiry'
                }
            },
            'greeting_friendly': {
                'content': '''Hi {customer_name}! 👋

Thanks for reaching out to {company_name}. I'm {agent_name}, and I'm happy to help!

I see you're having some trouble with {issue_category}. No worries - we'll get this sorted out for you!

Can you tell me a bit more about {specific_question}?

Cheers,
{agent_name}''',
                'metadata': {
                    'tone': 'friendly',
                    'avg_response_time': '1-2 minutes',
                    'customer_satisfaction': 4.5,
                    'use_case': 'general_inquiry'
                }
            },
            'technical_issue': {
                'content': '''Hello {customer_name},

Thank you for reporting this technical issue. I'm {agent_name} from our technical support team.

**Issue Summary:**
{issue_description}

**Troubleshooting Steps:**
{troubleshooting_steps}

**Expected Resolution:**
{resolution_timeline}

If these steps don't resolve the issue, I'll escalate this to our engineering team immediately.

Please let me know if you need any clarification on these steps.

Best regards,
{agent_name}
Technical Support Team''',
                'metadata': {
                    'tone': 'technical',
                    'avg_response_time': '5-7 minutes',
                    'customer_satisfaction': 4.3,
                    'use_case': 'technical_support'
                }
            },
            'billing_inquiry': {
                'content': '''Hello {customer_name},

Thank you for contacting us about your billing inquiry. I'm {agent_name} from the billing department.

I've reviewed your account and here's what I found:

**Account Details:**
- Account ID: {account_id}
- Current Balance: {current_balance}
- Last Payment: {last_payment}

{billing_explanation}

**Next Steps:**
{next_steps}

Is there anything else I can help clarify about your billing?

Best regards,
{agent_name}
Billing Support''',
                'metadata': {
                    'tone': 'professional',
                    'avg_response_time': '3-4 minutes',
                    'customer_satisfaction': 4.1,
                    'use_case': 'billing'
                }
            },
            'escalation_response': {
                'content': '''Dear {customer_name},

I sincerely apologize for the inconvenience you've experienced with {issue_category}.

I completely understand your frustration, and I want to make this right for you.

**Immediate Actions:**
{immediate_actions}

**Long-term Resolution:**
{long_term_plan}

**Compensation:**
{compensation_offer}

I've personally escalated this to our management team, and {manager_name} will follow up with you within {follow_up_time}.

Your satisfaction is our top priority, and we're committed to resolving this issue to your complete satisfaction.

Sincerely,
{agent_name}
Senior Customer Support''',
                'metadata': {
                    'tone': 'apologetic',
                    'avg_response_time': '10-15 minutes',
                    'customer_satisfaction': 3.9,
                    'use_case': 'escalation'
                }
            }
        }

        for name, template in templates.items():
            self.promptly.save(
                name=f'cs_{name}',
                content=template['content'],
                metadata=template['metadata']
            )
            print(f"✓ Created template: cs_{name}")

        print(f"\n✓ Created {len(templates)} response templates")

    def evaluate_response_quality(self):
        """Evaluate response quality across multiple dimensions"""
        print("\n[2/6] Evaluating Response Quality")
        print("-" * 80)

        # Quality dimensions
        dimensions = ['empathy', 'clarity', 'accuracy', 'professionalism', 'completeness']

        # Test scenarios
        test_scenarios = [
            {
                'template': 'cs_greeting_friendly',
                'inputs': {
                    'customer_name': 'Sarah Johnson',
                    'company_name': 'TechCorp',
                    'agent_name': 'Mike',
                    'issue_category': 'login issues',
                    'specific_question': 'what error message you're seeing'
                },
                'expected_scores': {
                    'empathy': 0.95,
                    'clarity': 0.90,
                    'accuracy': 0.85,
                    'professionalism': 0.88,
                    'completeness': 0.82
                }
            },
            {
                'template': 'cs_technical_issue',
                'inputs': {
                    'customer_name': 'David Chen',
                    'agent_name': 'Lisa',
                    'issue_description': 'Application crashes on startup',
                    'troubleshooting_steps': '1. Clear cache\n2. Restart device\n3. Reinstall app',
                    'resolution_timeline': '15-30 minutes'
                },
                'expected_scores': {
                    'empathy': 0.75,
                    'clarity': 0.95,
                    'accuracy': 0.98,
                    'professionalism': 0.95,
                    'completeness': 0.93
                }
            },
            {
                'template': 'cs_escalation_response',
                'inputs': {
                    'customer_name': 'Jennifer Williams',
                    'issue_category': 'order delay',
                    'immediate_actions': 'Expedited shipping at no cost',
                    'long_term_plan': 'Improved inventory management',
                    'compensation_offer': '25% discount on next purchase',
                    'manager_name': 'Robert Smith',
                    'follow_up_time': '24 hours',
                    'agent_name': 'Amanda'
                },
                'expected_scores': {
                    'empathy': 0.98,
                    'clarity': 0.92,
                    'accuracy': 0.90,
                    'professionalism': 0.97,
                    'completeness': 0.95
                }
            }
        ]

        evaluation_results = []

        for scenario in test_scenarios:
            template_name = scenario['template']
            print(f"\nEvaluating: {template_name}")

            # Simulate evaluation
            for dimension, expected_score in scenario['expected_scores'].items():
                # Add some realistic variance
                import random
                actual_score = expected_score + random.uniform(-0.05, 0.05)
                actual_score = max(0, min(1, actual_score))

                # Record evaluation
                self.promptly.eval_prompt(
                    name=template_name,
                    test_case=json.dumps({
                        'dimension': dimension,
                        'inputs': scenario['inputs']
                    }),
                    score=actual_score
                )

                print(f"  {dimension:15s}: {actual_score:.3f} (expected: {expected_score:.3f})")

            # Calculate overall score
            overall_score = sum(scenario['expected_scores'].values()) / len(scenario['expected_scores'])
            evaluation_results.append({
                'template': template_name,
                'overall_score': overall_score,
                'dimensions': scenario['expected_scores']
            })

        # Print summary
        print("\n" + "-" * 80)
        print("Quality Evaluation Summary:")
        for result in evaluation_results:
            print(f"  {result['template']:30s}: {result['overall_score']:.3f}")

        self.results.extend(evaluation_results)

    def ab_test_greetings(self):
        """A/B test different greeting templates"""
        print("\n[3/6] A/B Testing Greeting Templates")
        print("-" * 80)

        variants = ['cs_greeting_formal', 'cs_greeting_friendly']

        # Simulate A/B test with 100 customers per variant
        import random

        results = {}
        for variant in variants:
            variant_data = self.promptly.get(variant)
            base_satisfaction = variant_data['metadata'].get('customer_satisfaction', 4.0)

            # Simulate customer responses
            responses = []
            for i in range(100):
                # Add realistic variance
                satisfaction = base_satisfaction + random.gauss(0, 0.3)
                satisfaction = max(1, min(5, satisfaction))
                responses.append(satisfaction)

            avg_satisfaction = sum(responses) / len(responses)
            response_rate = 0.85 + random.uniform(-0.1, 0.1)

            results[variant] = {
                'avg_satisfaction': avg_satisfaction,
                'response_rate': response_rate,
                'sample_size': len(responses)
            }

            print(f"\n{variant}:")
            print(f"  Average Satisfaction: {avg_satisfaction:.2f}/5.0")
            print(f"  Response Rate: {response_rate:.1%}")
            print(f"  Sample Size: {len(responses)}")

        # Determine winner
        winner = max(results.items(), key=lambda x: x[1]['avg_satisfaction'])
        print(f"\n{'=' * 80}")
        print(f"Winner: {winner[0]} (satisfaction: {winner[1]['avg_satisfaction']:.2f})")
        print(f"{'=' * 80}")

    def track_response_times(self):
        """Track and analyze response times"""
        print("\n[4/6] Tracking Response Times")
        print("-" * 80)

        import random

        # Simulate response times for each template
        templates = ['cs_greeting_formal', 'cs_greeting_friendly', 'cs_technical_issue',
                     'cs_billing_inquiry', 'cs_escalation_response']

        response_time_data = {}

        for template in templates:
            template_data = self.promptly.get(template)
            metadata = template_data['metadata']

            # Parse avg_response_time (e.g., "2-3 minutes")
            time_range = metadata.get('avg_response_time', '1-2 minutes')
            time_parts = time_range.split('-')
            min_time = float(time_parts[0])
            max_time = float(time_parts[1].split()[0])

            # Simulate 50 response times
            times = [random.uniform(min_time, max_time) for _ in range(50)]

            avg_time = sum(times) / len(times)
            min_time_actual = min(times)
            max_time_actual = max(times)

            response_time_data[template] = {
                'avg': avg_time,
                'min': min_time_actual,
                'max': max_time_actual,
                'samples': len(times)
            }

            print(f"\n{template}:")
            print(f"  Average: {avg_time:.1f} minutes")
            print(f"  Min: {min_time_actual:.1f} minutes")
            print(f"  Max: {max_time_actual:.1f} minutes")

        # Find fastest template
        fastest = min(response_time_data.items(), key=lambda x: x[1]['avg'])
        print(f"\n{'=' * 80}")
        print(f"Fastest Template: {fastest[0]} ({fastest[1]['avg']:.1f} min average)")
        print(f"{'=' * 80}")

    def analyze_satisfaction_correlation(self):
        """Analyze correlation between response quality and satisfaction"""
        print("\n[5/6] Analyzing Quality-Satisfaction Correlation")
        print("-" * 80)

        # Get all templates with satisfaction scores
        templates = ['cs_greeting_formal', 'cs_greeting_friendly', 'cs_technical_issue',
                     'cs_billing_inquiry', 'cs_escalation_response']

        correlation_data = []

        for template in templates:
            template_data = self.promptly.get(template)
            satisfaction = template_data['metadata'].get('customer_satisfaction', 4.0)

            # Find quality score from results
            quality_score = 0.85  # Default
            for result in self.results:
                if result['template'] == template:
                    quality_score = result['overall_score']
                    break

            correlation_data.append({
                'template': template,
                'quality_score': quality_score,
                'satisfaction': satisfaction
            })

            print(f"{template:30s}: Quality={quality_score:.3f}, Satisfaction={satisfaction:.1f}")

        # Calculate simple correlation
        if len(correlation_data) >= 2:
            qualities = [d['quality_score'] for d in correlation_data]
            satisfactions = [d['satisfaction'] for d in correlation_data]

            # Simple correlation coefficient
            mean_q = sum(qualities) / len(qualities)
            mean_s = sum(satisfactions) / len(satisfactions)

            numerator = sum((q - mean_q) * (s - mean_s) for q, s in zip(qualities, satisfactions))
            denom_q = sum((q - mean_q) ** 2 for q in qualities) ** 0.5
            denom_s = sum((s - mean_s) ** 2 for s in satisfactions) ** 0.5

            if denom_q > 0 and denom_s > 0:
                correlation = numerator / (denom_q * denom_s)
                print(f"\n{'=' * 80}")
                print(f"Quality-Satisfaction Correlation: {correlation:.3f}")
                print(f"{'=' * 80}")

    def generate_report(self):
        """Generate comprehensive report"""
        print("\n[6/6] Generating Report")
        print("-" * 80)

        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_templates': 5,
                'total_evaluations': len(self.results),
                'avg_quality_score': sum(r['overall_score'] for r in self.results) / len(self.results) if self.results else 0,
            },
            'results': self.results,
            'recommendations': [
                'Use greeting_friendly for general inquiries (higher satisfaction)',
                'technical_issue template has highest accuracy scores',
                'escalation_response shows strong empathy but needs faster response time',
                'Consider adding more personalization variables to all templates',
                'Implement real-time quality monitoring for all responses'
            ]
        }

        # Save report
        report_path = self.demo_dir / 'customer_service_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Report saved to: {report_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("CUSTOMER SERVICE DEMO SUMMARY")
        print("=" * 80)
        print(f"Total Templates: {report['summary']['total_templates']}")
        print(f"Total Evaluations: {report['summary']['total_evaluations']}")
        print(f"Average Quality Score: {report['summary']['avg_quality_score']:.3f}")
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"  {i}. {rec}")

    def run(self):
        """Run complete demo"""
        try:
            self.setup()
            self.create_response_templates()
            self.evaluate_response_quality()
            self.ab_test_greetings()
            self.track_response_times()
            self.analyze_satisfaction_correlation()
            self.generate_report()

            print("\n" + "=" * 80)
            print("✓ DEMO COMPLETED SUCCESSFULLY")
            print("=" * 80)
            return True

        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    demo = CustomerServiceDemo()
    success = demo.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
