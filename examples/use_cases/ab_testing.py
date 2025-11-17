#!/usr/bin/env python3
"""
A/B Testing Use Case
====================
Systematic prompt optimization through controlled experimentation.

This demo shows:
- Multi-variant testing of prompts
- Statistical significance calculation
- Performance metric tracking
- Winner selection with confidence intervals
- Continuous optimization workflow

Requirements:
    pip install promptly

Usage:
    python examples/use_cases/ab_testing.py
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from Promptly.promptly.promptly import Promptly


class ABTestingDemo:
    """A/B testing framework demo"""

    def __init__(self):
        self.demo_dir = Path(__file__).parent / 'ab_testing_output'
        self.demo_dir.mkdir(exist_ok=True)
        self.promptly = None
        self.experiments = []

    def setup(self):
        """Initialize repository"""
        print("=" * 80)
        print("A/B TESTING FRAMEWORK DEMO".center(80))
        print("=" * 80)

        repo_path = self.demo_dir / 'ab_test_repo'
        repo_path.mkdir(exist_ok=True)
        self.promptly = Promptly(root_dir=str(repo_path))

        if not (repo_path / '.promptly').exists():
            self.promptly.init()
            print("✓ Initialized repository")

    def create_test_variants(self):
        """Create multiple variants for A/B testing"""
        print("\n[1/6] Creating Test Variants")
        print("-" * 80)

        # Test Case 1: Email subject lines
        email_variants = {
            'variant_a_control': {
                'name': 'email_subject_control',
                'content': 'You have a new message from {sender_name}',
                'metadata': {'variant': 'A', 'type': 'control', 'hypothesis': 'baseline'}
            },
            'variant_b_urgency': {
                'name': 'email_subject_urgency',
                'content': '[Action Required] Message from {sender_name} - Respond Today',
                'metadata': {'variant': 'B', 'type': 'treatment', 'hypothesis': 'urgency increases opens'}
            },
            'variant_c_personalized': {
                'name': 'email_subject_personalized',
                'content': '{recipient_name}, {sender_name} shared something important with you',
                'metadata': {'variant': 'C', 'type': 'treatment', 'hypothesis': 'personalization increases engagement'}
            },
            'variant_d_benefit': {
                'name': 'email_subject_benefit',
                'content': 'See what {sender_name} wants you to know (2 min read)',
                'metadata': {'variant': 'D', 'type': 'treatment', 'hypothesis': 'clear benefit increases clicks'}
            }
        }

        for variant_id, variant_data in email_variants.items():
            self.promptly.save(
                name=variant_data['name'],
                content=variant_data['content'],
                metadata=variant_data['metadata']
            )
            print(f"✓ Created variant: {variant_data['name']}")

        # Test Case 2: Call-to-action buttons
        cta_variants = {
            'cta_control': {
                'name': 'cta_button_control',
                'content': 'Click here to {action}',
                'metadata': {'variant': 'A', 'type': 'control'}
            },
            'cta_action': {
                'name': 'cta_button_action',
                'content': '{action} Now →',
                'metadata': {'variant': 'B', 'type': 'treatment'}
            },
            'cta_benefit': {
                'name': 'cta_button_benefit',
                'content': 'Get Your {benefit} - {action}',
                'metadata': {'variant': 'C', 'type': 'treatment'}
            }
        }

        for variant_id, variant_data in cta_variants.items():
            self.promptly.save(
                name=variant_data['name'],
                content=variant_data['content'],
                metadata=variant_data['metadata']
            )
            print(f"✓ Created variant: {variant_data['name']}")

    def run_ab_experiment(self):
        """Run A/B experiment with traffic split"""
        print("\n[2/6] Running A/B Experiment")
        print("-" * 80)

        # Email subject line experiment
        variants = [
            'email_subject_control',
            'email_subject_urgency',
            'email_subject_personalized',
            'email_subject_benefit'
        ]

        # Simulate traffic: 1000 users per variant
        sample_size = 1000
        traffic_split = {v: sample_size for v in variants}

        print(f"Experiment: Email Subject Line Optimization")
        print(f"Total Users: {sample_size * len(variants)}")
        print(f"Variants: {len(variants)}")
        print(f"Users per Variant: {sample_size}")

        # Simulate performance metrics
        results = {
            'email_subject_control': {
                'opens': int(sample_size * 0.22),  # 22% baseline
                'clicks': int(sample_size * 0.08),
                'conversions': int(sample_size * 0.03)
            },
            'email_subject_urgency': {
                'opens': int(sample_size * 0.28),  # 28% with urgency
                'clicks': int(sample_size * 0.12),
                'conversions': int(sample_size * 0.05)
            },
            'email_subject_personalized': {
                'opens': int(sample_size * 0.35),  # 35% with personalization (winner)
                'clicks': int(sample_size * 0.15),
                'conversions': int(sample_size * 0.07)
            },
            'email_subject_benefit': {
                'opens': int(sample_size * 0.26),  # 26% with benefit
                'clicks': int(sample_size * 0.10),
                'conversions': int(sample_size * 0.04)
            }
        }

        print("\nResults:")
        print(f"{'Variant':<30} {'Open Rate':<15} {'Click Rate':<15} {'Conv. Rate':<15}")
        print("-" * 75)

        for variant, metrics in results.items():
            open_rate = metrics['opens'] / sample_size
            click_rate = metrics['clicks'] / sample_size
            conv_rate = metrics['conversions'] / sample_size

            print(f"{variant:<30} {open_rate:<15.1%} {click_rate:<15.1%} {conv_rate:<15.1%}")

            # Record results
            self.promptly.eval_prompt(
                name=variant,
                test_case=json.dumps({'sample_size': sample_size}),
                score=conv_rate
            )

        self.experiments.append({
            'name': 'email_subject_optimization',
            'variants': list(results.keys()),
            'results': results,
            'sample_size': sample_size,
            'metric': 'conversion_rate'
        })

    def calculate_statistical_significance(self):
        """Calculate statistical significance of results"""
        print("\n[3/6] Statistical Significance Analysis")
        print("-" * 80)

        # Get last experiment
        experiment = self.experiments[-1]
        results = experiment['results']
        sample_size = experiment['sample_size']

        control = 'email_subject_control'
        control_conv = results[control]['conversions'] / sample_size

        print(f"Control: {control}")
        print(f"Control Conversion Rate: {control_conv:.2%}")
        print()

        significant_variants = []

        for variant, metrics in results.items():
            if variant == control:
                continue

            variant_conv = metrics['conversions'] / sample_size
            lift = (variant_conv - control_conv) / control_conv

            # Simplified z-test calculation
            pooled_conv = (metrics['conversions'] + results[control]['conversions']) / (2 * sample_size)
            se = (2 * pooled_conv * (1 - pooled_conv) / sample_size) ** 0.5
            z_score = (variant_conv - control_conv) / se if se > 0 else 0

            # Two-tailed test at 95% confidence (z > 1.96)
            significant = abs(z_score) > 1.96
            p_value = 2 * (1 - self._normal_cdf(abs(z_score)))

            print(f"Variant: {variant}")
            print(f"  Conversion Rate: {variant_conv:.2%}")
            print(f"  Lift: {lift:+.1%}")
            print(f"  Z-Score: {z_score:.2f}")
            print(f"  P-Value: {p_value:.4f}")
            print(f"  Significant: {'YES ✓' if significant else 'NO'}")
            print()

            if significant and lift > 0:
                significant_variants.append({
                    'variant': variant,
                    'lift': lift,
                    'p_value': p_value
                })

        if significant_variants:
            winner = max(significant_variants, key=lambda x: x['lift'])
            print(f"{'=' * 80}")
            print(f"WINNER: {winner['variant']}")
            print(f"Lift: {winner['lift']:+.1%} (statistically significant, p={winner['p_value']:.4f})")
            print(f"{'=' * 80}")
        else:
            print("No statistically significant difference found")

    def _normal_cdf(self, x: float) -> float:
        """Approximate cumulative distribution function of standard normal"""
        # Using erf approximation
        import math
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def multivariate_testing(self):
        """Run multivariate test with multiple factors"""
        print("\n[4/6] Multivariate Testing")
        print("-" * 80)

        print("Testing multiple factors simultaneously:")
        print("  Factor 1: Subject Line (Control vs Personalized)")
        print("  Factor 2: Send Time (Morning vs Evening)")
        print("  Factor 3: Sender Name (Team vs Individual)")
        print()

        # 2x2x2 = 8 combinations
        combinations = [
            ('control', 'morning', 'team', 0.18),
            ('control', 'morning', 'individual', 0.21),
            ('control', 'evening', 'team', 0.22),
            ('control', 'evening', 'individual', 0.24),
            ('personalized', 'morning', 'team', 0.28),
            ('personalized', 'morning', 'individual', 0.32),
            ('personalized', 'evening', 'team', 0.30),
            ('personalized', 'evening', 'individual', 0.35),  # Best combination
        ]

        print(f"{'Subject':<15} {'Time':<12} {'Sender':<12} {'Conv. Rate':<12}")
        print("-" * 55)

        for subject, time, sender, conv_rate in combinations:
            print(f"{subject:<15} {time:<12} {sender:<12} {conv_rate:<12.1%}")

        # Find best combination
        best = max(combinations, key=lambda x: x[3])
        baseline = combinations[0][3]  # Control + morning + team

        print(f"\nBest Combination:")
        print(f"  Subject: {best[0]}")
        print(f"  Send Time: {best[1]}")
        print(f"  Sender: {best[2]}")
        print(f"  Conversion Rate: {best[3]:.1%}")
        print(f"  Lift vs Baseline: {(best[3] - baseline) / baseline:+.1%}")

    def sequential_testing(self):
        """Demonstrate sequential/continuous testing"""
        print("\n[5/6] Sequential Testing")
        print("-" * 80)

        print("Monitoring experiment in real-time with early stopping...")
        print()

        # Simulate daily results
        days = 14
        control_conv_rates = [0.22 + random.uniform(-0.02, 0.02) for _ in range(days)]
        variant_conv_rates = [0.30 + random.uniform(-0.03, 0.03) for _ in range(days)]

        cumulative_control = []
        cumulative_variant = []

        control_sum = 0
        variant_sum = 0

        print(f"{'Day':<8} {'Control':<12} {'Variant':<12} {'Lift':<12} {'Decision':<20}")
        print("-" * 65)

        for day in range(1, days + 1):
            control_sum += control_conv_rates[day - 1]
            variant_sum += variant_conv_rates[day - 1]

            avg_control = control_sum / day
            avg_variant = variant_sum / day

            lift = (avg_variant - avg_control) / avg_control

            # Early stopping criteria
            if day >= 7:  # Minimum 7 days
                if abs(lift) > 0.25 and day >= 7:  # 25% lift threshold
                    decision = "STOP - Clear Winner"
                elif day >= days:
                    decision = "STOP - Time Limit"
                else:
                    decision = "Continue"
            else:
                decision = "Continue (min 7 days)"

            print(f"{day:<8} {avg_control:<12.2%} {avg_variant:<12.2%} {lift:<12.1%} {decision:<20}")

            if "STOP - Clear Winner" in decision:
                print(f"\n✓ Early stopping triggered on day {day}")
                print(f"  Variant is winning with {lift:+.1%} lift")
                break

    def optimization_recommendations(self):
        """Generate optimization recommendations"""
        print("\n[6/6] Optimization Recommendations")
        print("-" * 80)

        recommendations = {
            'immediate_actions': [
                {
                    'action': 'Deploy winning variant: email_subject_personalized',
                    'expected_impact': '+133% conversion rate improvement',
                    'confidence': '95%',
                    'priority': 'HIGH'
                },
                {
                    'action': 'Implement send time optimization (evening)',
                    'expected_impact': '+18% additional lift',
                    'confidence': '90%',
                    'priority': 'HIGH'
                }
            ],
            'follow_up_tests': [
                {
                    'test': 'Test different personalization elements',
                    'variants': 'First name, Full name, Company name',
                    'hypothesis': 'Different personalization types may perform better for different segments'
                },
                {
                    'test': 'Optimize urgency language',
                    'variants': 'Time-based urgency vs Scarcity vs FOMO',
                    'hypothesis': 'Different urgency types resonate with different audiences'
                }
            ],
            'learnings': [
                'Personalization significantly increases engagement (35% open rate vs 22% baseline)',
                'Evening send times outperform morning (+8% relative lift)',
                'Individual sender names perform better than team names (+13% lift)',
                'Combination of factors multiplicative, not additive',
                'Statistical significance achieved with 1000 users per variant'
            ]
        }

        print("Immediate Actions:")
        for i, action in enumerate(recommendations['immediate_actions'], 1):
            print(f"\n  {i}. {action['action']}")
            print(f"     Expected Impact: {action['expected_impact']}")
            print(f"     Confidence: {action['confidence']}")
            print(f"     Priority: {action['priority']}")

        print("\n\nFollow-up Tests:")
        for i, test in enumerate(recommendations['follow_up_tests'], 1):
            print(f"\n  {i}. {test['test']}")
            print(f"     Variants: {test['variants']}")
            print(f"     Hypothesis: {test['hypothesis']}")

        print("\n\nKey Learnings:")
        for i, learning in enumerate(recommendations['learnings'], 1):
            print(f"  {i}. {learning}")

        # Save recommendations
        report_path = self.demo_dir / 'optimization_recommendations.json'
        with open(report_path, 'w') as f:
            json.dump(recommendations, f, indent=2)

        print(f"\n✓ Recommendations saved: {report_path}")

    def run(self):
        """Run complete demo"""
        try:
            self.setup()
            self.create_test_variants()
            self.run_ab_experiment()
            self.calculate_statistical_significance()
            self.multivariate_testing()
            self.sequential_testing()
            self.optimization_recommendations()

            print("\n" + "=" * 80)
            print("✓ A/B TESTING DEMO COMPLETED")
            print("=" * 80)
            return True
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    demo = ABTestingDemo()
    success = demo.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
