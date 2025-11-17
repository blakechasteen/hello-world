#!/usr/bin/env python3
"""
Business Logic Evaluator

Custom evaluators based on business rules and domain-specific requirements.
Supports configurable thresholds, weights, and industry-specific validation.

Examples:
- Customer service response quality
- Financial compliance checking
- Medical accuracy validation
"""

from typing import Dict, Any, Optional, List, Callable
import re
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from promptly.plugins.base import BaseEvaluator


class BusinessLogicEvaluator(BaseEvaluator):
    """
    Base class for business logic evaluators.

    Provides framework for custom scoring based on business rules with:
    - Configurable rule sets
    - Weighted scoring
    - Threshold-based validation
    - Detailed metrics reporting
    """

    def __init__(self, name: str, description: str,
                 rules: Optional[List[Dict[str, Any]]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 min_passing_score: float = 0.7):
        """
        Initialize business logic evaluator

        Args:
            name: Evaluator name
            description: Evaluator description
            rules: List of business rules to evaluate
            weights: Weight for each rule category
            min_passing_score: Minimum score to pass validation (0.0-1.0)
        """
        super().__init__(name, description)
        self.rules = rules or []
        self.weights = weights or {}
        self.min_passing_score = min_passing_score

    def evaluate_rule(self, rule: Dict[str, Any], text: str,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single business rule

        Args:
            rule: Rule configuration
            text: Text to evaluate
            context: Optional context

        Returns:
            Dict with rule evaluation results
        """
        rule_type = rule.get('type')
        passed = False
        score = 0.0
        details = {}

        if rule_type == 'contains':
            # Check if text contains required keywords
            keywords = rule.get('keywords', [])
            found = [kw for kw in keywords if kw.lower() in text.lower()]
            passed = len(found) >= rule.get('min_matches', 1)
            score = len(found) / max(1, len(keywords))
            details = {'found_keywords': found, 'required': keywords}

        elif rule_type == 'excludes':
            # Check if text excludes forbidden keywords
            keywords = rule.get('keywords', [])
            found = [kw for kw in keywords if kw.lower() in text.lower()]
            passed = len(found) == 0
            score = 1.0 if passed else 0.0
            details = {'found_forbidden': found, 'forbidden': keywords}

        elif rule_type == 'length':
            # Check length constraints
            min_len = rule.get('min', 0)
            max_len = rule.get('max', float('inf'))
            length = len(text)
            passed = min_len <= length <= max_len
            score = 1.0 if passed else 0.5
            details = {'length': length, 'min': min_len, 'max': max_len}

        elif rule_type == 'regex':
            # Check regex pattern
            pattern = rule.get('pattern', '')
            matches = re.findall(pattern, text, re.IGNORECASE)
            required_matches = rule.get('min_matches', 1)
            passed = len(matches) >= required_matches
            score = min(1.0, len(matches) / required_matches)
            details = {'matches': len(matches), 'pattern': pattern}

        elif rule_type == 'custom':
            # Custom validation function
            validator = rule.get('validator')
            if validator and callable(validator):
                result = validator(text, context)
                passed = result.get('passed', False)
                score = result.get('score', 0.0)
                details = result.get('details', {})

        return {
            'rule_name': rule.get('name', 'unnamed'),
            'passed': passed,
            'score': score,
            'details': details
        }

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate text against all business rules

        Args:
            actual: Text to evaluate
            expected: Expected output (for reference)
            context: Optional context

        Returns:
            float: Weighted score between 0.0 and 1.0
        """
        if not actual:
            return 0.0

        # Evaluate all rules
        results = []
        for rule in self.rules:
            result = self.evaluate_rule(rule, actual, context)
            results.append(result)

        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0

        for result in results:
            rule_name = result['rule_name']
            weight = self.weights.get(rule_name, 1.0)
            total_score += result['score'] * weight
            total_weight += weight

        final_score = total_score / max(1.0, total_weight)
        return max(0.0, min(1.0, final_score))

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed business rule metrics"""
        results = []
        for rule in self.rules:
            result = self.evaluate_rule(rule, actual, context)
            results.append(result)

        score = self.evaluate(actual, expected, context)
        passed = score >= self.min_passing_score

        return {
            'score': score,
            'passed': passed,
            'min_passing_score': self.min_passing_score,
            'rule_results': results,
            'rules_passed': sum(1 for r in results if r['passed']),
            'total_rules': len(results)
        }


class CustomerServiceEvaluator(BusinessLogicEvaluator):
    """
    Customer service response quality evaluator.

    Validates responses for:
    - Professional tone and empathy
    - Required elements (greeting, resolution, closing)
    - Forbidden phrases (dismissive language)
    - Length constraints
    - Action items and next steps
    """

    def __init__(self, min_length: int = 50, max_length: int = 500):
        """
        Initialize customer service evaluator

        Args:
            min_length: Minimum response length
            max_length: Maximum response length
        """
        # Define customer service rules
        rules = [
            {
                'name': 'greeting',
                'type': 'regex',
                'pattern': r'(hello|hi|greetings|dear|thank you for)',
                'min_matches': 1,
                'description': 'Must include a greeting'
            },
            {
                'name': 'empathy',
                'type': 'contains',
                'keywords': [
                    'understand', 'appreciate', 'apologize', 'sorry',
                    'thank you', 'value', 'important'
                ],
                'min_matches': 1,
                'description': 'Must show empathy'
            },
            {
                'name': 'resolution',
                'type': 'contains',
                'keywords': [
                    'resolve', 'fix', 'solution', 'help', 'assist',
                    'address', 'handle', 'take care'
                ],
                'min_matches': 1,
                'description': 'Must offer resolution'
            },
            {
                'name': 'closing',
                'type': 'regex',
                'pattern': r'(best regards|sincerely|thanks|regards|feel free)',
                'min_matches': 1,
                'description': 'Must include professional closing'
            },
            {
                'name': 'no_dismissive',
                'type': 'excludes',
                'keywords': [
                    'not my problem', 'not my job', 'can\'t help',
                    'nothing we can do', 'tough luck', 'deal with it'
                ],
                'description': 'Must not contain dismissive language'
            },
            {
                'name': 'length',
                'type': 'length',
                'min': min_length,
                'max': max_length,
                'description': f'Must be between {min_length}-{max_length} characters'
            },
            {
                'name': 'next_steps',
                'type': 'contains',
                'keywords': [
                    'will', 'I will', 'we will', 'going to',
                    'next step', 'follow up', 'contact you'
                ],
                'min_matches': 1,
                'description': 'Must include next steps'
            }
        ]

        # Define weights for each rule
        weights = {
            'greeting': 0.8,
            'empathy': 1.5,  # Higher weight for empathy
            'resolution': 2.0,  # Highest weight for resolution
            'closing': 0.8,
            'no_dismissive': 2.0,  # Critical rule
            'length': 0.5,
            'next_steps': 1.2
        }

        super().__init__(
            name='customer_service',
            description='Customer service response quality evaluator',
            rules=rules,
            weights=weights,
            min_passing_score=0.75
        )


class FinancialComplianceEvaluator(BusinessLogicEvaluator):
    """
    Financial compliance evaluator.

    Validates financial content for:
    - Required disclaimers
    - Risk warnings
    - Regulatory compliance
    - Forbidden claims
    - Accuracy of financial data
    """

    def __init__(self, jurisdiction: str = 'US'):
        """
        Initialize financial compliance evaluator

        Args:
            jurisdiction: Regulatory jurisdiction ('US', 'EU', 'UK')
        """
        self.jurisdiction = jurisdiction

        # Define compliance rules (US example)
        rules = [
            {
                'name': 'risk_disclosure',
                'type': 'contains',
                'keywords': [
                    'risk', 'past performance', 'no guarantee',
                    'may lose', 'not guaranteed', 'disclaimer'
                ],
                'min_matches': 2,
                'description': 'Must include risk disclosures'
            },
            {
                'name': 'no_guaranteed_returns',
                'type': 'excludes',
                'keywords': [
                    'guaranteed return', 'guaranteed profit',
                    'risk-free', 'cannot lose', 'always profitable'
                ],
                'description': 'Must not promise guaranteed returns'
            },
            {
                'name': 'sec_disclaimer',
                'type': 'regex',
                'pattern': r'(not financial advice|consult.*professional|consult.*advisor)',
                'min_matches': 1,
                'description': 'Must include professional advice disclaimer'
            },
            {
                'name': 'no_insider_trading',
                'type': 'excludes',
                'keywords': [
                    'insider information', 'confidential tip',
                    'before public', 'secret information'
                ],
                'description': 'Must not reference insider information'
            },
            {
                'name': 'accurate_numbers',
                'type': 'custom',
                'validator': self._validate_financial_data,
                'description': 'Financial data must be accurate'
            }
        ]

        weights = {
            'risk_disclosure': 2.0,
            'no_guaranteed_returns': 3.0,  # Critical compliance issue
            'sec_disclaimer': 2.0,
            'no_insider_trading': 3.0,  # Critical compliance issue
            'accurate_numbers': 1.5
        }

        super().__init__(
            name=f'financial_compliance_{jurisdiction}',
            description=f'Financial compliance evaluator ({jurisdiction})',
            rules=rules,
            weights=weights,
            min_passing_score=0.9  # High bar for compliance
        )

    def _validate_financial_data(self, text: str,
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate financial data accuracy

        Args:
            text: Text containing financial data
            context: Context with expected financial data

        Returns:
            Dict with validation results
        """
        # Extract numbers from text
        numbers = re.findall(r'[\$€£]?\s*\d+(?:,\d{3})*(?:\.\d{2})?', text)

        # If context provides expected values, validate them
        if context and 'expected_values' in context:
            expected = context['expected_values']
            # Simple validation: check if expected values are present
            all_present = all(
                any(str(exp) in num for num in numbers)
                for exp in expected
            )
            return {
                'passed': all_present,
                'score': 1.0 if all_present else 0.0,
                'details': {
                    'found_numbers': numbers,
                    'expected_values': expected
                }
            }

        # If no context, just check that numbers are properly formatted
        return {
            'passed': True,
            'score': 1.0,
            'details': {'found_numbers': numbers}
        }


class MedicalAccuracyEvaluator(BusinessLogicEvaluator):
    """
    Medical accuracy and safety evaluator.

    Validates medical content for:
    - Medical disclaimers
    - Contraindication warnings
    - Professional consultation recommendations
    - Accuracy of medical terms
    - Safety warnings
    """

    def __init__(self, require_disclaimer: bool = True):
        """
        Initialize medical accuracy evaluator

        Args:
            require_disclaimer: Whether medical disclaimer is required
        """
        rules = [
            {
                'name': 'medical_disclaimer',
                'type': 'contains',
                'keywords': [
                    'consult your doctor', 'medical professional',
                    'healthcare provider', 'not medical advice',
                    'seek medical attention'
                ],
                'min_matches': 1 if require_disclaimer else 0,
                'description': 'Must include medical disclaimer'
            },
            {
                'name': 'contraindications',
                'type': 'contains',
                'keywords': [
                    'side effect', 'contraindication', 'caution',
                    'warning', 'risk', 'adverse'
                ],
                'min_matches': 0,  # Optional but scored
                'description': 'Should mention potential risks'
            },
            {
                'name': 'no_diagnosis',
                'type': 'excludes',
                'keywords': [
                    'you have', 'you definitely', 'you are diagnosed',
                    'I diagnose', 'this is definitely'
                ],
                'description': 'Must not make definitive diagnoses'
            },
            {
                'name': 'no_prescription',
                'type': 'excludes',
                'keywords': [
                    'take this medication', 'I prescribe',
                    'you should take', 'dosage for you is'
                ],
                'description': 'Must not prescribe medication'
            },
            {
                'name': 'emergency_warning',
                'type': 'custom',
                'validator': self._check_emergency_language,
                'description': 'Emergency situations must direct to 911/ER'
            }
        ]

        weights = {
            'medical_disclaimer': 2.0,
            'contraindications': 1.0,
            'no_diagnosis': 3.0,  # Critical - no AI diagnosis
            'no_prescription': 3.0,  # Critical - no AI prescriptions
            'emergency_warning': 2.5
        }

        super().__init__(
            name='medical_accuracy',
            description='Medical accuracy and safety evaluator',
            rules=rules,
            weights=weights,
            min_passing_score=0.85
        )

    def _check_emergency_language(self, text: str,
                                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if emergency situations are handled appropriately

        Args:
            text: Text to check
            context: Optional context

        Returns:
            Dict with validation results
        """
        # Emergency keywords
        emergency_terms = [
            'chest pain', 'heart attack', 'stroke', 'bleeding',
            'unconscious', 'overdose', 'severe pain', 'emergency'
        ]

        has_emergency = any(term in text.lower() for term in emergency_terms)

        if has_emergency:
            # Must direct to emergency services
            has_911_directive = any(
                phrase in text.lower() for phrase in [
                    'call 911', 'emergency room', 'seek immediate',
                    'go to er', 'emergency medical'
                ]
            )

            return {
                'passed': has_911_directive,
                'score': 1.0 if has_911_directive else 0.0,
                'details': {
                    'has_emergency_terms': True,
                    'has_emergency_directive': has_911_directive
                }
            }

        # No emergency terms, passes by default
        return {
            'passed': True,
            'score': 1.0,
            'details': {'has_emergency_terms': False}
        }


# Example usage and factory function
def create_business_evaluator(evaluator_type: str,
                              **kwargs) -> BusinessLogicEvaluator:
    """
    Factory function to create business logic evaluators

    Args:
        evaluator_type: Type of evaluator ('customer_service', 'financial', 'medical')
        **kwargs: Additional configuration parameters

    Returns:
        BusinessLogicEvaluator instance
    """
    evaluators = {
        'customer_service': CustomerServiceEvaluator,
        'financial': FinancialComplianceEvaluator,
        'medical': MedicalAccuracyEvaluator
    }

    evaluator_class = evaluators.get(evaluator_type)
    if not evaluator_class:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")

    return evaluator_class(**kwargs)


if __name__ == '__main__':
    # Example usage
    print("=" * 80)
    print("Business Logic Evaluator Examples")
    print("=" * 80)

    # Example 1: Customer Service
    print("\n1. Customer Service Evaluator")
    print("-" * 80)
    cs_evaluator = CustomerServiceEvaluator()

    good_response = """
    Hello! Thank you for contacting us.

    I understand how frustrating this situation must be, and I sincerely apologize
    for the inconvenience. I value your business and want to help resolve this issue.

    I will immediately look into your account and work to fix this problem. I'll
    follow up with you within 24 hours with a complete resolution.

    Please feel free to reach out if you have any other questions.

    Best regards,
    Support Team
    """

    bad_response = "Can't help you. Not my problem. Deal with it."

    print("\nGood Response:")
    metrics = cs_evaluator.get_metrics(good_response, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Passed: {metrics['passed']}")
    print(f"Rules Passed: {metrics['rules_passed']}/{metrics['total_rules']}")

    print("\nBad Response:")
    metrics = cs_evaluator.get_metrics(bad_response, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Passed: {metrics['passed']}")
    print(f"Rules Passed: {metrics['rules_passed']}/{metrics['total_rules']}")

    # Example 2: Financial Compliance
    print("\n\n2. Financial Compliance Evaluator")
    print("-" * 80)
    fin_evaluator = FinancialComplianceEvaluator()

    compliant_text = """
    This investment opportunity has shown strong past performance, but past
    performance does not guarantee future results. All investments carry risk,
    and you may lose money. This is not financial advice - please consult with
    a qualified financial advisor before making investment decisions.
    """

    non_compliant_text = """
    This is a guaranteed return investment! You cannot lose money with this
    risk-free opportunity. Guaranteed 20% annual returns!
    """

    print("\nCompliant Text:")
    metrics = fin_evaluator.get_metrics(compliant_text, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Passed: {metrics['passed']}")

    print("\nNon-Compliant Text:")
    metrics = fin_evaluator.get_metrics(non_compliant_text, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Passed: {metrics['passed']}")

    # Example 3: Medical Accuracy
    print("\n\n3. Medical Accuracy Evaluator")
    print("-" * 80)
    med_evaluator = MedicalAccuracyEvaluator()

    safe_text = """
    These symptoms could indicate several conditions. I recommend consulting your
    doctor or healthcare provider for a proper evaluation. They can provide a
    diagnosis and recommend appropriate treatment. This is not medical advice.
    """

    unsafe_text = """
    You definitely have the flu. Take these antibiotics immediately at 500mg
    twice daily. You are diagnosed with influenza type A.
    """

    print("\nSafe Medical Text:")
    metrics = med_evaluator.get_metrics(safe_text, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Passed: {metrics['passed']}")

    print("\nUnsafe Medical Text:")
    metrics = med_evaluator.get_metrics(unsafe_text, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Passed: {metrics['passed']}")

    print("\n" + "=" * 80)
