#!/usr/bin/env python3
"""
Comprehensive Test Suite for Custom Evaluators

Tests all custom evaluator implementations:
- Business Logic Evaluators
- Multi-Model Consensus Evaluator
- Structured Output Evaluators
- Domain-Specific Evaluators
- Human-in-the-Loop Evaluator
"""

import unittest
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_evaluators.business_logic import (
    CustomerServiceEvaluator,
    FinancialComplianceEvaluator,
    MedicalAccuracyEvaluator
)

from custom_evaluators.consensus import (
    MultiModelConsensusEvaluator,
    ConsensusStrategy,
    MockProvider
)

from custom_evaluators.structured import (
    JSONValidator,
    XMLValidator,
    YAMLValidator,
    SQLValidator
)

from custom_evaluators.domain_specific import (
    MedicalTerminologyEvaluator,
    LegalCitationEvaluator,
    CodeSecurityEvaluator,
    BrandVoiceEvaluator
)

from custom_evaluators.hitl import (
    HumanInTheLoopEvaluator,
    ReviewQueue,
    ReviewStatus,
    InterAnnotatorAgreement
)


class TestBusinessLogicEvaluators(unittest.TestCase):
    """Test business logic evaluators"""

    def test_customer_service_good_response(self):
        """Test customer service evaluator with good response"""
        evaluator = CustomerServiceEvaluator()

        good_response = """
        Hello! Thank you for contacting us.
        I understand your concern and apologize for the inconvenience.
        I will help resolve this issue immediately.
        Please feel free to reach out with any questions.
        Best regards
        """

        score = evaluator.evaluate(good_response, "")
        self.assertGreater(score, 0.7, "Good response should score > 0.7")

        metrics = evaluator.get_metrics(good_response, "")
        self.assertTrue(metrics['passed'], "Good response should pass")

    def test_customer_service_bad_response(self):
        """Test customer service evaluator with bad response"""
        evaluator = CustomerServiceEvaluator()

        bad_response = "Can't help. Not my problem."

        score = evaluator.evaluate(bad_response, "")
        self.assertLess(score, 0.5, "Bad response should score < 0.5")

    def test_financial_compliance_compliant(self):
        """Test financial compliance with compliant text"""
        evaluator = FinancialComplianceEvaluator()

        compliant = """
        Past performance does not guarantee future results. All investments
        carry risk and you may lose money. This is not financial advice.
        Please consult with a qualified financial advisor.
        """

        score = evaluator.evaluate(compliant, "")
        self.assertGreater(score, 0.8, "Compliant text should score > 0.8")

    def test_financial_compliance_non_compliant(self):
        """Test financial compliance with non-compliant text"""
        evaluator = FinancialComplianceEvaluator()

        non_compliant = "Guaranteed returns! Risk-free investment! Cannot lose!"

        score = evaluator.evaluate(non_compliant, "")
        self.assertLess(score, 0.5, "Non-compliant text should score < 0.5")

    def test_medical_accuracy_safe_text(self):
        """Test medical accuracy with safe text"""
        evaluator = MedicalAccuracyEvaluator()

        safe_text = """
        These symptoms could indicate various conditions. Please consult your
        doctor or healthcare provider for proper evaluation. This is not
        medical advice.
        """

        score = evaluator.evaluate(safe_text, "")
        self.assertGreater(score, 0.7, "Safe medical text should score > 0.7")

    def test_medical_accuracy_unsafe_text(self):
        """Test medical accuracy with unsafe text"""
        evaluator = MedicalAccuracyEvaluator()

        unsafe_text = "You definitely have cancer. Take this medication I prescribe."

        score = evaluator.evaluate(unsafe_text, "")
        self.assertLess(score, 0.5, "Unsafe medical text should score < 0.5")


class TestConsensusEvaluator(unittest.TestCase):
    """Test multi-model consensus evaluator"""

    def test_mean_consensus(self):
        """Test mean consensus strategy"""
        providers = [
            MockProvider("model_a", bias=0.8),
            MockProvider("model_b", bias=0.7),
            MockProvider("model_c", bias=0.6)
        ]

        evaluator = MultiModelConsensusEvaluator(
            providers=providers,
            strategy=ConsensusStrategy.MEAN
        )

        score = evaluator.evaluate("test output", "expected output")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_weighted_consensus(self):
        """Test weighted consensus strategy"""
        providers = [
            MockProvider("model_a", bias=0.9),
            MockProvider("model_b", bias=0.5)
        ]

        evaluator = MultiModelConsensusEvaluator(
            providers=providers,
            strategy=ConsensusStrategy.WEIGHTED,
            weights={"model_a": 2.0, "model_b": 1.0}
        )

        score = evaluator.evaluate("test output", "expected output")
        # Score should be closer to model_a due to higher weight
        self.assertGreater(score, 0.5)

    def test_consensus_metrics(self):
        """Test consensus metrics"""
        providers = [
            MockProvider("model_a", bias=0.8),
            MockProvider("model_b", bias=0.8),
            MockProvider("model_c", bias=0.8)
        ]

        evaluator = MultiModelConsensusEvaluator(providers=providers)

        metrics = evaluator.get_metrics("test", "expected")
        self.assertIn('score', metrics)
        self.assertIn('individual_scores', metrics)
        self.assertIn('disagreement', metrics)
        self.assertEqual(len(metrics['individual_scores']), 3)

    def test_cache_functionality(self):
        """Test caching functionality"""
        providers = [MockProvider("model", bias=0.7)]

        evaluator = MultiModelConsensusEvaluator(
            providers=providers,
            cache_enabled=True
        )

        # First call
        metrics1 = evaluator.get_metrics("test", "expected")
        self.assertFalse(metrics1['cache_hit'])

        # Second call (should hit cache)
        metrics2 = evaluator.get_metrics("test", "expected")
        self.assertTrue(metrics2['cache_hit'])
        self.assertEqual(metrics1['score'], metrics2['score'])


class TestStructuredEvaluators(unittest.TestCase):
    """Test structured output evaluators"""

    def test_json_validator_valid(self):
        """Test JSON validator with valid JSON"""
        schema = {
            "type": "object",
            "required": ["name", "email"],
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            }
        }

        validator = JSONValidator(schema=schema, required_fields=["name", "email"])

        valid_json = '{"name": "John", "email": "john@example.com"}'

        score = validator.evaluate(valid_json, "")
        self.assertEqual(score, 1.0, "Valid JSON should score 1.0")

    def test_json_validator_invalid(self):
        """Test JSON validator with invalid JSON"""
        validator = JSONValidator(required_fields=["name"])

        invalid_json = '{"invalid": json}'

        score = validator.evaluate(invalid_json, "")
        self.assertEqual(score, 0.0, "Invalid JSON should score 0.0")

    def test_json_validator_missing_fields(self):
        """Test JSON validator with missing required fields"""
        validator = JSONValidator(required_fields=["name", "email"])

        partial_json = '{"name": "John"}'

        score = validator.evaluate(partial_json, "")
        self.assertLess(score, 1.0, "JSON with missing fields should score < 1.0")

    def test_sql_validator_safe(self):
        """Test SQL validator with safe query"""
        validator = SQLValidator(
            allowed_operations={'SELECT'},
            forbidden_operations={'DROP', 'DELETE'}
        )

        safe_query = "SELECT * FROM users WHERE id = 1"

        score = validator.evaluate(safe_query, "")
        self.assertEqual(score, 1.0, "Safe SQL should score 1.0")

    def test_sql_validator_unsafe(self):
        """Test SQL validator with unsafe query"""
        validator = SQLValidator(forbidden_operations={'DROP'})

        unsafe_query = "DROP TABLE users"

        score = validator.evaluate(unsafe_query, "")
        self.assertLess(score, 0.5, "Unsafe SQL should score < 0.5")

    def test_yaml_validator_valid(self):
        """Test YAML validator with valid YAML"""
        validator = YAMLValidator(required_keys=['name', 'version'])

        valid_yaml = """
name: my-app
version: 1.0.0
dependencies:
  - package1
"""

        score = validator.evaluate(valid_yaml, "")
        self.assertEqual(score, 1.0, "Valid YAML should score 1.0")

    def test_yaml_validator_missing_keys(self):
        """Test YAML validator with missing keys"""
        validator = YAMLValidator(required_keys=['name', 'version'])

        incomplete_yaml = "name: my-app"

        score = validator.evaluate(incomplete_yaml, "")
        self.assertLess(score, 1.0, "YAML with missing keys should score < 1.0")


class TestDomainSpecificEvaluators(unittest.TestCase):
    """Test domain-specific evaluators"""

    def test_medical_terminology_professional(self):
        """Test medical terminology with professional language"""
        evaluator = MedicalTerminologyEvaluator()

        professional = """
        The patient presents with acute dyspnea and tachycardia. Cardiac
        examination reveals abnormal findings. Recommend cardiology consultation.
        """

        score = evaluator.evaluate(professional, "")
        self.assertGreater(score, 0.6, "Professional medical text should score > 0.6")

    def test_medical_terminology_unprofessional(self):
        """Test medical terminology with unprofessional language"""
        evaluator = MedicalTerminologyEvaluator()

        unprofessional = "The patient has a tummy ache and their heart is beating really fast."

        score = evaluator.evaluate(unprofessional, "")
        self.assertLess(score, 0.5, "Unprofessional text should score < 0.5")

    def test_legal_citation_proper(self):
        """Test legal citation evaluator with proper citations"""
        evaluator = LegalCitationEvaluator()

        proper_legal = """
        Introduction: This case concerns 42 USC § 1983.
        Analysis: The court in Miranda v. Arizona, 384 U.S. 436 (1966), held...
        Conclusion: Therefore, the motion should be granted.
        """

        score = evaluator.evaluate(proper_legal, "")
        self.assertGreater(score, 0.7, "Proper legal text should score > 0.7")

    def test_code_security_secure(self):
        """Test code security with secure code"""
        evaluator = CodeSecurityEvaluator(language="python")

        secure_code = '''
def get_user(user_id: int):
    """Get user by ID."""
    try:
        query = "SELECT * FROM users WHERE id = ?"
        return cursor.execute(query, (user_id,)).fetchone()
    except Exception as e:
        logger.error(e)
        raise
'''

        score = evaluator.evaluate(secure_code, "")
        self.assertGreater(score, 0.7, "Secure code should score > 0.7")

    def test_code_security_insecure(self):
        """Test code security with insecure code"""
        evaluator = CodeSecurityEvaluator(language="python")

        insecure_code = '''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = %s" % user_id
    return cursor.execute(query).fetchone()
'''

        score = evaluator.evaluate(insecure_code, "")
        self.assertLess(score, 0.7, "Insecure code should score < 0.7")

    def test_brand_voice_on_brand(self):
        """Test brand voice with on-brand content"""
        evaluator = BrandVoiceEvaluator()

        on_brand = """
        We're excited to help you! Our team is here to support you every step
        of the way. Together, we'll find the perfect solution. Thank you!
        """

        score = evaluator.evaluate(on_brand, "")
        self.assertGreater(score, 0.6, "On-brand content should score > 0.6")

    def test_brand_voice_off_brand(self):
        """Test brand voice with off-brand content"""
        evaluator = BrandVoiceEvaluator()

        off_brand = "This is expensive and complicated. Can't help you."

        score = evaluator.evaluate(off_brand, "")
        self.assertLess(score, 0.5, "Off-brand content should score < 0.5")


class TestHITLEvaluator(unittest.TestCase):
    """Test human-in-the-loop evaluator"""

    def setUp(self):
        """Set up test fixtures"""
        self.queue_path = Path('/tmp/test_hitl_queue.json')
        if self.queue_path.exists():
            self.queue_path.unlink()

    def tearDown(self):
        """Clean up test fixtures"""
        if self.queue_path.exists():
            self.queue_path.unlink()

    def test_review_queue_add(self):
        """Test adding items to review queue"""
        queue = ReviewQueue(backend="file", config={'file_path': str(self.queue_path)})

        item_id = queue.add("test output", "expected output", {"type": "test"})
        self.assertIsNotNone(item_id)

        item = queue.get(item_id)
        self.assertIsNotNone(item)
        self.assertEqual(item.actual, "test output")
        self.assertEqual(item.status, ReviewStatus.PENDING)

    def test_review_queue_update(self):
        """Test updating review items"""
        queue = ReviewQueue(backend="file", config={'file_path': str(self.queue_path)})

        item_id = queue.add("test", "expected")
        queue.update(item_id, ReviewStatus.APPROVED, score=0.9, feedback="Good", reviewer="test")

        item = queue.get(item_id)
        self.assertEqual(item.status, ReviewStatus.APPROVED)
        self.assertEqual(item.score, 0.9)
        self.assertEqual(item.feedback, "Good")

    def test_review_queue_pending(self):
        """Test getting pending reviews"""
        queue = ReviewQueue(backend="file", config={'file_path': str(self.queue_path)})

        # Add multiple items
        queue.add("test1", "expected1")
        queue.add("test2", "expected2")
        queue.add("test3", "expected3")

        pending = queue.get_pending(limit=10)
        self.assertGreaterEqual(len(pending), 3)

    def test_hitl_evaluator_auto_approve(self):
        """Test HITL evaluator with auto-approve"""
        queue = ReviewQueue(backend="file", config={'file_path': str(self.queue_path)})
        evaluator = HumanInTheLoopEvaluator(
            queue=queue,
            auto_approve_threshold=0.9
        )

        score = evaluator.evaluate(
            "high quality",
            "expected",
            context={'pre_score': 0.95}
        )

        self.assertEqual(score, 0.95, "Should auto-approve with high score")

    def test_hitl_evaluator_pending(self):
        """Test HITL evaluator with pending review"""
        queue = ReviewQueue(backend="file", config={'file_path': str(self.queue_path)})
        evaluator = HumanInTheLoopEvaluator(queue=queue)

        score = evaluator.evaluate("medium quality", "expected")
        self.assertEqual(score, 0.5, "Should return 0.5 for pending review")

    def test_inter_annotator_agreement_perfect(self):
        """Test inter-annotator agreement with perfect agreement"""
        ratings1 = [5, 4, 3, 2, 1]
        ratings2 = [5, 4, 3, 2, 1]

        agreement = InterAnnotatorAgreement()

        percentage = agreement.percentage_agreement(ratings1, ratings2)
        self.assertEqual(percentage, 1.0, "Perfect agreement should be 1.0")

        kappa = agreement.cohens_kappa(ratings1, ratings2)
        self.assertEqual(kappa, 1.0, "Perfect agreement should have kappa = 1.0")

    def test_inter_annotator_agreement_partial(self):
        """Test inter-annotator agreement with partial agreement"""
        ratings1 = [5, 4, 3, 5, 2]
        ratings2 = [5, 4, 4, 5, 2]

        agreement = InterAnnotatorAgreement()

        percentage = agreement.percentage_agreement(ratings1, ratings2)
        self.assertEqual(percentage, 0.8, "80% agreement")

        kappa = agreement.cohens_kappa(ratings1, ratings2)
        self.assertGreater(kappa, 0.0, "Should have positive kappa")


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple evaluators"""

    def test_pipeline_evaluation(self):
        """Test evaluation pipeline with multiple evaluators"""
        # Customer service pipeline
        text = """
        Hello! Thank you for contacting us. I understand your concern and
        will help resolve this immediately. Best regards.
        """

        # Test with multiple evaluators
        cs_eval = CustomerServiceEvaluator()
        cs_score = cs_eval.evaluate(text, "")
        self.assertGreater(cs_score, 0.7)

        # Combine with brand voice
        brand_eval = BrandVoiceEvaluator()
        brand_score = brand_eval.evaluate(text, "")

        # Combined score
        combined_score = (cs_score * 0.6 + brand_score * 0.4)
        self.assertGreater(combined_score, 0.5)

    def test_structured_with_business_logic(self):
        """Test structured validation with business logic"""
        json_text = '{"name": "John", "email": "john@example.com", "age": 30}'

        # Validate structure
        json_eval = JSONValidator(required_fields=["name", "email"])
        json_score = json_eval.evaluate(json_text, "")
        self.assertGreater(json_score, 0.8)

        # Additional business logic validation could be added here


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBusinessLogicEvaluators))
    suite.addTests(loader.loadTestsFromTestCase(TestConsensusEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestStructuredEvaluators))
    suite.addTests(loader.loadTestsFromTestCase(TestDomainSpecificEvaluators))
    suite.addTests(loader.loadTestsFromTestCase(TestHITLEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
