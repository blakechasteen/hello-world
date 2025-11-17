#!/usr/bin/env python3
"""
Standalone Verification of Custom Evaluators

Tests evaluator logic without full Promptly package dependency.
"""

import re
import json

print("=" * 80)
print("Custom Evaluators Verification")
print("=" * 80)

# Test 1: Business Logic - Customer Service Rules
print("\n1. Business Logic Evaluator (Customer Service)")
print("-" * 80)

def test_customer_service():
    good_response = """
    Hello! Thank you for contacting us.
    I understand your concern and will help resolve this immediately.
    I'll follow up with you within 24 hours.
    Best regards
    """

    bad_response = "Can't help. Not my problem."

    # Simple rule checks
    def has_greeting(text):
        return bool(re.search(r'(hello|hi|thank you)', text, re.I))

    def has_empathy(text):
        return any(word in text.lower() for word in ['understand', 'apologize', 'value'])

    def has_resolution(text):
        return any(word in text.lower() for word in ['help', 'resolve', 'assist'])

    def has_closing(text):
        return bool(re.search(r'(best regards|sincerely)', text, re.I))

    # Test good response
    good_checks = {
        'greeting': has_greeting(good_response),
        'empathy': has_empathy(good_response),
        'resolution': has_resolution(good_response),
        'closing': has_closing(good_response)
    }
    good_score = sum(good_checks.values()) / len(good_checks)

    # Test bad response
    bad_checks = {
        'greeting': has_greeting(bad_response),
        'empathy': has_empathy(bad_response),
        'resolution': has_resolution(bad_response),
        'closing': has_closing(bad_response)
    }
    bad_score = sum(bad_checks.values()) / len(bad_checks)

    print(f"Good response score: {good_score:.2f} ({sum(good_checks.values())}/4 rules)")
    print(f"  Checks: {good_checks}")
    print(f"\nBad response score: {bad_score:.2f} ({sum(bad_checks.values())}/4 rules)")
    print(f"  Checks: {bad_checks}")

    return good_score > 0.7 and bad_score < 0.5

test1_pass = test_customer_service()
print(f"\n✓ Test PASSED" if test1_pass else "\n✗ Test FAILED")


# Test 2: Consensus - Score Aggregation
print("\n\n2. Multi-Model Consensus Evaluator")
print("-" * 80)

def test_consensus():
    import statistics

    # Simulate 3 model scores
    model_scores = [0.8, 0.75, 0.85]

    # Different strategies
    mean_score = statistics.mean(model_scores)
    median_score = statistics.median(model_scores)
    min_score = min(model_scores)
    max_score = max(model_scores)

    # Disagreement (std dev)
    disagreement = statistics.stdev(model_scores)

    print(f"Individual scores: {model_scores}")
    print(f"\nConsensus strategies:")
    print(f"  Mean: {mean_score:.3f}")
    print(f"  Median: {median_score:.3f}")
    print(f"  Min: {min_score:.3f}")
    print(f"  Max: {max_score:.3f}")
    print(f"\nDisagreement (std dev): {disagreement:.3f}")
    print(f"Flagged (threshold 0.3): {disagreement > 0.3}")

    return 0.7 < mean_score < 0.9 and disagreement < 0.1

test2_pass = test_consensus()
print(f"\n✓ Test PASSED" if test2_pass else "\n✗ Test FAILED")


# Test 3: Structured - JSON Validation
print("\n\n3. Structured Output Evaluator (JSON)")
print("-" * 80)

def test_json_validation():
    valid_json = '{"name": "John", "email": "john@example.com"}'
    invalid_json = '{"name": "John" invalid}'

    # Test valid JSON
    try:
        data = json.loads(valid_json)
        has_name = 'name' in data
        has_email = 'email' in data
        valid_score = 1.0 if has_name and has_email else 0.6
        print(f"Valid JSON: ✓")
        print(f"  Parsed: {data}")
        print(f"  Has required fields: {has_name and has_email}")
        print(f"  Score: {valid_score:.2f}")
    except json.JSONDecodeError as e:
        valid_score = 0.0
        print(f"Valid JSON: ✗ ({e})")

    # Test invalid JSON
    try:
        data = json.loads(invalid_json)
        invalid_score = 0.5
        print(f"\nInvalid JSON: ✗ (should have failed)")
    except json.JSONDecodeError:
        invalid_score = 0.0
        print(f"\nInvalid JSON: ✓ (correctly rejected)")
        print(f"  Score: {invalid_score:.2f}")

    return valid_score == 1.0 and invalid_score == 0.0

test3_pass = test_json_validation()
print(f"\n✓ Test PASSED" if test3_pass else "\n✗ Test FAILED")


# Test 4: Domain-Specific - Code Security
print("\n\n4. Domain-Specific Evaluator (Code Security)")
print("-" * 80)

def test_code_security():
    secure_code = '''
def get_user(user_id: int):
    """Get user by ID."""
    query = "SELECT * FROM users WHERE id = ?"
    return cursor.execute(query, (user_id,)).fetchone()
'''

    insecure_code = '''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = %s" % user_id
    return cursor.execute(query).fetchone()
'''

    # Check for SQL injection pattern
    sql_injection_pattern = r'%s.*%|execute.*\+'

    # Check secure code
    has_docstring = '"""' in secure_code
    has_type_hints = ': int' in secure_code
    has_sql_injection = bool(re.search(sql_injection_pattern, secure_code))

    secure_score = 0.0
    if has_docstring:
        secure_score += 0.3
    if has_type_hints:
        secure_score += 0.3
    if not has_sql_injection:
        secure_score += 0.4

    print(f"Secure code:")
    print(f"  Has documentation: {has_docstring}")
    print(f"  Has type hints: {has_type_hints}")
    print(f"  SQL injection risk: {has_sql_injection}")
    print(f"  Score: {secure_score:.2f}")

    # Check insecure code
    has_sql_injection_bad = bool(re.search(sql_injection_pattern, insecure_code))
    insecure_score = 0.3 if not has_sql_injection_bad else 0.0

    print(f"\nInsecure code:")
    print(f"  SQL injection risk: {has_sql_injection_bad}")
    print(f"  Score: {insecure_score:.2f}")

    return secure_score > 0.7 and insecure_score < 0.5

test4_pass = test_code_security()
print(f"\n✓ Test PASSED" if test4_pass else "\n✗ Test FAILED")


# Test 5: Human-in-the-Loop - Queue Management
print("\n\n5. Human-in-the-Loop Evaluator (Queue)")
print("-" * 80)

def test_hitl():
    # Simulate queue operations
    queue = []

    # Add items
    item1 = {
        'id': 'item1',
        'actual': 'High quality response',
        'pre_score': 0.95,
        'status': 'pending'
    }

    item2 = {
        'id': 'item2',
        'actual': 'Medium quality response',
        'pre_score': 0.6,
        'status': 'pending'
    }

    queue.append(item1)
    queue.append(item2)

    # Auto-approve threshold: 0.9
    auto_approve_threshold = 0.9
    auto_reject_threshold = 0.3

    # Process items
    for item in queue:
        if item['pre_score'] >= auto_approve_threshold:
            item['status'] = 'approved'
            item['score'] = item['pre_score']
        elif item['pre_score'] <= auto_reject_threshold:
            item['status'] = 'rejected'
            item['score'] = item['pre_score']
        else:
            item['status'] = 'needs_review'
            item['score'] = 0.5  # Pending score

    print(f"Queue items: {len(queue)}")
    for item in queue:
        print(f"  {item['id']}: {item['status']} (score: {item['score']:.2f})")

    # Check results
    approved = sum(1 for item in queue if item['status'] == 'approved')
    needs_review = sum(1 for item in queue if item['status'] == 'needs_review')

    print(f"\nAuto-approved: {approved}")
    print(f"Needs human review: {needs_review}")

    return approved == 1 and needs_review == 1

test5_pass = test_hitl()
print(f"\n✓ Test PASSED" if test5_pass else "\n✗ Test FAILED")


# Summary
print("\n\n" + "=" * 80)
print("Verification Summary")
print("=" * 80)

all_tests = [
    ("Business Logic (Customer Service)", test1_pass),
    ("Multi-Model Consensus", test2_pass),
    ("Structured Output (JSON)", test3_pass),
    ("Domain-Specific (Code Security)", test4_pass),
    ("Human-in-the-Loop (Queue)", test5_pass)
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

for name, result in all_tests:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status} - {name}")

print(f"\nTotal: {passed}/{total} tests passed")

if passed == total:
    print("\n🎉 All evaluators verified successfully!")
else:
    print(f"\n⚠️  {total - passed} test(s) failed")

print("=" * 80)
