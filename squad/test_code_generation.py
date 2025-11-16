#!/usr/bin/env python3
"""
Squad Code Generation Test Suite
=================================
Comprehensive tests for LLM-powered code generation endpoints.

Tests all capabilities:
- Code generation
- Code refactoring
- Bug fixing
- Test generation
- Code review
- Code explanation

Author: Claude Code
Date: November 16, 2025
"""

import requests
import json
import time
from typing import Dict, List, Any
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)


class CodeGenerationTester:
    """Test harness for Squad code generation endpoints"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []

    def log(self, message: str, level: str = "INFO"):
        """Pretty logging with colors"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "INFO": Fore.CYAN,
            "SUCCESS": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED
        }
        color = colors.get(level, Fore.WHITE)
        print(f"[{timestamp}] [{color}{level}{Style.RESET_ALL}] {message}")

    def test_health(self) -> bool:
        """Test /health endpoint"""
        self.log("Testing /health endpoint...", "INFO")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy" and data.get("llm_ready"):
                    provider = data.get("llm_provider", {})
                    self.log(f"✅ Health check passed - Provider: {provider.get('provider')}, Model: {provider.get('model')}", "SUCCESS")
                    self.results.append({"test": "health_check", "status": "pass", "provider": provider})
                    return True
                else:
                    self.log(f"⚠️  Server healthy but LLM not ready: {data}", "WARNING")
                    return False
        except requests.exceptions.ConnectionError:
            self.log("❌ Cannot connect to server - is it running?", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ Health check failed: {e}", "ERROR")
            return False

        return False

    def test_generate_code(self):
        """Test /generate endpoint"""
        self.log("Testing /generate endpoint...", "INFO")

        # Test case: Generate a Python function
        payload = {
            "description": "Create a Python function that checks if a number is prime",
            "language": "python"
        }

        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=30
            )
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                code = data.get("code", "")
                explanation = data.get("explanation", "")
                confidence = data.get("confidence", 0)

                # Validate response
                if "def" in code and "prime" in code.lower():
                    self.log(f"✅ Code generation passed (confidence: {confidence:.2f}, {duration:.0f}ms)", "SUCCESS")
                    self.log(f"   Generated {len(code)} chars of code", "INFO")
                    self.results.append({
                        "test": "generate_code",
                        "status": "pass",
                        "confidence": confidence,
                        "duration_ms": duration,
                        "code_length": len(code)
                    })
                else:
                    self.log(f"⚠️  Code generated but may be incomplete", "WARNING")
                    self.results.append({"test": "generate_code", "status": "partial"})
            else:
                self.log(f"❌ Generation failed: HTTP {response.status_code}", "ERROR")
                self.results.append({"test": "generate_code", "status": "fail"})

        except Exception as e:
            self.log(f"❌ Generation error: {e}", "ERROR")
            self.results.append({"test": "generate_code", "status": "error", "error": str(e)})

    def test_refactor_code(self):
        """Test /refactor endpoint"""
        self.log("Testing /refactor endpoint...", "INFO")

        # Test case: Refactor a simple function
        code = """
def calc(x, y, op):
    if op == '+':
        return x + y
    elif op == '-':
        return x - y
    elif op == '*':
        return x * y
    elif op == '/':
        return x / y
"""

        payload = {
            "code": code,
            "instructions": "Refactor to use match/case pattern and add type hints",
            "language": "python"
        }

        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/refactor",
                json=payload,
                timeout=30
            )
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                refactored_code = data.get("code", "")
                diff = data.get("diff")
                confidence = data.get("confidence", 0)

                # Validate refactoring
                if refactored_code and len(refactored_code) > 0:
                    self.log(f"✅ Refactoring passed (confidence: {confidence:.2f}, {duration:.0f}ms)", "SUCCESS")
                    if diff:
                        self.log(f"   Generated diff with {len(diff)} chars", "INFO")
                    self.results.append({
                        "test": "refactor_code",
                        "status": "pass",
                        "confidence": confidence,
                        "duration_ms": duration,
                        "has_diff": diff is not None
                    })
                else:
                    self.log(f"⚠️  Refactoring returned empty code", "WARNING")
                    self.results.append({"test": "refactor_code", "status": "partial"})
            else:
                self.log(f"❌ Refactoring failed: HTTP {response.status_code}", "ERROR")
                self.results.append({"test": "refactor_code", "status": "fail"})

        except Exception as e:
            self.log(f"❌ Refactoring error: {e}", "ERROR")
            self.results.append({"test": "refactor_code", "status": "error", "error": str(e)})

    def test_fix_code(self):
        """Test /fix endpoint"""
        self.log("Testing /fix endpoint...", "INFO")

        # Test case: Fix a buggy function
        buggy_code = """
def divide_numbers(a, b):
    return a / b  # Bug: No zero division check
"""

        payload = {
            "code": buggy_code,
            "error_message": "Fix potential division by zero error",
            "language": "python"
        }

        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/fix",
                json=payload,
                timeout=30
            )
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                fixed_code = data.get("code", "")
                explanation = data.get("explanation", "")
                confidence = data.get("confidence", 0)

                # Validate fix
                if "if" in fixed_code or "try" in fixed_code or "zero" in fixed_code.lower():
                    self.log(f"✅ Bug fix passed (confidence: {confidence:.2f}, {duration:.0f}ms)", "SUCCESS")
                    self.results.append({
                        "test": "fix_code",
                        "status": "pass",
                        "confidence": confidence,
                        "duration_ms": duration
                    })
                else:
                    self.log(f"⚠️  Fix may not address the issue", "WARNING")
                    self.results.append({"test": "fix_code", "status": "partial"})
            else:
                self.log(f"❌ Fix failed: HTTP {response.status_code}", "ERROR")
                self.results.append({"test": "fix_code", "status": "fail"})

        except Exception as e:
            self.log(f"❌ Fix error: {e}", "ERROR")
            self.results.append({"test": "fix_code", "status": "error", "error": str(e)})

    def test_generate_tests(self):
        """Test /tests endpoint"""
        self.log("Testing /tests endpoint...", "INFO")

        # Test case: Generate tests for a function
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

        payload = {
            "code": code,
            "language": "python",
            "test_framework": "pytest"
        }

        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/tests",
                json=payload,
                timeout=30
            )
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                test_code = data.get("code", "")
                confidence = data.get("confidence", 0)

                # Validate test generation
                if "test_" in test_code or "def test" in test_code:
                    self.log(f"✅ Test generation passed (confidence: {confidence:.2f}, {duration:.0f}ms)", "SUCCESS")
                    self.results.append({
                        "test": "generate_tests",
                        "status": "pass",
                        "confidence": confidence,
                        "duration_ms": duration
                    })
                else:
                    self.log(f"⚠️  Tests generated but may be incomplete", "WARNING")
                    self.results.append({"test": "generate_tests", "status": "partial"})
            else:
                self.log(f"❌ Test generation failed: HTTP {response.status_code}", "ERROR")
                self.results.append({"test": "generate_tests", "status": "fail"})

        except Exception as e:
            self.log(f"❌ Test generation error: {e}", "ERROR")
            self.results.append({"test": "generate_tests", "status": "error", "error": str(e)})

    def test_review_code(self):
        """Test /review endpoint"""
        self.log("Testing /review endpoint...", "INFO")

        # Test case: Review problematic code
        code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result
"""

        payload = {
            "code": code,
            "language": "python"
        }

        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/review",
                json=payload,
                timeout=30
            )
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                explanation = data.get("explanation", "")
                confidence = data.get("confidence", 0)

                # Validate review
                if len(explanation) > 50:  # Substantial review
                    self.log(f"✅ Code review passed (confidence: {confidence:.2f}, {duration:.0f}ms)", "SUCCESS")
                    self.log(f"   Review length: {len(explanation)} chars", "INFO")
                    self.results.append({
                        "test": "review_code",
                        "status": "pass",
                        "confidence": confidence,
                        "duration_ms": duration,
                        "review_length": len(explanation)
                    })
                else:
                    self.log(f"⚠️  Review too short", "WARNING")
                    self.results.append({"test": "review_code", "status": "partial"})
            else:
                self.log(f"❌ Code review failed: HTTP {response.status_code}", "ERROR")
                self.results.append({"test": "review_code", "status": "fail"})

        except Exception as e:
            self.log(f"❌ Review error: {e}", "ERROR")
            self.results.append({"test": "review_code", "status": "error", "error": str(e)})

    def test_explain_code(self):
        """Test /explain endpoint"""
        self.log("Testing /explain endpoint...", "INFO")

        # Test case: Explain a complex algorithm
        code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

        payload = {
            "code": code,
            "language": "python",
            "question": "How does this quicksort implementation work?"
        }

        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/explain",
                json=payload,
                timeout=30
            )
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                explanation = data.get("explanation", "")
                confidence = data.get("confidence", 0)

                # Validate explanation
                if "quicksort" in explanation.lower() or "pivot" in explanation.lower():
                    self.log(f"✅ Code explanation passed (confidence: {confidence:.2f}, {duration:.0f}ms)", "SUCCESS")
                    self.log(f"   Explanation length: {len(explanation)} chars", "INFO")
                    self.results.append({
                        "test": "explain_code",
                        "status": "pass",
                        "confidence": confidence,
                        "duration_ms": duration,
                        "explanation_length": len(explanation)
                    })
                else:
                    self.log(f"⚠️  Explanation may be incomplete", "WARNING")
                    self.results.append({"test": "explain_code", "status": "partial"})
            else:
                self.log(f"❌ Explanation failed: HTTP {response.status_code}", "ERROR")
                self.results.append({"test": "explain_code", "status": "fail"})

        except Exception as e:
            self.log(f"❌ Explanation error: {e}", "ERROR")
            self.results.append({"test": "explain_code", "status": "error", "error": str(e)})

    def test_legacy_query(self):
        """Test /query endpoint (legacy)"""
        self.log("Testing /query endpoint (legacy)...", "INFO")

        payload = {
            "text": "What is Thompson Sampling?",
            "mode": "direct"
        }

        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=30
            )
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                confidence = data.get("confidence", 0)

                self.log(f"✅ Legacy query passed (confidence: {confidence:.2f}, {duration:.0f}ms)", "SUCCESS")
                self.results.append({
                    "test": "legacy_query",
                    "status": "pass",
                    "confidence": confidence,
                    "duration_ms": duration
                })
            else:
                self.log(f"❌ Legacy query failed: HTTP {response.status_code}", "ERROR")
                self.results.append({"test": "legacy_query", "status": "fail"})

        except Exception as e:
            self.log(f"❌ Legacy query error: {e}", "ERROR")
            self.results.append({"test": "legacy_query", "status": "error", "error": str(e)})

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}Test Summary{Style.RESET_ALL}")
        print("=" * 80)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "pass")
        partial = sum(1 for r in self.results if r["status"] == "partial")
        failed = sum(1 for r in self.results if r["status"] in ["fail", "error"])

        print(f"\nTotal tests: {total}")
        print(f"{Fore.GREEN}Passed: {passed}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Partial: {partial}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {failed}{Style.RESET_ALL}")

        if passed == total:
            print(f"\n{Fore.GREEN}🎉 All tests passed!{Style.RESET_ALL}")
        elif passed + partial == total:
            print(f"\n{Fore.YELLOW}⚠️  All tests completed with some warnings{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}❌ Some tests failed{Style.RESET_ALL}")

        # Save results to JSON
        with open("test_code_generation_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "passed": passed,
                    "partial": partial,
                    "failed": failed
                },
                "results": self.results
            }, f, indent=2)

        print(f"\n{Fore.CYAN}Results saved to: test_code_generation_results.json{Style.RESET_ALL}")

    def run_all_tests(self) -> bool:
        """Run all tests and return success status"""
        print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Squad Code Generation Test Suite{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

        # Check health first
        if not self.test_health():
            self.log("❌ Server not ready - aborting tests", "ERROR")
            return False

        print()  # Blank line

        # Run all code generation tests
        self.test_generate_code()
        print()

        self.test_refactor_code()
        print()

        self.test_fix_code()
        print()

        self.test_generate_tests()
        print()

        self.test_review_code()
        print()

        self.test_explain_code()
        print()

        self.test_legacy_query()

        # Print summary
        self.print_summary()

        # Return success if all passed
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "pass")
        return passed == total


def main():
    """Main entry point"""
    tester = CodeGenerationTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
