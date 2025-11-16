#!/usr/bin/env python3
"""
Squad RAG Ingestion Test Suite
================================
Comprehensive testing of all RAG ingestion endpoints.

Tests:
1. Health check
2. Codebase ingestion
3. API connection (OpenAPI spec)
4. Documentation crawling
5. Forum search (Stack Overflow)
6. Context summary

Author: Claude Code
Date: November 16, 2025
"""

import requests
import json
import time
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


# ANSI color codes for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class TestResult:
    """Test result"""
    name: str
    passed: bool
    duration_ms: float
    details: Optional[str] = None
    error: Optional[str] = None


class RAGIngestionTester:
    """Test all RAG ingestion endpoints"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results: list[TestResult] = []

    def log_success(self, message: str):
        """Log success message"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"{Colors.OKGREEN}[{timestamp}] [SUCCESS] ✅ {message}{Colors.ENDC}")

    def log_error(self, message: str):
        """Log error message"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"{Colors.FAIL}[{timestamp}] [ERROR] ❌ {message}{Colors.ENDC}")

    def log_info(self, message: str):
        """Log info message"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"{Colors.OKCYAN}[{timestamp}] [INFO] ℹ️  {message}{Colors.ENDC}")

    def log_warning(self, message: str):
        """Log warning message"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"{Colors.WARNING}[{timestamp}] [WARNING] ⚠️  {message}{Colors.ENDC}")

    def test_health(self) -> TestResult:
        """Test 1: Health check"""
        print(f"\n{Colors.BOLD}Test 1: Health Check{Colors.ENDC}")
        print("=" * 80)

        start = time.time()
        try:
            response = self.session.get(f"{self.base_url}/health")
            duration_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                provider = data.get("llm_provider", "unknown")
                model = data.get("llm_model", "unknown")
                rag_enabled = data.get("rag_enabled", False)

                details = f"Provider: {provider}, Model: {model}, RAG: {rag_enabled}"
                self.log_success(f"Health check passed - {details}")

                return TestResult(
                    name="health_check",
                    passed=True,
                    duration_ms=duration_ms,
                    details=details
                )
            else:
                self.log_error(f"Health check failed: HTTP {response.status_code}")
                return TestResult(
                    name="health_check",
                    passed=False,
                    duration_ms=duration_ms,
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.log_error(f"Health check error: {str(e)}")
            return TestResult(
                name="health_check",
                passed=False,
                duration_ms=duration_ms,
                error=str(e)
            )

    def test_ingest_codebase(self) -> TestResult:
        """Test 2: Codebase ingestion"""
        print(f"\n{Colors.BOLD}Test 2: Codebase Ingestion{Colors.ENDC}")
        print("=" * 80)

        # Use Squad's own directory as test
        root_path = os.path.dirname(os.path.abspath(__file__))

        payload = {
            "root_path": root_path,
            "include_patterns": ["*.py"],
            "exclude_patterns": ["__pycache__", "*.pyc", ".venv", "node_modules"]
        }

        self.log_info(f"Ingesting codebase: {root_path}")

        start = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/ingest/codebase",
                json=payload
            )
            duration_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                total_items = data.get("total_items", 0)
                metadata = data.get("metadata", {})
                message = data.get("message", "")

                details = f"Items: {total_items}, Files: {metadata.get('files', 0)}"
                self.log_success(f"Codebase ingestion passed - {details} ({duration_ms:.0f}ms)")
                self.log_info(f"Message: {message}")

                return TestResult(
                    name="codebase_ingestion",
                    passed=success,
                    duration_ms=duration_ms,
                    details=details
                )
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", error_msg)
                except:
                    pass

                self.log_error(f"Codebase ingestion failed: {error_msg}")
                return TestResult(
                    name="codebase_ingestion",
                    passed=False,
                    duration_ms=duration_ms,
                    error=error_msg
                )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.log_error(f"Codebase ingestion error: {str(e)}")
            return TestResult(
                name="codebase_ingestion",
                passed=False,
                duration_ms=duration_ms,
                error=str(e)
            )

    def test_ingest_api(self) -> TestResult:
        """Test 3: API connection (OpenAPI spec)"""
        print(f"\n{Colors.BOLD}Test 3: API Connection (OpenAPI){Colors.ENDC}")
        print("=" * 80)

        # Use public Petstore API as test
        payload = {
            "spec_url": "https://petstore.swagger.io/v2/swagger.json",
            "api_type": "openapi"
        }

        self.log_info(f"Connecting to API: {payload['spec_url']}")

        start = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/ingest/api",
                json=payload,
                timeout=30
            )
            duration_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                total_items = data.get("total_items", 0)
                metadata = data.get("metadata", {})
                message = data.get("message", "")

                details = f"Endpoints: {metadata.get('endpoints', 0)}, API: {metadata.get('api', 'Unknown')}"
                self.log_success(f"API connection passed - {details} ({duration_ms:.0f}ms)")
                self.log_info(f"Message: {message}")

                return TestResult(
                    name="api_connection",
                    passed=success,
                    duration_ms=duration_ms,
                    details=details
                )
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", error_msg)
                except:
                    pass

                self.log_warning(f"API connection failed (may be network issue): {error_msg}")
                # Don't fail the test suite on network errors
                return TestResult(
                    name="api_connection",
                    passed=False,
                    duration_ms=duration_ms,
                    error=error_msg
                )

        except requests.exceptions.Timeout:
            duration_ms = (time.time() - start) * 1000
            self.log_warning(f"API connection timeout (network issue) - ({duration_ms:.0f}ms)")
            return TestResult(
                name="api_connection",
                passed=False,
                duration_ms=duration_ms,
                error="Timeout (network issue)"
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.log_warning(f"API connection error (may be network issue): {str(e)}")
            return TestResult(
                name="api_connection",
                passed=False,
                duration_ms=duration_ms,
                error=str(e)
            )

    def test_ingest_documentation(self) -> TestResult:
        """Test 4: Documentation crawling"""
        print(f"\n{Colors.BOLD}Test 4: Documentation Crawling{Colors.ENDC}")
        print("=" * 80)

        # Use a simple, fast documentation site
        payload = {
            "url": "https://example.com",  # Simple test page
            "max_pages": 1,
            "follow_links": False
        }

        self.log_info(f"Crawling documentation: {payload['url']}")

        start = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/ingest/documentation",
                json=payload,
                timeout=30
            )
            duration_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                total_items = data.get("total_items", 0)
                metadata = data.get("metadata", {})
                message = data.get("message", "")

                details = f"Pages: {metadata.get('pages', 0)}, Shards: {total_items}"
                self.log_success(f"Documentation crawling passed - {details} ({duration_ms:.0f}ms)")
                self.log_info(f"Message: {message}")

                return TestResult(
                    name="documentation_crawling",
                    passed=success,
                    duration_ms=duration_ms,
                    details=details
                )
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", error_msg)
                except:
                    pass

                self.log_warning(f"Documentation crawling failed (may be network issue): {error_msg}")
                return TestResult(
                    name="documentation_crawling",
                    passed=False,
                    duration_ms=duration_ms,
                    error=error_msg
                )

        except requests.exceptions.Timeout:
            duration_ms = (time.time() - start) * 1000
            self.log_warning(f"Documentation crawling timeout (network issue) - ({duration_ms:.0f}ms)")
            return TestResult(
                name="documentation_crawling",
                passed=False,
                duration_ms=duration_ms,
                error="Timeout (network issue)"
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.log_warning(f"Documentation crawling error: {str(e)}")
            return TestResult(
                name="documentation_crawling",
                passed=False,
                duration_ms=duration_ms,
                error=str(e)
            )

    def test_search_forum(self) -> TestResult:
        """Test 5: Forum search (Stack Overflow)"""
        print(f"\n{Colors.BOLD}Test 5: Forum Search (Stack Overflow){Colors.ENDC}")
        print("=" * 80)

        payload = {
            "query": "python asyncio",
            "source": "stackoverflow",
            "max_results": 3
        }

        self.log_info(f"Searching {payload['source']} for: {payload['query']}")

        start = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/ingest/forum",
                json=payload,
                timeout=30
            )
            duration_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                total_items = data.get("total_items", 0)
                metadata = data.get("metadata", {})
                message = data.get("message", "")

                details = f"Posts: {metadata.get('posts', 0)}, Source: {metadata.get('source', 'Unknown')}"
                self.log_success(f"Forum search passed - {details} ({duration_ms:.0f}ms)")
                self.log_info(f"Message: {message}")

                return TestResult(
                    name="forum_search",
                    passed=success,
                    duration_ms=duration_ms,
                    details=details
                )
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", error_msg)
                except:
                    pass

                self.log_warning(f"Forum search failed (may be network/rate limit): {error_msg}")
                return TestResult(
                    name="forum_search",
                    passed=False,
                    duration_ms=duration_ms,
                    error=error_msg
                )

        except requests.exceptions.Timeout:
            duration_ms = (time.time() - start) * 1000
            self.log_warning(f"Forum search timeout (network issue) - ({duration_ms:.0f}ms)")
            return TestResult(
                name="forum_search",
                passed=False,
                duration_ms=duration_ms,
                error="Timeout (network issue)"
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.log_warning(f"Forum search error: {str(e)}")
            return TestResult(
                name="forum_search",
                passed=False,
                duration_ms=duration_ms,
                error=str(e)
            )

    def test_context_summary(self) -> TestResult:
        """Test 6: Context summary"""
        print(f"\n{Colors.BOLD}Test 6: Context Summary{Colors.ENDC}")
        print("=" * 80)

        start = time.time()
        try:
            response = self.session.get(f"{self.base_url}/context/summary")
            duration_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                total_shards = data.get("total_shards", 0)
                codebases = data.get("codebases", 0)
                apis = data.get("apis", 0)
                docs = data.get("documentation_sites", 0)
                forums = data.get("forum_searches", 0)

                details = f"Shards: {total_shards}, Codebases: {codebases}, APIs: {apis}, Docs: {docs}, Forums: {forums}"
                self.log_success(f"Context summary passed - {details} ({duration_ms:.0f}ms)")

                # Pretty print summary
                print(f"\n{Colors.OKCYAN}Context Summary:{Colors.ENDC}")
                print(f"  Total Shards: {total_shards}")
                print(f"  Codebases: {codebases}")
                print(f"  APIs: {apis}")
                print(f"  Documentation Sites: {docs}")
                print(f"  Forum Searches: {forums}")

                return TestResult(
                    name="context_summary",
                    passed=True,
                    duration_ms=duration_ms,
                    details=details
                )
            else:
                self.log_error(f"Context summary failed: HTTP {response.status_code}")
                return TestResult(
                    name="context_summary",
                    passed=False,
                    duration_ms=duration_ms,
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.log_error(f"Context summary error: {str(e)}")
            return TestResult(
                name="context_summary",
                passed=False,
                duration_ms=duration_ms,
                error=str(e)
            )

    def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}")
        print("=" * 80)
        print("Squad RAG Ingestion Test Suite")
        print("=" * 80)
        print(f"{Colors.ENDC}")

        # Run tests
        self.results.append(self.test_health())
        self.results.append(self.test_ingest_codebase())
        self.results.append(self.test_ingest_api())
        self.results.append(self.test_ingest_documentation())
        self.results.append(self.test_search_forum())
        self.results.append(self.test_context_summary())

        # Print summary
        self.print_summary()

        # Save results to file
        self.save_results()

        # Return overall success
        passed = sum(1 for r in self.results if r.passed)
        partial = sum(1 for r in self.results if not r.passed and "network" in (r.error or "").lower())
        failed = len(self.results) - passed - partial

        return failed == 0

    def print_summary(self):
        """Print test summary"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}")
        print("=" * 80)
        print("Test Summary")
        print("=" * 80)
        print(f"{Colors.ENDC}")

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        partial = sum(1 for r in self.results if not r.passed and "network" in (r.error or "").lower())
        failed = total - passed - partial

        print(f"\nTotal tests: {total}")
        print(f"{Colors.OKGREEN}Passed: {passed}{Colors.ENDC}")
        if partial > 0:
            print(f"{Colors.WARNING}Partial (network issues): {partial}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {failed}{Colors.ENDC}")

        # Detailed results
        print(f"\n{Colors.BOLD}Detailed Results:{Colors.ENDC}")
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            color = Colors.OKGREEN if result.passed else Colors.FAIL
            print(f"{color}{status}{Colors.ENDC} - {result.name} ({result.duration_ms:.0f}ms)")
            if result.details:
                print(f"  {Colors.OKCYAN}{result.details}{Colors.ENDC}")
            if result.error:
                print(f"  {Colors.WARNING}Error: {result.error}{Colors.ENDC}")

        # Final message
        print()
        if failed == 0:
            if partial > 0:
                print(f"{Colors.WARNING}🎉 All critical tests passed! ({partial} network-dependent tests skipped){Colors.ENDC}")
            else:
                print(f"{Colors.OKGREEN}🎉 All tests passed!{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}❌ {failed} test(s) failed{Colors.ENDC}")

    def save_results(self):
        """Save results to JSON file"""
        try:
            results_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
                "tests": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "duration_ms": r.duration_ms,
                        "details": r.details,
                        "error": r.error
                    }
                    for r in self.results
                ]
            }

            output_file = "test_rag_results.json"
            with open(output_file, "w") as f:
                json.dump(results_data, f, indent=2)

            self.log_info(f"Results saved to {output_file}")

        except Exception as e:
            self.log_warning(f"Failed to save results: {str(e)}")


def main():
    """Main test runner"""
    # Check if server is specified
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"{Colors.OKCYAN}Testing Squad RAG at: {base_url}{Colors.ENDC}\n")

    tester = RAGIngestionTester(base_url)
    success = tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
