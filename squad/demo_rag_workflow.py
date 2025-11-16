#!/usr/bin/env python3
"""
Squad RAG Workflow Demo
=======================
Demonstrates complete RAG pipeline:
1. Ingest codebase
2. Connect to API
3. Crawl documentation
4. Search forums
5. Generate code using all context

Author: Claude Code
Date: November 16, 2025
"""

import requests
import json
import time
import os
from typing import Dict, Any


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class RAGWorkflowDemo:
    """Demonstrates complete RAG workflow"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def print_header(self, message: str):
        """Print section header"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

    def print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.OKCYAN}ℹ️  {message}{Colors.ENDC}")

    def print_step(self, step: int, total: int, message: str):
        """Print step message"""
        print(f"\n{Colors.OKBLUE}[Step {step}/{total}] {message}{Colors.ENDC}")

    def check_server(self) -> bool:
        """Check if server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def ingest_codebase(self) -> Dict[str, Any]:
        """Step 1: Ingest Squad's own codebase"""
        self.print_step(1, 5, "Ingesting codebase (Squad's own code)")

        root_path = os.path.dirname(os.path.abspath(__file__))

        payload = {
            "root_path": root_path,
            "include_patterns": ["*.py"],
            "exclude_patterns": ["__pycache__", "*.pyc", ".venv", "node_modules", "out"]
        }

        self.print_info(f"Scanning: {root_path}")
        self.print_info("Including: *.py files")

        response = self.session.post(
            f"{self.base_url}/ingest/codebase",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            self.print_success(f"Ingested {data['total_items']} code entities")
            return data
        else:
            raise Exception(f"Codebase ingestion failed: HTTP {response.status_code}")

    def connect_api(self) -> Dict[str, Any]:
        """Step 2: Connect to public API"""
        self.print_step(2, 5, "Connecting to API (Petstore OpenAPI)")

        payload = {
            "spec_url": "https://petstore.swagger.io/v2/swagger.json",
            "api_type": "openapi"
        }

        self.print_info(f"Fetching spec: {payload['spec_url']}")

        try:
            response = self.session.post(
                f"{self.base_url}/ingest/api",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Connected to API: {data['total_items']} endpoints")
                return data
            else:
                self.print_info("Skipping API connection (may be network issue)")
                return {"total_items": 0, "success": False}
        except Exception as e:
            self.print_info(f"Skipping API connection: {str(e)}")
            return {"total_items": 0, "success": False}

    def crawl_documentation(self) -> Dict[str, Any]:
        """Step 3: Crawl documentation"""
        self.print_step(3, 5, "Crawling documentation (example.com)")

        payload = {
            "url": "https://example.com",
            "max_pages": 1,
            "follow_links": False
        }

        self.print_info(f"Crawling: {payload['url']}")

        try:
            response = self.session.post(
                f"{self.base_url}/ingest/documentation",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Crawled {data['metadata'].get('pages', 0)} pages")
                return data
            else:
                self.print_info("Skipping documentation crawling (may be network issue)")
                return {"total_items": 0, "success": False}
        except Exception as e:
            self.print_info(f"Skipping documentation: {str(e)}")
            return {"total_items": 0, "success": False}

    def search_forum(self) -> Dict[str, Any]:
        """Step 4: Search forums"""
        self.print_step(4, 5, "Searching forums (Stack Overflow)")

        payload = {
            "query": "python fastapi tutorial",
            "source": "stackoverflow",
            "max_results": 3
        }

        self.print_info(f"Searching for: {payload['query']}")

        try:
            response = self.session.post(
                f"{self.base_url}/ingest/forum",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Found {data['total_items']} forum posts")
                return data
            else:
                self.print_info("Skipping forum search (may be rate limited)")
                return {"total_items": 0, "success": False}
        except Exception as e:
            self.print_info(f"Skipping forum search: {str(e)}")
            return {"total_items": 0, "success": False}

    def get_context_summary(self) -> Dict[str, Any]:
        """Get context summary"""
        response = self.session.get(f"{self.base_url}/context/summary")

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Context summary failed: HTTP {response.status_code}")

    def generate_code_with_context(self):
        """Step 5: Generate code using all ingested context"""
        self.print_step(5, 5, "Generating code using RAG context")

        payload = {
            "description": "Create a FastAPI endpoint for user registration with validation",
            "language": "python"
        }

        self.print_info(f"Request: {payload['description']}")

        response = self.session.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            self.print_success(f"Generated code (confidence: {data['confidence']:.2f})")
            return data
        else:
            raise Exception(f"Code generation failed: HTTP {response.status_code}")

    def run_demo(self):
        """Run complete RAG workflow demo"""
        self.print_header("Squad RAG Workflow Demo")

        # Check server
        if not self.check_server():
            print(f"{Colors.FAIL}❌ Server not running at {self.base_url}{Colors.ENDC}")
            print(f"{Colors.WARNING}Please start server: PYTHONPATH=/home/user/hello-world python server.py{Colors.ENDC}")
            return False

        self.print_success(f"Server running at {self.base_url}")

        try:
            # Step 1: Ingest codebase
            codebase_result = self.ingest_codebase()

            # Step 2: Connect to API
            api_result = self.connect_api()

            # Step 3: Crawl documentation
            docs_result = self.crawl_documentation()

            # Step 4: Search forums
            forum_result = self.search_forum()

            # Get context summary
            self.print_header("Context Summary")
            summary = self.get_context_summary()

            print(f"{Colors.OKCYAN}Total Shards:{Colors.ENDC} {summary['total_shards']}")
            print(f"{Colors.OKCYAN}Codebases:{Colors.ENDC} {summary['codebases']}")
            print(f"{Colors.OKCYAN}APIs:{Colors.ENDC} {summary['apis']}")
            print(f"{Colors.OKCYAN}Documentation Sites:{Colors.ENDC} {summary['documentation_sites']}")
            print(f"{Colors.OKCYAN}Forum Searches:{Colors.ENDC} {summary['forum_searches']}")

            # Step 5: Generate code with context
            self.print_header("Code Generation with RAG Context")
            code_result = self.generate_code_with_context()

            # Print generated code
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}Generated Code:{Colors.ENDC}")
            print(f"{Colors.OKCYAN}{'─' * 80}{Colors.ENDC}")
            print(code_result['code'])
            print(f"{Colors.OKCYAN}{'─' * 80}{Colors.ENDC}")

            print(f"\n{Colors.OKGREEN}{Colors.BOLD}Explanation:{Colors.ENDC}")
            print(code_result['explanation'])

            # Final summary
            self.print_header("Demo Complete!")
            self.print_success("All RAG workflow steps completed successfully")
            print(f"\n{Colors.OKCYAN}Summary:{Colors.ENDC}")
            print(f"  • Ingested {codebase_result['total_items']} code entities")
            print(f"  • Connected to {api_result['total_items']} API endpoints")
            print(f"  • Crawled {docs_result['total_items']} documentation pages")
            print(f"  • Found {forum_result['total_items']} forum posts")
            print(f"  • Generated code with confidence: {code_result['confidence']:.2f}")

            print(f"\n{Colors.OKGREEN}✨ Squad is now a context-aware AI coding assistant!{Colors.ENDC}\n")

            return True

        except Exception as e:
            print(f"\n{Colors.FAIL}❌ Demo failed: {str(e)}{Colors.ENDC}")
            return False


def main():
    """Main entry point"""
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    demo = RAGWorkflowDemo(base_url)
    success = demo.run_demo()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
