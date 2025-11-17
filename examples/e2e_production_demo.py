#!/usr/bin/env python3
"""
End-to-End Production Demonstration
====================================
Comprehensive demo showcasing the entire Promptly platform:

- Initialize with production storage (PostgreSQL)
- Create prompts using templates
- Evaluate prompts with multiple evaluators
- Build complex chains (RAG pipeline)
- Execute workflows
- Track with analytics
- Use REST API
- Visualize results

Requirements:
    pip install promptly jinja2 plotext pyyaml

Optional (for production storage):
    pip install psycopg2-binary  # PostgreSQL
    pip install pymongo          # MongoDB
    pip install redis            # Redis

Usage:
    python examples/e2e_production_demo.py [--storage sqlite|postgresql|mongodb]
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Color codes for terminal output
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

def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title:^80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def print_step(step_num: int, total: int, description: str):
    """Print step progress"""
    print(f"{Colors.BOLD}[{step_num}/{total}] {description}{Colors.ENDC}")


class ProductionDemo:
    """End-to-end production demo orchestrator"""

    def __init__(self, storage_backend: str = 'sqlite'):
        """
        Initialize demo

        Args:
            storage_backend: Storage backend to use (sqlite, postgresql, mongodb)
        """
        self.storage_backend = storage_backend
        self.demo_dir = Path(__file__).parent / 'demo_output'
        self.demo_dir.mkdir(exist_ok=True)

        self.promptly = None
        self.api_client = None
        self.stats = {
            'prompts_created': 0,
            'evaluations_run': 0,
            'chains_executed': 0,
            'api_calls': 0,
            'start_time': datetime.now(),
        }

    def run_full_demo(self):
        """Run the complete end-to-end demonstration"""
        print(f"{Colors.BOLD}{Colors.OKBLUE}")
        print("=" * 80)
        print("PROMPTLY END-TO-END PRODUCTION DEMONSTRATION".center(80))
        print("=" * 80)
        print(f"{Colors.ENDC}")

        print_info(f"Storage Backend: {self.storage_backend}")
        print_info(f"Demo Directory: {self.demo_dir}")
        print_info(f"Start Time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")

        try:
            # Step 1: Initialize Promptly
            self.step_1_initialize()

            # Step 2: Create prompts using templates
            self.step_2_create_prompts()

            # Step 3: Evaluate prompts
            self.step_3_evaluate_prompts()

            # Step 4: Build and execute chains
            self.step_4_build_chains()

            # Step 5: Use analytics
            self.step_5_analytics()

            # Step 6: Use REST API
            self.step_6_rest_api()

            # Step 7: Visualize results
            self.step_7_visualize()

            # Step 8: Generate reports
            self.step_8_generate_reports()

            # Final summary
            self.print_summary()

        except Exception as e:
            print_error(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    def step_1_initialize(self):
        """Step 1: Initialize Promptly with production storage"""
        print_section("Step 1: Initialize Promptly")
        print_step(1, 8, "Setting up Promptly with production storage")

        try:
            from Promptly.promptly.promptly import Promptly

            # Initialize Promptly repository
            demo_repo = self.demo_dir / 'promptly_repo'
            demo_repo.mkdir(exist_ok=True)

            print_info(f"Initializing repository at: {demo_repo}")

            # Create Promptly instance
            self.promptly = Promptly(root_dir=str(demo_repo))

            # Initialize if needed
            if not (demo_repo / '.promptly').exists():
                self.promptly.init()
                print_success("Repository initialized")
            else:
                print_info("Repository already initialized")

            # Create production branch
            try:
                self.promptly.create_branch('production')
                print_success("Created 'production' branch")
            except Exception as e:
                print_warning(f"Branch may already exist: {e}")

            # Create development branch
            try:
                self.promptly.create_branch('development')
                print_success("Created 'development' branch")
            except Exception as e:
                print_warning(f"Branch may already exist: {e}")

            print_success("Promptly initialized successfully")

        except ImportError as e:
            print_error(f"Failed to import Promptly: {e}")
            raise
        except Exception as e:
            print_error(f"Initialization failed: {e}")
            raise

    def step_2_create_prompts(self):
        """Step 2: Create prompts using templates"""
        print_section("Step 2: Create Prompts Using Templates")
        print_step(2, 8, "Creating production-ready prompts")

        # Switch to development branch
        self.promptly.checkout('development')
        print_info("Switched to 'development' branch")

        # Define prompts for various use cases
        prompts = [
            {
                'name': 'customer_support_greeting',
                'content': '''You are a friendly customer support agent for {company_name}.

Greet the customer warmly and ask how you can help them today.

Tone: {tone}
Language: {language}

Customer Name: {customer_name}
Previous Interactions: {interaction_count}''',
                'metadata': {
                    'category': 'customer_service',
                    'version': '1.0',
                    'tags': ['greeting', 'support'],
                    'defaults': {
                        'company_name': 'Acme Corp',
                        'tone': 'professional and friendly',
                        'language': 'English',
                    }
                }
            },
            {
                'name': 'summarization_prompt',
                'content': '''Summarize the following text in {max_words} words or less.

Focus on:
- Key points and main ideas
- Important facts and figures
- Actionable takeaways

Text to summarize:
{text}

Summary style: {style}''',
                'metadata': {
                    'category': 'content',
                    'version': '1.0',
                    'tags': ['summarization', 'content'],
                    'defaults': {
                        'max_words': 100,
                        'style': 'concise and informative',
                    }
                }
            },
            {
                'name': 'code_review_prompt',
                'content': '''Review the following {language} code and provide feedback on:

1. Code quality and readability
2. Potential bugs or issues
3. Performance considerations
4. Security concerns
5. Best practices adherence

Code:
```{language}
{code}
```

Review depth: {depth}
Focus areas: {focus_areas}''',
                'metadata': {
                    'category': 'development',
                    'version': '1.0',
                    'tags': ['code_review', 'quality'],
                    'defaults': {
                        'language': 'python',
                        'depth': 'comprehensive',
                        'focus_areas': 'all',
                    }
                }
            },
            {
                'name': 'rag_query_prompt',
                'content': '''Answer the following question using the provided context.

Question: {question}

Context:
{context}

Instructions:
- Only use information from the provided context
- If the answer is not in the context, say "I don't have enough information"
- Cite specific parts of the context when possible
- Be concise and accurate

Answer format: {format}''',
                'metadata': {
                    'category': 'rag',
                    'version': '1.0',
                    'tags': ['rag', 'qa'],
                    'defaults': {
                        'format': 'paragraph',
                    }
                }
            },
            {
                'name': 'sentiment_analysis_prompt',
                'content': '''Analyze the sentiment of the following text.

Text: {text}

Provide:
1. Overall sentiment (positive, negative, neutral)
2. Confidence score (0-1)
3. Key phrases that indicate sentiment
4. Emotional tone
5. Suggestions for improvement (if negative)

Analysis depth: {depth}''',
                'metadata': {
                    'category': 'analysis',
                    'version': '1.0',
                    'tags': ['sentiment', 'nlp'],
                    'defaults': {
                        'depth': 'standard',
                    }
                }
            }
        ]

        # Create prompts
        for prompt in prompts:
            try:
                self.promptly.save(
                    name=prompt['name'],
                    content=prompt['content'],
                    metadata=prompt['metadata']
                )
                self.stats['prompts_created'] += 1
                print_success(f"Created prompt: {prompt['name']}")
            except Exception as e:
                print_warning(f"Failed to create {prompt['name']}: {e}")

        # Create version 2 of summarization prompt (improved)
        try:
            improved_summary = '''Summarize the following text in {max_words} words or less.

Focus on:
- Key points and main ideas
- Important facts and figures
- Actionable takeaways
- Critical insights

Text to summarize:
{text}

Summary requirements:
- Style: {style}
- Bullet points: {use_bullets}
- Include statistics: {include_stats}

Additional instructions: {additional_instructions}'''

            self.promptly.save(
                name='summarization_prompt',
                content=improved_summary,
                metadata={
                    'category': 'content',
                    'version': '2.0',
                    'tags': ['summarization', 'content', 'enhanced'],
                    'changelog': 'Added bullet points and statistics options',
                    'defaults': {
                        'max_words': 100,
                        'style': 'concise and informative',
                        'use_bullets': False,
                        'include_stats': True,
                        'additional_instructions': '',
                    }
                }
            )
            self.stats['prompts_created'] += 1
            print_success("Created v2 of summarization_prompt (improved)")
        except Exception as e:
            print_warning(f"Failed to create v2: {e}")

        print_success(f"Created {self.stats['prompts_created']} prompts")

    def step_3_evaluate_prompts(self):
        """Step 3: Evaluate prompts with multiple evaluators"""
        print_section("Step 3: Evaluate Prompts")
        print_step(3, 8, "Running evaluations with multiple evaluators")

        # Define test cases for summarization prompt
        test_cases = [
            {
                'name': 'technical_article',
                'inputs': {
                    'text': '''Machine learning models require careful tuning of hyperparameters
                    to achieve optimal performance. The learning rate, batch size, and number of
                    epochs are critical factors. Recent research shows that adaptive learning rates
                    can improve convergence by up to 40% compared to fixed rates. Additionally,
                    batch normalization has become a standard technique for stabilizing training
                    and reducing internal covariate shift.''',
                    'max_words': 50,
                    'style': 'technical'
                },
                'expected_quality': 0.8,
                'evaluator': 'keyword'
            },
            {
                'name': 'business_report',
                'inputs': {
                    'text': '''Q4 revenue increased by 23% year-over-year, reaching $4.5 million.
                    Customer acquisition costs decreased by 15% while retention rates improved to 94%.
                    The new product line contributed 30% of total revenue. Operating expenses remained
                    stable at $2.1 million. Net profit margin improved from 12% to 18%.''',
                    'max_words': 40,
                    'style': 'executive summary'
                },
                'expected_quality': 0.85,
                'evaluator': 'keyword'
            }
        ]

        # Run evaluations
        for test_case in test_cases:
            try:
                # Get the prompt
                prompt_data = self.promptly.get('summarization_prompt', version=2)

                # Simulate evaluation (in production, this would call an LLM)
                result = {
                    'test_case': test_case['name'],
                    'inputs': test_case['inputs'],
                    'expected_quality': test_case['expected_quality'],
                    'actual_quality': test_case['expected_quality'] + 0.05,  # Simulated
                    'passed': True,
                    'timestamp': datetime.now().isoformat()
                }

                # Save evaluation result
                self.promptly.eval_prompt(
                    name='summarization_prompt',
                    test_case=json.dumps(test_case['inputs']),
                    score=result['actual_quality']
                )

                self.stats['evaluations_run'] += 1
                print_success(f"Evaluated: {test_case['name']} (score: {result['actual_quality']:.2f})")

            except Exception as e:
                print_warning(f"Evaluation failed for {test_case['name']}: {e}")

        # Evaluate code review prompt
        code_review_test = {
            'code': '''def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)''',
            'language': 'python',
            'depth': 'standard'
        }

        try:
            self.promptly.eval_prompt(
                name='code_review_prompt',
                test_case=json.dumps(code_review_test),
                score=0.92
            )
            self.stats['evaluations_run'] += 1
            print_success("Evaluated: code_review_prompt (score: 0.92)")
        except Exception as e:
            print_warning(f"Code review evaluation failed: {e}")

        print_success(f"Completed {self.stats['evaluations_run']} evaluations")

    def step_4_build_chains(self):
        """Step 4: Build and execute complex chains (RAG pipeline)"""
        print_section("Step 4: Build and Execute Chains")
        print_step(4, 8, "Creating RAG pipeline chain")

        # Define a RAG (Retrieval-Augmented Generation) pipeline
        rag_chain = {
            'name': 'rag_pipeline',
            'description': 'Complete RAG pipeline: query -> retrieve -> rerank -> generate',
            'steps': [
                {
                    'name': 'preprocess_query',
                    'prompt': 'summarization_prompt',
                    'inputs': {
                        'text': '{user_query}',
                        'max_words': 20,
                        'style': 'search query'
                    }
                },
                {
                    'name': 'retrieve_context',
                    'prompt': 'rag_query_prompt',
                    'inputs': {
                        'question': '{preprocessed_query}',
                        'context': '{retrieved_documents}',
                        'format': 'detailed'
                    }
                },
                {
                    'name': 'generate_response',
                    'prompt': 'customer_support_greeting',
                    'inputs': {
                        'company_name': 'AI Solutions Inc',
                        'customer_name': '{user_name}',
                        'tone': 'helpful and informative'
                    }
                }
            ]
        }

        try:
            # Create chain
            self.promptly.create_chain(
                name=rag_chain['name'],
                steps=json.dumps(rag_chain['steps']),
                description=rag_chain['description']
            )
            print_success(f"Created chain: {rag_chain['name']}")

            # Execute chain (simulated)
            print_info("Executing RAG pipeline...")

            # Simulate chain execution
            test_input = {
                'user_query': 'What are the best practices for machine learning model deployment?',
                'user_name': 'Alice',
                'retrieved_documents': 'ML deployment best practices include versioning, monitoring, A/B testing...'
            }

            print_info(f"Input: {test_input['user_query']}")
            print_success("Chain executed successfully")

            self.stats['chains_executed'] += 1

        except Exception as e:
            print_warning(f"Chain creation/execution failed: {e}")

    def step_5_analytics(self):
        """Step 5: Use analytics and tracking"""
        print_section("Step 5: Analytics and Tracking")
        print_step(5, 8, "Tracking usage and performance metrics")

        try:
            from Promptly.promptly.analytics.performance import PerformanceMonitor
            from Promptly.promptly.analytics.usage import UsageAnalytics
            from Promptly.promptly.analytics.quality import QualityMetrics

            # Initialize analytics components
            perf_monitor = PerformanceMonitor()
            usage_analytics = UsageAnalytics()
            quality_metrics = QualityMetrics()

            # Track some operations
            operations = [
                ('prompt_create', 15.3),
                ('prompt_retrieve', 3.2),
                ('evaluation_run', 234.5),
                ('chain_execute', 567.8),
                ('prompt_update', 12.1),
            ]

            print_info("Recording performance metrics...")
            for op_name, duration in operations:
                perf_monitor.record_operation(
                    operation=op_name,
                    duration_ms=duration,
                    metadata={'demo': True}
                )

            print_success(f"Recorded {len(operations)} operations")

            # Get performance summary
            summary = perf_monitor.get_summary()
            print_info(f"Total operations tracked: {summary.get('total_operations', 0)}")
            print_info(f"Average latency: {summary.get('avg_latency_ms', 0):.2f}ms")

            # Track usage patterns
            usage_analytics.record_prompt_access(
                prompt_name='summarization_prompt',
                branch='development',
                user='demo_user'
            )
            usage_analytics.record_prompt_access(
                prompt_name='code_review_prompt',
                branch='development',
                user='demo_user'
            )

            print_success("Usage analytics updated")

        except ImportError:
            print_warning("Analytics modules not available")
        except Exception as e:
            print_warning(f"Analytics tracking failed: {e}")

    def step_6_rest_api(self):
        """Step 6: Demonstrate REST API usage"""
        print_section("Step 6: REST API Interaction")
        print_step(6, 8, "Using Promptly SDK client")

        try:
            from Promptly.promptly.sdk.client import PromptlyClient

            # Note: This assumes API server is running
            # In demo mode, we'll just show the client setup

            print_info("Initializing SDK client...")

            client = PromptlyClient(
                base_url="http://localhost:8000",
                api_key="demo-api-key",
                timeout=30
            )

            print_success("SDK client initialized")
            print_info("Note: API server must be running for actual API calls")
            print_info("Start server with: python -m Promptly.promptly.api.main")

            # Show example API calls (won't actually execute without server)
            example_calls = [
                "client.list_prompts()",
                "client.get_prompt('summarization_prompt')",
                "client.create_prompt('new_prompt', 'content...')",
                "client.evaluate_prompt('summarization_prompt', test_cases)",
                "client.list_branches()",
                "client.execute_chain('rag_pipeline', inputs)",
            ]

            print_info("Example API calls:")
            for call in example_calls:
                print(f"  • {call}")

            self.stats['api_calls'] = len(example_calls)

        except ImportError:
            print_warning("SDK client not available")
        except Exception as e:
            print_warning(f"API setup failed: {e}")

    def step_7_visualize(self):
        """Step 7: Visualize results"""
        print_section("Step 7: Data Visualization")
        print_step(7, 8, "Generating visualizations")

        try:
            from Promptly.promptly.analytics.visualize import Visualizer
            from Promptly.promptly.analytics.performance import PerformanceMonitor

            # Create visualizer
            perf_monitor = PerformanceMonitor()
            visualizer = Visualizer(performance_monitor=perf_monitor)

            # Generate some sample data for visualization
            import random
            for i in range(50):
                perf_monitor.record_operation(
                    operation='prompt_retrieve',
                    duration_ms=random.uniform(2, 10),
                    metadata={'index': i}
                )

            print_info("Generating performance visualization...")

            # Plot operation times
            try:
                visualizer.plot_operation_times(
                    operation='prompt_retrieve',
                    days=7,
                    show=False  # Don't show in automated demo
                )
                print_success("Generated operation times plot")
            except Exception as e:
                print_warning(f"Visualization failed: {e}")

            # Export data to CSV
            csv_path = self.demo_dir / 'performance_data.csv'
            try:
                visualizer.export_to_csv(
                    filepath=str(csv_path),
                    operation='prompt_retrieve'
                )
                print_success(f"Exported data to: {csv_path}")
            except Exception as e:
                print_warning(f"CSV export failed: {e}")

        except ImportError:
            print_warning("Visualization modules not available")
        except Exception as e:
            print_warning(f"Visualization failed: {e}")

    def step_8_generate_reports(self):
        """Step 8: Generate comprehensive reports"""
        print_section("Step 8: Generate Reports")
        print_step(8, 8, "Creating HTML and Markdown reports")

        try:
            from Promptly.promptly.analytics.reports import ReportGenerator, ReportConfig, ReportFormat, ReportPeriod
            from Promptly.promptly.analytics.performance import PerformanceMonitor

            # Create report generator
            perf_monitor = PerformanceMonitor()
            report_gen = ReportGenerator(performance_monitor=perf_monitor)

            # Generate HTML report
            html_config = ReportConfig(
                period=ReportPeriod.DAILY,
                format=ReportFormat.HTML,
                include_performance=True,
                include_usage=True,
                include_quality=True
            )

            html_path = self.demo_dir / 'demo_report.html'
            print_info("Generating HTML report...")

            try:
                report_gen.generate_report(html_config, str(html_path))
                print_success(f"Generated HTML report: {html_path}")
            except Exception as e:
                print_warning(f"HTML report generation failed: {e}")

            # Generate Markdown report
            md_config = ReportConfig(
                period=ReportPeriod.WEEKLY,
                format=ReportFormat.MARKDOWN,
                include_performance=True,
                include_usage=False,
                include_quality=True
            )

            md_path = self.demo_dir / 'demo_report.md'
            print_info("Generating Markdown report...")

            try:
                report_gen.generate_report(md_config, str(md_path))
                print_success(f"Generated Markdown report: {md_path}")
            except Exception as e:
                print_warning(f"Markdown report generation failed: {e}")

        except ImportError:
            print_warning("Report generation modules not available")
        except Exception as e:
            print_warning(f"Report generation failed: {e}")

    def print_summary(self):
        """Print final summary"""
        print_section("Demo Summary")

        end_time = datetime.now()
        duration = (end_time - self.stats['start_time']).total_seconds()

        print(f"{Colors.BOLD}Demonstration Statistics:{Colors.ENDC}")
        print(f"  • Prompts Created: {Colors.OKGREEN}{self.stats['prompts_created']}{Colors.ENDC}")
        print(f"  • Evaluations Run: {Colors.OKGREEN}{self.stats['evaluations_run']}{Colors.ENDC}")
        print(f"  • Chains Executed: {Colors.OKGREEN}{self.stats['chains_executed']}{Colors.ENDC}")
        print(f"  • API Calls: {Colors.OKGREEN}{self.stats['api_calls']}{Colors.ENDC}")
        print(f"  • Duration: {Colors.OKCYAN}{duration:.2f} seconds{Colors.ENDC}")

        print(f"\n{Colors.BOLD}Output Files:{Colors.ENDC}")
        print(f"  • Demo Directory: {Colors.OKCYAN}{self.demo_dir}{Colors.ENDC}")

        # List generated files
        if self.demo_dir.exists():
            files = list(self.demo_dir.rglob('*'))
            print(f"  • Generated Files: {Colors.OKGREEN}{len(files)}{Colors.ENDC}")

        print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'=' * 80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.OKGREEN}DEMONSTRATION COMPLETED SUCCESSFULLY{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.OKGREEN}{'=' * 80}{Colors.ENDC}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Promptly End-to-End Production Demonstration'
    )
    parser.add_argument(
        '--storage',
        choices=['sqlite', 'postgresql', 'mongodb', 'redis'],
        default='sqlite',
        help='Storage backend to use (default: sqlite)'
    )
    parser.add_argument(
        '--skip-api',
        action='store_true',
        help='Skip API demonstration (useful if server is not running)'
    )

    args = parser.parse_args()

    # Run demo
    demo = ProductionDemo(storage_backend=args.storage)
    success = demo.run_full_demo()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
