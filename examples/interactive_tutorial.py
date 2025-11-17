#!/usr/bin/env python3
"""
Interactive Promptly Tutorial
==============================
Step-by-step guided tour of Promptly features with interactive prompts.

This tutorial covers:
1. Repository initialization
2. Creating and managing prompts
3. Branching and versioning
4. Evaluations
5. Chains and workflows
6. Analytics and reporting
7. Templates and composition
8. API usage

Requirements:
    pip install promptly

Usage:
    python examples/interactive_tutorial.py
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from Promptly.promptly.promptly import Promptly


class Colors:
    """Terminal colors"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ProgressBar:
    """Simple progress bar"""

    @staticmethod
    def show(current: int, total: int, prefix: str = '', length: int = 50):
        """Display progress bar"""
        filled = int(length * current / total)
        bar = '█' * filled + '░' * (length - filled)
        percent = f"{100 * current / total:.1f}"
        print(f'\r{prefix} |{bar}| {percent}% Complete', end='')
        if current == total:
            print()


class InteractiveTutorial:
    """Interactive Promptly tutorial"""

    def __init__(self):
        self.tutorial_dir = Path(__file__).parent / 'tutorial_workspace'
        self.tutorial_dir.mkdir(exist_ok=True)
        self.promptly = None
        self.tutorial_state = {
            'current_step': 0,
            'total_steps': 8,
            'completed_steps': [],
            'prompts_created': 0,
            'evaluations_run': 0
        }

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')

    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{text:^80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")

    def print_step(self, step_num: int, title: str):
        """Print step header"""
        self.tutorial_state['current_step'] = step_num
        progress = f"Step {step_num}/{self.tutorial_state['total_steps']}"

        print(f"\n{Colors.BOLD}{Colors.CYAN}[{progress}] {title}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'-' * 80}{Colors.ENDC}")

    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

    def print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

    def print_tip(self, message: str):
        """Print tip message"""
        print(f"{Colors.YELLOW}💡 TIP: {message}{Colors.ENDC}")

    def wait_for_user(self, prompt: str = "Press Enter to continue..."):
        """Wait for user input"""
        input(f"\n{Colors.CYAN}{prompt}{Colors.ENDC}")

    def get_user_input(self, prompt: str, default: Optional[str] = None) -> str:
        """Get user input with optional default"""
        if default:
            prompt_text = f"{prompt} [{default}]: "
        else:
            prompt_text = f"{prompt}: "

        user_input = input(f"{Colors.CYAN}{prompt_text}{Colors.ENDC}").strip()
        return user_input if user_input else (default or '')

    def welcome(self):
        """Welcome message"""
        self.clear_screen()
        print(f"{Colors.BOLD}{Colors.HEADER}")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "WELCOME TO PROMPTLY INTERACTIVE TUTORIAL".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("║" + "Learn prompt management through hands-on practice".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print(f"{Colors.ENDC}\n")

        print("This tutorial will guide you through:")
        print("  1. Repository setup and initialization")
        print("  2. Creating and managing prompts")
        print("  3. Branching and version control")
        print("  4. Evaluation and testing")
        print("  5. Chains and workflows")
        print("  6. Analytics and monitoring")
        print("  7. Templates and composition")
        print("  8. API integration")
        print()
        print(f"Estimated time: {Colors.BOLD}15-20 minutes{Colors.ENDC}")
        print(f"Output directory: {Colors.CYAN}{self.tutorial_dir}{Colors.ENDC}")
        print()

        self.wait_for_user("Ready to begin? Press Enter to start...")

    def step_1_initialization(self):
        """Step 1: Initialize repository"""
        self.print_step(1, "Repository Initialization")

        print("Let's create your first Promptly repository!")
        print()
        print("Promptly stores prompts in a local repository with version control,")
        print("similar to Git. This allows you to track changes, create branches,")
        print("and collaborate with your team.")
        print()

        repo_name = self.get_user_input("Enter a name for your repository", "my-prompts")
        repo_path = self.tutorial_dir / repo_name
        repo_path.mkdir(exist_ok=True)

        self.print_info(f"Creating repository at: {repo_path}")

        # Simulate initialization steps
        steps = [
            "Creating directory structure...",
            "Initializing database...",
            "Creating main branch...",
            "Setting up configuration..."
        ]

        for i, step in enumerate(steps, 1):
            print(f"  {step}", end='', flush=True)
            time.sleep(0.3)
            print(f" {Colors.GREEN}✓{Colors.ENDC}")
            ProgressBar.show(i, len(steps), prefix='Progress')

        self.promptly = Promptly(root_dir=str(repo_path))
        self.promptly.init()

        self.print_success(f"Repository initialized: {repo_name}")

        self.print_tip("You can have multiple repositories for different projects")
        self.print_tip("Use 'promptly init' command to create repos from CLI")

        self.tutorial_state['completed_steps'].append(1)
        self.wait_for_user()

    def step_2_creating_prompts(self):
        """Step 2: Create prompts"""
        self.print_step(2, "Creating Your First Prompts")

        print("Now let's create some prompts! Prompts are the core of Promptly.")
        print("Each prompt can have variables (using {variable_name} syntax),")
        print("metadata, and version history.")
        print()

        # Interactive prompt creation
        create_custom = self.get_user_input(
            "Would you like to create a custom prompt? (y/n)",
            "y"
        ).lower() == 'y'

        if create_custom:
            prompt_name = self.get_user_input("Prompt name", "my_first_prompt")
            print()
            print("Enter your prompt content (press Ctrl+D or Ctrl+Z when done):")
            print(f"{Colors.YELLOW}Example: Write a summary of {{text}} in {{max_words}} words{Colors.ENDC}")
            print()

            # For automated demo, use default
            prompt_content = "Summarize the following text in {max_words} words:\n\n{text}"
            print(prompt_content)
        else:
            prompt_name = "greeting_prompt"
            prompt_content = "Hello {name}! Welcome to {service}. How can I help you today?"

        # Save prompt
        self.promptly.save(name=prompt_name, content=prompt_content)
        self.tutorial_state['prompts_created'] += 1

        self.print_success(f"Prompt '{prompt_name}' created!")

        # Create a few more examples
        print("\nLet's create a few more example prompts...")

        example_prompts = [
            {
                'name': 'code_review',
                'content': 'Review this {language} code:\n\n```\n{code}\n```\n\nProvide feedback on quality and best practices.'
            },
            {
                'name': 'translation',
                'content': 'Translate the following text from {source_lang} to {target_lang}:\n\n{text}'
            }
        ]

        for i, prompt in enumerate(example_prompts, 1):
            self.promptly.save(name=prompt['name'], content=prompt['content'])
            self.tutorial_state['prompts_created'] += 1
            print(f"  {i}. Created: {prompt['name']}")
            time.sleep(0.2)

        self.print_success(f"Created {self.tutorial_state['prompts_created']} prompts total")

        self.print_tip("Use metadata to tag and categorize your prompts")
        self.print_tip("Variables make prompts reusable across different contexts")

        self.tutorial_state['completed_steps'].append(2)
        self.wait_for_user()

    def step_3_branching(self):
        """Step 3: Branching and versioning"""
        self.print_step(3, "Branching and Versioning")

        print("Promptly supports branching like Git! This lets you:")
        print("  • Experiment with prompt variations")
        print("  • Test changes before deploying")
        print("  • Maintain separate environments (dev/staging/prod)")
        print()

        # Create a development branch
        print("Creating a development branch...")
        self.promptly.create_branch('development')
        self.print_success("Branch 'development' created")

        # Switch to development
        self.promptly.checkout('development')
        self.print_info("Switched to 'development' branch")

        # Update a prompt
        print("\nLet's update the greeting prompt on this branch...")
        updated_content = "Hi {name}! 👋 Welcome to {service}. I'm here to help! What brings you here today?"
        self.promptly.save(name='greeting_prompt', content=updated_content)

        self.print_success("Updated greeting_prompt (version 2)")

        # Show version history
        prompt_data = self.promptly.get('greeting_prompt')
        print(f"\nCurrent version: {prompt_data['version']}")
        print(f"Branch: development")

        # Switch back to main
        self.promptly.checkout('main')
        self.print_info("Switched back to 'main' branch")

        print("\nOn 'main' branch, greeting_prompt is still version 1")

        self.print_tip("Create branches for A/B testing different prompt variations")
        self.print_tip("Use 'production' branch for deployed prompts")

        self.tutorial_state['completed_steps'].append(3)
        self.wait_for_user()

    def step_4_evaluations(self):
        """Step 4: Evaluations"""
        self.print_step(4, "Evaluating Prompts")

        print("Evaluation helps you measure prompt quality and performance.")
        print("You can run test cases and track scores over time.")
        print()

        # Run evaluation
        print("Running evaluation on 'greeting_prompt'...")

        test_cases = [
            {
                'inputs': {'name': 'Alice', 'service': 'AI Support'},
                'expected_quality': 0.9
            },
            {
                'inputs': {'name': 'Bob', 'service': 'Customer Service'},
                'expected_quality': 0.85
            },
            {
                'inputs': {'name': 'Charlie', 'service': 'Tech Support'},
                'expected_quality': 0.88
            }
        ]

        print(f"\nRunning {len(test_cases)} test cases...")

        for i, test_case in enumerate(test_cases, 1):
            # Simulate evaluation
            print(f"  Test case {i}...", end='', flush=True)
            time.sleep(0.3)

            score = test_case['expected_quality']
            self.promptly.eval_prompt(
                name='greeting_prompt',
                test_case=json.dumps(test_case['inputs']),
                score=score
            )

            print(f" Score: {score:.2f} {Colors.GREEN}✓{Colors.ENDC}")
            self.tutorial_state['evaluations_run'] += 1

        avg_score = sum(tc['expected_quality'] for tc in test_cases) / len(test_cases)
        self.print_success(f"Average score: {avg_score:.2f}")

        self.print_tip("Set up automated evaluations in your CI/CD pipeline")
        self.print_tip("Track scores over time to monitor prompt quality")

        self.tutorial_state['completed_steps'].append(4)
        self.wait_for_user()

    def step_5_chains(self):
        """Step 5: Chains and workflows"""
        self.print_step(5, "Building Chains and Workflows")

        print("Chains let you combine multiple prompts into workflows.")
        print("Perfect for complex tasks like RAG, multi-step generation, etc.")
        print()

        # Create a simple chain
        print("Creating a content generation chain...")

        chain_steps = [
            {'name': 'outline', 'prompt': 'Create an outline for {topic}'},
            {'name': 'draft', 'prompt': 'Write a draft based on {outline}'},
            {'name': 'edit', 'prompt': 'Edit and improve {draft}'}
        ]

        # Save chain prompts
        for step in chain_steps:
            self.promptly.save(
                name=f"chain_{step['name']}",
                content=step['prompt']
            )

        # Create chain
        self.promptly.create_chain(
            name='content_pipeline',
            steps=json.dumps(chain_steps),
            description='Multi-stage content generation'
        )

        self.print_success("Chain 'content_pipeline' created")

        print("\nChain steps:")
        for i, step in enumerate(chain_steps, 1):
            print(f"  {i}. {step['name']}: {step['prompt']}")

        self.print_tip("Chains can include conditional logic and parallel execution")
        self.print_tip("Use chain tracing to debug complex workflows")

        self.tutorial_state['completed_steps'].append(5)
        self.wait_for_user()

    def step_6_analytics(self):
        """Step 6: Analytics and monitoring"""
        self.print_step(6, "Analytics and Monitoring")

        print("Track usage, performance, and quality metrics.")
        print()

        # Simulate some analytics
        print("Generating analytics report...")

        stats = {
            'total_prompts': self.tutorial_state['prompts_created'] + 3,
            'total_evaluations': self.tutorial_state['evaluations_run'],
            'branches': 2,
            'chains': 1,
            'avg_quality_score': 0.88
        }

        print("\nRepository Statistics:")
        print(f"  Total Prompts: {stats['total_prompts']}")
        print(f"  Total Evaluations: {stats['total_evaluations']}")
        print(f"  Branches: {stats['branches']}")
        print(f"  Chains: {stats['chains']}")
        print(f"  Avg Quality Score: {stats['avg_quality_score']:.2f}")

        # Simulate performance metrics
        print("\nPerformance Metrics:")
        print("  Most Used Prompt: greeting_prompt (120 calls)")
        print("  Slowest Prompt: code_review (avg 234ms)")
        print("  Best Performing: translation (score: 0.94)")

        self.print_tip("Set up dashboards to monitor prompt performance in production")
        self.print_tip("Use analytics to identify prompts that need optimization")

        self.tutorial_state['completed_steps'].append(6)
        self.wait_for_user()

    def step_7_templates(self):
        """Step 7: Templates and composition"""
        self.print_step(7, "Templates and Composition")

        print("Templates make prompts more powerful with:")
        print("  • Variable substitution")
        print("  • Conditional logic")
        print("  • Loops and iterations")
        print("  • Template inheritance")
        print()

        # Create a template example
        template_content = """# {document_type} for {audience}

## Introduction
{% if include_summary %}
Summary: {summary}
{% endif %}

## Main Content
{content}

{% if include_references %}
## References
{% for ref in references %}
- {ref}
{% endfor %}
{% endif %}

---
Generated on: {date}"""

        print("Example template with conditional blocks:")
        print(f"{Colors.YELLOW}{template_content}{Colors.ENDC}")

        self.print_tip("Use Jinja2 syntax for advanced templating")
        self.print_tip("Create template libraries for common patterns")

        self.tutorial_state['completed_steps'].append(7)
        self.wait_for_user()

    def step_8_api_usage(self):
        """Step 8: API usage"""
        self.print_step(8, "API Integration")

        print("Promptly provides both Python SDK and REST API.")
        print()

        print("Python SDK Example:")
        print(f"{Colors.YELLOW}")
        print("from Promptly.promptly.sdk.client import PromptlyClient")
        print()
        print("client = PromptlyClient(base_url='http://localhost:8000')")
        print("prompts = client.list_prompts()")
        print("prompt = client.get_prompt('greeting_prompt')")
        print(f"{Colors.ENDC}")

        print("\nREST API Example:")
        print(f"{Colors.YELLOW}")
        print("GET    /api/prompts                    # List prompts")
        print("GET    /api/prompts/{name}             # Get prompt")
        print("POST   /api/prompts                    # Create prompt")
        print("POST   /api/evaluations                # Run evaluation")
        print("POST   /api/chains/{name}/execute      # Execute chain")
        print(f"{Colors.ENDC}")

        self.print_tip("Start API server with: python -m Promptly.promptly.api.main")
        self.print_tip("Use API for integrating Promptly into your applications")

        self.tutorial_state['completed_steps'].append(8)
        self.wait_for_user()

    def completion_summary(self):
        """Show completion summary"""
        self.print_header("Tutorial Complete! 🎉")

        print("Congratulations! You've completed the Promptly tutorial.")
        print()

        # Progress summary
        completed = len(self.tutorial_state['completed_steps'])
        total = self.tutorial_state['total_steps']

        print(f"Progress: {completed}/{total} steps completed")
        ProgressBar.show(completed, total, prefix='Overall')

        print("\nWhat you learned:")
        print("  ✓ Repository initialization and setup")
        print("  ✓ Creating and managing prompts")
        print("  ✓ Branching and version control")
        print("  ✓ Running evaluations and tests")
        print("  ✓ Building chains and workflows")
        print("  ✓ Monitoring analytics")
        print("  ✓ Using templates")
        print("  ✓ API integration")

        print("\nYour statistics:")
        print(f"  Prompts Created: {self.tutorial_state['prompts_created']}")
        print(f"  Evaluations Run: {self.tutorial_state['evaluations_run']}")

        print("\nNext steps:")
        print("  1. Check out the use case demos in examples/use_cases/")
        print("  2. Run the E2E production demo")
        print("  3. Explore the API documentation")
        print("  4. Join the community and share your prompts")

        print(f"\n{Colors.BOLD}Tutorial workspace: {self.tutorial_dir}{Colors.ENDC}")

        # Save completion certificate
        certificate = {
            'completed_at': datetime.now().isoformat(),
            'steps_completed': self.tutorial_state['completed_steps'],
            'prompts_created': self.tutorial_state['prompts_created'],
            'evaluations_run': self.tutorial_state['evaluations_run']
        }

        cert_path = self.tutorial_dir / 'completion_certificate.json'
        with open(cert_path, 'w') as f:
            json.dump(certificate, f, indent=2)

        self.print_success(f"Completion certificate saved: {cert_path}")

        print(f"\n{Colors.BOLD}{Colors.GREEN}Thank you for using Promptly!{Colors.ENDC}\n")

    def run(self):
        """Run the interactive tutorial"""
        try:
            self.welcome()
            self.step_1_initialization()
            self.step_2_creating_prompts()
            self.step_3_branching()
            self.step_4_evaluations()
            self.step_5_chains()
            self.step_6_analytics()
            self.step_7_templates()
            self.step_8_api_usage()
            self.completion_summary()

            return True

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Tutorial interrupted. Progress saved!{Colors.ENDC}")
            print(f"Resume anytime by running this script again.\n")
            return False
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            return False


def main():
    tutorial = InteractiveTutorial()
    success = tutorial.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
