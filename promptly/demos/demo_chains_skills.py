"""
Demo 3: Chains & Skills

Demonstrates advanced Promptly features:
- Creating multi-step chains
- Variable flow between chain steps
- Creating reusable skills with schemas
- Running chains/skills with inputs
- Attached files for skills

Interactive flow:
[1] Create a prompt chain
[2] Run a chain with input
[3] Create a skill template
[4] Run a skill
[5] View chain/skill details
[Q] Return to main menu
"""

import asyncio
from typing import TYPE_CHECKING, Dict, Any, List

if TYPE_CHECKING:
    from .interactive_demo import DemoRunner


# Sample chain definitions
SAMPLE_CHAINS = {
    "research-and-summarize": {
        "name": "research-and-summarize",
        "description": "Research a topic and then summarize findings",
        "steps": [
            {
                "name": "research",
                "prompt": "research",
                "output_var": "research_output"
            },
            {
                "name": "summarize",
                "prompt": "summarize",
                "inputs": {
                    "text": "{{research_output}}",
                    "length": "3"
                }
            }
        ]
    },
    "code-review-pipeline": {
        "name": "code-review-pipeline",
        "description": "Review code, identify issues, suggest fixes",
        "steps": [
            {
                "name": "analyze",
                "prompt": "code-review",
                "inputs": {"language": "{{language}}", "code": "{{code}}"},
                "output_var": "analysis"
            },
            {
                "name": "suggest-fixes",
                "prompt": "suggest-fixes",
                "inputs": {"analysis": "{{analysis}}", "code": "{{code}}"},
                "output_var": "suggestions"
            },
            {
                "name": "final-report",
                "prompt": "format-report",
                "inputs": {"analysis": "{{analysis}}", "suggestions": "{{suggestions}}"}
            }
        ]
    },
    "translate-and-verify": {
        "name": "translate-and-verify",
        "description": "Translate text and verify accuracy",
        "steps": [
            {
                "name": "translate",
                "prompt": "translate",
                "inputs": {
                    "text": "{{text}}",
                    "source_lang": "{{source}}",
                    "target_lang": "{{target}}"
                },
                "output_var": "translation"
            },
            {
                "name": "back-translate",
                "prompt": "translate",
                "inputs": {
                    "text": "{{translation}}",
                    "source_lang": "{{target}}",
                    "target_lang": "{{source}}"
                },
                "output_var": "back_translation"
            },
            {
                "name": "compare",
                "prompt": "compare-texts",
                "inputs": {
                    "original": "{{text}}",
                    "back_translated": "{{back_translation}}"
                }
            }
        ]
    }
}


# Sample skill definitions
SAMPLE_SKILLS = {
    "code-documenter": {
        "name": "code-documenter",
        "description": "Auto-document code with docstrings",
        "template": """Generate comprehensive documentation for this {{language}} code.

Include:
- Module/class docstring
- Function/method docstrings with parameters and return types
- Inline comments for complex logic
- Usage examples

Code:
```{{language}}
{{code}}
```

Documentation:""",
        "inputs": {
            "language": {"type": "string", "description": "Programming language"},
            "code": {"type": "string", "description": "Code to document"}
        },
        "outputs": {
            "documentation": {"type": "string", "description": "Generated documentation"}
        }
    },
    "test-generator": {
        "name": "test-generator",
        "description": "Generate unit tests for code",
        "template": """Generate comprehensive unit tests for this {{language}} code.

Use {{framework}} testing framework.
Cover edge cases and error conditions.

Code:
```{{language}}
{{code}}
```

Tests:""",
        "inputs": {
            "language": {"type": "string", "description": "Programming language"},
            "code": {"type": "string", "description": "Code to test"},
            "framework": {"type": "string", "description": "Testing framework", "default": "pytest"}
        },
        "outputs": {
            "tests": {"type": "string", "description": "Generated tests"}
        }
    },
    "api-spec-generator": {
        "name": "api-spec-generator",
        "description": "Generate OpenAPI spec from code",
        "template": """Analyze this {{language}} API code and generate an OpenAPI 3.0 specification.

Include:
- Endpoint paths
- Request/response schemas
- Parameters and their types
- Example values

Code:
```{{language}}
{{code}}
```

OpenAPI Spec:""",
        "inputs": {
            "language": {"type": "string", "description": "Programming language"},
            "code": {"type": "string", "description": "API code"}
        },
        "outputs": {
            "openapi_spec": {"type": "string", "description": "OpenAPI 3.0 specification"}
        }
    }
}


async def run_chains_demo(runner: "DemoRunner"):
    """
    Run the chains and skills demo.

    Args:
        runner: DemoRunner instance for UI
    """
    runner.print_info("Welcome to Chains & Skills!")
    runner.print()
    runner.print("[dim]Chains: Multi-step prompt sequences with variable flow[/dim]")
    runner.print("[dim]Skills: Reusable templates with input/output schemas[/dim]")
    runner.print()

    # Ensure Promptly is initialized
    if not hasattr(runner, 'promptly') or runner.promptly is None:
        await runner.setup_sample_data()

    promptly = runner.promptly

    while True:
        # Show demo menu
        runner.print()
        runner.print_table(
            "⛓️ Chains & Skills Demo",
            ["Option", "Action"],
            [
                ["1", "Create a prompt chain"],
                ["2", "Run a chain with input"],
                ["3", "Create a skill template"],
                ["4", "Run a skill"],
                ["5", "View chain/skill details"],
                ["Q", "Return to main menu"],
            ]
        )

        choice = runner.prompt_choice("Select action", ["1", "2", "3", "4", "5", "q", "Q"], "1")

        if choice.lower() == 'q':
            break

        try:
            if choice == "1":
                await demo_create_chain(runner, promptly)
            elif choice == "2":
                await demo_run_chain(runner, promptly)
            elif choice == "3":
                await demo_create_skill(runner, promptly)
            elif choice == "4":
                await demo_run_skill(runner, promptly)
            elif choice == "5":
                await demo_view_details(runner, promptly)
        except Exception as e:
            runner.print_error(f"Error: {e}")


async def demo_create_chain(runner: "DemoRunner", promptly):
    """Demonstrate creating a prompt chain."""
    runner.print()
    runner.print_info("Creating a Prompt Chain")
    runner.print()

    runner.print("[bold]What is a Chain?[/bold]")
    runner.print("[dim]A chain connects multiple prompts where output of one[/dim]")
    runner.print("[dim]becomes input to the next. Variables flow between steps.[/dim]")
    runner.print()

    # Show available chains
    runner.print("[bold]Sample Chains:[/bold]")
    for i, (name, chain) in enumerate(SAMPLE_CHAINS.items(), 1):
        runner.print(f"  [{i}] {name}: {chain['description']}")

    runner.print()
    choice = runner.prompt_choice("Select a chain to explore", ["1", "2", "3"], "1")

    chain_names = list(SAMPLE_CHAINS.keys())
    chain_name = chain_names[int(choice) - 1]
    chain = SAMPLE_CHAINS[chain_name]

    # Show chain structure
    runner.print()
    runner.print(f"[bold cyan]Chain: {chain_name}[/bold cyan]")
    runner.print(f"[dim]{chain['description']}[/dim]")
    runner.print()

    runner.print("[bold]Steps:[/bold]")
    for i, step in enumerate(chain['steps'], 1):
        runner.print(f"\n  [cyan]Step {i}: {step['name']}[/cyan]")
        runner.print(f"    Prompt: {step['prompt']}")

        if 'inputs' in step:
            runner.print(f"    Inputs: {step['inputs']}")

        if 'output_var' in step:
            runner.print(f"    Output → [green]${{{{ {step['output_var']} }}}}[/green]")

    # Visualize flow
    runner.print()
    runner.print("[bold]Variable Flow:[/bold]")

    flow_parts = []
    for step in chain['steps']:
        if 'output_var' in step:
            flow_parts.append(f"[{step['name']}] → ${{{step['output_var']}}}")
        else:
            flow_parts.append(f"[{step['name']}] → [output]")

    runner.print(f"  {' → '.join(flow_parts)}")

    # Create in Promptly (if supported)
    runner.print()
    if runner.prompt_confirm("Save this chain?", default=True):
        try:
            if hasattr(promptly, 'create_chain'):
                promptly.create_chain(chain_name, chain['steps'])
                runner.print_success(f"Chain '{chain_name}' saved!")
            else:
                runner.print_info("Chain definition stored (chains API demo)")
        except Exception as e:
            runner.print_warning(f"Could not save chain: {e}")


async def demo_run_chain(runner: "DemoRunner", promptly):
    """Demonstrate running a chain."""
    runner.print()
    runner.print_info("Running a Chain")
    runner.print()

    # Show available chains
    runner.print("[bold]Available Chains:[/bold]")
    for i, name in enumerate(SAMPLE_CHAINS.keys(), 1):
        runner.print(f"  [{i}] {name}")

    runner.print()
    choice = runner.prompt_choice("Select chain to run", ["1", "2", "3"], "1")

    chain_names = list(SAMPLE_CHAINS.keys())
    chain_name = chain_names[int(choice) - 1]
    chain = SAMPLE_CHAINS[chain_name]

    runner.print()
    runner.print(f"[bold]Running: {chain_name}[/bold]")
    runner.print()

    # Collect inputs
    runner.print("[bold]Provide inputs:[/bold]")

    inputs = {}
    if chain_name == "research-and-summarize":
        inputs['topic'] = runner.prompt_input("Topic to research", "quantum computing")
    elif chain_name == "code-review-pipeline":
        inputs['language'] = runner.prompt_input("Programming language", "Python")
        inputs['code'] = runner.prompt_input("Code to review", "def add(a, b): return a + b")
    elif chain_name == "translate-and-verify":
        inputs['text'] = runner.prompt_input("Text to translate", "Hello, world!")
        inputs['source'] = runner.prompt_input("Source language", "English")
        inputs['target'] = runner.prompt_input("Target language", "Spanish")

    runner.print()

    # Simulate chain execution
    runner.print("[bold]Executing chain...[/bold]")
    runner.print()

    context = inputs.copy()

    for i, step in enumerate(chain['steps'], 1):
        runner.print(f"[cyan]Step {i}/{len(chain['steps'])}: {step['name']}[/cyan]")

        with runner.show_progress(f"Running {step['name']}") as progress:
            task = progress.add_task("Processing...", total=1)

            # Simulate processing time
            await asyncio.sleep(0.5)

            # Simulate output
            if runner.mock_mode or True:  # Always mock for demo
                output = f"[Simulated output for {step['name']}]"
                if step['name'] == 'research':
                    output = f"Research findings on {context.get('topic', 'topic')}:\n1. Key finding one\n2. Key finding two\n3. Key finding three"
                elif step['name'] == 'translate':
                    output = f"¡{context.get('text', 'Hello')}!"
                elif step['name'] == 'analyze':
                    output = f"Code analysis:\n- Clean function definition\n- Consider adding type hints"

            progress.update(task, advance=1)

        runner.print(f"  Output: [dim]{output[:60]}...[/dim]")

        if 'output_var' in step:
            context[step['output_var']] = output
            runner.print(f"  Stored in: [green]${{{step['output_var']}}}[/green]")

        runner.print()

    runner.print_success("Chain execution complete!")

    # Show final output
    runner.print()
    runner.print("[bold]Final Output:[/bold]")
    last_step = chain['steps'][-1]
    if 'output_var' in last_step:
        runner.print(f"[green]{context.get(last_step['output_var'], 'N/A')}[/green]")
    else:
        runner.print("[dim]Chain completed successfully[/dim]")


async def demo_create_skill(runner: "DemoRunner", promptly):
    """Demonstrate creating a skill template."""
    runner.print()
    runner.print_info("Creating a Skill Template")
    runner.print()

    runner.print("[bold]What is a Skill?[/bold]")
    runner.print("[dim]A skill is a validated prompt template with defined[/dim]")
    runner.print("[dim]input/output schemas. Think of it as a function.[/dim]")
    runner.print()

    # Show available skills
    runner.print("[bold]Sample Skills:[/bold]")
    for i, (name, skill) in enumerate(SAMPLE_SKILLS.items(), 1):
        runner.print(f"  [{i}] {name}: {skill['description']}")

    runner.print()
    choice = runner.prompt_choice("Select a skill to explore", ["1", "2", "3"], "1")

    skill_names = list(SAMPLE_SKILLS.keys())
    skill_name = skill_names[int(choice) - 1]
    skill = SAMPLE_SKILLS[skill_name]

    # Show skill definition
    runner.print()
    runner.print(f"[bold cyan]Skill: {skill_name}[/bold cyan]")
    runner.print(f"[dim]{skill['description']}[/dim]")
    runner.print()

    # Show inputs
    runner.print("[bold]Inputs:[/bold]")
    for input_name, input_spec in skill['inputs'].items():
        required = "required" if 'default' not in input_spec else f"default: {input_spec['default']}"
        runner.print(f"  • {input_name}: {input_spec['type']} ({required})")
        runner.print(f"    [dim]{input_spec['description']}[/dim]")

    runner.print()

    # Show outputs
    runner.print("[bold]Outputs:[/bold]")
    for output_name, output_spec in skill['outputs'].items():
        runner.print(f"  • {output_name}: {output_spec['type']}")
        runner.print(f"    [dim]{output_spec['description']}[/dim]")

    runner.print()

    # Show template
    runner.print("[bold]Template:[/bold]")
    runner.print_prompt(skill_name, skill['template'])

    # Save skill
    runner.print()
    if runner.prompt_confirm("Save this skill?", default=True):
        try:
            if hasattr(promptly, 'skill_add'):
                promptly.skill_add(skill_name, skill['template'], skill['inputs'], skill['outputs'])
                runner.print_success(f"Skill '{skill_name}' saved!")
            else:
                runner.print_info("Skill definition stored (skills API demo)")
        except Exception as e:
            runner.print_warning(f"Could not save skill: {e}")


async def demo_run_skill(runner: "DemoRunner", promptly):
    """Demonstrate running a skill."""
    runner.print()
    runner.print_info("Running a Skill")
    runner.print()

    # Show available skills
    runner.print("[bold]Available Skills:[/bold]")
    for i, name in enumerate(SAMPLE_SKILLS.keys(), 1):
        runner.print(f"  [{i}] {name}")

    runner.print()
    choice = runner.prompt_choice("Select skill to run", ["1", "2", "3"], "1")

    skill_names = list(SAMPLE_SKILLS.keys())
    skill_name = skill_names[int(choice) - 1]
    skill = SAMPLE_SKILLS[skill_name]

    runner.print()
    runner.print(f"[bold]Running: {skill_name}[/bold]")
    runner.print()

    # Collect inputs
    runner.print("[bold]Provide inputs:[/bold]")

    inputs = {}
    for input_name, input_spec in skill['inputs'].items():
        default = input_spec.get('default', '')
        if input_name == 'code':
            default = "def hello():\n    print('Hello!')"
        elif input_name == 'language':
            default = "Python"

        inputs[input_name] = runner.prompt_input(
            f"{input_name} ({input_spec['description']})",
            default
        )

    runner.print()

    # Render template
    rendered = skill['template']
    for key, value in inputs.items():
        rendered = rendered.replace("{{" + key + "}}", value)

    runner.print("[bold]Rendered Prompt:[/bold]")
    runner.print_prompt(f"{skill_name} (rendered)", rendered)

    # Simulate execution
    runner.print()
    with runner.show_progress("Executing skill") as progress:
        task = progress.add_task("Processing...", total=1)
        await asyncio.sleep(1)
        progress.update(task, advance=1)

    # Simulate output
    runner.print()
    runner.print("[bold]Skill Output:[/bold]")

    mock_output = f"""```{inputs.get('language', 'python')}
def hello():
    \"\"\"
    Prints a greeting message.

    Returns:
        None
    \"\"\"
    print('Hello!')  # Output greeting
```

Usage:
```python
hello()  # Prints: Hello!
```"""

    runner.print(f"[green]{mock_output}[/green]")


async def demo_view_details(runner: "DemoRunner", promptly):
    """View chain or skill details."""
    runner.print()
    runner.print_info("View Chain/Skill Details")
    runner.print()

    choice = runner.prompt_choice(
        "View chains or skills?",
        ["chains", "skills", "c", "s"],
        "chains"
    )

    if choice in ['chains', 'c']:
        runner.print()
        runner.print("[bold]All Chains:[/bold]")
        runner.print()

        for name, chain in SAMPLE_CHAINS.items():
            runner.print(f"[cyan]{name}[/cyan]")
            runner.print(f"  Description: {chain['description']}")
            runner.print(f"  Steps: {len(chain['steps'])}")
            step_names = [s['name'] for s in chain['steps']]
            runner.print(f"  Flow: {' → '.join(step_names)}")
            runner.print()

    else:
        runner.print()
        runner.print("[bold]All Skills:[/bold]")
        runner.print()

        for name, skill in SAMPLE_SKILLS.items():
            runner.print(f"[cyan]{name}[/cyan]")
            runner.print(f"  Description: {skill['description']}")
            runner.print(f"  Inputs: {', '.join(skill['inputs'].keys())}")
            runner.print(f"  Outputs: {', '.join(skill['outputs'].keys())}")
            runner.print()


# Direct execution for testing
if __name__ == "__main__":
    from .interactive_demo import DemoRunner

    async def main():
        runner = DemoRunner()
        await runner.setup_sample_data()
        await run_chains_demo(runner)

    asyncio.run(main())
