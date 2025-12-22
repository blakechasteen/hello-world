"""
Demo 4: LLM Judge Evaluation

Demonstrates Promptly's evaluation system:
- Evaluating prompts with different criteria
- 6 judging methodologies comparison
- Constitutional AI evaluation
- G-Eval methodology
- Pairwise comparison of outputs
- Score breakdown visualization

Interactive flow:
[1] Quick quality evaluation
[2] Full comprehensive evaluation
[3] Constitutional AI evaluation
[4] Pairwise comparison (A vs B)
[5] G-Eval methodology
[6] Compare judging methods
[Q] Return to main menu
"""

import asyncio
from typing import TYPE_CHECKING, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .interactive_demo import DemoRunner


# Evaluation criteria definitions
class JudgeCriteria(Enum):
    """12 evaluation criteria."""
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    HELPFULNESS = "helpfulness"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONCISENESS = "conciseness"
    CREATIVITY = "creativity"
    SAFETY = "safety"
    BIAS = "bias"
    TONE = "tone"
    ENGAGEMENT = "engagement"


CRITERIA_DESCRIPTIONS = {
    JudgeCriteria.RELEVANCE: "How well the response addresses the prompt",
    JudgeCriteria.COHERENCE: "Logical flow and consistency of the response",
    JudgeCriteria.FLUENCY: "Grammar, spelling, and readability",
    JudgeCriteria.HELPFULNESS: "Practical value and usefulness",
    JudgeCriteria.ACCURACY: "Factual correctness of information",
    JudgeCriteria.COMPLETENESS: "Coverage of all aspects of the prompt",
    JudgeCriteria.CONCISENESS: "Brevity without sacrificing clarity",
    JudgeCriteria.CREATIVITY: "Originality and novel approaches",
    JudgeCriteria.SAFETY: "Avoidance of harmful content",
    JudgeCriteria.BIAS: "Fairness and balance in perspective",
    JudgeCriteria.TONE: "Appropriate voice and style",
    JudgeCriteria.ENGAGEMENT: "Interest and readability",
}


class JudgingMethod(Enum):
    """6 judging methodologies."""
    SINGLE_SCORE = "single_score"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    PAIRWISE = "pairwise"
    REFERENCE_BASED = "reference_based"
    GEVAL = "g_eval"
    CONSTITUTIONAL = "constitutional"


METHOD_DESCRIPTIONS = {
    JudgingMethod.SINGLE_SCORE: "Simple 0-100 scoring",
    JudgingMethod.CHAIN_OF_THOUGHT: "Step-by-step reasoning before scoring",
    JudgingMethod.PAIRWISE: "Compare two outputs, pick winner",
    JudgingMethod.REFERENCE_BASED: "Compare against gold reference",
    JudgingMethod.GEVAL: "Multi-aspect evaluation with rubrics",
    JudgingMethod.CONSTITUTIONAL: "Evaluate against principles",
}


# Sample prompts and responses for evaluation
SAMPLE_EVALUATIONS = [
    {
        "prompt": "Explain what machine learning is in simple terms.",
        "response_a": "Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance on tasks without being explicitly programmed. It works by identifying patterns in training data and using these patterns to make predictions or decisions on new data.",
        "response_b": "ML is AI that learns stuff from data. It's like how you learn things, but for computers. Pretty cool tech that's used everywhere now.",
        "reference": "Machine learning is a branch of artificial intelligence where computers learn to recognize patterns in data. Instead of following fixed rules, ML systems improve automatically through experience, similar to how humans learn from practice.",
    },
    {
        "prompt": "Write a haiku about coding.",
        "response_a": "Fingers dance on keys,\nLogic blooms in silent night,\nBugs hide in the dawn.",
        "response_b": "Code is fun to write\nProgramming all day long yeah\nI like computers",
        "reference": None,
    },
    {
        "prompt": "What are the benefits of exercise?",
        "response_a": "Exercise offers numerous physical and mental benefits: improved cardiovascular health, stronger muscles and bones, better weight management, enhanced mood through endorphin release, reduced stress and anxiety, improved sleep quality, increased energy levels, and reduced risk of chronic diseases like diabetes and heart disease.",
        "response_b": "Exercise is good for you. It makes you healthy and strong. You should do it every day.",
        "reference": "Regular exercise provides cardiovascular benefits, muscle strengthening, weight control, improved mental health, better sleep, and reduced chronic disease risk.",
    }
]


# Constitutional AI principles
CONSTITUTIONAL_PRINCIPLES = [
    "Be helpful while avoiding harm",
    "Provide accurate, factual information",
    "Respect user autonomy and privacy",
    "Avoid reinforcing stereotypes or biases",
    "Acknowledge uncertainty when appropriate",
    "Encourage critical thinking over blind acceptance",
]


async def run_judge_demo(runner: "DemoRunner"):
    """
    Run the LLM Judge demo.

    Args:
        runner: DemoRunner instance for UI
    """
    runner.print_info("Welcome to LLM Judge Evaluation!")
    runner.print()
    runner.print("[dim]Evaluate prompt responses using 12 criteria and 6 methods[/dim]")
    runner.print()

    while True:
        # Show demo menu
        runner.print()
        runner.print_table(
            "⚖️ LLM Judge Demo",
            ["Option", "Action"],
            [
                ["1", "Quick quality evaluation"],
                ["2", "Full comprehensive evaluation"],
                ["3", "Constitutional AI evaluation"],
                ["4", "Pairwise comparison (A vs B)"],
                ["5", "G-Eval methodology"],
                ["6", "Compare judging methods"],
                ["Q", "Return to main menu"],
            ]
        )

        choice = runner.prompt_choice("Select action", ["1", "2", "3", "4", "5", "6", "q", "Q"], "1")

        if choice.lower() == 'q':
            break

        try:
            if choice == "1":
                await demo_quick_eval(runner)
            elif choice == "2":
                await demo_full_eval(runner)
            elif choice == "3":
                await demo_constitutional(runner)
            elif choice == "4":
                await demo_pairwise(runner)
            elif choice == "5":
                await demo_geval(runner)
            elif choice == "6":
                await demo_compare_methods(runner)
        except Exception as e:
            runner.print_error(f"Error: {e}")


async def demo_quick_eval(runner: "DemoRunner"):
    """Quick single-score evaluation."""
    runner.print()
    runner.print_info("Quick Quality Evaluation")
    runner.print()

    # Use sample or get input
    sample = SAMPLE_EVALUATIONS[0]

    runner.print("[bold]Prompt:[/bold]")
    runner.print(f"[cyan]{sample['prompt']}[/cyan]")
    runner.print()

    runner.print("[bold]Response:[/bold]")
    runner.print(sample['response_a'])
    runner.print()

    # Simulate evaluation
    runner.print("[bold]Evaluating...[/bold]")

    with runner.show_progress("Evaluating response") as progress:
        task = progress.add_task("Analyzing...", total=1)
        await asyncio.sleep(0.5)
        progress.update(task, advance=1)

    # Generate mock scores
    import random
    scores = {
        "overall": 85,
        "relevance": 92,
        "coherence": 88,
        "helpfulness": 82,
        "accuracy": 90,
    }

    runner.print()
    runner.print("[bold green]Evaluation Results:[/bold green]")
    runner.print()

    # Visual score bar
    for criterion, score in scores.items():
        bar_len = int(score / 5)  # 0-20 chars
        bar = "█" * bar_len + "░" * (20 - bar_len)

        if score >= 80:
            color = "green"
        elif score >= 60:
            color = "yellow"
        else:
            color = "red"

        runner.print(f"  {criterion:12} [{color}]{bar}[/{color}] {score}/100")

    runner.print()
    runner.print(f"[bold]Overall Score: [green]{scores['overall']}/100[/green][/bold]")


async def demo_full_eval(runner: "DemoRunner"):
    """Full comprehensive evaluation with all 12 criteria."""
    runner.print()
    runner.print_info("Full Comprehensive Evaluation")
    runner.print()

    runner.print("[bold]12 Evaluation Criteria:[/bold]")
    for i, (criterion, desc) in enumerate(CRITERIA_DESCRIPTIONS.items(), 1):
        runner.print(f"  {i:2}. {criterion.value:15} - {desc}")

    runner.print()

    # Use sample
    sample = SAMPLE_EVALUATIONS[2]

    runner.print("[bold]Prompt:[/bold]")
    runner.print(f"[cyan]{sample['prompt']}[/cyan]")
    runner.print()

    runner.print("[bold]Response:[/bold]")
    runner.print(sample['response_a'])
    runner.print()

    # Simulate comprehensive evaluation
    runner.print("[bold]Running comprehensive evaluation...[/bold]")

    with runner.show_progress("Evaluating all criteria") as progress:
        task = progress.add_task("Processing...", total=12)

        scores = {}
        for criterion in JudgeCriteria:
            await asyncio.sleep(0.1)

            # Mock score
            import random
            base = 75 if criterion in [JudgeCriteria.RELEVANCE, JudgeCriteria.ACCURACY] else 70
            scores[criterion] = min(100, max(0, base + random.randint(-10, 20)))

            progress.update(task, advance=1)

    runner.print()
    runner.print("[bold green]Full Evaluation Results:[/bold green]")
    runner.print()

    # Build table data
    rows = []
    for criterion, score in scores.items():
        if score >= 80:
            indicator = "[green]●[/green]"
        elif score >= 60:
            indicator = "[yellow]●[/yellow]"
        else:
            indicator = "[red]●[/red]"

        rows.append([criterion.value, str(score), indicator, CRITERIA_DESCRIPTIONS[criterion][:40]])

    runner.print_table(
        "Score Breakdown",
        ["Criterion", "Score", "", "Description"],
        rows
    )

    # Calculate overall
    overall = sum(scores.values()) / len(scores)
    runner.print()
    runner.print(f"[bold]Overall Score: [green]{overall:.1f}/100[/green][/bold]")


async def demo_constitutional(runner: "DemoRunner"):
    """Constitutional AI evaluation."""
    runner.print()
    runner.print_info("Constitutional AI Evaluation")
    runner.print()

    runner.print("[bold]What is Constitutional AI?[/bold]")
    runner.print("[dim]Evaluates responses against a set of ethical principles,[/dim]")
    runner.print("[dim]checking for alignment with human values and safety.[/dim]")
    runner.print()

    runner.print("[bold]Principles:[/bold]")
    for i, principle in enumerate(CONSTITUTIONAL_PRINCIPLES, 1):
        runner.print(f"  {i}. {principle}")

    runner.print()

    sample = SAMPLE_EVALUATIONS[0]

    runner.print("[bold]Prompt:[/bold]")
    runner.print(f"[cyan]{sample['prompt']}[/cyan]")
    runner.print()

    runner.print("[bold]Response:[/bold]")
    runner.print(sample['response_a'])
    runner.print()

    # Evaluate against principles
    runner.print("[bold]Evaluating against principles...[/bold]")
    runner.print()

    results = []
    for i, principle in enumerate(CONSTITUTIONAL_PRINCIPLES, 1):
        with runner.show_progress(f"Checking principle {i}") as progress:
            task = progress.add_task("Analyzing...", total=1)
            await asyncio.sleep(0.2)
            progress.update(task, advance=1)

        # Mock result
        import random
        passed = random.random() > 0.15  # 85% pass rate
        score = random.randint(80, 100) if passed else random.randint(40, 70)

        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        runner.print(f"  {status} {principle}")

        results.append({"principle": principle, "passed": passed, "score": score})

    runner.print()

    # Summary
    passed_count = sum(1 for r in results if r['passed'])
    avg_score = sum(r['score'] for r in results) / len(results)

    runner.print(f"[bold]Constitutional Compliance:[/bold]")
    runner.print(f"  Principles Passed: {passed_count}/{len(CONSTITUTIONAL_PRINCIPLES)}")
    runner.print(f"  Average Score: {avg_score:.1f}/100")

    if passed_count == len(CONSTITUTIONAL_PRINCIPLES):
        runner.print("[bold green]  Status: FULLY COMPLIANT[/bold green]")
    elif passed_count >= 4:
        runner.print("[bold yellow]  Status: MOSTLY COMPLIANT[/bold yellow]")
    else:
        runner.print("[bold red]  Status: NEEDS IMPROVEMENT[/bold red]")


async def demo_pairwise(runner: "DemoRunner"):
    """Pairwise comparison evaluation."""
    runner.print()
    runner.print_info("Pairwise Comparison")
    runner.print()

    runner.print("[bold]What is Pairwise Comparison?[/bold]")
    runner.print("[dim]Compare two responses and determine which is better.[/dim]")
    runner.print("[dim]Useful for A/B testing and preference learning.[/dim]")
    runner.print()

    sample = SAMPLE_EVALUATIONS[0]

    runner.print("[bold]Prompt:[/bold]")
    runner.print(f"[cyan]{sample['prompt']}[/cyan]")
    runner.print()

    runner.print("[bold cyan]Response A:[/bold cyan]")
    runner.print(sample['response_a'])
    runner.print()

    runner.print("[bold magenta]Response B:[/bold magenta]")
    runner.print(sample['response_b'])
    runner.print()

    # Simulate comparison
    runner.print("[bold]Comparing responses...[/bold]")

    with runner.show_progress("Analyzing both responses") as progress:
        task = progress.add_task("Comparing...", total=1)
        await asyncio.sleep(0.8)
        progress.update(task, advance=1)

    runner.print()
    runner.print("[bold green]Comparison Results:[/bold green]")
    runner.print()

    # Mock results
    criteria_results = {
        "Relevance": ("A", 92, 78),
        "Coherence": ("A", 88, 65),
        "Helpfulness": ("A", 85, 60),
        "Completeness": ("A", 90, 55),
        "Fluency": ("A", 88, 72),
    }

    runner.print("[bold]Per-Criterion Comparison:[/bold]")
    for criterion, (winner, score_a, score_b) in criteria_results.items():
        winner_mark = "[cyan]←[/cyan]" if winner == "A" else "[magenta]→[/magenta]"
        runner.print(f"  {criterion:15} A: {score_a:3} vs B: {score_b:3}  {winner_mark}")

    runner.print()
    runner.print("[bold]Winner: [cyan]Response A[/cyan][/bold]")
    runner.print()

    runner.print("[bold]Reasoning:[/bold]")
    runner.print("[dim]Response A is preferred because:[/dim]")
    runner.print("[dim]• More comprehensive explanation of the concept[/dim]")
    runner.print("[dim]• Better technical accuracy[/dim]")
    runner.print("[dim]• More professional and polished language[/dim]")


async def demo_geval(runner: "DemoRunner"):
    """G-Eval methodology."""
    runner.print()
    runner.print_info("G-Eval Methodology")
    runner.print()

    runner.print("[bold]What is G-Eval?[/bold]")
    runner.print("[dim]A framework for NLG evaluation using LLMs with:[/dim]")
    runner.print("[dim]• Detailed evaluation rubrics[/dim]")
    runner.print("[dim]• Chain-of-thought prompting[/dim]")
    runner.print("[dim]• Multi-aspect scoring[/dim]")
    runner.print()

    sample = SAMPLE_EVALUATIONS[1]  # Haiku

    runner.print("[bold]Prompt:[/bold]")
    runner.print(f"[cyan]{sample['prompt']}[/cyan]")
    runner.print()

    runner.print("[bold]Response:[/bold]")
    runner.print(sample['response_a'])
    runner.print()

    # G-Eval aspects
    aspects = [
        {
            "name": "Form Adherence",
            "rubric": "5-7-5 syllable structure, proper haiku form",
            "score": 95
        },
        {
            "name": "Imagery",
            "rubric": "Vivid sensory language, concrete images",
            "score": 90
        },
        {
            "name": "Theme Relevance",
            "rubric": "Connection to coding/programming",
            "score": 88
        },
        {
            "name": "Emotional Impact",
            "rubric": "Evokes feeling or insight",
            "score": 85
        },
        {
            "name": "Originality",
            "rubric": "Fresh perspective, avoids clichés",
            "score": 82
        }
    ]

    runner.print("[bold]G-Eval Analysis:[/bold]")
    runner.print()

    for aspect in aspects:
        runner.print(f"[cyan]{aspect['name']}[/cyan]")
        runner.print(f"  Rubric: [dim]{aspect['rubric']}[/dim]")

        # Visual bar
        score = aspect['score']
        bar_len = int(score / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        color = "green" if score >= 80 else "yellow" if score >= 60 else "red"

        runner.print(f"  Score:  [{color}]{bar}[/{color}] {score}/100")
        runner.print()

    avg_score = sum(a['score'] for a in aspects) / len(aspects)
    runner.print(f"[bold]G-Eval Overall: [green]{avg_score:.1f}/100[/green][/bold]")


async def demo_compare_methods(runner: "DemoRunner"):
    """Compare different judging methods."""
    runner.print()
    runner.print_info("Compare Judging Methods")
    runner.print()

    runner.print("[bold]6 Judging Methods:[/bold]")
    for method, desc in METHOD_DESCRIPTIONS.items():
        runner.print(f"  • {method.value:20} - {desc}")

    runner.print()

    sample = SAMPLE_EVALUATIONS[0]

    runner.print("[bold]Prompt:[/bold]")
    runner.print(f"[cyan]{sample['prompt']}[/cyan]")
    runner.print()

    runner.print("[bold]Response:[/bold]")
    runner.print(sample['response_a'][:100] + "...")
    runner.print()

    runner.print("[bold]Running all methods...[/bold]")
    runner.print()

    # Simulate running all methods
    method_results = {}
    for method in JudgingMethod:
        with runner.show_progress(f"Running {method.value}") as progress:
            task = progress.add_task("Evaluating...", total=1)
            await asyncio.sleep(0.3)
            progress.update(task, advance=1)

        # Mock results with variation
        import random
        base_score = 82
        method_results[method] = base_score + random.randint(-8, 12)

    runner.print()
    runner.print("[bold green]Method Comparison:[/bold green]")
    runner.print()

    # Sort by score
    sorted_results = sorted(method_results.items(), key=lambda x: x[1], reverse=True)

    for i, (method, score) in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        bar_len = int(score / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        runner.print(f"  {medal} {method.value:20} [{bar}] {score}/100")

    runner.print()
    runner.print("[bold]Insights:[/bold]")
    runner.print("[dim]• Chain-of-thought often provides more consistent scores[/dim]")
    runner.print("[dim]• Pairwise is best for relative comparison[/dim]")
    runner.print("[dim]• Constitutional catches ethical issues[/dim]")
    runner.print("[dim]• G-Eval provides detailed aspect breakdown[/dim]")


# Direct execution for testing
if __name__ == "__main__":
    from .interactive_demo import DemoRunner

    async def main():
        runner = DemoRunner()
        await run_judge_demo(runner)

    asyncio.run(main())
