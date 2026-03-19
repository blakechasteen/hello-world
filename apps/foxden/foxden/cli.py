"""CLI entrypoint for foxden."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _resolve_persona_dir(custom: str | None) -> Path:
    if custom:
        return Path(custom)
    # Default: personas/ next to the package.
    return Path(__file__).parent.parent / "personas"


def _build_backend(backend: str, model: str | None):
    if backend == "ollama":
        from .llm.ollama import OllamaBackend

        return OllamaBackend(model=model or "qwen3:30b")
    elif backend == "anthropic":
        from .llm.anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=model or "claude-sonnet-4-20250514")
    elif backend == "hololoom":
        from .llm.hololoom_backend import HoloLoomBackend

        return HoloLoomBackend(model=model or "qwen3:30b")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="foxden",
        description="Agent-based software testing — a focus group on a chip",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Run a focus group session")
    run_p.add_argument("target", help="Target to test (command, URL, etc.)")
    run_p.add_argument(
        "-i", "--interface", default="cli", choices=["cli", "web", "api"],
        help="Interface type (default: cli)",
    )
    run_p.add_argument(
        "-b", "--backend", default="ollama", choices=["ollama", "anthropic", "hololoom"],
        help="LLM backend (default: ollama)",
    )
    run_p.add_argument("-m", "--model", help="Model name override")
    run_p.add_argument(
        "-p", "--personas", nargs="*",
        help="Persona names to use (default: all)",
    )
    run_p.add_argument("--persona-dir", help="Custom persona directory")
    run_p.add_argument(
        "-c", "--copies", type=int, default=1,
        help="Copies of each persona (default: 1)",
    )
    run_p.add_argument(
        "-n", "--concurrency", type=int, default=5,
        help="Max parallel sessions (default: 5)",
    )
    run_p.add_argument(
        "--max-steps", type=int, default=50,
        help="Max steps per session (default: 50)",
    )
    run_p.add_argument(
        "-o", "--output", help="Save report JSON to this path",
    )

    # Web-specific options.
    run_p.add_argument(
        "--headless", action="store_true", default=True,
        help="Run browser in headless mode (default: true)",
    )
    run_p.add_argument(
        "--headed", action="store_true",
        help="Run browser with visible window",
    )
    run_p.add_argument(
        "--screenshots", help="Directory to save screenshots",
    )
    run_p.add_argument(
        "--viewport", default="1280x720",
        help="Browser viewport WxH (default: 1280x720)",
    )

    # Synthesis options.
    run_p.add_argument(
        "--synthesize", action="store_true",
        help="Run LLM synthesis agent on results for a richer report",
    )
    run_p.add_argument(
        "--synthesis-output", help="Save synthesized report JSON to this path",
    )

    # API-specific options.
    run_p.add_argument(
        "--openapi", help="Path or URL to OpenAPI spec (JSON or YAML)",
    )
    run_p.add_argument(
        "--header", action="append", dest="api_headers", metavar="KEY:VALUE",
        help="HTTP header for API requests (repeatable, e.g. --header 'Authorization:Bearer tok')",
    )

    # Managed mode options.
    run_p.add_argument(
        "--managed", action="store_true",
        help="Self-organizing mode: agents manage the focus group (recruit, moderate, amplify)",
    )
    run_p.add_argument(
        "--no-recruit", action="store_true",
        help="Skip auto-recruitment in managed mode (use provided personas only)",
    )
    run_p.add_argument(
        "--token-budget", type=int, default=None, metavar="N",
        help="Total token budget for managed session",
    )
    run_p.add_argument(
        "--time-budget", type=float, default=None, metavar="MINUTES",
        help="Time budget in minutes for managed session",
    )

    # CI options.
    run_p.add_argument(
        "--fail-on-issues", type=int, default=None, metavar="N",
        help="Exit with code 1 if more than N issues found (for CI)",
    )
    run_p.add_argument(
        "--compare", metavar="BASELINE",
        help="Compare results against a previous report JSON (shows regressions)",
    )

    # --- tutorial ---
    tut_p = sub.add_parser("tutorial", help="Follow a tutorial and report divergences")
    tut_p.add_argument("tutorial_path", help="Path to tutorial/doc file (markdown, text)")
    tut_p.add_argument("target", help="Target to test (command, URL, etc.)")
    tut_p.add_argument(
        "-i", "--interface", default="cli", choices=["cli", "web", "api"],
        help="Interface type (default: cli)",
    )
    tut_p.add_argument(
        "-b", "--backend", default="ollama", choices=["ollama", "anthropic", "hololoom"],
        help="LLM backend (default: ollama)",
    )
    tut_p.add_argument("-m", "--model", help="Model name override")
    tut_p.add_argument("-o", "--output", help="Save result JSON to this path")
    tut_p.add_argument("--headed", action="store_true", help="Run browser with visible window")
    tut_p.add_argument("--viewport", default="1280x720", help="Browser viewport WxH")

    # --- diff ---
    diff_p = sub.add_parser("diff", help="Compare two focus group reports")
    diff_p.add_argument("before", help="Path to baseline report JSON")
    diff_p.add_argument("after", help="Path to new report JSON")

    # --- list ---
    list_p = sub.add_parser("list", help="List available personas")
    list_p.add_argument("--persona-dir", help="Custom persona directory")

    args = parser.parse_args(argv)

    if args.command == "list":
        from .core.persona import load_personas

        persona_dir = _resolve_persona_dir(args.persona_dir)
        personas = load_personas(persona_dir)
        if not personas:
            print(f"No personas found in {persona_dir}")
            return
        for name, p in personas.items():
            tags = ", ".join(p.tags) if p.tags else ""
            print(f"  {name:<25} [{tags}]")
            for g in p.goals:
                print(f"    - {g}")
        return

    if args.command == "run":
        exit_code = asyncio.run(_run(args))
        sys.exit(exit_code or 0)

    if args.command == "tutorial":
        asyncio.run(_tutorial(args))
        return

    if args.command == "diff":
        _diff(args)
        return

    parser.print_help()


async def _run(args) -> int | None:
    from .core.focus_group import build_focus_group

    persona_dir = _resolve_persona_dir(args.persona_dir)
    backend = _build_backend(args.backend, args.model)

    # Build interface kwargs.
    interface_kwargs = {}
    if args.interface == "web":
        interface_kwargs["headless"] = not args.headed
        if args.screenshots:
            interface_kwargs["screenshot_dir"] = args.screenshots
        if args.viewport:
            w, h = args.viewport.split("x")
            interface_kwargs["viewport_width"] = int(w)
            interface_kwargs["viewport_height"] = int(h)
    elif args.interface == "api":
        if args.api_headers:
            headers = {}
            for h in args.api_headers:
                key, _, val = h.partition(":")
                headers[key.strip()] = val.strip()
            interface_kwargs["api_headers"] = headers
        if args.openapi:
            from .interfaces.api import load_openapi_spec
            interface_kwargs["openapi_spec"] = await load_openapi_spec(args.openapi)

    # Managed mode: self-organizing focus group.
    if args.managed:
        from .managed.group import ManagedFocusGroup, ManagedConfig
        from .managed.facilitator import FacilitatorConfig
        from .core.persona import load_personas

        facilitator_config = FacilitatorConfig()
        if args.token_budget is not None:
            facilitator_config.token_budget = args.token_budget
        if args.time_budget is not None:
            facilitator_config.time_budget_seconds = args.time_budget * 60

        managed_config = ManagedConfig(
            recruiter=not args.no_recruit,
            facilitator_config=facilitator_config,
            max_steps=args.max_steps,
            concurrency=args.concurrency,
        )

        # Resolve personas if specific names were requested.
        initial_personas = None
        if args.personas:
            all_personas = load_personas(persona_dir)
            initial_personas = [
                all_personas[name] for name in args.personas if name in all_personas
            ]
            if not initial_personas:
                print(f"No matching personas found for: {args.personas}")
                return 1

        managed = ManagedFocusGroup(
            target=args.target,
            interface=args.interface,
            backend=backend,
            config=managed_config,
            persona_dir=persona_dir,
            personas=initial_personas,
            copies=args.copies,
            **interface_kwargs,
        )

        features = []
        if managed_config.recruiter:
            features.append("recruit")
        if managed_config.moderator:
            features.append("moderate")
        if managed_config.amplifier:
            features.append("amplify")
        if managed_config.pollinator:
            features.append("pollinate")
        print(f"Starting managed focus group: [{', '.join(features)}]")
        print(f"  interface={args.interface}, backend={args.backend}")
        if args.token_budget:
            print(f"  token budget: {args.token_budget:,}")
        if args.time_budget:
            print(f"  time budget: {args.time_budget:.0f}m")
        print()

        report = await managed.run()
        print()
        print(report.summary)

        if args.output:
            report.save(args.output)
            print(f"\nReport saved to {args.output}")

        # CI gate.
        if args.fail_on_issues is not None:
            total_issues = sum(len(t.issues) for t in report.transcripts)
            if total_issues > args.fail_on_issues:
                print(f"\nCI FAIL: {total_issues} issues found (threshold: {args.fail_on_issues})")
                return 1
            else:
                print(f"\nCI PASS: {total_issues} issues (threshold: {args.fail_on_issues})")

        return 0

    # Standard mode: fixed roster, no management layer.
    group = build_focus_group(
        target=args.target,
        interface=args.interface,
        persona_dir=persona_dir,
        backend=backend,
        persona_names=args.personas,
        copies=args.copies,
        concurrency=args.concurrency,
        max_steps=args.max_steps,
        **interface_kwargs,
    )

    total = len(group.personas) * group.copies
    print(f"Starting focus group: {total} agents, interface={args.interface}, backend={args.backend}")
    print()

    completed = 0

    def on_complete(transcript):
        nonlocal completed
        completed += 1
        print(f"  [{completed}/{total}] {transcript.persona_name} finished — "
              f"{len(transcript.issues)} issues, {len(transcript.confusion_points)} confusion points")

    report = await group.run(on_complete=on_complete)

    print()
    print(report.summary)

    if args.output:
        report.save(args.output)
        print(f"\nReport saved to {args.output}")

    # Synthesis.
    if args.synthesize and report.transcripts:
        from .core.synthesis import synthesize

        print("\nRunning synthesis agent...")
        synth = await synthesize(report.transcripts, backend)
        print()
        print(synth.summary)

        if args.synthesis_output:
            import json as json_mod

            out = Path(args.synthesis_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json_mod.dump(synth.raw_json, f, indent=2)
            print(f"\nSynthesized report saved to {args.synthesis_output}")

    # Compare against baseline.
    if args.compare and args.output:
        from .core.replay import RunSnapshot, compare

        before = RunSnapshot.load(args.compare)
        after = RunSnapshot.load(args.output)
        diff = compare(before, after)
        print()
        print(diff.summary)

        if diff.new_issues:
            print(f"\nWARNING: {len(diff.new_issues)} new issue(s) found since baseline")

    # CI gate.
    if args.fail_on_issues is not None:
        total_issues = sum(len(t.issues) for t in report.transcripts)
        if total_issues > args.fail_on_issues:
            print(f"\nCI FAIL: {total_issues} issues found (threshold: {args.fail_on_issues})")
            return 1
        else:
            print(f"\nCI PASS: {total_issues} issues (threshold: {args.fail_on_issues})")

    return 0


async def _tutorial(args):
    from .core.tutorial import follow_tutorial, load_tutorial

    backend = _build_backend(args.backend, args.model)
    steps = load_tutorial(args.tutorial_path)

    if not steps:
        print(f"No steps found in {args.tutorial_path}")
        return

    print(f"Following tutorial: {len(steps)} steps detected")
    print()

    # Build the interface.
    if args.interface == "cli":
        from .interfaces.cli import CLIInterface
        interface = CLIInterface(command=args.target)
        hint = "cli"
    elif args.interface == "web":
        from .interfaces.web import WebInterface
        w, h = args.viewport.split("x")
        interface = WebInterface(
            url=args.target,
            headless=not args.headed,
            viewport_width=int(w),
            viewport_height=int(h),
        )
        hint = "web"
    elif args.interface == "api":
        from .interfaces.api import APIInterface
        interface = APIInterface(base_url=args.target)
        hint = "api"
    else:
        raise ValueError(f"Unknown interface: {args.interface}")

    result = await follow_tutorial(steps, interface, backend, interface_hint=hint)

    print(result.summary)

    if args.output:
        import json as json_mod

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "steps_completed": result.steps_completed,
            "steps_total": result.steps_total,
            "success_rate": result.success_rate,
            "divergences": [
                {
                    "step": d.step,
                    "instruction": d.instruction,
                    "expected": d.expected,
                    "actual": d.actual,
                    "severity": d.severity,
                }
                for d in result.divergences
            ],
            "transcript": result.transcript.to_dict(),
        }
        with open(out, "w") as f:
            json_mod.dump(data, f, indent=2)
        print(f"\nResult saved to {args.output}")


def _diff(args):
    from .core.replay import RunSnapshot, compare

    before = RunSnapshot.load(args.before)
    after = RunSnapshot.load(args.after)
    diff = compare(before, after)
    print(diff.summary)


if __name__ == "__main__":
    main()
