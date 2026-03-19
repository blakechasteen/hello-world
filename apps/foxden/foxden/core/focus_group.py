"""Focus Group — orchestrate multiple agent sessions in parallel."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..core.persona import Persona, load_personas
from ..core.report import Report
from ..core.session import run_session
from ..interfaces.base import TestInterface
from ..llm.backend import LLMBackend
from ..recording.transcript import Transcript


@dataclass
class FocusGroup:
    """Run N personas against a target, collect findings."""

    personas: list[Persona]
    interface_factory: Callable[[], TestInterface]
    backend: LLMBackend
    copies: int = 1  # how many copies of each persona to run
    max_steps: int = 50
    concurrency: int = 5  # max parallel sessions

    async def run(self, on_complete: Callable[[Transcript], None] | None = None) -> Report:
        """Run all sessions and produce a report."""
        sem = asyncio.Semaphore(self.concurrency)

        async def _run_one(persona: Persona, copy_id: int) -> Transcript:
            async with sem:
                interface = self.interface_factory()
                transcript = await run_session(
                    persona, interface, self.backend, max_steps=self.max_steps
                )
                if on_complete:
                    on_complete(transcript)
                return transcript

        tasks = [
            _run_one(persona, copy_id)
            for persona in self.personas
            for copy_id in range(self.copies)
        ]

        transcripts = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures.
        good: list[Transcript] = []
        for t in transcripts:
            if isinstance(t, Transcript):
                good.append(t)
            else:
                print(f"Session failed: {t}")

        return Report(transcripts=good)


def build_focus_group(
    target: str,
    interface: str,
    persona_dir: str | Path,
    backend: LLMBackend,
    *,
    persona_names: list[str] | None = None,
    copies: int = 1,
    concurrency: int = 5,
    max_steps: int = 50,
    **interface_kwargs,
) -> FocusGroup:
    """Convenience builder for common configurations."""
    all_personas = load_personas(persona_dir)

    if persona_names:
        personas = [all_personas[n] for n in persona_names if n in all_personas]
    else:
        personas = list(all_personas.values())

    if interface == "cli":
        from ..interfaces.cli import CLIInterface

        def factory() -> TestInterface:
            return CLIInterface(command=target, **interface_kwargs)

    elif interface == "web":
        from ..interfaces.web import WebInterface

        headless = interface_kwargs.pop("headless", True)
        screenshot_dir = interface_kwargs.pop("screenshot_dir", None)
        viewport_width = interface_kwargs.pop("viewport_width", 1280)
        viewport_height = interface_kwargs.pop("viewport_height", 720)

        def factory() -> TestInterface:
            return WebInterface(
                url=target,
                headless=headless,
                screenshot_dir=screenshot_dir,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                **interface_kwargs,
            )

    elif interface == "api":
        from ..interfaces.api import APIInterface

        api_headers = interface_kwargs.pop("api_headers", {})
        openapi_spec = interface_kwargs.pop("openapi_spec", None)

        def factory() -> TestInterface:
            return APIInterface(
                base_url=target,
                headers=api_headers,
                openapi_spec=openapi_spec,
                **interface_kwargs,
            )
    else:
        raise ValueError(f"Unknown interface: {interface}")

    return FocusGroup(
        personas=personas,
        interface_factory=factory,
        backend=backend,
        copies=copies,
        concurrency=concurrency,
        max_steps=max_steps,
    )
