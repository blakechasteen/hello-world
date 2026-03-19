"""Managed Focus Group — the full self-organizing test session.

Composes all management agents:
  - Facilitator: watches the big picture, detects saturation
  - Moderator: intervenes in individual sessions
  - Recruiter: selects and generates personas
  - Amplifier: scales up promising personas
  - Pollinator: shares findings across sessions

Usage:
    group = ManagedFocusGroup(
        target="http://localhost:3000",
        interface="web",
        backend=OllamaBackend(model="qwen3:30b"),
    )
    report = await group.run()
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.persona import Persona, load_personas
from ..core.report import Report
from ..interfaces.base import TestInterface
from ..llm.backend import LLMBackend
from ..recording.transcript import Transcript
from .amplifier import Amplifier, AmplifierConfig
from .events import Event, EventBus, EventKind
from .facilitator import Facilitator, FacilitatorConfig
from .moderator import Moderator, ModeratorConfig
from .pollinator import Pollinator, PollinatorConfig
from .session_bridge import ManagedSession


@dataclass
class ManagedConfig:
    """Top-level config for a managed focus group."""

    # Which managers to enable.
    facilitator: bool = True
    moderator: bool = True
    recruiter: bool = True
    amplifier: bool = True
    pollinator: bool = True

    # Sub-configs (None = use defaults).
    facilitator_config: FacilitatorConfig | None = None
    moderator_config: ModeratorConfig | None = None
    amplifier_config: AmplifierConfig | None = None
    pollinator_config: PollinatorConfig | None = None

    # Session defaults.
    max_steps: int = 50
    concurrency: int = 5


class ManagedFocusGroup:
    """Self-organizing focus group with management agents."""

    def __init__(
        self,
        target: str,
        interface: str,
        backend: LLMBackend,
        *,
        config: ManagedConfig | None = None,
        persona_dir: str | Path | None = None,
        personas: list[Persona] | None = None,
        copies: int = 1,
        interface_factory: Callable[[], TestInterface] | None = None,
        **interface_kwargs,
    ):
        self.target = target
        self.interface_type = interface
        self.backend = backend
        self.config = config or ManagedConfig()
        self.interface_kwargs = interface_kwargs
        self._copies = copies

        # Event bus.
        self.bus = EventBus()

        # Persona source.
        self._persona_dir = Path(persona_dir) if persona_dir else Path(__file__).parent.parent.parent / "personas"
        self._initial_personas = personas
        self._interface_factory = interface_factory

        # Management agents.
        self._managers = []

        # State.
        self._transcripts: list[Transcript] = []
        self._complete = asyncio.Event()
        self._spawn_queue: asyncio.Queue[Persona] = asyncio.Queue()

    def _build_interface(self) -> TestInterface:
        """Build an interface instance."""
        if self._interface_factory:
            return self._interface_factory()

        if self.interface_type == "cli":
            from ..interfaces.cli import CLIInterface
            return CLIInterface(command=self.target, **self.interface_kwargs)
        elif self.interface_type == "web":
            from ..interfaces.web import WebInterface
            return WebInterface(url=self.target, **self.interface_kwargs)
        elif self.interface_type == "api":
            from ..interfaces.api import APIInterface
            return APIInterface(base_url=self.target, **self.interface_kwargs)
        else:
            raise ValueError(f"Unknown interface: {self.interface_type}")

    async def run(
        self,
        on_complete: Callable[[Transcript], None] | None = None,
    ) -> Report:
        """Run the managed focus group."""

        # --- 1. Recruit ---
        if self._initial_personas:
            personas = self._initial_personas
            copies = {p.name: self._copies for p in personas}
        elif self.config.recruiter:
            from .recruiter import Recruiter
            recruiter = Recruiter(self.bus, self.backend, self._persona_dir)
            recruiter.attach()
            self._managers.append(recruiter)
            plan = await recruiter.recruit(self.target, self.interface_type)
            personas = plan.personas
            copies = plan.copies
            print(f"  Recruiter: {plan.software_type}")
            for name, reason in plan.reasoning.items():
                print(f"    {name} (x{copies.get(name, 1)}): {reason}")
            print()
        else:
            all_personas = load_personas(self._persona_dir)
            personas = list(all_personas.values())
            copies = {p.name: self._copies for p in personas}

        # --- 2. Attach managers ---
        if self.config.facilitator:
            facilitator = Facilitator(
                self.bus, self.backend, self.config.facilitator_config
            )
            facilitator.attach()
            self._managers.append(facilitator)

        if self.config.moderator:
            moderator = Moderator(
                self.bus, self.backend, self.config.moderator_config
            )
            moderator.attach()
            self._managers.append(moderator)

        if self.config.amplifier:
            amplifier = Amplifier(self.bus, self.config.amplifier_config)
            amplifier.attach()
            self._managers.append(amplifier)

        if self.config.pollinator:
            pollinator = Pollinator(self.bus, self.config.pollinator_config)
            pollinator.attach()
            self._managers.append(pollinator)

        # Listen for group completion.
        self.bus.on(EventKind.GROUP_COMPLETE, self._on_complete)

        # Listen for spawn requests from amplifier.
        self.bus.on(EventKind.SPAWN_AGENT, self._on_spawn)

        # --- 3. Run sessions ---
        sem = asyncio.Semaphore(self.config.concurrency)

        async def _run_one(persona: Persona) -> Transcript:
            async with sem:
                session_id = uuid.uuid4().hex[:12]
                interface = self._build_interface()
                session = ManagedSession(
                    session_id=session_id,
                    persona=persona,
                    interface=interface,
                    backend=self.backend,
                    bus=self.bus,
                    max_steps=self.config.max_steps,
                )
                transcript = await session.run()
                self._transcripts.append(transcript)
                if on_complete:
                    on_complete(transcript)
                return transcript

        # Initial batch.
        tasks = []
        for persona in personas:
            n = copies.get(persona.name, 1)
            for _ in range(n):
                tasks.append(asyncio.create_task(_run_one(persona)))

        # Spawn loop: also process amplifier requests.
        spawn_task = asyncio.create_task(self._spawn_loop(_run_one, tasks, on_complete))

        # Wait for all initial tasks + any spawned ones.
        done = False
        while not done:
            if tasks:
                finished, _ = await asyncio.wait(
                    tasks, timeout=2.0, return_when=asyncio.FIRST_COMPLETED
                )
                for t in finished:
                    tasks.remove(t)
                    # Absorb exceptions.
                    try:
                        t.result()
                    except Exception as e:
                        print(f"  Session failed: {e}")

            # Check if we're done.
            if not tasks and self._spawn_queue.empty():
                done = True
            if self._complete.is_set() and not tasks:
                done = True

        spawn_task.cancel()
        try:
            await spawn_task
        except asyncio.CancelledError:
            pass

        # --- 4. Detach managers ---
        for m in self._managers:
            m.detach()

        # --- 5. Report ---
        report = Report(transcripts=self._transcripts)

        # Print management stats.
        stats = self.bus.stats
        if stats:
            print(f"\n  Event bus stats: {stats}")

        for m in self._managers:
            if hasattr(m, "leaderboard"):
                board = m.leaderboard
                if board:
                    print(f"\n  Amplifier leaderboard:")
                    for entry in board[:5]:
                        print(f"    {entry['persona']}: {entry['issues']} issues "
                              f"({entry['hit_rate']} hit rate, {entry['sessions']} sessions)")

            if hasattr(m, "sharing_stats"):
                ss = m.sharing_stats
                if ss["total_shares"] > 0:
                    print(f"\n  Pollinator: {ss['total_shares']} cross-pollinations "
                          f"across {ss['total_findings']} findings")

        return report

    async def _on_complete(self, event: Event) -> None:
        reason = event.data.get("reason", "unknown")
        print(f"\n  Facilitator: wrapping up ({reason})")
        self._complete.set()
        # Terminate all active sessions.
        await self.bus.emit(Event(
            kind=EventKind.TERMINATE_SESSION,
            session_id="*",
            data={"reason": reason},
        ))

    async def _on_spawn(self, event: Event) -> None:
        persona_name = event.persona
        # Find the persona.
        all_personas = load_personas(self._persona_dir)
        if persona_name in all_personas:
            await self._spawn_queue.put(all_personas[persona_name])
            print(f"  Amplifier: spawning another {persona_name} "
                  f"(hit rate: {event.data.get('hit_rate', '?')})")

    async def _spawn_loop(self, run_fn, tasks, on_complete) -> None:
        """Process the spawn queue."""
        try:
            while True:
                persona = await asyncio.wait_for(self._spawn_queue.get(), timeout=1.0)
                task = asyncio.create_task(run_fn(persona))
                tasks.append(task)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
