"""Web interface — test web applications via Playwright browser automation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .base import Action, Observation


@dataclass
class WebInterface:
    """Interact with a web application via a real browser."""

    url: str
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout: float = 30_000  # ms
    screenshots: bool = True
    screenshot_dir: str | None = None

    # Private state — set during start().
    _playwright: Any = field(default=None, init=False, repr=False)
    _browser: Any = field(default=None, init=False, repr=False)
    _page: Any = field(default=None, init=False, repr=False)
    _screenshot_count: int = field(default=0, init=False)

    async def start(self) -> Observation:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError("pip install foxden[web]  (requires playwright)")

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._page = await self._browser.new_page(
            viewport={"width": self.viewport_width, "height": self.viewport_height}
        )
        self._page.set_default_timeout(self.timeout)

        await self._page.goto(self.url, wait_until="domcontentloaded")
        return await self._observe("Navigated to " + self.url)

    async def act(self, action: Action) -> Observation:
        page = self._page
        kind = action.kind
        value = action.value
        meta = action.metadata or {}

        try:
            if kind == "click":
                await page.click(value, timeout=self.timeout)
                return await self._observe(f"Clicked: {value}")

            elif kind == "type":
                selector = meta.get("selector", value)
                text = meta.get("text", value)
                if selector != text:
                    await page.fill(selector, text)
                    return await self._observe(f"Typed '{text}' into {selector}")
                else:
                    await page.keyboard.type(text)
                    return await self._observe(f"Typed: {text}")

            elif kind == "navigate":
                await page.goto(value, wait_until="domcontentloaded")
                return await self._observe(f"Navigated to {value}")

            elif kind == "press":
                await page.keyboard.press(value)
                return await self._observe(f"Pressed key: {value}")

            elif kind == "scroll":
                direction = value.lower()
                if direction == "down":
                    await page.evaluate("window.scrollBy(0, 500)")
                elif direction == "up":
                    await page.evaluate("window.scrollBy(0, -500)")
                return await self._observe(f"Scrolled {direction}")

            elif kind == "wait":
                ms = int(value) if value.isdigit() else 1000
                await page.wait_for_timeout(ms)
                return await self._observe(f"Waited {ms}ms")

            elif kind == "select":
                selector = meta.get("selector", "")
                await page.select_option(selector, value)
                return await self._observe(f"Selected '{value}' in {selector}")

            elif kind == "hover":
                await page.hover(value, timeout=self.timeout)
                return await self._observe(f"Hovered: {value}")

            elif kind == "back":
                await page.go_back(wait_until="domcontentloaded")
                return await self._observe("Navigated back")

            else:
                return Observation(
                    content=f"Unknown action kind: {kind}. "
                    f"Available: click, type, navigate, press, scroll, wait, select, hover, back"
                )

        except Exception as e:
            return Observation(
                content=f"ACTION FAILED: {kind} '{value}' — {type(e).__name__}: {e}",
                raw={"error": str(e), "action_kind": kind},
            )

    async def _observe(self, action_summary: str) -> Observation:
        """Capture the current page state for the agent."""
        page = self._page
        parts = [f"ACTION: {action_summary}", ""]

        # Page info.
        title = await page.title()
        url = page.url
        parts.append(f"PAGE: {title}")
        parts.append(f"URL: {url}")
        parts.append("")

        # Accessibility snapshot — gives the agent a structured view of the page.
        try:
            snapshot = await page.accessibility.snapshot()
            if snapshot:
                tree_text = _format_a11y_tree(snapshot, max_depth=4)
                parts.append("ACCESSIBILITY TREE:")
                parts.append(tree_text)
        except Exception:
            pass

        # Visible text (fallback / supplement).
        try:
            visible = await page.evaluate(
                """() => {
                    const sel = document.querySelectorAll(
                        'h1,h2,h3,h4,h5,h6,p,a,button,input,select,textarea,label,li,td,th,span'
                    );
                    const items = [];
                    for (const el of sel) {
                        const tag = el.tagName.toLowerCase();
                        const text = el.innerText?.trim().slice(0, 200);
                        if (!text) continue;
                        const attrs = [];
                        if (el.href) attrs.push('href=' + el.href);
                        if (el.type) attrs.push('type=' + el.type);
                        if (el.name) attrs.push('name=' + el.name);
                        if (el.id) attrs.push('id=' + el.id);
                        if (el.placeholder) attrs.push('placeholder=' + el.placeholder);
                        const attrStr = attrs.length ? ' [' + attrs.join(', ') + ']' : '';
                        items.push(tag + attrStr + ': ' + text);
                        if (items.length > 80) break;
                    }
                    return items.join('\\n');
                }"""
            )
            if visible and visible.strip():
                parts.append("")
                parts.append("VISIBLE ELEMENTS:")
                parts.append(visible)
        except Exception:
            pass

        # Screenshot.
        screenshot_bytes = None
        if self.screenshots:
            try:
                screenshot_bytes = await page.screenshot(type="png")
                self._screenshot_count += 1
                if self.screenshot_dir:
                    import os
                    os.makedirs(self.screenshot_dir, exist_ok=True)
                    path = os.path.join(
                        self.screenshot_dir, f"screenshot_{self._screenshot_count:04d}.png"
                    )
                    with open(path, "wb") as f:
                        f.write(screenshot_bytes)
                    parts.append(f"\nSCREENSHOT saved: {path}")
            except Exception:
                pass

        return Observation(
            content="\n".join(parts),
            raw={"title": title, "url": url},
            screenshot=screenshot_bytes,
        )

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()


def _format_a11y_tree(node: dict, indent: int = 0, max_depth: int = 4) -> str:
    """Format a Playwright accessibility snapshot into readable text."""
    if indent // 2 >= max_depth:
        return ""

    lines = []
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    parts = []
    if role:
        parts.append(role)
    if name:
        parts.append(f'"{name}"')
    if value:
        parts.append(f"value={value}")

    # Include useful state.
    for attr in ("checked", "pressed", "selected", "disabled", "expanded"):
        if node.get(attr):
            parts.append(attr)

    if parts:
        prefix = "  " * indent
        lines.append(f"{prefix}{' '.join(parts)}")

    for child in node.get("children", []):
        child_text = _format_a11y_tree(child, indent + 1, max_depth)
        if child_text:
            lines.append(child_text)

    return "\n".join(lines)
