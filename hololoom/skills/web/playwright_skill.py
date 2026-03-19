"""Playwright Skill - Browser automation and testing"""

import time
from typing import Any

from hololoom.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillInput,
    SkillMetadata,
    SkillOutput,
    SkillStatus,
)

# Check for playwright availability
try:
    from playwright.async_api import Browser, Page, async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class PlaywrightSkill(BaseSkill):
    """Browser automation with Playwright."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="playwright",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.WEB,
            tags=["browser", "automation", "testing", "scraping", "e2e"],
            description_short="Browser automation and testing with Playwright",
            description_long="Automate Chrome/Firefox/WebKit browsers for testing, scraping, screenshots, PDFs, and E2E workflows. Supports headless mode, device emulation, and network interception.",
            requires_code_execution=True,
            requires_network=True,
            external_dependencies=["playwright"],
            expected_latency_ms="1000-30000ms",
            token_usage="200-2000 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = ["navigate", "screenshot", "pdf", "scrape", "fill_form", "click", "wait_for"]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")
        if skill_input.operation == "navigate" and "url" not in skill_input.parameters:
            raise ValueError("navigate requires 'url' parameter")
        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()

        # Check for playwright availability
        if not PLAYWRIGHT_AVAILABLE:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message="playwright not available. Install with: pip install playwright && playwright install",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=["Missing dependency: playwright"]
            )

        try:
            # Dispatch to operation handler
            if skill_input.operation == "navigate":
                result = await self._navigate(skill_input.parameters)
            elif skill_input.operation == "screenshot":
                result = await self._screenshot(skill_input.parameters)
            elif skill_input.operation == "pdf":
                result = await self._pdf(skill_input.parameters)
            elif skill_input.operation == "scrape":
                result = await self._scrape(skill_input.parameters)
            elif skill_input.operation == "fill_form":
                result = await self._fill_form(skill_input.parameters)
            elif skill_input.operation == "click":
                result = await self._click(skill_input.parameters)
            elif skill_input.operation == "wait_for":
                result = await self._wait_for(skill_input.parameters)
            else:
                raise ValueError(f"Unknown operation: {skill_input.operation}")

            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"Playwright '{skill_input.operation}' completed",
                execution_time_ms=(time.time() - start_time) * 1000,
                details=result
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Playwright operation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _navigate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Navigate to URL."""
        url = params["url"]
        headless = params.get("headless", True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            await page.goto(url)

            title = await page.title()
            await browser.close()

            return {
                "operation": "navigate",
                "url": url,
                "title": title,
                "loaded": True
            }

    async def _screenshot(self, params: dict[str, Any]) -> dict[str, Any]:
        """Take screenshot of page."""
        url = params["url"]
        path = params.get("path", "screenshot.png")
        full_page = params.get("full_page", False)
        headless = params.get("headless", True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            await page.goto(url)

            await page.screenshot(path=path, full_page=full_page)
            await browser.close()

            # Get file size
            import os
            size_kb = os.path.getsize(path) / 1024

            return {
                "operation": "screenshot",
                "url": url,
                "path": path,
                "format": "png",
                "size_kb": round(size_kb, 2),
                "full_page": full_page
            }

    async def _pdf(self, params: dict[str, Any]) -> dict[str, Any]:
        """Save page as PDF."""
        url = params["url"]
        path = params.get("path", "page.pdf")
        headless = params.get("headless", True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            await page.goto(url)

            await page.pdf(path=path)
            await browser.close()

            # Get file size
            import os
            size_kb = os.path.getsize(path) / 1024

            return {
                "operation": "pdf",
                "url": url,
                "path": path,
                "size_kb": round(size_kb, 2)
            }

    async def _scrape(self, params: dict[str, Any]) -> dict[str, Any]:
        """Scrape text content from page."""
        url = params["url"]
        selector = params.get("selector", "body")
        headless = params.get("headless", True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            await page.goto(url)

            # Get text content
            text = await page.locator(selector).inner_text()

            # Count links and images
            links = await page.locator("a").count()
            images = await page.locator("img").count()

            await browser.close()

            return {
                "operation": "scrape",
                "url": url,
                "selector": selector,
                "text": text[:1000],  # Limit to 1000 chars
                "text_length": len(text),
                "links": links,
                "images": images
            }

    async def _fill_form(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fill form fields."""
        url = params["url"]
        fields = params["fields"]  # {selector: value}
        submit_selector = params.get("submit_selector")
        headless = params.get("headless", True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            await page.goto(url)

            # Fill each field
            for selector, value in fields.items():
                await page.fill(selector, value)

            # Submit if selector provided
            submitted = False
            if submit_selector:
                await page.click(submit_selector)
                submitted = True

            await browser.close()

            return {
                "operation": "fill_form",
                "url": url,
                "fields_filled": len(fields),
                "submitted": submitted
            }

    async def _click(self, params: dict[str, Any]) -> dict[str, Any]:
        """Click element."""
        url = params["url"]
        selector = params["selector"]
        headless = params.get("headless", True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            await page.goto(url)

            await page.click(selector)
            await browser.close()

            return {
                "operation": "click",
                "url": url,
                "selector": selector,
                "clicked": True
            }

    async def _wait_for(self, params: dict[str, Any]) -> dict[str, Any]:
        """Wait for element to appear."""
        url = params["url"]
        selector = params["selector"]
        timeout = params.get("timeout", 30000)  # 30 seconds default
        headless = params.get("headless", True)

        start_time = time.time()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            await page.goto(url)

            await page.wait_for_selector(selector, timeout=timeout)
            wait_time_ms = (time.time() - start_time) * 1000

            await browser.close()

            return {
                "operation": "wait_for",
                "url": url,
                "selector": selector,
                "found": True,
                "wait_time_ms": round(wait_time_ms, 2)
            }


from hololoom.skills.base import register_skill

register_skill(PlaywrightSkill())
