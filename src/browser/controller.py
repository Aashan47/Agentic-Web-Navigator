"""
Playwright Browser Controller for the Agentic Web Navigator.
Provides async browser automation with comprehensive error handling.
"""

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Callable
from datetime import datetime
import traceback

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    ElementHandle,
    TimeoutError as PlaywrightTimeout,
    Error as PlaywrightError,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from src.utils.logger import agent_logger as logger
from src.utils.state import ActionType, ActionStatus, global_state


@dataclass
class ActionResult:
    """Result of a browser action."""
    success: bool
    action_type: ActionType
    message: str
    error: Optional[str] = None
    traceback: Optional[str] = None
    screenshot_base64: Optional[str] = None
    url_before: Optional[str] = None
    url_after: Optional[str] = None
    duration_ms: Optional[float] = None


class BrowserController:
    """
    Async Playwright browser controller with error handling and retry logic.
    Wraps all Playwright operations with try-except for LLM feedback.
    """

    def __init__(
        self,
        headless: bool = None,
        viewport_width: int = None,
        viewport_height: int = None,
        timeout_ms: int = None,
        slow_mo: int = None,
        user_agent: str = None,
        screenshot_dir: Path = Path("screenshots")
    ):
        # Use settings defaults if not specified
        self.headless = headless if headless is not None else settings.browser.headless
        self.viewport_width = viewport_width or settings.browser.viewport_width
        self.viewport_height = viewport_height or settings.browser.viewport_height
        self.timeout_ms = timeout_ms or settings.browser.timeout_ms
        self.slow_mo = slow_mo or settings.browser.slow_mo
        self.user_agent = user_agent or settings.browser.user_agent
        self.screenshot_dir = screenshot_dir

        # Playwright objects
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

        # State
        self._is_initialized = False

        # Ensure screenshot directory exists
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> ActionResult:
        """Initialize the browser and create a new page."""
        start_time = datetime.now()
        try:
            logger.info("Initializing Playwright browser...")

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--no-sandbox",
                ]
            )

            self._context = await self._browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height},
                user_agent=self.user_agent,
                java_script_enabled=True,
            )

            # Set default timeout
            self._context.set_default_timeout(self.timeout_ms)

            self._page = await self._context.new_page()
            self._is_initialized = True

            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.success("Browser initialized successfully")

            return ActionResult(
                success=True,
                action_type=ActionType.NAVIGATE,
                message="Browser initialized successfully",
                duration_ms=duration
            )

        except Exception as e:
            error_tb = traceback.format_exc()
            logger.error(f"Failed to initialize browser: {e}", exception=e)
            return ActionResult(
                success=False,
                action_type=ActionType.NAVIGATE,
                message="Failed to initialize browser",
                error=str(e),
                traceback=error_tb
            )

    async def close(self):
        """Close the browser and cleanup resources."""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()

            self._is_initialized = False
            logger.info("Browser closed successfully")
        except Exception as e:
            logger.error(f"Error closing browser: {e}", exception=e)

    async def _ensure_initialized(self):
        """Ensure browser is initialized before operations."""
        if not self._is_initialized or not self._page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

    def _wrap_action(self, action_type: ActionType):
        """Decorator factory for wrapping actions with error handling."""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs) -> ActionResult:
                await self._ensure_initialized()
                start_time = datetime.now()
                url_before = self._page.url

                try:
                    result = await func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds() * 1000

                    # Wait a bit for any navigation/changes
                    await asyncio.sleep(settings.agent.action_delay_ms / 1000)

                    url_after = self._page.url

                    return ActionResult(
                        success=True,
                        action_type=action_type,
                        message=result if isinstance(result, str) else "Action completed",
                        url_before=url_before,
                        url_after=url_after,
                        duration_ms=duration
                    )

                except PlaywrightTimeout as e:
                    error_tb = traceback.format_exc()
                    logger.error(f"Timeout during {action_type.value}: {e}")
                    return ActionResult(
                        success=False,
                        action_type=action_type,
                        message=f"Timeout: Element not found or action timed out",
                        error=str(e),
                        traceback=error_tb,
                        url_before=url_before
                    )

                except PlaywrightError as e:
                    error_tb = traceback.format_exc()
                    logger.error(f"Playwright error during {action_type.value}: {e}")
                    return ActionResult(
                        success=False,
                        action_type=action_type,
                        message=f"Browser error: {str(e)[:200]}",
                        error=str(e),
                        traceback=error_tb,
                        url_before=url_before
                    )

                except Exception as e:
                    error_tb = traceback.format_exc()
                    logger.error(f"Unexpected error during {action_type.value}: {e}", exception=e)
                    return ActionResult(
                        success=False,
                        action_type=action_type,
                        message=f"Unexpected error: {str(e)[:200]}",
                        error=str(e),
                        traceback=error_tb,
                        url_before=url_before
                    )

            return wrapper
        return decorator

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> ActionResult:
        """Navigate to a URL."""
        @self._wrap_action(ActionType.NAVIGATE)
        async def _navigate():
            logger.action("navigate", {"url": url, "wait_until": wait_until})
            await self._page.goto(url, wait_until=wait_until)
            return f"Navigated to {url}"

        return await _navigate()

    async def click(
        self,
        selector: str,
        timeout_ms: int = None,
        force: bool = False
    ) -> ActionResult:
        """Click on an element."""
        @self._wrap_action(ActionType.CLICK)
        async def _click():
            logger.action("click", {"selector": selector, "force": force})
            timeout = timeout_ms or self.timeout_ms
            await self._page.click(selector, timeout=timeout, force=force)
            return f"Clicked element: {selector}"

        return await _click()

    async def click_by_text(self, text: str, tag: str = None) -> ActionResult:
        """Click an element by its visible text content."""
        @self._wrap_action(ActionType.CLICK)
        async def _click_by_text():
            logger.action("click_by_text", {"text": text, "tag": tag})

            if tag:
                selector = f"{tag}:has-text(\"{text}\")"
            else:
                selector = f"text=\"{text}\""

            await self._page.click(selector)
            return f"Clicked element with text: {text}"

        return await _click_by_text()

    async def type_text(
        self,
        selector: str,
        text: str,
        clear_first: bool = True,
        delay_ms: int = 50
    ) -> ActionResult:
        """Type text into an input field."""
        @self._wrap_action(ActionType.TYPE)
        async def _type():
            logger.action("type", {"selector": selector, "text": text[:50], "clear_first": clear_first})

            if clear_first:
                await self._page.fill(selector, "")
                await asyncio.sleep(0.1)

            await self._page.type(selector, text, delay=delay_ms)
            return f"Typed text into: {selector}"

        return await _type()

    async def fill(self, selector: str, text: str) -> ActionResult:
        """Fill an input field (faster than type, no key events)."""
        @self._wrap_action(ActionType.TYPE)
        async def _fill():
            logger.action("fill", {"selector": selector, "text": text[:50]})
            await self._page.fill(selector, text)
            return f"Filled: {selector}"

        return await _fill()

    async def select_option(
        self,
        selector: str,
        value: str = None,
        label: str = None,
        index: int = None
    ) -> ActionResult:
        """Select an option from a dropdown."""
        @self._wrap_action(ActionType.CLICK)
        async def _select():
            logger.action("select", {"selector": selector, "value": value, "label": label})

            if value:
                await self._page.select_option(selector, value=value)
            elif label:
                await self._page.select_option(selector, label=label)
            elif index is not None:
                await self._page.select_option(selector, index=index)

            return f"Selected option in: {selector}"

        return await _select()

    async def scroll(
        self,
        direction: str = "down",
        amount: int = 500,
        selector: str = None
    ) -> ActionResult:
        """Scroll the page or a specific element."""
        @self._wrap_action(ActionType.SCROLL)
        async def _scroll():
            logger.action("scroll", {"direction": direction, "amount": amount, "selector": selector})

            if selector:
                element = await self._page.query_selector(selector)
                if element:
                    await element.scroll_into_view_if_needed()
                    return f"Scrolled element into view: {selector}"

            # Calculate scroll delta
            delta_y = amount if direction == "down" else -amount
            delta_x = 0
            if direction == "right":
                delta_x = amount
                delta_y = 0
            elif direction == "left":
                delta_x = -amount
                delta_y = 0

            await self._page.mouse.wheel(delta_x, delta_y)
            return f"Scrolled {direction} by {amount}px"

        return await _scroll()

    async def scroll_to_bottom(self) -> ActionResult:
        """Scroll to the bottom of the page."""
        @self._wrap_action(ActionType.SCROLL)
        async def _scroll_bottom():
            logger.action("scroll_to_bottom", {})
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            return "Scrolled to bottom of page"

        return await _scroll_bottom()

    async def scroll_to_top(self) -> ActionResult:
        """Scroll to the top of the page."""
        @self._wrap_action(ActionType.SCROLL)
        async def _scroll_top():
            logger.action("scroll_to_top", {})
            await self._page.evaluate("window.scrollTo(0, 0)")
            return "Scrolled to top of page"

        return await _scroll_top()

    async def wait(self, milliseconds: int) -> ActionResult:
        """Wait for a specified duration."""
        @self._wrap_action(ActionType.WAIT)
        async def _wait():
            logger.action("wait", {"milliseconds": milliseconds})
            await asyncio.sleep(milliseconds / 1000)
            return f"Waited {milliseconds}ms"

        return await _wait()

    async def wait_for_selector(
        self,
        selector: str,
        state: str = "visible",
        timeout_ms: int = None
    ) -> ActionResult:
        """Wait for an element to appear."""
        @self._wrap_action(ActionType.WAIT)
        async def _wait_selector():
            logger.action("wait_for_selector", {"selector": selector, "state": state})
            timeout = timeout_ms or self.timeout_ms
            await self._page.wait_for_selector(selector, state=state, timeout=timeout)
            return f"Element appeared: {selector}"

        return await _wait_selector()

    async def wait_for_navigation(self, timeout_ms: int = None) -> ActionResult:
        """Wait for page navigation to complete."""
        @self._wrap_action(ActionType.WAIT)
        async def _wait_nav():
            logger.action("wait_for_navigation", {})
            timeout = timeout_ms or self.timeout_ms
            await self._page.wait_for_load_state("domcontentloaded", timeout=timeout)
            return "Navigation completed"

        return await _wait_nav()

    async def press_key(self, key: str) -> ActionResult:
        """Press a keyboard key."""
        @self._wrap_action(ActionType.TYPE)
        async def _press():
            logger.action("press_key", {"key": key})
            await self._page.keyboard.press(key)
            return f"Pressed key: {key}"

        return await _press()

    async def go_back(self) -> ActionResult:
        """Navigate back in history."""
        @self._wrap_action(ActionType.BACK)
        async def _back():
            logger.action("go_back", {})
            await self._page.go_back()
            return "Navigated back"

        return await _back()

    async def refresh(self) -> ActionResult:
        """Refresh the current page."""
        @self._wrap_action(ActionType.REFRESH)
        async def _refresh():
            logger.action("refresh", {})
            await self._page.reload()
            return "Page refreshed"

        return await _refresh()

    async def take_screenshot(
        self,
        full_page: bool = False,
        quality: int = None
    ) -> ActionResult:
        """Take a screenshot and return as base64."""
        await self._ensure_initialized()
        start_time = datetime.now()

        try:
            quality = quality or settings.agent.screenshot_quality

            screenshot_bytes = await self._page.screenshot(
                full_page=full_page,
                type="jpeg",
                quality=quality
            )

            screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"screenshot_{timestamp}.jpg"
            filepath = self.screenshot_dir / filename

            with open(filepath, "wb") as f:
                f.write(screenshot_bytes)

            duration = (datetime.now() - start_time).total_seconds() * 1000

            return ActionResult(
                success=True,
                action_type=ActionType.SCREENSHOT,
                message=f"Screenshot saved: {filename}",
                screenshot_base64=screenshot_base64,
                duration_ms=duration
            )

        except Exception as e:
            error_tb = traceback.format_exc()
            logger.error(f"Failed to take screenshot: {e}", exception=e)
            return ActionResult(
                success=False,
                action_type=ActionType.SCREENSHOT,
                message="Failed to capture screenshot",
                error=str(e),
                traceback=error_tb
            )

    async def get_page_content(self) -> tuple[str, str, str]:
        """
        Get current page HTML, URL, and title.

        Returns:
            Tuple of (html, url, title)
        """
        await self._ensure_initialized()

        html = await self._page.content()
        url = self._page.url
        title = await self._page.title()

        return html, url, title

    async def get_element_info(self, selector: str) -> Optional[dict]:
        """Get information about a specific element."""
        await self._ensure_initialized()

        try:
            element = await self._page.query_selector(selector)
            if not element:
                return None

            # Get bounding box
            box = await element.bounding_box()

            # Get attributes
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            text = await element.text_content()
            is_visible = await element.is_visible()
            is_enabled = await element.is_enabled()

            return {
                "tag": tag_name,
                "text": text[:100] if text else "",
                "bounding_box": box,
                "is_visible": is_visible,
                "is_enabled": is_enabled,
                "selector": selector
            }

        except Exception as e:
            logger.debug(f"Could not get element info for {selector}: {e}")
            return None

    async def evaluate_js(self, script: str) -> Any:
        """Execute JavaScript in the page context."""
        await self._ensure_initialized()
        return await self._page.evaluate(script)

    async def handle_dialog(self, action: str = "accept", prompt_text: str = None):
        """
        Set up a handler for dialogs (alerts, confirms, prompts).

        Args:
            action: "accept" or "dismiss"
            prompt_text: Text to enter for prompt dialogs
        """
        async def dialog_handler(dialog):
            logger.observation(f"Dialog appeared: {dialog.type} - {dialog.message}")
            if action == "accept":
                if prompt_text and dialog.type == "prompt":
                    await dialog.accept(prompt_text)
                else:
                    await dialog.accept()
            else:
                await dialog.dismiss()

        self._page.on("dialog", dialog_handler)

    @property
    def page(self) -> Optional[Page]:
        """Get the current page object for advanced operations."""
        return self._page

    @property
    def current_url(self) -> str:
        """Get the current page URL."""
        return self._page.url if self._page else ""

    @property
    def is_ready(self) -> bool:
        """Check if browser is ready for operations."""
        return self._is_initialized and self._page is not None


# Convenience function for quick browser setup
async def create_browser(headless: bool = False) -> BrowserController:
    """Create and initialize a browser controller."""
    controller = BrowserController(headless=headless)
    await controller.initialize()
    return controller
