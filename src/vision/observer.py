"""
Vision Observer for the Agentic Web Navigator.
Combines screenshot capture with DOM processing for unified page observation.
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.browser.controller import BrowserController, ActionResult
from src.vision.dom_processor import DOMProcessor, DOMSnapshot, InteractiveElement
from src.utils.logger import agent_logger as logger
from src.utils.state import global_state


@dataclass
class PageObservation:
    """
    Complete observation of the current page state.
    Used as input for the LLM to make decisions.
    """
    url: str
    title: str
    screenshot_base64: Optional[str]
    dom_snapshot: DOMSnapshot
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_interactive_elements(self) -> list[InteractiveElement]:
        """Get all interactive elements from the DOM."""
        return self.dom_snapshot.elements

    def get_accessibility_tree(self) -> str:
        """Get the simplified accessibility tree."""
        return self.dom_snapshot.accessibility_tree

    def get_element_by_id(self, element_id: int) -> Optional[InteractiveElement]:
        """Find an element by its assigned ID."""
        for elem in self.dom_snapshot.elements:
            if elem.element_id == element_id:
                return elem
        return None

    def to_llm_context(self, include_screenshot: bool = True) -> dict:
        """
        Convert observation to a format suitable for LLM consumption.
        Optimized for minimal token usage while preserving essential info.
        """
        context = {
            "url": self.url,
            "title": self.title,
            "accessibility_tree": self.dom_snapshot.accessibility_tree,
            "interactive_elements_count": len(self.dom_snapshot.elements),
            "dom_hash": self.dom_snapshot.dom_hash,
        }

        if include_screenshot and self.screenshot_base64:
            context["screenshot_base64"] = self.screenshot_base64

        if self.error:
            context["error"] = self.error

        return context

    def get_compact_elements_list(self) -> str:
        """Get a compact string list of interactive elements."""
        lines = []
        for elem in self.dom_snapshot.elements:
            lines.append(elem.to_compact_string())
        return "\n".join(lines)


class PageObserver:
    """
    Observes the current page state by capturing screenshots and processing DOM.
    Provides the "eyes" for the navigation agent.
    """

    def __init__(
        self,
        browser: BrowserController,
        dom_processor: Optional[DOMProcessor] = None,
        capture_screenshots: bool = True,
        full_page_screenshots: bool = False
    ):
        self.browser = browser
        self.dom_processor = dom_processor or DOMProcessor()
        self.capture_screenshots = capture_screenshots
        self.full_page_screenshots = full_page_screenshots

        # Cache for last observation
        self._last_observation: Optional[PageObservation] = None

    async def observe(self, force_screenshot: bool = False) -> PageObservation:
        """
        Capture the current page state.

        Args:
            force_screenshot: Always capture screenshot even if disabled

        Returns:
            PageObservation with screenshot and DOM snapshot
        """
        logger.info("Observing page state...")

        error_msg = None
        screenshot_b64 = None

        # Get page content
        try:
            html, url, title = await self.browser.get_page_content()
        except Exception as e:
            logger.error(f"Failed to get page content: {e}", exception=e)
            error_msg = f"Failed to get page content: {str(e)}"
            html, url, title = "", "", ""

        # Process DOM
        try:
            dom_snapshot = self.dom_processor.process(html, url, title)
            logger.dom_summary(
                dom_snapshot.raw_html_length,
                dom_snapshot.get_interactive_count(),
                dom_snapshot.title
            )
        except Exception as e:
            logger.error(f"Failed to process DOM: {e}", exception=e)
            error_msg = (error_msg or "") + f" DOM processing error: {str(e)}"
            # Create empty snapshot on error
            dom_snapshot = DOMSnapshot(
                url=url,
                title=title,
                elements=[],
                accessibility_tree="[DOM PROCESSING ERROR]",
                dom_hash="error",
                raw_html_length=len(html),
                processed_length=0
            )

        # Capture screenshot
        if self.capture_screenshots or force_screenshot:
            try:
                screenshot_result = await self.browser.take_screenshot(
                    full_page=self.full_page_screenshots
                )
                if screenshot_result.success:
                    screenshot_b64 = screenshot_result.screenshot_base64
                else:
                    logger.warning(f"Screenshot capture failed: {screenshot_result.error}")
            except Exception as e:
                logger.error(f"Failed to capture screenshot: {e}", exception=e)

        # Record page state in global state
        global_state.record_page_state(
            url=url,
            title=title,
            dom_hash=dom_snapshot.dom_hash,
            interactive_elements_count=dom_snapshot.get_interactive_count()
        )

        observation = PageObservation(
            url=url,
            title=title,
            screenshot_base64=screenshot_b64,
            dom_snapshot=dom_snapshot,
            error=error_msg
        )

        self._last_observation = observation
        return observation

    async def observe_element(self, selector: str) -> Optional[dict]:
        """Get detailed information about a specific element."""
        return await self.browser.get_element_info(selector)

    async def wait_for_change(
        self,
        timeout_ms: int = 5000,
        poll_interval_ms: int = 500
    ) -> bool:
        """
        Wait for the page to change (DOM hash changes).

        Returns:
            True if page changed, False if timeout
        """
        if not self._last_observation:
            return True

        original_hash = self._last_observation.dom_snapshot.dom_hash
        elapsed = 0

        while elapsed < timeout_ms:
            await self.browser.wait(poll_interval_ms)
            elapsed += poll_interval_ms

            # Get current DOM hash
            try:
                html, url, _ = await self.browser.get_page_content()
                current_snapshot = self.dom_processor.process(html, url)

                if current_snapshot.dom_hash != original_hash:
                    logger.observation("Page content changed")
                    return True

            except Exception:
                pass

        logger.observation(f"No change detected after {timeout_ms}ms")
        return False

    async def find_element_for_action(
        self,
        element_id: int = None,
        text: str = None,
        tag: str = None
    ) -> Optional[InteractiveElement]:
        """
        Find an interactive element by various criteria.

        Args:
            element_id: The assigned element ID from observation
            text: Text content to search for
            tag: HTML tag to filter by

        Returns:
            InteractiveElement if found
        """
        if not self._last_observation:
            await self.observe()

        elements = self._last_observation.dom_snapshot.elements

        # Search by ID
        if element_id is not None:
            for elem in elements:
                if elem.element_id == element_id:
                    return elem

        # Search by text
        if text:
            text_lower = text.lower()
            for elem in elements:
                if tag and elem.tag != tag:
                    continue
                if text_lower in elem.text.lower():
                    return elem
                # Also check aria-label
                if text_lower in elem.attributes.get("aria-label", "").lower():
                    return elem

        return None

    @property
    def last_observation(self) -> Optional[PageObservation]:
        """Get the most recent observation."""
        return self._last_observation

    def get_elements_by_type(self, element_type: str) -> list[InteractiveElement]:
        """Get all elements of a specific type from last observation."""
        if not self._last_observation:
            return []

        return [
            elem for elem in self._last_observation.dom_snapshot.elements
            if elem.element_type == element_type
        ]

    def get_links(self) -> list[InteractiveElement]:
        """Get all links from last observation."""
        return self.get_elements_by_type("link")

    def get_buttons(self) -> list[InteractiveElement]:
        """Get all buttons from last observation."""
        if not self._last_observation:
            return []

        return [
            elem for elem in self._last_observation.dom_snapshot.elements
            if elem.tag == "button" or elem.attributes.get("role") == "button"
        ]

    def get_inputs(self) -> list[InteractiveElement]:
        """Get all input fields from last observation."""
        if not self._last_observation:
            return []

        return [
            elem for elem in self._last_observation.dom_snapshot.elements
            if elem.tag in ("input", "textarea")
        ]
