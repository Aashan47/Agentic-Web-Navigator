"""
Self-Correction and Error Recovery System.
Handles popups, dynamic loading, and failed actions with intelligent recovery.
"""

import re
import asyncio
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.state import AgentState, ToolCall, ToolName
from src.browser.controller import BrowserController
from src.vision.observer import PageObserver
from src.utils.logger import agent_logger as logger


class ErrorType(str, Enum):
    """Classification of error types for targeted recovery."""
    TIMEOUT = "timeout"
    ELEMENT_NOT_FOUND = "element_not_found"
    ELEMENT_INTERCEPTED = "element_intercepted"  # Something blocking the element
    ELEMENT_NOT_VISIBLE = "element_not_visible"
    ELEMENT_NOT_ENABLED = "element_not_enabled"
    NAVIGATION_ERROR = "navigation_error"
    DYNAMIC_CONTENT = "dynamic_content"  # Content still loading
    POPUP_BLOCKING = "popup_blocking"
    UNKNOWN = "unknown"


@dataclass
class RecoveryAction:
    """A recovery action to attempt."""
    action: ToolCall
    reason: str
    confidence: float


class ErrorClassifier:
    """Classifies errors to determine appropriate recovery strategy."""

    ERROR_PATTERNS = {
        ErrorType.TIMEOUT: [
            r"timeout",
            r"waiting for.*failed",
            r"exceeded.*timeout",
        ],
        ErrorType.ELEMENT_NOT_FOUND: [
            r"element.*not found",
            r"no.*element.*matches",
            r"selector.*failed",
            r"cannot find",
        ],
        ErrorType.ELEMENT_INTERCEPTED: [
            r"intercept",
            r"other element.*receive.*click",
            r"obscured",
            r"blocked",
            r"overlay",
        ],
        ErrorType.ELEMENT_NOT_VISIBLE: [
            r"not visible",
            r"hidden",
            r"display.*none",
            r"visibility.*hidden",
        ],
        ErrorType.ELEMENT_NOT_ENABLED: [
            r"disabled",
            r"not enabled",
            r"readonly",
        ],
        ErrorType.NAVIGATION_ERROR: [
            r"navigation.*failed",
            r"net::err",
            r"page.*crashed",
            r"connection.*refused",
        ],
        ErrorType.DYNAMIC_CONTENT: [
            r"stale.*element",
            r"element.*detached",
            r"dom.*changed",
        ],
    }

    @classmethod
    def classify(cls, error_message: str) -> ErrorType:
        """Classify an error message into an ErrorType."""
        if not error_message:
            return ErrorType.UNKNOWN

        error_lower = error_message.lower()

        for error_type, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_lower, re.IGNORECASE):
                    return error_type

        return ErrorType.UNKNOWN


class PopupHandler:
    """Specialized handler for detecting and dismissing popups."""

    # Common popup dismiss button patterns
    DISMISS_PATTERNS = [
        # Cookie consent
        ("cookie", ["accept", "accept all", "agree", "ok", "got it", "i understand", "allow"]),
        # Newsletter
        ("newsletter", ["no thanks", "close", "x", "dismiss", "not now", "skip"]),
        # Notifications
        ("notification", ["block", "no thanks", "not now", "deny"]),
        # Login prompts
        ("login", ["close", "x", "not now", "skip", "continue as guest"]),
        # Age verification
        ("age", ["yes", "i am", "confirm", "enter"]),
        # Generic overlays
        ("overlay", ["close", "x", "dismiss", "continue"]),
    ]

    # CSS selectors commonly used for popup close buttons
    CLOSE_BUTTON_SELECTORS = [
        '[aria-label*="close" i]',
        '[aria-label*="dismiss" i]',
        'button[class*="close"]',
        'button[class*="dismiss"]',
        '[data-dismiss]',
        '[data-close]',
        '.modal-close',
        '.popup-close',
        '.overlay-close',
        'button:has-text("×")',
        'button:has-text("X")',
        '[role="dialog"] button',
    ]

    # Selectors for cookie consent specifically
    COOKIE_SELECTORS = [
        '#onetrust-accept-btn-handler',
        '[id*="cookie"] button[id*="accept"]',
        '[class*="cookie"] button[class*="accept"]',
        '[data-testid*="cookie"] button',
        '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
        '.cc-accept',
        '#accept-cookies',
        'button[aria-label*="cookie" i]',
    ]

    def __init__(self, browser: BrowserController, observer: PageObserver):
        self.browser = browser
        self.observer = observer

    async def detect_and_dismiss(self, state: AgentState) -> Optional[RecoveryAction]:
        """
        Detect popups in the current page and suggest dismissal action.

        Returns RecoveryAction if popup found, None otherwise.
        """
        logger.info("Scanning for popups...")

        # First, try common cookie consent selectors
        for selector in self.COOKIE_SELECTORS:
            try:
                elem_info = await self.browser.get_element_info(selector)
                if elem_info and elem_info.get("is_visible"):
                    logger.observation(f"Found cookie consent: {selector}")
                    return RecoveryAction(
                        action=ToolCall(
                            tool=ToolName.CLICK,
                            parameters={"selector": selector},
                            reasoning="Dismissing cookie consent banner"
                        ),
                        reason="Cookie consent banner detected",
                        confidence=0.9
                    )
            except Exception:
                pass

        # Try generic close button selectors
        for selector in self.CLOSE_BUTTON_SELECTORS:
            try:
                elem_info = await self.browser.get_element_info(selector)
                if elem_info and elem_info.get("is_visible"):
                    logger.observation(f"Found close button: {selector}")
                    return RecoveryAction(
                        action=ToolCall(
                            tool=ToolName.CLICK,
                            parameters={"selector": selector},
                            reasoning="Clicking close button on overlay"
                        ),
                        reason="Overlay close button detected",
                        confidence=0.7
                    )
            except Exception:
                pass

        # Search in accessibility tree for dismiss buttons
        elements = state.interactive_elements or []
        for elem in elements:
            text_lower = elem.get("text", "").lower()
            # Look for common dismiss text
            dismiss_words = ["accept", "agree", "ok", "close", "dismiss", "got it", "x"]
            if any(word == text_lower or word in text_lower.split() for word in dismiss_words):
                # Check if it might be part of a popup
                attrs = elem.get("attributes", {})
                classes = attrs.get("class", "").lower()
                if any(hint in classes for hint in ["modal", "popup", "overlay", "banner", "consent", "cookie"]):
                    return RecoveryAction(
                        action=ToolCall(
                            tool=ToolName.CLICK,
                            parameters={"element_id": elem.get("id")},
                            reasoning=f"Clicking '{elem.get('text')}' to dismiss popup"
                        ),
                        reason=f"Found dismiss button: {elem.get('text')}",
                        confidence=0.75
                    )

        # Try pressing Escape as last resort
        return RecoveryAction(
            action=ToolCall(
                tool=ToolName.PRESS_KEY,
                parameters={"key": "Escape"},
                reasoning="Pressing Escape to dismiss potential overlay"
            ),
            reason="Attempting Escape key to dismiss overlay",
            confidence=0.4
        )

    async def wait_for_loading(self, timeout_ms: int = 5000) -> bool:
        """Wait for dynamic content to finish loading."""
        logger.info("Waiting for dynamic content...")

        # Check for common loading indicators
        loading_selectors = [
            '[class*="loading"]',
            '[class*="spinner"]',
            '[class*="skeleton"]',
            '[aria-busy="true"]',
            '.loader',
            '#loading',
        ]

        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) * 1000 < timeout_ms:
            loading_found = False

            for selector in loading_selectors:
                try:
                    elem_info = await self.browser.get_element_info(selector)
                    if elem_info and elem_info.get("is_visible"):
                        loading_found = True
                        break
                except Exception:
                    pass

            if not loading_found:
                logger.success("Page appears to have finished loading")
                return True

            await asyncio.sleep(0.5)

        logger.warning("Timeout waiting for loading to complete")
        return False


class RecoveryManager:
    """
    Manages error recovery strategies and self-correction.
    """

    def __init__(
        self,
        browser: BrowserController,
        observer: PageObserver
    ):
        self.browser = browser
        self.observer = observer
        self.popup_handler = PopupHandler(browser, observer)
        self.error_classifier = ErrorClassifier()

    async def recover(
        self,
        state: AgentState,
        error: str,
        failed_action: dict
    ) -> Tuple[bool, Optional[RecoveryAction]]:
        """
        Attempt to recover from an error.

        Args:
            state: Current agent state
            error: Error message from failed action
            failed_action: The action that failed

        Returns:
            Tuple of (should_retry_same_action, recovery_action)
        """
        error_type = self.error_classifier.classify(error)
        logger.info(f"Error classified as: {error_type.value}")

        if error_type == ErrorType.ELEMENT_INTERCEPTED:
            return await self._handle_intercepted(state)

        elif error_type == ErrorType.ELEMENT_NOT_FOUND:
            return await self._handle_not_found(state, failed_action)

        elif error_type == ErrorType.TIMEOUT:
            return await self._handle_timeout(state)

        elif error_type == ErrorType.DYNAMIC_CONTENT:
            return await self._handle_dynamic_content(state)

        elif error_type == ErrorType.ELEMENT_NOT_VISIBLE:
            return await self._handle_not_visible(state, failed_action)

        elif error_type == ErrorType.NAVIGATION_ERROR:
            return await self._handle_navigation_error(state)

        else:
            # Unknown error - try popup handler as default
            recovery = await self.popup_handler.detect_and_dismiss(state)
            return False, recovery

    async def _handle_intercepted(self, state: AgentState) -> Tuple[bool, Optional[RecoveryAction]]:
        """Handle element click intercepted errors - usually popups."""
        logger.thought("Element was intercepted - likely a popup blocking it")

        recovery = await self.popup_handler.detect_and_dismiss(state)
        if recovery:
            return True, recovery  # Retry same action after dismissing popup

        return False, None

    async def _handle_not_found(
        self,
        state: AgentState,
        failed_action: dict
    ) -> Tuple[bool, Optional[RecoveryAction]]:
        """Handle element not found errors."""
        logger.thought("Element not found - page may have changed")

        # First, wait for possible dynamic loading
        await self.popup_handler.wait_for_loading(3000)

        # Try scrolling to make element visible
        scroll_action = RecoveryAction(
            action=ToolCall(
                tool=ToolName.SCROLL,
                parameters={"direction": "down", "amount": 300},
                reasoning="Scrolling to find element"
            ),
            reason="Element may be below viewport",
            confidence=0.5
        )

        return False, scroll_action

    async def _handle_timeout(self, state: AgentState) -> Tuple[bool, Optional[RecoveryAction]]:
        """Handle timeout errors."""
        logger.thought("Action timed out - page may be slow or element missing")

        # Wait for loading
        loaded = await self.popup_handler.wait_for_loading(5000)

        if loaded:
            return True, None  # Retry same action

        # Try refreshing the page
        return False, RecoveryAction(
            action=ToolCall(
                tool=ToolName.WAIT,
                parameters={"milliseconds": 2000},
                reasoning="Waiting for page to stabilize"
            ),
            reason="Waiting for slow page",
            confidence=0.6
        )

    async def _handle_dynamic_content(self, state: AgentState) -> Tuple[bool, Optional[RecoveryAction]]:
        """Handle stale element / dynamic content errors."""
        logger.thought("DOM changed - waiting for stability")

        await asyncio.sleep(1)
        return True, None  # Just retry

    async def _handle_not_visible(
        self,
        state: AgentState,
        failed_action: dict
    ) -> Tuple[bool, Optional[RecoveryAction]]:
        """Handle element not visible errors."""
        logger.thought("Element not visible - may need to scroll")

        # Try scrolling
        return False, RecoveryAction(
            action=ToolCall(
                tool=ToolName.SCROLL,
                parameters={"direction": "down", "amount": 400},
                reasoning="Scrolling to make element visible"
            ),
            reason="Element below viewport",
            confidence=0.6
        )

    async def _handle_navigation_error(self, state: AgentState) -> Tuple[bool, Optional[RecoveryAction]]:
        """Handle navigation errors."""
        logger.thought("Navigation failed - may need to retry or go back")

        return False, RecoveryAction(
            action=ToolCall(
                tool=ToolName.GO_BACK,
                parameters={},
                reasoning="Going back after navigation failure"
            ),
            reason="Navigation failed, returning to previous page",
            confidence=0.7
        )


class ReflexionEngine:
    """
    Implements reflexion - agent critiquing its own failures.
    """

    def __init__(self, oracle):
        """
        Args:
            oracle: The Oracle instance for LLM-based analysis
        """
        self.oracle = oracle

    async def analyze_failure_pattern(
        self,
        state: AgentState,
        failures: list[dict]
    ) -> dict:
        """
        Analyze a pattern of failures and suggest a different approach.

        Args:
            state: Current agent state
            failures: List of failed actions with errors

        Returns:
            Analysis result with suggested alternative
        """
        if len(failures) < 2:
            return {"analysis": "Not enough failures to analyze", "alternative": None}

        # Check for repeated same action
        unique_actions = set()
        for f in failures:
            action_sig = f"{f.get('tool', '')}:{f.get('parameters', {}).get('selector', '')}"
            unique_actions.add(action_sig)

        if len(unique_actions) == 1:
            # Same action failing repeatedly
            return await self._suggest_alternative_for_repeated_failure(state, failures[0])

        else:
            # Different actions failing
            return await self._suggest_alternative_approach(state, failures)

    async def _suggest_alternative_for_repeated_failure(
        self,
        state: AgentState,
        failed_action: dict
    ) -> dict:
        """Suggest alternative when same action fails repeatedly."""
        tool = failed_action.get("tool", "")
        params = failed_action.get("parameters", {})

        # If it's a click that keeps failing, suggest alternatives
        if tool == ToolName.CLICK.value:
            return {
                "analysis": "Click action keeps failing on the same element",
                "root_cause": "Element may be unclickable or selector is wrong",
                "alternative": {
                    "tool": ToolName.PRESS_KEY.value,
                    "parameters": {"key": "Tab"},
                    "reasoning": "Try keyboard navigation instead"
                }
            }

        elif tool == ToolName.TYPE.value:
            return {
                "analysis": "Typing keeps failing on input field",
                "root_cause": "Input may not be focused or is disabled",
                "alternative": {
                    "tool": ToolName.CLICK.value,
                    "parameters": params,
                    "reasoning": "Click to focus input first"
                }
            }

        return {
            "analysis": f"Action '{tool}' keeps failing",
            "root_cause": "Unknown",
            "alternative": None
        }

    async def _suggest_alternative_approach(
        self,
        state: AgentState,
        failures: list[dict]
    ) -> dict:
        """Suggest a completely different approach when multiple actions fail."""
        # Use the Oracle for more sophisticated analysis
        if self.oracle:
            return await self.oracle.reflexion(
                failed_action=failures[-1],
                failure_count=len(failures),
                errors=[f.get("error", "") for f in failures],
                state=state
            )

        return {
            "analysis": "Multiple different actions have failed",
            "root_cause": "Page may have changed or goal may not be achievable",
            "alternative": {
                "tool": ToolName.SCROLL.value,
                "parameters": {"direction": "top"},
                "reasoning": "Return to top and re-analyze page"
            }
        }
