"""
Executor Node - Executes tool calls via Playwright.
Bridges the Oracle's decisions with browser actions.
"""

from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.state import AgentState, ToolCall, ToolName
from src.browser.controller import BrowserController, ActionResult
from src.vision.dom_processor import InteractiveElement
from src.utils.logger import agent_logger as logger


class Executor:
    """
    Executes tool calls from the Oracle using the Playwright browser controller.
    Handles parameter resolution and error reporting.
    """

    def __init__(self, browser: BrowserController, elements: list[InteractiveElement] = None):
        self.browser = browser
        self.elements = elements or []

    def update_elements(self, elements: list[InteractiveElement]):
        """Update the current page elements for selector resolution."""
        self.elements = elements

    def _resolve_selector(self, params: dict) -> Optional[str]:
        """
        Resolve a selector from various parameter formats.

        Supports:
        - element_id: Looks up in current elements list
        - selector: Direct CSS selector
        - text: Text-based selection
        """
        # Direct selector
        if "selector" in params:
            return params["selector"]

        # Element ID lookup
        if "element_id" in params:
            elem_id = params["element_id"]
            for elem in self.elements:
                if elem.element_id == elem_id:
                    logger.debug(f"Resolved element {elem_id} to selector: {elem.selector}")
                    return elem.selector
            logger.warning(f"Element ID {elem_id} not found in current elements")
            return None

        # Text-based selection
        if "text" in params:
            text = params["text"]
            # First try to find in elements
            for elem in self.elements:
                if text.lower() in elem.text.lower():
                    return elem.selector
            # Fall back to text selector
            return f'text="{text}"'

        return None

    async def execute(self, tool_call: ToolCall, state: AgentState) -> ActionResult:
        """
        Execute a tool call and return the result.

        Args:
            tool_call: The tool call from the Oracle
            state: Current agent state

        Returns:
            ActionResult with success/failure and details
        """
        logger.info(f"Executing: {tool_call.tool.value}")
        params = tool_call.parameters

        try:
            if tool_call.tool == ToolName.CLICK:
                return await self._execute_click(params)

            elif tool_call.tool == ToolName.TYPE:
                return await self._execute_type(params)

            elif tool_call.tool == ToolName.SCROLL:
                return await self._execute_scroll(params)

            elif tool_call.tool == ToolName.WAIT:
                return await self._execute_wait(params)

            elif tool_call.tool == ToolName.NAVIGATE:
                return await self._execute_navigate(params)

            elif tool_call.tool == ToolName.GO_BACK:
                return await self.browser.go_back()

            elif tool_call.tool == ToolName.PRESS_KEY:
                return await self._execute_press_key(params)

            elif tool_call.tool == ToolName.SELECT_OPTION:
                return await self._execute_select(params)

            elif tool_call.tool == ToolName.FINISH:
                return self._execute_finish(params)

            else:
                return ActionResult(
                    success=False,
                    action_type=tool_call.tool,
                    message=f"Unknown tool: {tool_call.tool}",
                    error="UnknownToolError"
                )

        except Exception as e:
            import traceback
            error_tb = traceback.format_exc()
            logger.error(f"Execution error: {e}", exception=e)
            return ActionResult(
                success=False,
                action_type=tool_call.tool,
                message=f"Execution failed: {str(e)}",
                error=str(e),
                traceback=error_tb
            )

    async def _execute_click(self, params: dict) -> ActionResult:
        """Execute a click action."""
        selector = self._resolve_selector(params)

        if not selector:
            return ActionResult(
                success=False,
                action_type=ToolName.CLICK,
                message="Could not resolve selector for click",
                error="SelectorResolutionError: No valid selector found"
            )

        # Try clicking
        result = await self.browser.click(selector, force=params.get("force", False))

        if not result.success and "text=" in selector:
            # Fallback: try click_by_text
            text = selector.replace('text="', '').rstrip('"')
            logger.debug(f"Retrying with click_by_text: {text}")
            result = await self.browser.click_by_text(text)

        return result

    async def _execute_type(self, params: dict) -> ActionResult:
        """Execute a type action."""
        selector = self._resolve_selector(params)
        text = params.get("text", "")
        clear_first = params.get("clear_first", True)

        if not selector:
            return ActionResult(
                success=False,
                action_type=ToolName.TYPE,
                message="Could not resolve selector for typing",
                error="SelectorResolutionError"
            )

        if not text:
            return ActionResult(
                success=False,
                action_type=ToolName.TYPE,
                message="No text provided to type",
                error="MissingParameterError: text is required"
            )

        # Use fill for faster input, type for more realistic simulation
        if params.get("simulate_typing", False):
            return await self.browser.type_text(selector, text, clear_first=clear_first)
        else:
            return await self.browser.fill(selector, text)

    async def _execute_scroll(self, params: dict) -> ActionResult:
        """Execute a scroll action."""
        # Scroll to element
        if "element_id" in params or "selector" in params:
            selector = self._resolve_selector(params)
            if selector:
                return await self.browser.scroll(selector=selector)

        # Scroll by direction
        direction = params.get("direction", "down")
        amount = params.get("amount", 500)

        if direction == "bottom":
            return await self.browser.scroll_to_bottom()
        elif direction == "top":
            return await self.browser.scroll_to_top()
        else:
            return await self.browser.scroll(direction=direction, amount=amount)

    async def _execute_wait(self, params: dict) -> ActionResult:
        """Execute a wait action."""
        # Wait for selector
        if "selector" in params or "element_id" in params:
            selector = self._resolve_selector(params)
            if selector:
                return await self.browser.wait_for_selector(selector)

        # Wait for duration
        ms = params.get("milliseconds", 1000)
        return await self.browser.wait(ms)

    async def _execute_navigate(self, params: dict) -> ActionResult:
        """Execute a navigation action."""
        url = params.get("url", "")

        if not url:
            return ActionResult(
                success=False,
                action_type=ToolName.NAVIGATE,
                message="No URL provided for navigation",
                error="MissingParameterError: url is required"
            )

        # Ensure URL has protocol
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        return await self.browser.navigate(url)

    async def _execute_press_key(self, params: dict) -> ActionResult:
        """Execute a key press action."""
        key = params.get("key", "Enter")
        return await self.browser.press_key(key)

    async def _execute_select(self, params: dict) -> ActionResult:
        """Execute a select option action."""
        selector = self._resolve_selector(params)

        if not selector:
            return ActionResult(
                success=False,
                action_type=ToolName.SELECT_OPTION,
                message="Could not resolve selector for select",
                error="SelectorResolutionError"
            )

        value = params.get("value")
        label = params.get("label")
        index = params.get("index")

        return await self.browser.select_option(
            selector,
            value=value,
            label=label,
            index=index
        )

    def _execute_finish(self, params: dict) -> ActionResult:
        """Execute finish action (marks task complete)."""
        success = params.get("success", True)
        reason = params.get("reason", "Task completed")

        logger.success(f"Agent finished: {reason}") if success else logger.warning(f"Agent finished (unsuccessful): {reason}")

        return ActionResult(
            success=True,  # The action itself succeeded
            action_type=ToolName.FINISH,
            message=reason,
            error=None if success else "Task could not be completed"
        )


class ExecutionError(Exception):
    """Custom exception for execution errors with context."""

    def __init__(self, message: str, tool: ToolName, selector: str = None, original_error: Exception = None):
        super().__init__(message)
        self.tool = tool
        self.selector = selector
        self.original_error = original_error

    def to_feedback(self) -> str:
        """Format error for LLM feedback."""
        feedback = f"Action '{self.tool.value}' failed: {str(self)}"
        if self.selector:
            feedback += f"\nSelector used: {self.selector}"
        if self.original_error:
            feedback += f"\nOriginal error: {type(self.original_error).__name__}: {self.original_error}"
        return feedback
