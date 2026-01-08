"""
LangGraph-based Agent for Web Navigation.
Implements a ReAct-style reasoning loop with self-correction.
"""

import asyncio
from typing import Literal, Optional, Annotated
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.state import AgentState, ToolCall, ToolName, ActionMemory, ReflexionEntry, create_initial_state
from src.agent.oracle import Oracle
from src.agent.executor import Executor
from src.browser.controller import BrowserController
from src.vision.observer import PageObserver
from src.vision.dom_processor import DOMProcessor
from src.utils.logger import agent_logger as logger
from src.utils.state import global_state, ActionType, ActionStatus
from config.settings import settings


class WebNavigatorGraph:
    """
    LangGraph-based web navigation agent.
    Implements: Oracle -> Act -> Observe -> Check Success loop
    """

    def __init__(
        self,
        browser: BrowserController,
        observer: PageObserver,
        oracle: Oracle = None,
        max_steps: int = None
    ):
        self.browser = browser
        self.observer = observer
        self.oracle = oracle or Oracle()
        self.executor = Executor(browser)
        self.max_steps = max_steps or settings.agent.max_steps

        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""

        # Create graph with AgentState schema
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("observe", self._observe_node)
        graph.add_node("oracle", self._oracle_node)
        graph.add_node("act", self._act_node)
        graph.add_node("check_success", self._check_success_node)
        graph.add_node("self_correct", self._self_correct_node)
        graph.add_node("handle_popup", self._handle_popup_node)

        # Set entry point
        graph.set_entry_point("observe")

        # Add edges
        graph.add_edge("observe", "oracle")
        graph.add_edge("act", "check_success")
        graph.add_edge("self_correct", "observe")
        graph.add_edge("handle_popup", "observe")

        # Conditional edges from oracle
        graph.add_conditional_edges(
            "oracle",
            self._route_after_oracle,
            {
                "act": "act",
                "end": END,
            }
        )

        # Conditional edges from check_success
        graph.add_conditional_edges(
            "check_success",
            self._route_after_check,
            {
                "observe": "observe",
                "self_correct": "self_correct",
                "handle_popup": "handle_popup",
                "end": END,
            }
        )

        return graph

    async def _observe_node(self, state: AgentState) -> AgentState:
        """
        Observe Node: Capture current page state.
        Takes screenshot and processes DOM.
        """
        logger.info(f"[Observe] Step {state.current_step}")

        try:
            observation = await self.observer.observe()

            # Update state with observation
            state.current_url = observation.url
            state.page_title = observation.title
            state.screenshot_base64 = observation.screenshot_base64
            state.accessibility_tree = observation.dom_snapshot.accessibility_tree
            state.interactive_elements = [e.to_dict() for e in observation.dom_snapshot.elements]
            state.dom_hash = observation.dom_snapshot.dom_hash

            # Update executor with current elements
            self.executor.update_elements(observation.dom_snapshot.elements)

            logger.observation(
                f"Page: {state.page_title[:50]}... | Elements: {len(state.interactive_elements)}"
            )

        except Exception as e:
            logger.error(f"Observation failed: {e}", exception=e)
            state.action_error = f"Observation error: {str(e)}"

        return state

    async def _oracle_node(self, state: AgentState) -> AgentState:
        """
        Oracle Node: Analyze state and decide on action.
        Uses Gemini to process screenshot + DOM.
        """
        logger.info(f"[Oracle] Analyzing...")

        try:
            thought, tool_call = await self.oracle.analyze(state)

            state.thought = thought

            if tool_call:
                state.tool_call = tool_call.to_dict()
                logger.action(tool_call.tool.value, tool_call.parameters)
            else:
                # No valid action, mark for completion
                state.tool_call = None
                state.is_complete = True
                state.completion_reason = "Oracle could not determine action"

        except Exception as e:
            logger.error(f"Oracle analysis failed: {e}", exception=e)
            state.action_error = f"Oracle error: {str(e)}"
            state.tool_call = None

        return state

    async def _act_node(self, state: AgentState) -> AgentState:
        """
        Act Node: Execute the tool call via Playwright.
        """
        logger.info(f"[Act] Executing action...")

        if not state.tool_call:
            state.action_success = False
            state.action_error = "No tool call to execute"
            return state

        try:
            # Reconstruct ToolCall from dict
            tool_call = ToolCall(
                tool=ToolName(state.tool_call["tool"]),
                parameters=state.tool_call.get("parameters", {}),
                confidence=state.tool_call.get("confidence", 0.8),
                reasoning=state.tool_call.get("reasoning", "")
            )

            # Execute action
            result = await self.executor.execute(tool_call, state)

            # Update state
            state.action_success = result.success
            state.action_error = result.error if not result.success else None
            state.action_traceback = result.traceback if not result.success else None
            state.total_actions += 1

            if result.success:
                state.successful_actions += 1
                state.reset_failure_count()
                logger.success(f"Action succeeded: {result.message}")
            else:
                selector = state.tool_call.get("parameters", {}).get("selector", "")
                state.record_failure(selector)
                logger.error(f"Action failed: {result.error}")

            # Add to memory
            memory_entry = ActionMemory(
                step=state.current_step,
                thought=state.thought,
                tool_call=state.tool_call,
                success=result.success,
                error=result.error,
                url=state.current_url
            )
            state.add_to_memory(memory_entry)

            # Record in global state
            global_state.record_action(
                action_type=ActionType(tool_call.tool.value.upper()) if tool_call.tool.value.upper() in [a.value for a in ActionType] else ActionType.CLICK,
                parameters=tool_call.parameters,
                status=ActionStatus.SUCCESS if result.success else ActionStatus.FAILED,
                error_message=result.error,
                url_before=result.url_before,
                url_after=result.url_after,
                thought=state.thought
            )

            # Check if this was a finish action
            if tool_call.tool == ToolName.FINISH:
                state.is_complete = True
                state.success = state.tool_call.get("parameters", {}).get("success", True)
                state.completion_reason = result.message

        except Exception as e:
            logger.error(f"Action execution error: {e}", exception=e)
            state.action_success = False
            state.action_error = str(e)
            state.record_failure()

        state.current_step += 1
        return state

    async def _check_success_node(self, state: AgentState) -> AgentState:
        """
        Check Success Node: Determine if goal is met or if we should continue.
        """
        logger.info(f"[Check] Step {state.current_step}/{state.max_steps}")

        # Check completion conditions
        if state.is_complete:
            logger.success(f"Task complete: {state.completion_reason}")
            return state

        if state.current_step >= state.max_steps:
            state.is_complete = True
            state.success = False
            state.completion_reason = f"Max steps ({state.max_steps}) reached"
            logger.warning(state.completion_reason)
            return state

        # Check for loop detection
        is_loop, loop_reason = global_state.detect_loop()
        if is_loop:
            state.is_in_recovery_mode = True
            logger.warning(f"Loop detected: {loop_reason}")

        return state

    async def _self_correct_node(self, state: AgentState) -> AgentState:
        """
        Self-Correction Node: Triggered when actions fail repeatedly.
        Uses reflexion to analyze failures and try alternatives.
        """
        logger.banner("Self-Correction Mode")
        state.self_corrections += 1

        # Check if we should trigger full reflexion
        if state.should_trigger_reflexion() and state.tool_call:
            logger.thought("Triggering reflexion analysis...")

            # Gather error history
            errors = [
                m.get("error") for m in state.action_memory
                if m.get("error") and not m.get("success")
            ][-3:]

            # Run reflexion
            reflexion_result = await self.oracle.reflexion(
                failed_action=state.tool_call,
                failure_count=state.consecutive_failures,
                errors=errors,
                state=state
            )

            # Record reflexion
            state.reflexion_history.append({
                "step": state.current_step,
                "analysis": reflexion_result.get("analysis"),
                "root_cause": reflexion_result.get("root_cause"),
                "timestamp": datetime.now().isoformat()
            })

            # If reflexion suggests skipping this element
            if reflexion_result.get("should_skip"):
                logger.warning("Reflexion suggests abandoning this approach")
                # Clear the problematic selector from failed list
                if state.tool_call:
                    selector = state.tool_call.get("parameters", {}).get("selector", "")
                    if selector in state.failed_selectors:
                        del state.failed_selectors[selector]

            # If there's an alternative approach, use it
            alt = reflexion_result.get("alternative_approach")
            if alt and alt.get("tool"):
                logger.observation(f"Trying alternative: {alt.get('tool')}")
                state.tool_call = alt
                state.is_in_recovery_mode = True
                state.recovery_attempts += 1

        # Reset some state for retry
        state.action_error = None
        state.action_traceback = None

        return state

    async def _handle_popup_node(self, state: AgentState) -> AgentState:
        """
        Handle Popup Node: Specifically detect and dismiss popups.
        """
        logger.info("[Popup Handler] Checking for blocking overlays...")

        try:
            # Use Oracle to detect popups
            popup_action = await self.oracle.analyze_for_popup(state)

            if popup_action:
                logger.observation(f"Found popup, attempting to dismiss with: {popup_action.tool.value}")

                result = await self.executor.execute(popup_action, state)

                if result.success:
                    state.popups_handled += 1
                    logger.success("Popup dismissed successfully")
                    # Wait a bit for popup to close
                    await asyncio.sleep(0.5)
                else:
                    logger.warning(f"Failed to dismiss popup: {result.error}")

        except Exception as e:
            logger.error(f"Popup handling error: {e}")

        # Reset recovery state
        state.is_in_recovery_mode = False
        state.recovery_attempts = 0

        return state

    def _route_after_oracle(self, state: AgentState) -> Literal["act", "end"]:
        """Route after Oracle analysis."""
        if state.is_complete:
            return "end"
        if state.tool_call:
            return "act"
        return "end"

    def _route_after_check(self, state: AgentState) -> Literal["observe", "self_correct", "handle_popup", "end"]:
        """Route after checking success."""
        if state.is_complete:
            return "end"

        # Check if we need self-correction
        if state.consecutive_failures >= 2:
            if state.recovery_attempts >= state.max_recovery_attempts:
                # Too many recovery attempts, check for popup
                return "handle_popup"
            return "self_correct"

        # Check if last action failed and might be popup-related
        if not state.action_success and state.action_error:
            error_lower = state.action_error.lower()
            if any(word in error_lower for word in ["intercept", "overlay", "blocked", "obscured"]):
                return "handle_popup"

        return "observe"

    async def run(self, goal: str, start_url: str = None) -> AgentState:
        """
        Run the agent to achieve a goal.

        Args:
            goal: Natural language goal description
            start_url: Optional starting URL

        Returns:
            Final AgentState with results
        """
        logger.banner(f"Starting Navigation: {goal}")

        # Initialize state
        state = create_initial_state(goal, self.max_steps)

        # Navigate to starting URL if provided
        if start_url:
            logger.info(f"Navigating to start URL: {start_url}")
            await self.browser.navigate(start_url)
            await asyncio.sleep(1)  # Wait for page load

        # Start global state session
        global_state.start_session(goal)

        # Run the graph
        try:
            final_state = await self.app.ainvoke(state)

            # Log final results
            logger.banner("Navigation Complete")
            logger.show_json({
                "success": final_state.success,
                "completion_reason": final_state.completion_reason,
                "total_steps": final_state.current_step,
                "total_actions": final_state.total_actions,
                "successful_actions": final_state.successful_actions,
                "self_corrections": final_state.self_corrections,
                "popups_handled": final_state.popups_handled,
            }, "Final Results")

            return final_state

        except Exception as e:
            logger.error(f"Graph execution error: {e}", exception=e)
            state.is_complete = True
            state.success = False
            state.completion_reason = f"Execution error: {str(e)}"
            return state


async def create_navigator(
    headless: bool = False,
    api_key: str = None
) -> WebNavigatorGraph:
    """
    Factory function to create a fully initialized navigator.

    Args:
        headless: Run browser in headless mode
        api_key: Google API key for Gemini

    Returns:
        Initialized WebNavigatorGraph
    """
    # Initialize browser
    browser = BrowserController(headless=headless)
    await browser.initialize()

    # Initialize observer
    dom_processor = DOMProcessor()
    observer = PageObserver(browser, dom_processor)

    # Initialize oracle
    oracle = Oracle(api_key=api_key)

    # Create navigator
    navigator = WebNavigatorGraph(
        browser=browser,
        observer=observer,
        oracle=oracle
    )

    return navigator
