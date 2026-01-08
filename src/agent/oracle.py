"""
Oracle Node - The "Brain" of the Web Navigator.
Uses Gemini 1.5 Pro to analyze page state and decide on actions.
"""

import json
import re
from typing import Optional
import google.generativeai as genai
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from src.agent.state import AgentState, ToolCall, ToolName
from src.utils.logger import agent_logger as logger


# System prompt for the Oracle
ORACLE_SYSTEM_PROMPT = """You are an expert web navigation AI agent. Your task is to help users accomplish goals by interacting with web pages.

## Your Capabilities
You can perform these actions:
- click_element: Click on buttons, links, or interactive elements
- type_text: Enter text into input fields
- scroll: Scroll the page up/down or to a specific element
- wait: Wait for elements to load
- navigate: Go to a specific URL
- go_back: Navigate back in browser history
- press_key: Press keyboard keys (Enter, Tab, Escape, etc.)
- select_option: Select from dropdown menus
- finish: Complete the task (use when goal is achieved OR impossible)

## Input Format
You receive:
1. The user's GOAL
2. Current page URL and title
3. An ACCESSIBILITY TREE listing interactive elements with IDs
4. Recent action MEMORY showing what you've tried
5. Any ERROR from the last action

## Output Format
You MUST respond with valid JSON containing:
```json
{
    "thought": "Your analysis of the current state and what to do next",
    "tool": "tool_name",
    "parameters": {
        "param1": "value1"
    },
    "confidence": 0.8,
    "reasoning": "Why this action will help achieve the goal"
}
```

## Tool Parameters
- click_element: {"element_id": int} OR {"selector": "css_selector"} OR {"text": "visible text"}
- type_text: {"element_id": int OR "selector": "...", "text": "text to type", "clear_first": true/false}
- scroll: {"direction": "up/down", "amount": 500} OR {"element_id": int}
- wait: {"milliseconds": 1000}
- navigate: {"url": "https://..."}
- go_back: {}
- press_key: {"key": "Enter/Tab/Escape/ArrowDown/etc."}
- select_option: {"element_id": int OR "selector": "...", "value": "..." OR "label": "..."}
- finish: {"success": true/false, "reason": "why task is complete or impossible"}

## Important Rules
1. ALWAYS analyze the accessibility tree to find the correct element
2. Use element IDs when available - they are the most reliable
3. If an element ID is not working, try using visible text or a CSS selector
4. If a button doesn't respond after 2 attempts, look for alternative approaches
5. Watch for pop-ups, cookie banners, or overlays that might block interaction
6. If you see a cookie consent banner, dismiss it first before proceeding
7. If the goal seems impossible or the website is inaccessible, use finish with success=false
8. Keep your thoughts concise but informative

## Common Patterns
- Cookie banners: Look for "Accept", "Accept All", "Agree", "OK", "Got it" buttons
- Login walls: May need to close or work around them
- Loading states: Use wait if content is still loading
- Search: Usually involves typing in input + clicking search button OR pressing Enter
"""

RECOVERY_PROMPT_ADDITION = """
## RECOVERY MODE
The previous action FAILED. Analyze the error and screenshot carefully:
- Is there a pop-up or overlay blocking the element?
- Did the page layout change?
- Is the element no longer visible?
- Should you try a different selector or approach?

Suggest a recovery action that addresses the specific failure."""

REFLEXION_PROMPT = """You are analyzing a repeated failure pattern. The same action has failed multiple times.

Previous failed action: {failed_action}
Failure count: {failure_count}
Error messages: {errors}

Analyze WHY this keeps failing and suggest a completely different approach to achieve the same sub-goal.
Consider:
1. Is this element actually interactive?
2. Is something blocking it (overlay, popup, loading spinner)?
3. Should we try a different element or navigation path entirely?
4. Is the goal achievable on this page?

Respond with JSON:
```json
{{
    "analysis": "Your analysis of why this keeps failing",
    "root_cause": "The likely root cause",
    "alternative_approach": {{
        "tool": "tool_name",
        "parameters": {{}},
        "reasoning": "Why this different approach should work"
    }},
    "should_skip": false
}}
```
Set should_skip=true if this sub-goal should be abandoned."""


class Oracle:
    """
    The Oracle analyzes page state and decides on actions using Gemini 1.5 Pro.
    Supports multimodal input (screenshot + text).
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        temperature: float = None,
        use_vision: bool = True
    ):
        self.api_key = api_key or settings.llm.google_api_key
        self.model_name = model_name or settings.llm.model_name
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        self.use_vision = use_vision

        # Configure Gemini
        if self.api_key:
            genai.configure(api_key=self.api_key)

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=settings.llm.max_output_tokens,
                top_p=settings.llm.top_p,
            )
        )

        logger.info(f"Oracle initialized with model: {self.model_name}")

    def _build_prompt(self, state: AgentState) -> str:
        """Build the prompt from current state."""
        prompt_parts = [
            f"## GOAL\n{state.goal}\n",
            f"## CURRENT PAGE\nURL: {state.current_url}\nTitle: {state.page_title}\n",
            f"## ACCESSIBILITY TREE\n{state.accessibility_tree[:6000]}\n",
            f"## MEMORY\n{state.get_memory_context()}\n",
        ]

        # Add error context if last action failed
        if state.action_error:
            prompt_parts.append(
                f"## LAST ACTION ERROR\n{state.action_error}\n"
                f"Traceback (partial): {state.action_traceback[:500] if state.action_traceback else 'N/A'}\n"
            )

        # Add recovery mode context
        if state.is_in_recovery_mode:
            prompt_parts.append(RECOVERY_PROMPT_ADDITION)

        prompt_parts.append("\n## YOUR RESPONSE (JSON only):")

        return "\n".join(prompt_parts)

    def _parse_response(self, response_text: str) -> Optional[ToolCall]:
        """Parse the LLM response into a ToolCall."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.error("No JSON found in response")
                    return None

            data = json.loads(json_str)

            # Extract thought for logging
            thought = data.get("thought", "")
            if thought:
                logger.thought(thought)

            # Parse tool call
            tool_name = data.get("tool", "").lower()

            # Map to ToolName enum
            tool_mapping = {
                "click_element": ToolName.CLICK,
                "click": ToolName.CLICK,
                "type_text": ToolName.TYPE,
                "type": ToolName.TYPE,
                "scroll": ToolName.SCROLL,
                "wait": ToolName.WAIT,
                "navigate": ToolName.NAVIGATE,
                "go_back": ToolName.GO_BACK,
                "press_key": ToolName.PRESS_KEY,
                "select_option": ToolName.SELECT_OPTION,
                "finish": ToolName.FINISH,
            }

            if tool_name not in tool_mapping:
                logger.error(f"Unknown tool: {tool_name}")
                return None

            return ToolCall(
                tool=tool_mapping[tool_name],
                parameters=data.get("parameters", {}),
                confidence=data.get("confidence", 0.8),
                reasoning=data.get("reasoning", "")
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None

    async def analyze(self, state: AgentState) -> tuple[str, Optional[ToolCall]]:
        """
        Analyze the current state and decide on an action.

        Args:
            state: Current agent state with page observation

        Returns:
            Tuple of (thought, tool_call)
        """
        logger.info(f"Oracle analyzing state (step {state.current_step})...")

        prompt = self._build_prompt(state)

        try:
            # Build content parts
            content_parts = [ORACLE_SYSTEM_PROMPT, prompt]

            # Add screenshot if available and vision is enabled
            if self.use_vision and state.screenshot_base64:
                import base64
                # Create image part for Gemini
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": state.screenshot_base64
                }
                content_parts.insert(1, image_part)
                logger.debug("Including screenshot in analysis")

            # Generate response
            response = await self.model.generate_content_async(
                content_parts,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=settings.llm.max_output_tokens,
                )
            )

            response_text = response.text
            logger.debug(f"Oracle response: {response_text[:500]}...")

            # Parse response
            tool_call = self._parse_response(response_text)

            if tool_call:
                thought = f"Decided to {tool_call.tool.value}: {tool_call.reasoning}"
                logger.action(tool_call.tool.value, tool_call.parameters)
                return thought, tool_call
            else:
                # Fallback: wait and retry observation
                logger.warning("Failed to parse Oracle response, defaulting to wait")
                return "Could not parse response, waiting...", ToolCall(
                    tool=ToolName.WAIT,
                    parameters={"milliseconds": 1000},
                    confidence=0.3,
                    reasoning="Fallback due to parse error"
                )

        except Exception as e:
            logger.error(f"Oracle analysis failed: {e}", exception=e)
            return f"Analysis error: {e}", None

    async def reflexion(
        self,
        failed_action: dict,
        failure_count: int,
        errors: list[str],
        state: AgentState
    ) -> dict:
        """
        Perform reflexion analysis on repeated failures.

        Returns analysis and alternative approach.
        """
        logger.banner("Reflexion Analysis")
        logger.thought(f"Analyzing repeated failure (count: {failure_count})")

        prompt = REFLEXION_PROMPT.format(
            failed_action=json.dumps(failed_action, indent=2),
            failure_count=failure_count,
            errors="\n".join(errors[-3:])  # Last 3 errors
        )

        try:
            content_parts = [prompt]

            # Include screenshot for visual analysis
            if self.use_vision and state.screenshot_base64:
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": state.screenshot_base64
                }
                content_parts.append(image_part)

            response = await self.model.generate_content_async(content_parts)
            response_text = response.text

            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group(0))
                logger.observation(f"Reflexion analysis: {result.get('analysis', 'N/A')}")
                logger.observation(f"Root cause: {result.get('root_cause', 'Unknown')}")
                return result

        except Exception as e:
            logger.error(f"Reflexion failed: {e}")

        return {
            "analysis": "Reflexion failed",
            "root_cause": "Unknown",
            "alternative_approach": None,
            "should_skip": False
        }

    async def analyze_for_popup(self, state: AgentState) -> Optional[ToolCall]:
        """
        Specifically analyze for popups/overlays that might be blocking.

        Returns a tool call to dismiss the popup if found.
        """
        popup_prompt = """Analyze this page for any popups, overlays, or banners that might be blocking interaction.

Look for:
- Cookie consent banners
- Newsletter signup popups
- Login/signup prompts
- Age verification
- Notification permission requests
- "Accept cookies" buttons
- Close (X) buttons on overlays

If you find a blocking element, respond with JSON to dismiss it:
```json
{
    "found_popup": true,
    "popup_type": "cookie_banner/newsletter/login/notification/other",
    "dismiss_action": {
        "tool": "click_element",
        "parameters": {"element_id": X or "selector": "..." or "text": "..."}
    }
}
```

If no blocking popup found:
```json
{
    "found_popup": false
}
```
"""
        try:
            content_parts = [popup_prompt, f"Accessibility tree:\n{state.accessibility_tree[:4000]}"]

            if self.use_vision and state.screenshot_base64:
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": state.screenshot_base64
                }
                content_parts.insert(1, image_part)

            response = await self.model.generate_content_async(content_parts)

            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                result = json.loads(json_match.group(0))

                if result.get("found_popup"):
                    logger.observation(f"Popup detected: {result.get('popup_type')}")
                    dismiss = result.get("dismiss_action", {})

                    return ToolCall(
                        tool=ToolName.CLICK,
                        parameters=dismiss.get("parameters", {}),
                        confidence=0.9,
                        reasoning=f"Dismissing {result.get('popup_type')} popup"
                    )

        except Exception as e:
            logger.debug(f"Popup analysis failed: {e}")

        return None
