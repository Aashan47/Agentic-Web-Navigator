"""
Agentic Web Navigator - Main Entry Point

An autonomous agent that takes natural language goals and executes
multi-step browser actions to achieve them using LangGraph and Gemini.

Usage:
    python main.py "Go to Amazon, find the cheapest 4k monitor, and add it to cart"
    python main.py --headless "Search for weather in London"
    python main.py --start-url "https://amazon.com" "Search for wireless headphones"
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.browser.controller import BrowserController
from src.vision.observer import PageObserver
from src.vision.dom_processor import DOMProcessor
from src.agent.oracle import Oracle
from src.agent.graph import WebNavigatorGraph
from src.utils.logger import agent_logger as logger
from src.utils.state import global_state
from config import settings


async def run_navigator(
    goal: str,
    start_url: str = None,
    headless: bool = False,
    max_steps: int = 50,
    api_key: str = None
):
    """
    Run the web navigator with a given goal.

    Args:
        goal: Natural language description of what to accomplish
        start_url: Optional URL to start from
        headless: Run browser in headless mode
        max_steps: Maximum steps before stopping
        api_key: Google API key for Gemini (uses env var if not provided)
    """
    logger.banner("Agentic Web Navigator")
    logger.info(f"Goal: {goal}")

    if start_url:
        logger.info(f"Start URL: {start_url}")

    # Initialize components
    browser = BrowserController(
        headless=headless,
        viewport_width=settings.browser.viewport_width,
        viewport_height=settings.browser.viewport_height
    )

    try:
        # Initialize browser
        result = await browser.initialize()
        if not result.success:
            logger.error(f"Failed to initialize browser: {result.error}")
            return None

        # Create DOM processor and observer
        dom_processor = DOMProcessor(
            max_elements=settings.dom.max_elements,
            max_text_length=settings.dom.max_text_length
        )

        observer = PageObserver(
            browser=browser,
            dom_processor=dom_processor,
            capture_screenshots=True
        )

        # Create Oracle (LLM interface)
        oracle = Oracle(
            api_key=api_key or settings.llm.google_api_key,
            model_name=settings.llm.model_name,
            use_vision=True
        )

        # Create the navigation agent
        navigator = WebNavigatorGraph(
            browser=browser,
            observer=observer,
            oracle=oracle,
            max_steps=max_steps
        )

        logger.thought(
            "Agent initialized with:\n"
            f"  - Browser: Chromium ({'headless' if headless else 'visible'})\n"
            f"  - LLM: {settings.llm.model_name}\n"
            f"  - Max steps: {max_steps}\n"
            "Starting navigation loop..."
        )

        # Run the navigation
        final_state = await navigator.run(goal, start_url)

        # Display final results
        logger.banner("Navigation Complete")

        if final_state.success:
            logger.success(f"Goal achieved: {final_state.completion_reason}")
        else:
            logger.warning(f"Goal not achieved: {final_state.completion_reason}")

        # Show statistics
        stats = {
            "success": final_state.success,
            "total_steps": final_state.current_step,
            "total_actions": final_state.total_actions,
            "successful_actions": final_state.successful_actions,
            "self_corrections": final_state.self_corrections,
            "popups_handled": final_state.popups_handled,
            "final_url": final_state.current_url,
        }
        logger.show_json(stats, "Session Statistics")

        return final_state

    except KeyboardInterrupt:
        logger.warning("Navigation interrupted by user")
        return None

    except Exception as e:
        logger.error(f"Navigator error: {e}", exception=e)
        return None

    finally:
        await browser.close()
        logger.info("Browser closed")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Agentic Web Navigator - Autonomous web browsing agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Search for the weather in Tokyo"
  python main.py --headless "Find Python tutorials on YouTube"
  python main.py --start-url "https://amazon.com" "Search for wireless headphones"
  python main.py --max-steps 30 "Navigate to Reddit and go to r/programming"
        """
    )

    parser.add_argument(
        "goal",
        nargs="?",
        default="Navigate to Google and search for 'web automation with AI'",
        help="Natural language goal for the agent"
    )

    parser.add_argument(
        "--start-url", "-u",
        type=str,
        default=None,
        help="Starting URL (default: determined by goal)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode"
    )

    parser.add_argument(
        "--max-steps", "-m",
        type=int,
        default=50,
        help="Maximum steps before stopping (default: 50)"
    )

    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="Google API key for Gemini (uses GOOGLE_API_KEY env var if not provided)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Run the navigator
    result = asyncio.run(run_navigator(
        goal=args.goal,
        start_url=args.start_url,
        headless=args.headless,
        max_steps=args.max_steps,
        api_key=args.api_key
    ))

    # Exit with appropriate code
    if result and result.success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
