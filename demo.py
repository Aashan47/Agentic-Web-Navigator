"""
Demo script for testing the Agentic Web Navigator components.
Tests Browser Controller, DOM Processor, and Vision Observer.
"""

import asyncio
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.browser.controller import BrowserController, create_browser
from src.vision.dom_processor import DOMProcessor
from src.vision.observer import PageObserver
from src.utils.logger import agent_logger as logger
from src.utils.state import global_state, ActionType, ActionStatus


async def demo_browser_controller():
    """Demonstrate browser controller functionality."""
    logger.banner("Browser Controller Demo")

    browser = BrowserController(headless=False)

    try:
        # Initialize browser
        result = await browser.initialize()
        if not result.success:
            logger.error(f"Failed to initialize: {result.error}")
            return

        # Navigate to a test page
        logger.thought("Navigating to example.com to test basic functionality")
        result = await browser.navigate("https://example.com")

        if result.success:
            logger.success(f"Navigation successful: {result.message}")
            logger.info(f"Current URL: {browser.current_url}")
        else:
            logger.error(f"Navigation failed: {result.error}")

        # Take a screenshot
        logger.thought("Taking a screenshot of the page")
        screenshot_result = await browser.take_screenshot()

        if screenshot_result.success:
            logger.success(f"Screenshot captured: {screenshot_result.message}")
        else:
            logger.error(f"Screenshot failed: {screenshot_result.error}")

        # Get page content
        html, url, title = await browser.get_page_content()
        logger.info(f"Page title: {title}")
        logger.info(f"HTML length: {len(html)} characters")

        # Wait a moment to see the browser
        await asyncio.sleep(2)

    finally:
        await browser.close()


async def demo_dom_processor():
    """Demonstrate DOM processor functionality."""
    logger.banner("DOM Processor Demo")

    # Sample HTML for testing
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title></head>
    <body>
        <nav>
            <a href="/home" id="home-link">Home</a>
            <a href="/products">Products</a>
            <a href="/contact">Contact</a>
        </nav>
        <main>
            <h1>Welcome to the Test Page</h1>
            <form id="search-form">
                <input type="text" name="q" placeholder="Search..." aria-label="Search input">
                <button type="submit" class="btn btn-primary">Search</button>
            </form>
            <div class="products">
                <button data-testid="add-to-cart" onclick="addToCart()">Add to Cart</button>
                <select name="quantity">
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                </select>
            </div>
            <script>console.log('This should be filtered out');</script>
            <style>.hidden { display: none; }</style>
            <div style="display: none;">Hidden content</div>
        </main>
    </body>
    </html>
    """

    processor = DOMProcessor(max_elements=100, max_text_length=80)

    logger.thought("Processing sample HTML to extract interactive elements")
    snapshot = processor.process(sample_html, "https://example.com/test", "Test Page")

    logger.info(f"Page title: {snapshot.title}")
    logger.info(f"DOM hash: {snapshot.dom_hash}")
    logger.info(f"Raw HTML: {snapshot.raw_html_length} chars -> Processed: {snapshot.processed_length} chars")
    logger.info(f"Compression: {100 - (snapshot.processed_length / snapshot.raw_html_length * 100):.1f}%")

    logger.banner("Interactive Elements Found")
    for elem in snapshot.elements:
        logger.info(elem.to_compact_string())

    logger.banner("Accessibility Tree")
    print(snapshot.accessibility_tree)


async def demo_vision_observer():
    """Demonstrate the vision observer with live browser."""
    logger.banner("Vision Observer Demo")

    browser = BrowserController(headless=False)

    try:
        await browser.initialize()

        # Create observer
        observer = PageObserver(
            browser=browser,
            capture_screenshots=True,
            full_page_screenshots=False
        )

        # Navigate and observe
        logger.thought("Navigating to Wikipedia to test real-world DOM processing")
        await browser.navigate("https://en.wikipedia.org")

        # Observe the page
        observation = await observer.observe()

        logger.info(f"URL: {observation.url}")
        logger.info(f"Title: {observation.title}")
        logger.info(f"Interactive elements: {len(observation.dom_snapshot.elements)}")
        logger.info(f"Has screenshot: {observation.screenshot_base64 is not None}")

        # Show some elements
        logger.banner("First 10 Interactive Elements")
        for elem in observation.dom_snapshot.elements[:10]:
            logger.info(elem.to_compact_string())

        # Find specific elements
        logger.banner("Element Search Demo")
        search_input = await observer.find_element_for_action(text="search", tag="input")
        if search_input:
            logger.success(f"Found search input: {search_input.to_compact_string()}")
            logger.info(f"Selector: {search_input.selector}")
        else:
            logger.warning("Search input not found")

        # Get element types
        links = observer.get_links()
        buttons = observer.get_buttons()
        inputs = observer.get_inputs()

        logger.info(f"Found {len(links)} links, {len(buttons)} buttons, {len(inputs)} input fields")

        await asyncio.sleep(3)

    finally:
        await browser.close()


async def demo_global_state():
    """Demonstrate global state tracking."""
    logger.banner("Global State Demo")

    # Start a session
    global_state.start_session("Navigate to Amazon and search for monitors")

    # Record some actions
    global_state.record_action(
        ActionType.NAVIGATE,
        {"url": "https://amazon.com"},
        ActionStatus.SUCCESS,
        url_before="about:blank",
        url_after="https://amazon.com",
        thought="Starting by navigating to Amazon homepage"
    )

    global_state.record_action(
        ActionType.TYPE,
        {"selector": "#search-input", "text": "4k monitor"},
        ActionStatus.SUCCESS,
        thought="Typing search query into search box"
    )

    global_state.record_action(
        ActionType.CLICK,
        {"selector": "#search-button"},
        ActionStatus.SUCCESS,
        thought="Clicking search button to submit query"
    )

    # Get stats
    stats = global_state.get_session_stats()
    logger.show_json(stats, "Session Statistics")

    # Check for loops
    is_loop, reason = global_state.detect_loop()
    logger.info(f"Loop detected: {is_loop}, Reason: {reason}")

    # Get history
    history = global_state.get_history_summary()
    logger.show_json({"history": history}, "Action History")


async def main():
    """Run all demos."""
    logger.banner("Agentic Web Navigator - Component Demo")

    print("\nSelect demo to run:")
    print("1. Browser Controller")
    print("2. DOM Processor")
    print("3. Vision Observer (requires browser)")
    print("4. Global State")
    print("5. Run All")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        await demo_browser_controller()
    elif choice == "2":
        await demo_dom_processor()
    elif choice == "3":
        await demo_vision_observer()
    elif choice == "4":
        await demo_global_state()
    elif choice == "5":
        await demo_dom_processor()  # No browser needed
        await demo_global_state()    # No browser needed
        await demo_browser_controller()
        await demo_vision_observer()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
