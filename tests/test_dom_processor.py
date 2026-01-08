"""Tests for the DOM Processor."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.dom_processor import DOMProcessor, DOMSnapshot, InteractiveElement


class TestDOMProcessor:
    """Test suite for DOMProcessor."""

    @pytest.fixture
    def processor(self):
        return DOMProcessor(max_elements=100, max_text_length=50)

    @pytest.fixture
    def sample_html(self):
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>
                <a href="/home" id="home-link">Home</a>
                <a href="/about">About</a>
            </nav>
            <main>
                <h1>Main Title</h1>
                <form id="test-form">
                    <input type="text" name="username" placeholder="Enter username">
                    <input type="password" name="password" placeholder="Enter password">
                    <button type="submit" class="btn primary">Login</button>
                </form>
                <button data-testid="action-btn" onclick="doSomething()">Action</button>
            </main>
            <script>console.log('should be removed');</script>
            <style>.hidden { display: none; }</style>
        </body>
        </html>
        """

    def test_process_returns_snapshot(self, processor, sample_html):
        """Test that process returns a DOMSnapshot."""
        result = processor.process(sample_html, "https://test.com", "Test")
        assert isinstance(result, DOMSnapshot)
        assert result.url == "https://test.com"

    def test_extracts_title(self, processor, sample_html):
        """Test title extraction."""
        result = processor.process(sample_html)
        assert result.title == "Test Page"

    def test_extracts_interactive_elements(self, processor, sample_html):
        """Test that interactive elements are extracted."""
        result = processor.process(sample_html)
        assert len(result.elements) > 0

        # Check that we have links
        links = [e for e in result.elements if e.tag == "a"]
        assert len(links) == 2

        # Check that we have inputs
        inputs = [e for e in result.elements if e.tag == "input"]
        assert len(inputs) == 2

        # Check that we have buttons
        buttons = [e for e in result.elements if e.tag == "button"]
        assert len(buttons) == 2

    def test_removes_script_tags(self, processor, sample_html):
        """Test that script tags are removed."""
        result = processor.process(sample_html)
        assert "console.log" not in result.accessibility_tree

    def test_removes_style_tags(self, processor, sample_html):
        """Test that style tags are removed."""
        result = processor.process(sample_html)
        assert ".hidden" not in result.accessibility_tree

    def test_generates_selectors(self, processor, sample_html):
        """Test that selectors are generated for elements."""
        result = processor.process(sample_html)

        # Find element with ID
        home_link = next((e for e in result.elements if "home" in e.text.lower()), None)
        assert home_link is not None
        assert "#home-link" in home_link.selector

        # Find element with data-testid
        action_btn = next((e for e in result.elements if e.attributes.get("data-testid")), None)
        assert action_btn is not None
        assert "data-testid" in action_btn.selector

    def test_generates_xpath(self, processor, sample_html):
        """Test that XPath selectors are generated."""
        result = processor.process(sample_html)

        for elem in result.elements:
            assert elem.xpath is not None
            assert elem.xpath.startswith("/")

    def test_calculates_dom_hash(self, processor, sample_html):
        """Test that DOM hash is calculated."""
        result = processor.process(sample_html)
        assert result.dom_hash is not None
        assert len(result.dom_hash) == 16

    def test_dom_hash_changes_with_content(self, processor):
        """Test that DOM hash changes when content changes."""
        html1 = "<html><body><button>Click 1</button></body></html>"
        html2 = "<html><body><button>Click 2</button></body></html>"

        result1 = processor.process(html1)
        result2 = processor.process(html2)

        assert result1.dom_hash != result2.dom_hash

    def test_element_to_compact_string(self, processor, sample_html):
        """Test compact string representation."""
        result = processor.process(sample_html)

        for elem in result.elements:
            compact = elem.to_compact_string()
            assert f"[{elem.element_id}]" in compact
            assert f"<{elem.tag}" in compact

    def test_max_elements_limit(self):
        """Test that max_elements limit is respected."""
        html = "<html><body>"
        for i in range(100):
            html += f'<button id="btn-{i}">Button {i}</button>'
        html += "</body></html>"

        processor = DOMProcessor(max_elements=10)
        result = processor.process(html)

        assert len(result.elements) <= 10

    def test_removes_hidden_elements(self, processor):
        """Test that hidden elements are removed."""
        html = """
        <html><body>
            <button style="display: none;">Hidden 1</button>
            <button hidden>Hidden 2</button>
            <button aria-hidden="true">Hidden 3</button>
            <button class="d-none">Hidden 4</button>
            <button>Visible</button>
        </body></html>
        """
        result = processor.process(html)

        # Only visible button should be extracted
        buttons = [e for e in result.elements if e.tag == "button"]
        assert len(buttons) == 1
        assert "Visible" in buttons[0].text


class TestInteractiveElement:
    """Test suite for InteractiveElement."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        elem = InteractiveElement(
            element_id=1,
            tag="button",
            element_type="submit",
            text="Click Me",
            selector="#btn",
            xpath="//button[@id='btn']",
            attributes={"id": "btn", "class": "primary"}
        )

        d = elem.to_dict()
        assert d["id"] == 1
        assert d["tag"] == "button"
        assert d["type"] == "submit"
        assert d["text"] == "Click Me"

    def test_to_compact_string(self):
        """Test compact string representation."""
        elem = InteractiveElement(
            element_id=5,
            tag="input",
            element_type="text",
            text="",
            selector="input[name='email']",
            xpath="//input[@name='email']",
            attributes={"name": "email", "placeholder": "Enter email"}
        )

        compact = elem.to_compact_string()
        assert "[5]" in compact
        assert "<input[text]>" in compact
        assert 'placeholder="Enter email"' in compact
