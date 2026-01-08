"""
DOM Processor for the Agentic Web Navigator.
Converts complex HTML DOM into a simplified accessibility tree optimized for LLM consumption.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, Any
from bs4 import BeautifulSoup, Tag, NavigableString


@dataclass
class InteractiveElement:
    """Represents an interactive element extracted from the DOM."""
    element_id: int  # Unique ID for this session
    tag: str
    element_type: Optional[str]  # input type, button type, etc.
    text: str  # Visible text content
    selector: str  # CSS selector to locate this element
    xpath: str  # XPath alternative selector
    attributes: dict[str, str]  # Relevant attributes
    is_visible: bool = True
    is_enabled: bool = True
    bounding_box: Optional[dict[str, float]] = None  # x, y, width, height

    def to_dict(self) -> dict:
        """Convert to dictionary for LLM context."""
        return {
            "id": self.element_id,
            "tag": self.tag,
            "type": self.element_type,
            "text": self.text[:100] if self.text else "",
            "selector": self.selector,
            "attributes": self.attributes,
        }

    def to_compact_string(self) -> str:
        """Compact string representation for minimal token usage."""
        type_str = f"[{self.element_type}]" if self.element_type else ""
        text_str = f'"{self.text[:50]}"' if self.text else ""
        attrs = []
        if self.attributes.get("placeholder"):
            attrs.append(f'placeholder="{self.attributes["placeholder"][:30]}"')
        if self.attributes.get("aria-label"):
            attrs.append(f'aria="{self.attributes["aria-label"][:30]}"')
        if self.attributes.get("name"):
            attrs.append(f'name="{self.attributes["name"]}"')

        attr_str = " ".join(attrs)
        return f"[{self.element_id}] <{self.tag}{type_str}> {text_str} {attr_str}".strip()


@dataclass
class DOMSnapshot:
    """Snapshot of the processed DOM."""
    url: str
    title: str
    elements: list[InteractiveElement]
    accessibility_tree: str  # Simplified text representation
    dom_hash: str  # For change detection
    raw_html_length: int
    processed_length: int
    timestamp: str = field(default_factory=lambda: __import__("datetime").datetime.now().isoformat())

    def get_interactive_count(self) -> int:
        return len(self.elements)


class DOMProcessor:
    """
    Processes raw HTML DOM into a simplified accessibility tree.
    Filters non-interactive elements and extracts actionable components.
    """

    # Tags to completely ignore
    EXCLUDED_TAGS = {
        "script", "style", "noscript", "svg", "path", "meta", "link",
        "head", "title", "br", "hr", "iframe", "object", "embed",
        "canvas", "video", "audio", "source", "track", "map", "area",
        "picture", "template", "slot", "math", "annotation"
    }

    # Tags that are interactive
    INTERACTIVE_TAGS = {
        "a", "button", "input", "select", "textarea", "option",
        "label", "details", "summary", "dialog"
    }

    # Tags that may contain important text
    TEXT_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6", "span", "div",
        "li", "td", "th", "article", "section", "header", "footer",
        "nav", "main", "aside", "figcaption", "blockquote", "pre", "code"
    }

    # Attributes to preserve
    IMPORTANT_ATTRS = {
        "id", "class", "name", "type", "value", "placeholder", "href",
        "src", "alt", "title", "aria-label", "aria-describedby",
        "role", "data-testid", "data-cy", "data-test"
    }

    def __init__(
        self,
        max_elements: int = 500,
        max_text_length: int = 100,
        include_aria: bool = True,
        include_data_attrs: bool = False
    ):
        self.max_elements = max_elements
        self.max_text_length = max_text_length
        self.include_aria = include_aria
        self.include_data_attrs = include_data_attrs
        self._element_counter = 0

    def process(self, html: str, url: str = "", title: str = "") -> DOMSnapshot:
        """
        Process raw HTML and return a DOMSnapshot with interactive elements.

        Args:
            html: Raw HTML string
            url: Current page URL
            title: Page title

        Returns:
            DOMSnapshot with processed elements and accessibility tree
        """
        self._element_counter = 0
        raw_length = len(html)

        soup = BeautifulSoup(html, "lxml")

        # Remove excluded tags entirely
        for tag in soup.find_all(self.EXCLUDED_TAGS):
            tag.decompose()

        # Remove hidden elements
        self._remove_hidden_elements(soup)

        # Extract interactive elements
        elements = self._extract_interactive_elements(soup)

        # Build accessibility tree
        tree = self._build_accessibility_tree(elements, soup)

        # Calculate DOM hash for change detection
        dom_hash = self._calculate_hash(tree)

        return DOMSnapshot(
            url=url,
            title=title or self._extract_title(soup),
            elements=elements,
            accessibility_tree=tree,
            dom_hash=dom_hash,
            raw_html_length=raw_length,
            processed_length=len(tree)
        )

    def _remove_hidden_elements(self, soup: BeautifulSoup):
        """Remove elements that are hidden via common patterns."""
        # Remove elements with display:none or visibility:hidden in style attribute
        for elem in soup.find_all(style=re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)):
            elem.decompose()

        # Remove elements with hidden attribute
        for elem in soup.find_all(attrs={"hidden": True}):
            elem.decompose()

        # Remove elements with aria-hidden="true"
        for elem in soup.find_all(attrs={"aria-hidden": "true"}):
            elem.decompose()

        # Remove common hidden class patterns
        hidden_classes = ["hidden", "d-none", "invisible", "sr-only", "visually-hidden"]
        for cls in hidden_classes:
            for elem in soup.find_all(class_=re.compile(rf"\b{cls}\b", re.I)):
                elem.decompose()

    def _extract_interactive_elements(self, soup: BeautifulSoup) -> list[InteractiveElement]:
        """Extract all interactive elements from the DOM."""
        elements = []

        # Find all interactive elements
        for tag_name in self.INTERACTIVE_TAGS:
            for elem in soup.find_all(tag_name):
                if len(elements) >= self.max_elements:
                    break

                interactive = self._parse_element(elem)
                if interactive:
                    elements.append(interactive)

        # Also find elements with click handlers or role="button"
        for elem in soup.find_all(attrs={"role": re.compile(r"button|link|checkbox|radio|menuitem|tab", re.I)}):
            if len(elements) >= self.max_elements:
                break
            if elem.name not in self.INTERACTIVE_TAGS:
                interactive = self._parse_element(elem)
                if interactive:
                    elements.append(interactive)

        # Find elements with onclick attribute
        for elem in soup.find_all(attrs={"onclick": True}):
            if len(elements) >= self.max_elements:
                break
            if elem.name not in self.INTERACTIVE_TAGS:
                interactive = self._parse_element(elem)
                if interactive:
                    elements.append(interactive)

        return elements

    def _parse_element(self, elem: Tag) -> Optional[InteractiveElement]:
        """Parse a single element into InteractiveElement."""
        if not isinstance(elem, Tag):
            return None

        self._element_counter += 1

        # Get text content
        text = self._get_element_text(elem)

        # Build selector
        selector = self._build_selector(elem)
        xpath = self._build_xpath(elem)

        # Extract relevant attributes
        attrs = {}
        for attr in self.IMPORTANT_ATTRS:
            if elem.has_attr(attr):
                value = elem[attr]
                if isinstance(value, list):
                    value = " ".join(value)
                attrs[attr] = str(value)[:self.max_text_length]

        # Include data attributes if configured
        if self.include_data_attrs:
            for key, value in elem.attrs.items():
                if key.startswith("data-") and key not in attrs:
                    if isinstance(value, list):
                        value = " ".join(value)
                    attrs[key] = str(value)[:50]

        # Determine element type
        elem_type = None
        if elem.name == "input":
            elem_type = attrs.get("type", "text")
        elif elem.name == "button":
            elem_type = attrs.get("type", "button")
        elif elem.name == "a":
            elem_type = "link"
        elif elem.name == "select":
            elem_type = "dropdown"
        elif elem.name == "textarea":
            elem_type = "textarea"

        return InteractiveElement(
            element_id=self._element_counter,
            tag=elem.name,
            element_type=elem_type,
            text=text,
            selector=selector,
            xpath=xpath,
            attributes=attrs
        )

    def _get_element_text(self, elem: Tag) -> str:
        """Extract meaningful text from an element."""
        # First check for value attribute (inputs)
        if elem.has_attr("value") and elem.name == "input":
            return str(elem["value"])[:self.max_text_length]

        # Check for aria-label
        if elem.has_attr("aria-label"):
            return str(elem["aria-label"])[:self.max_text_length]

        # Check for title
        if elem.has_attr("title"):
            return str(elem["title"])[:self.max_text_length]

        # Get direct text content
        text = elem.get_text(strip=True, separator=" ")
        return text[:self.max_text_length] if text else ""

    def _build_selector(self, elem: Tag) -> str:
        """Build a CSS selector for the element."""
        # Priority: id > data-testid > name > class combination > tag with index

        # ID selector (most reliable)
        if elem.has_attr("id"):
            elem_id = elem["id"]
            if isinstance(elem_id, list):
                elem_id = elem_id[0]
            # Escape special characters in ID
            safe_id = re.sub(r'([^\w-])', r'\\\1', elem_id)
            return f"#{safe_id}"

        # data-testid (common in React apps)
        if elem.has_attr("data-testid"):
            return f'[data-testid="{elem["data-testid"]}"]'

        # data-cy (Cypress test attribute)
        if elem.has_attr("data-cy"):
            return f'[data-cy="{elem["data-cy"]}"]'

        # Name attribute (for forms)
        if elem.has_attr("name"):
            return f'{elem.name}[name="{elem["name"]}"]'

        # Build class-based selector
        if elem.has_attr("class"):
            classes = elem["class"]
            if isinstance(classes, list):
                # Filter out common utility classes that are too generic
                generic = {"container", "row", "col", "wrapper", "content", "text", "item"}
                specific_classes = [c for c in classes if c.lower() not in generic and len(c) > 2]
                if specific_classes:
                    class_selector = ".".join(specific_classes[:3])  # Max 3 classes
                    return f"{elem.name}.{class_selector}"

        # Fall back to tag with placeholder or type for inputs
        if elem.name == "input":
            if elem.has_attr("placeholder"):
                return f'input[placeholder="{elem["placeholder"][:30]}"]'
            if elem.has_attr("type"):
                return f'input[type="{elem["type"]}"]'

        # For links, use href pattern
        if elem.name == "a" and elem.has_attr("href"):
            href = elem["href"]
            if href and href != "#" and not href.startswith("javascript:"):
                # Use partial href match
                return f'a[href*="{href[:50]}"]'

        # Last resort: tag with text content
        text = self._get_element_text(elem)
        if text:
            # Use contains text selector (pseudo-selector, may need JS)
            return f'{elem.name}:has-text("{text[:30]}")'

        return f"{elem.name}"

    def _build_xpath(self, elem: Tag) -> str:
        """Build an XPath selector for the element."""
        if elem.has_attr("id"):
            elem_id = elem["id"]
            if isinstance(elem_id, list):
                elem_id = elem_id[0]
            return f'//*[@id="{elem_id}"]'

        if elem.has_attr("data-testid"):
            return f'//*[@data-testid="{elem["data-testid"]}"]'

        if elem.has_attr("name"):
            return f'//{elem.name}[@name="{elem["name"]}"]'

        # Build path from root
        parts = []
        current = elem
        while current.parent and current.parent.name:
            if current.name == "[document]":
                break
            siblings = current.parent.find_all(current.name, recursive=False)
            if len(siblings) > 1:
                index = siblings.index(current) + 1
                parts.append(f"{current.name}[{index}]")
            else:
                parts.append(current.name)
            current = current.parent

        parts.reverse()
        return "/" + "/".join(parts) if parts else f"//{elem.name}"

    def _build_accessibility_tree(
        self,
        elements: list[InteractiveElement],
        soup: BeautifulSoup
    ) -> str:
        """
        Build a simplified accessibility tree string for LLM consumption.
        Optimized for minimal tokens while preserving essential information.
        """
        lines = []
        lines.append("=== INTERACTIVE ELEMENTS ===")

        for elem in elements:
            lines.append(elem.to_compact_string())

        # Add page structure summary
        lines.append("\n=== PAGE STRUCTURE ===")

        # Extract headings for context
        for level in range(1, 4):
            for heading in soup.find_all(f"h{level}")[:5]:
                text = heading.get_text(strip=True)[:80]
                if text:
                    lines.append(f"H{level}: {text}")

        # Extract main navigation links
        nav = soup.find("nav")
        if nav:
            lines.append("\nNAVIGATION:")
            for link in nav.find_all("a")[:10]:
                text = link.get_text(strip=True)[:40]
                if text:
                    lines.append(f"  - {text}")

        # Extract form summary
        forms = soup.find_all("form")
        if forms:
            lines.append(f"\nFORMS: {len(forms)} form(s) on page")

        return "\n".join(lines)

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)

        # Fallback to h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)[:100]

        return "Untitled Page"

    def _calculate_hash(self, content: str) -> str:
        """Calculate hash for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_element_by_id(self, elements: list[InteractiveElement], elem_id: int) -> Optional[InteractiveElement]:
        """Find an element by its assigned ID."""
        for elem in elements:
            if elem.element_id == elem_id:
                return elem
        return None
