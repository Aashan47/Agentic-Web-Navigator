# Agentic Web Navigator

An autonomous browser agent that executes multi-step web actions from natural language instructions. Built with Python, LangGraph, Playwright, and Google Gemini 1.5 Pro.

## What It Does

Give the agent a goal like "Go to Amazon and search for wireless headphones" and it will:

1. Launch a browser
2. Analyze the page (screenshot + DOM)
3. Decide what action to take (click, type, scroll, etc.)
4. Execute the action
5. Handle errors, popups, and unexpected UI states
6. Repeat until the goal is achieved or deemed impossible

The agent uses Gemini 1.5 Pro's vision capabilities to understand page layouts and make decisions based on both visual and structural information.

## Requirements

- Python 3.10+
- Google API key with Gemini 1.5 Pro access
- Chrome/Chromium browser

## Installation

```bash
# Clone and enter directory
cd Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install browser
playwright install chromium

# Configure API key
cp .env.example .env
```

Edit `.env` and add your Google API key:

```
LLM_GOOGLE_API_KEY=your-api-key-here
```

## Usage

### Basic Usage

```bash
# Run with a goal
python main.py "Search for Python tutorials on YouTube"

# Run in headless mode (no visible browser)
python main.py --headless "Check the weather in Tokyo"

# Start from a specific URL
python main.py --start-url "https://amazon.com" "Search for mechanical keyboards"

# Limit the number of steps
python main.py --max-steps 20 "Find the r/programming subreddit"
```

### Run Evaluation Benchmark

The project includes 10 benchmark tasks to measure agent performance.

```bash
# Run all benchmarks
python evaluate.py

# Run in headless mode
python evaluate.py --headless

# Run specific tasks
python evaluate.py --tasks "task_001,task_002,task_003"
```

Results are saved to `evaluation_results/` as JSON files.

### Demo Individual Components

```bash
python demo.py
```

Select from:
1. Browser Controller - test Playwright integration
2. DOM Processor - test accessibility tree generation
3. Vision Observer - test screenshot + DOM capture
4. Global State - test action tracking and loop detection

## Project Structure

```
Project/
├── src/
│   ├── agent/
│   │   ├── state.py        # Agent state schema
│   │   ├── oracle.py       # Gemini LLM integration
│   │   ├── executor.py     # Action execution
│   │   ├── graph.py        # LangGraph state machine
│   │   └── recovery.py     # Error recovery and self-correction
│   ├── browser/
│   │   └── controller.py   # Playwright browser control
│   ├── vision/
│   │   ├── dom_processor.py # HTML to accessibility tree
│   │   └── observer.py      # Page observation
│   └── utils/
│       ├── logger.py       # Terminal output formatting
│       └── state.py        # Global state management
├── config/
│   └── settings.py         # Configuration via environment variables
├── benchmarks/
│   └── tasks.json          # Evaluation benchmark tasks
├── main.py                 # CLI entry point
├── evaluate.py             # Benchmark runner
└── demo.py                 # Component demos
```

## How It Works

### Agent Loop

The agent runs a ReAct-style loop implemented as a LangGraph state machine:

```
[Observe] -> [Oracle] -> [Act] -> [Check Success] -> [Observe] ...
                 |                       |
                 v                       v
              [Finish]            [Self-Correct]
```

**Observe**: Captures a screenshot and processes the DOM into a simplified accessibility tree. The tree lists interactive elements (buttons, links, inputs) with unique IDs and CSS selectors.

**Oracle**: Sends the screenshot and accessibility tree to Gemini 1.5 Pro. The model analyzes the current state against the goal and outputs a structured action:

```json
{
  "thought": "I need to click the search button to submit the query",
  "tool": "click_element",
  "parameters": {"element_id": 15},
  "confidence": 0.9
}
```

**Act**: Executes the action via Playwright. All actions are wrapped in try-except blocks that capture errors for feedback.

**Check Success**: Determines if the goal is complete, if we should continue, or if self-correction is needed.

### Self-Correction

When actions fail, the agent attempts recovery:

1. **Error Classification**: Categorizes the error (timeout, element not found, element intercepted, etc.)

2. **Popup Detection**: Scans for cookie banners, newsletter popups, login prompts. Uses common selectors and text patterns to find dismiss buttons.

3. **Reflexion**: If the same action fails twice, the agent analyzes the failure pattern and suggests an alternative approach.

4. **Loop Detection**: Tracks action signatures to detect repetitive patterns (same action repeated, A-B-A-B oscillation).

### DOM Processing

Raw HTML is converted to a compact accessibility tree to reduce token usage:

```
=== INTERACTIVE ELEMENTS ===
[1] <a[link]> "Home" href="/home"
[2] <input[text]> placeholder="Search..." name="q"
[3] <button[submit]> "Search" class="btn-primary"
[4] <a[link]> "Sign In" href="/login"

=== PAGE STRUCTURE ===
H1: Welcome to Example Site
NAVIGATION:
  - Home
  - Products
  - Contact
FORMS: 1 form(s) on page
```

Each element gets a unique ID that the Oracle can reference in actions.

## Configuration

Environment variables (set in `.env` or export directly):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_GOOGLE_API_KEY` | (required) | Google API key |
| `LLM_MODEL_NAME` | gemini-1.5-pro | Model to use |
| `LLM_TEMPERATURE` | 0.1 | Sampling temperature |
| `BROWSER_HEADLESS` | false | Run without visible browser |
| `BROWSER_VIEWPORT_WIDTH` | 1280 | Browser width |
| `BROWSER_VIEWPORT_HEIGHT` | 720 | Browser height |
| `BROWSER_TIMEOUT_MS` | 30000 | Action timeout |
| `AGENT_MAX_STEPS` | 50 | Max steps per goal |
| `AGENT_MAX_RETRIES` | 3 | Retries per action |
| `DOM_MAX_ELEMENTS` | 500 | Max elements to extract |
| `LOG_LEVEL` | INFO | Logging verbosity |

## Benchmark Tasks

The evaluation suite includes 10 tasks across different websites:

| Task | Goal | Difficulty |
|------|------|------------|
| Google Search | Search for "artificial intelligence news" | Easy |
| Wikipedia Article | Search for "Machine Learning" | Easy |
| Weather Check | Check weather for London on weather.com | Medium |
| GitHub Search | Search for "langchain" repository | Easy |
| BBC News | Navigate to Technology section | Medium |
| Amazon Search | Search for "4K monitor" | Medium |
| YouTube Search | Search for "python tutorial" | Medium |
| Stack Overflow | Search for "async await python" | Medium |
| Reddit Navigation | Go to r/programming | Medium |
| eBay Search | Search for "iPhone 15" | Medium |

Success is determined by URL patterns or page content matching.

## Limitations

- Requires a valid Google API key with Gemini access
- Some websites block automated browsers (CAPTCHAs, bot detection)
- Complex multi-page workflows may exceed step limits
- Vision model costs can add up for long sessions
- Does not handle file uploads or downloads
- No support for authentication/login flows beyond dismissing prompts

## Extending

### Adding New Tools

Edit `src/agent/state.py` to add new tool types:

```python
class ToolName(str, Enum):
    # ... existing tools
    HOVER = "hover"
```

Then implement the tool in `src/agent/executor.py`:

```python
async def _execute_hover(self, params: dict) -> ActionResult:
    selector = self._resolve_selector(params)
    await self.browser.page.hover(selector)
    return ActionResult(success=True, ...)
```

### Adding Benchmark Tasks

Edit `benchmarks/tasks.json`:

```json
{
  "id": "task_011",
  "name": "New Task",
  "goal": "Do something specific",
  "start_url": "https://example.com",
  "success_criteria": {
    "type": "url_contains",
    "value": "success-indicator"
  },
  "difficulty": "medium",
  "timeout_seconds": 90
}
```

### Custom Recovery Strategies

Add new error handlers in `src/agent/recovery.py`:

```python
async def _handle_captcha(self, state: AgentState) -> Tuple[bool, Optional[RecoveryAction]]:
    # Custom logic for CAPTCHA handling
    pass
```

## Troubleshooting

**"Browser not initialized"**: Run `playwright install chromium` to install the browser.

**"API key not found"**: Set `LLM_GOOGLE_API_KEY` in `.env` or pass `--api-key` flag.

**"Timeout waiting for element"**: The page may be slow or the element doesn't exist. Check the selector in the logs.

**"Element click intercepted"**: Something is blocking the element (popup, overlay). The agent should auto-recover, but some sites have persistent overlays.

**High token usage**: Reduce `DOM_MAX_ELEMENTS` or disable screenshots in `PageObserver` for text-only mode.

## License

MIT
