"""
Rich + Loguru logging utility for the Agentic Web Navigator.
Provides beautiful terminal output for agent's thought process.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn


# Initialize Rich console
console = Console()


class AgentLogger:
    """
    Custom logger that combines Loguru's flexibility with Rich's beautiful output.
    Designed to display the agent's thought process clearly in the terminal.
    """

    def __init__(
        self,
        level: str = "INFO",
        log_to_file: bool = True,
        log_dir: Path = Path("logs"),
        app_name: str = "WebNavigator"
    ):
        self.app_name = app_name
        self.console = console
        self.log_dir = log_dir

        # Remove default logger
        logger.remove()

        # Add custom format for terminal (minimal, Rich handles styling)
        logger.add(
            sys.stderr,
            format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True,
        )

        # Add file logging if enabled
        if log_to_file:
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"{app_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                rotation="10 MB",
                retention="7 days",
            )

        self._logger = logger

    def thought(self, message: str, title: str = "Agent Thought"):
        """Display the agent's thought/reasoning process."""
        panel = Panel(
            Text(message, style="italic cyan"),
            title=f"[bold blue]{title}[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)
        self._logger.info(f"[THOUGHT] {message}")

    def action(self, action_type: str, details: dict[str, Any]):
        """Display an action being taken by the agent."""
        table = Table(title=f"[bold green]Action: {action_type}[/bold green]", border_style="green")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")

        for key, value in details.items():
            table.add_row(key, str(value)[:100])  # Truncate long values

        self.console.print(table)
        self._logger.info(f"[ACTION] {action_type}: {details}")

    def observation(self, message: str, data: Any = None):
        """Display an observation from the environment."""
        panel = Panel(
            Text(message, style="yellow"),
            title="[bold yellow]Observation[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
        self.console.print(panel)
        self._logger.debug(f"[OBSERVATION] {message}")

        if data:
            self._logger.debug(f"[OBSERVATION DATA] {data}")

    def error(self, message: str, exception: Exception = None):
        """Display an error with optional exception details."""
        error_text = Text(message, style="bold red")
        if exception:
            error_text.append(f"\n\nException: {type(exception).__name__}: {str(exception)}", style="red")

        panel = Panel(
            error_text,
            title="[bold red]Error[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
        self.console.print(panel)
        self._logger.error(f"[ERROR] {message}", exc_info=exception)

    def success(self, message: str):
        """Display a success message."""
        panel = Panel(
            Text(message, style="bold green"),
            title="[bold green]Success[/bold green]",
            border_style="green",
        )
        self.console.print(panel)
        self._logger.info(f"[SUCCESS] {message}")

    def step(self, step_num: int, total_steps: int, description: str):
        """Display current step progress."""
        progress_bar = f"[{'=' * step_num}{' ' * (total_steps - step_num)}]"
        self.console.print(
            f"[bold magenta]Step {step_num}/{total_steps}[/bold magenta] {progress_bar} {description}"
        )
        self._logger.info(f"[STEP {step_num}/{total_steps}] {description}")

    def dom_summary(self, elements_count: int, interactive_count: int, page_title: str):
        """Display a summary of processed DOM."""
        table = Table(title="[bold cyan]DOM Summary[/bold cyan]", border_style="cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Page Title", page_title[:60])
        table.add_row("Total Elements", str(elements_count))
        table.add_row("Interactive Elements", str(interactive_count))
        self.console.print(table)
        self._logger.debug(f"[DOM] Title: {page_title}, Elements: {elements_count}, Interactive: {interactive_count}")

    def show_json(self, data: dict, title: str = "JSON Data"):
        """Display JSON data with syntax highlighting."""
        import json
        json_str = json.dumps(data, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        panel = Panel(syntax, title=f"[bold]{title}[/bold]", border_style="white")
        self.console.print(panel)

    def info(self, message: str):
        """Standard info logging."""
        self.console.print(f"[dim cyan]INFO:[/dim cyan] {message}")
        self._logger.info(message)

    def debug(self, message: str):
        """Debug logging (only to file by default)."""
        self._logger.debug(message)

    def warning(self, message: str):
        """Warning logging."""
        self.console.print(f"[bold yellow]WARNING:[/bold yellow] {message}")
        self._logger.warning(message)

    def banner(self, text: str):
        """Display a banner/header."""
        self.console.print()
        self.console.rule(f"[bold magenta]{text}[/bold magenta]", style="magenta")
        self.console.print()

    def get_spinner(self, description: str = "Processing..."):
        """Get a Rich progress spinner context manager."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        )


# Global logger instance
agent_logger = AgentLogger()
