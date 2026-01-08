"""
Evaluation Runner for the Agentic Web Navigator.
Runs benchmark tasks and calculates success metrics.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.browser.controller import BrowserController
from src.vision.observer import PageObserver
from src.vision.dom_processor import DOMProcessor
from src.agent.graph import WebNavigatorGraph
from src.agent.oracle import Oracle
from src.utils.logger import agent_logger as logger
from rich.table import Table
from rich.console import Console

console = Console()


@dataclass
class TaskResult:
    """Result of a single benchmark task."""
    task_id: str
    task_name: str
    goal: str
    success: bool
    completion_reason: str
    total_steps: int
    total_actions: int
    successful_actions: int
    self_corrections: int
    popups_handled: int
    duration_seconds: float
    final_url: str
    error: Optional[str] = None
    screenshot_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "success": self.success,
            "completion_reason": self.completion_reason,
            "total_steps": self.total_steps,
            "self_corrections": self.self_corrections,
            "duration_seconds": self.duration_seconds,
            "error": self.error
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    success_rate: float
    average_steps: float
    average_duration: float
    total_self_corrections: int
    total_popups_handled: int
    results: list[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_tasks": self.total_tasks,
                "successful_tasks": self.successful_tasks,
                "failed_tasks": self.failed_tasks,
                "success_rate": f"{self.success_rate:.1f}%",
                "average_steps": round(self.average_steps, 1),
                "average_duration_seconds": round(self.average_duration, 1),
                "total_self_corrections": self.total_self_corrections,
                "total_popups_handled": self.total_popups_handled,
            },
            "results": [r.to_dict() for r in self.results]
        }


class SuccessCriteriaChecker:
    """Checks if success criteria are met."""

    @staticmethod
    def check(criteria: dict, final_url: str, page_content: str = "") -> bool:
        """
        Check if success criteria are met.

        Args:
            criteria: Success criteria from task definition
            final_url: The final URL after task execution
            page_content: Optional page content for text matching

        Returns:
            True if criteria are met
        """
        criteria_type = criteria.get("type", "")
        value = criteria.get("value", "")

        if criteria_type == "url_contains":
            return value.lower() in final_url.lower()

        elif criteria_type == "url_exact":
            return final_url.lower() == value.lower()

        elif criteria_type == "page_contains_text":
            return value.lower() in page_content.lower()

        elif criteria_type == "url_regex":
            import re
            return bool(re.search(value, final_url))

        return False


class EvaluationRunner:
    """
    Runs evaluation benchmarks and generates reports.
    """

    def __init__(
        self,
        tasks_file: Path = Path("benchmarks/tasks.json"),
        output_dir: Path = Path("evaluation_results"),
        headless: bool = True,
        api_key: str = None
    ):
        self.tasks_file = tasks_file
        self.output_dir = output_dir
        self.headless = headless
        self.api_key = api_key
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_tasks(self) -> list[dict]:
        """Load benchmark tasks from JSON file."""
        with open(self.tasks_file, "r") as f:
            data = json.load(f)
        return data.get("tasks", [])

    async def run_single_task(
        self,
        task: dict,
        navigator: WebNavigatorGraph
    ) -> TaskResult:
        """Run a single benchmark task."""
        task_id = task.get("id", "unknown")
        task_name = task.get("name", "Unnamed Task")
        goal = task.get("goal", "")
        start_url = task.get("start_url", "")
        criteria = task.get("success_criteria", {})
        timeout = task.get("timeout_seconds", 120)

        logger.banner(f"Task: {task_name}")
        logger.info(f"Goal: {goal}")
        logger.info(f"Start URL: {start_url}")

        start_time = datetime.now()
        error = None
        final_url = ""

        try:
            # Run the navigation task with timeout
            final_state = await asyncio.wait_for(
                navigator.run(goal, start_url),
                timeout=timeout
            )

            final_url = final_state.current_url

            # Check success criteria
            criteria_met = SuccessCriteriaChecker.check(criteria, final_url)

            # Determine overall success
            success = final_state.success and criteria_met

            if not criteria_met and final_state.success:
                error = f"Task completed but success criteria not met. URL: {final_url}"

            duration = (datetime.now() - start_time).total_seconds()

            return TaskResult(
                task_id=task_id,
                task_name=task_name,
                goal=goal,
                success=success,
                completion_reason=final_state.completion_reason,
                total_steps=final_state.current_step,
                total_actions=final_state.total_actions,
                successful_actions=final_state.successful_actions,
                self_corrections=final_state.self_corrections,
                popups_handled=final_state.popups_handled,
                duration_seconds=duration,
                final_url=final_url,
                error=error
            )

        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            return TaskResult(
                task_id=task_id,
                task_name=task_name,
                goal=goal,
                success=False,
                completion_reason="Task timed out",
                total_steps=0,
                total_actions=0,
                successful_actions=0,
                self_corrections=0,
                popups_handled=0,
                duration_seconds=duration,
                final_url=final_url,
                error=f"Timeout after {timeout} seconds"
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Task failed with error: {e}", exception=e)
            return TaskResult(
                task_id=task_id,
                task_name=task_name,
                goal=goal,
                success=False,
                completion_reason=f"Error: {str(e)}",
                total_steps=0,
                total_actions=0,
                successful_actions=0,
                self_corrections=0,
                popups_handled=0,
                duration_seconds=duration,
                final_url=final_url,
                error=str(e)
            )

    async def run_evaluation(self, task_ids: list[str] = None) -> EvaluationReport:
        """
        Run the full evaluation suite.

        Args:
            task_ids: Optional list of specific task IDs to run.
                      If None, runs all tasks.

        Returns:
            EvaluationReport with results
        """
        logger.banner("Starting Evaluation Suite")

        tasks = self.load_tasks()

        # Filter tasks if specific IDs provided
        if task_ids:
            tasks = [t for t in tasks if t.get("id") in task_ids]

        logger.info(f"Running {len(tasks)} benchmark tasks")

        results: list[TaskResult] = []

        # Initialize browser once for all tasks
        browser = BrowserController(headless=self.headless)
        await browser.initialize()

        try:
            dom_processor = DOMProcessor()
            observer = PageObserver(browser, dom_processor)
            oracle = Oracle(api_key=self.api_key)

            navigator = WebNavigatorGraph(
                browser=browser,
                observer=observer,
                oracle=oracle,
                max_steps=30  # Limit steps for evaluation
            )

            for i, task in enumerate(tasks):
                logger.info(f"\n[{i+1}/{len(tasks)}] Running task: {task.get('name')}")

                result = await self.run_single_task(task, navigator)
                results.append(result)

                # Log result
                status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
                console.print(f"  Result: {status} | Steps: {result.total_steps} | Self-corrections: {result.self_corrections}")

                # Small delay between tasks
                await asyncio.sleep(2)

        finally:
            await browser.close()

        # Generate report
        report = self._generate_report(results)

        # Save report
        self._save_report(report)

        # Display report
        self._display_report(report)

        return report

    def _generate_report(self, results: list[TaskResult]) -> EvaluationReport:
        """Generate evaluation report from results."""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        total_steps = sum(r.total_steps for r in results)
        total_duration = sum(r.duration_seconds for r in results)
        total_self_corrections = sum(r.self_corrections for r in results)
        total_popups = sum(r.popups_handled for r in results)

        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_tasks=total,
            successful_tasks=successful,
            failed_tasks=failed,
            success_rate=(successful / total * 100) if total > 0 else 0,
            average_steps=total_steps / total if total > 0 else 0,
            average_duration=total_duration / total if total > 0 else 0,
            total_self_corrections=total_self_corrections,
            total_popups_handled=total_popups,
            results=results
        )

    def _save_report(self, report: EvaluationReport):
        """Save report to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"evaluation_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Report saved to: {filepath}")

    def _display_report(self, report: EvaluationReport):
        """Display the evaluation report in the terminal."""
        logger.banner("Evaluation Report")

        # Summary table
        summary_table = Table(title="Summary", border_style="blue")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")

        summary_table.add_row("Total Tasks", str(report.total_tasks))
        summary_table.add_row("Successful", f"[green]{report.successful_tasks}[/green]")
        summary_table.add_row("Failed", f"[red]{report.failed_tasks}[/red]")
        summary_table.add_row("Success Rate", f"[bold]{report.success_rate:.1f}%[/bold]")
        summary_table.add_row("Avg Steps", f"{report.average_steps:.1f}")
        summary_table.add_row("Avg Duration", f"{report.average_duration:.1f}s")
        summary_table.add_row("Self-Corrections", str(report.total_self_corrections))
        summary_table.add_row("Popups Handled", str(report.total_popups_handled))

        console.print(summary_table)

        # Results table
        results_table = Table(title="Task Results", border_style="green")
        results_table.add_column("Task", style="cyan", max_width=25)
        results_table.add_column("Status", style="white")
        results_table.add_column("Steps", style="white")
        results_table.add_column("Self-Corr", style="white")
        results_table.add_column("Duration", style="white")

        for result in report.results:
            status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
            results_table.add_row(
                result.task_name[:25],
                status,
                str(result.total_steps),
                str(result.self_corrections),
                f"{result.duration_seconds:.1f}s"
            )

        console.print(results_table)

        # Print failures
        failures = [r for r in report.results if not r.success]
        if failures:
            console.print("\n[bold red]Failed Tasks:[/bold red]")
            for f in failures:
                console.print(f"  - {f.task_name}: {f.completion_reason}")
                if f.error:
                    console.print(f"    Error: {f.error}")


async def main():
    """Main entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Agentic Web Navigator evaluation")
    parser.add_argument("--tasks", type=str, help="Comma-separated task IDs to run")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--api-key", type=str, help="Google API key")
    parser.add_argument("--tasks-file", type=str, default="benchmarks/tasks.json", help="Path to tasks JSON")

    args = parser.parse_args()

    task_ids = args.tasks.split(",") if args.tasks else None

    runner = EvaluationRunner(
        tasks_file=Path(args.tasks_file),
        headless=args.headless,
        api_key=args.api_key
    )

    report = await runner.run_evaluation(task_ids)

    # Exit with appropriate code
    sys.exit(0 if report.success_rate >= 35 else 1)


if __name__ == "__main__":
    asyncio.run(main())
