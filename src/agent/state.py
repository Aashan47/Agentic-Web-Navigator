"""
Agent State for the LangGraph-based Web Navigator.
Defines the state schema that flows through the reasoning loop.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import json


class ToolName(str, Enum):
    """Available tools for the agent."""
    CLICK = "click_element"
    TYPE = "type_text"
    SCROLL = "scroll"
    WAIT = "wait"
    NAVIGATE = "navigate"
    GO_BACK = "go_back"
    PRESS_KEY = "press_key"
    SELECT_OPTION = "select_option"
    FINISH = "finish"


class ToolCall(BaseModel):
    """Structured tool call from the Oracle."""
    tool: ToolName
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reasoning: str = Field(default="")

    def to_dict(self) -> dict:
        return {
            "tool": self.tool.value,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


class ActionMemory(BaseModel):
    """Memory of a single action for context."""
    step: int
    thought: str
    tool_call: dict
    success: bool
    error: Optional[str] = None
    url: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_context_string(self) -> str:
        """Format for LLM context."""
        status = "SUCCESS" if self.success else f"FAILED: {self.error}"
        return f"Step {self.step}: {self.tool_call['tool']} -> {status}"


class ReflexionEntry(BaseModel):
    """Entry for the reflexion system when actions fail repeatedly."""
    failed_action: dict
    failure_count: int
    analysis: str
    suggested_alternative: Optional[dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentState(BaseModel):
    """
    The complete state that flows through the LangGraph.
    Contains all information needed for decision-making.
    """
    # Goal and progress
    goal: str
    current_step: int = 0
    max_steps: int = 50

    # Current observation
    current_url: str = ""
    page_title: str = ""
    screenshot_base64: Optional[str] = None
    accessibility_tree: str = ""
    interactive_elements: list[dict] = Field(default_factory=list)
    dom_hash: str = ""

    # Oracle output
    thought: str = ""
    tool_call: Optional[dict] = None

    # Action result
    action_success: bool = False
    action_error: Optional[str] = None
    action_traceback: Optional[str] = None

    # Memory (last N actions for context)
    action_memory: list[dict] = Field(default_factory=list)
    memory_limit: int = 5

    # Self-correction state
    consecutive_failures: int = 0
    max_consecutive_failures: int = 3
    reflexion_history: list[dict] = Field(default_factory=list)
    is_in_recovery_mode: bool = False
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3

    # Completion state
    is_complete: bool = False
    completion_reason: str = ""
    success: bool = False

    # Error tracking for specific elements
    failed_selectors: dict[str, int] = Field(default_factory=dict)

    # Metrics
    total_actions: int = 0
    successful_actions: int = 0
    self_corrections: int = 0
    popups_handled: int = 0

    class Config:
        arbitrary_types_allowed = True

    def add_to_memory(self, memory_entry: ActionMemory):
        """Add an action to memory, maintaining the limit."""
        self.action_memory.append(memory_entry.model_dump())
        if len(self.action_memory) > self.memory_limit:
            self.action_memory.pop(0)

    def get_memory_context(self) -> str:
        """Get formatted memory for LLM context."""
        if not self.action_memory:
            return "No previous actions."

        lines = ["Recent actions:"]
        for entry in self.action_memory:
            mem = ActionMemory(**entry)
            lines.append(f"  - {mem.to_context_string()}")
        return "\n".join(lines)

    def record_failure(self, selector: str = None):
        """Record a failure, tracking repeated failures on same element."""
        self.consecutive_failures += 1
        if selector:
            self.failed_selectors[selector] = self.failed_selectors.get(selector, 0) + 1

    def reset_failure_count(self):
        """Reset consecutive failure count on success."""
        self.consecutive_failures = 0
        self.is_in_recovery_mode = False
        self.recovery_attempts = 0

    def should_trigger_reflexion(self) -> bool:
        """Check if we should trigger reflexion due to repeated failures."""
        return self.consecutive_failures >= 2

    def is_element_problematic(self, selector: str) -> bool:
        """Check if an element has failed multiple times."""
        return self.failed_selectors.get(selector, 0) >= 2

    def to_llm_context(self) -> dict:
        """Convert state to context for LLM consumption."""
        return {
            "goal": self.goal,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "current_url": self.current_url,
            "page_title": self.page_title,
            "accessibility_tree": self.accessibility_tree[:8000],  # Limit size
            "memory": self.get_memory_context(),
            "consecutive_failures": self.consecutive_failures,
            "is_in_recovery_mode": self.is_in_recovery_mode,
            "last_error": self.action_error if not self.action_success else None,
        }


def create_initial_state(goal: str, max_steps: int = 50) -> AgentState:
    """Factory function to create initial agent state."""
    return AgentState(
        goal=goal,
        max_steps=max_steps,
        current_step=0
    )
