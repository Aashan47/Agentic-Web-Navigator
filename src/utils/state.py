"""
Global State management for the Agentic Web Navigator.
Tracks action history, detects infinite loops, and manages navigation state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum
from collections import deque
import hashlib
import json


class ActionType(str, Enum):
    """Types of actions the agent can perform."""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    NAVIGATE = "navigate"
    SCREENSHOT = "screenshot"
    FINISH = "finish"
    BACK = "back"
    REFRESH = "refresh"


class ActionStatus(str, Enum):
    """Status of an action execution."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


@dataclass
class ActionRecord:
    """Record of a single action taken by the agent."""
    action_type: ActionType
    parameters: dict[str, Any]
    status: ActionStatus
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    retry_count: int = 0
    url_before: Optional[str] = None
    url_after: Optional[str] = None
    thought: Optional[str] = None  # Agent's reasoning for this action

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action_type": self.action_type.value,
            "parameters": self.parameters,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "url_before": self.url_before,
            "url_after": self.url_after,
            "thought": self.thought,
        }

    def action_signature(self) -> str:
        """Generate a unique signature for loop detection."""
        sig_data = f"{self.action_type.value}:{json.dumps(self.parameters, sort_keys=True)}"
        return hashlib.md5(sig_data.encode()).hexdigest()[:16]


@dataclass
class PageState:
    """Snapshot of the current page state."""
    url: str
    title: str
    dom_hash: str  # Hash of simplified DOM for change detection
    screenshot_path: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    interactive_elements_count: int = 0


class GlobalState:
    """
    Global state manager for the web navigation agent.
    Tracks history, detects loops, and manages navigation state.
    """

    def __init__(
        self,
        max_history: int = 100,
        loop_detection_window: int = 10,
        max_same_action_count: int = 3
    ):
        self.max_history = max_history
        self.loop_detection_window = loop_detection_window
        self.max_same_action_count = max_same_action_count

        # Action history (deque for efficient pop from front)
        self.action_history: deque[ActionRecord] = deque(maxlen=max_history)

        # Page state history
        self.page_states: deque[PageState] = deque(maxlen=50)

        # Current state
        self.current_goal: Optional[str] = None
        self.current_step: int = 0
        self.total_errors: int = 0
        self.start_time: Optional[datetime] = None
        self.is_finished: bool = False
        self.finish_reason: Optional[str] = None

        # Loop detection
        self._action_signature_counts: dict[str, int] = {}
        self._recent_signatures: deque[str] = deque(maxlen=loop_detection_window)

    def start_session(self, goal: str):
        """Initialize a new navigation session."""
        self.current_goal = goal
        self.current_step = 0
        self.total_errors = 0
        self.start_time = datetime.now()
        self.is_finished = False
        self.finish_reason = None
        self.action_history.clear()
        self.page_states.clear()
        self._action_signature_counts.clear()
        self._recent_signatures.clear()

    def record_action(
        self,
        action_type: ActionType,
        parameters: dict[str, Any],
        status: ActionStatus,
        error_message: Optional[str] = None,
        url_before: Optional[str] = None,
        url_after: Optional[str] = None,
        thought: Optional[str] = None
    ) -> ActionRecord:
        """Record an action and update state."""
        self.current_step += 1

        record = ActionRecord(
            action_type=action_type,
            parameters=parameters,
            status=status,
            error_message=error_message,
            url_before=url_before,
            url_after=url_after,
            thought=thought
        )

        self.action_history.append(record)

        if status == ActionStatus.FAILED:
            self.total_errors += 1

        # Update loop detection
        sig = record.action_signature()
        self._recent_signatures.append(sig)
        self._action_signature_counts[sig] = self._action_signature_counts.get(sig, 0) + 1

        return record

    def record_page_state(
        self,
        url: str,
        title: str,
        dom_hash: str,
        screenshot_path: Optional[str] = None,
        interactive_elements_count: int = 0
    ) -> PageState:
        """Record the current page state."""
        state = PageState(
            url=url,
            title=title,
            dom_hash=dom_hash,
            screenshot_path=screenshot_path,
            interactive_elements_count=interactive_elements_count
        )
        self.page_states.append(state)
        return state

    def detect_loop(self) -> tuple[bool, Optional[str]]:
        """
        Detect if the agent is stuck in a loop.
        Returns (is_loop, reason).
        """
        if len(self._recent_signatures) < 3:
            return False, None

        # Check for repeated identical actions
        recent_list = list(self._recent_signatures)
        for sig in set(recent_list[-self.loop_detection_window:]):
            count = recent_list.count(sig)
            if count >= self.max_same_action_count:
                return True, f"Same action repeated {count} times in last {self.loop_detection_window} actions"

        # Check for oscillating pattern (A-B-A-B)
        if len(recent_list) >= 4:
            last_four = recent_list[-4:]
            if last_four[0] == last_four[2] and last_four[1] == last_four[3] and last_four[0] != last_four[1]:
                return True, "Detected oscillating action pattern (A-B-A-B)"

        # Check for page state not changing
        if len(self.page_states) >= 3:
            recent_states = list(self.page_states)[-3:]
            if all(s.dom_hash == recent_states[0].dom_hash for s in recent_states):
                # Check if actions were taken but nothing changed
                recent_actions = list(self.action_history)[-3:]
                if all(a.status == ActionStatus.SUCCESS for a in recent_actions):
                    return True, "Page state unchanged after multiple successful actions"

        return False, None

    def mark_finished(self, reason: str, success: bool = True):
        """Mark the navigation session as finished."""
        self.is_finished = True
        self.finish_reason = reason
        status = "SUCCESS" if success else "INCOMPLETE"
        return f"[{status}] {reason}"

    def get_history_summary(self, last_n: int = 10) -> list[dict]:
        """Get a summary of recent actions for the LLM context."""
        recent = list(self.action_history)[-last_n:]
        return [action.to_dict() for action in recent]

    def get_failed_actions(self) -> list[ActionRecord]:
        """Get all failed actions for analysis."""
        return [a for a in self.action_history if a.status == ActionStatus.FAILED]

    def get_session_stats(self) -> dict:
        """Get statistics about the current session."""
        elapsed = None
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        success_count = sum(
            1 for a in self.action_history if a.status == ActionStatus.SUCCESS
        )
        failed_count = sum(
            1 for a in self.action_history if a.status == ActionStatus.FAILED
        )

        return {
            "goal": self.current_goal,
            "total_steps": self.current_step,
            "successful_actions": success_count,
            "failed_actions": failed_count,
            "total_errors": self.total_errors,
            "elapsed_seconds": elapsed,
            "is_finished": self.is_finished,
            "finish_reason": self.finish_reason,
            "pages_visited": len(set(s.url for s in self.page_states)),
        }

    def should_stop(self, max_steps: int) -> tuple[bool, Optional[str]]:
        """Check if the agent should stop execution."""
        if self.is_finished:
            return True, self.finish_reason

        if self.current_step >= max_steps:
            return True, f"Maximum steps ({max_steps}) reached"

        is_loop, loop_reason = self.detect_loop()
        if is_loop:
            return True, f"Loop detected: {loop_reason}"

        if self.total_errors >= 10:
            return True, "Too many errors accumulated"

        return False, None


# Global state instance
global_state = GlobalState()
