"""Runtime executor — agent action model, tool executor, and output parser.

This module is the v0.3.0 successor to core/tools.py.

Design rules (unchanged from v0.2.0):
  - Agents never execute tools directly — only ToolExecutor can run tools.
  - Agent output must be strict JSON; parse_agent_output validates it.
  - All steps emit structured log events for full observability.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from coretex.registry.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent Action Model
# ---------------------------------------------------------------------------


class AgentAction:
    """Represents a structured action emitted by the worker agent.

    Attributes:
        action: The action type — ``"respond"`` or ``"tool"``.
        tool: The tool name to call (only when ``action == "tool"``).
        args: Keyword arguments to pass to the tool function.
        content: The direct response content (only when ``action == "respond"``).
    """

    def __init__(
        self,
        action: Optional[str],
        tool: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> None:
        self.action = action
        self.tool = tool
        self.args = args or {}
        self.content = content

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentAction:
        """Construct an AgentAction from a parsed JSON dict."""
        logger.info(
            "event=agent_action_parsed action=%s tool=%s",
            data.get("action"),
            data.get("tool"),
        )
        return cls(
            action=data.get("action"),
            tool=data.get("tool"),
            args=data.get("args"),
            content=data.get("content"),
        )


# ---------------------------------------------------------------------------
# Tool Executor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """The only component that can run tools. Dispatches on AgentAction.action.

    Supported action types:
        ``respond`` — return ``action.content`` directly without executing any tool.
        ``tool``    — look up ``action.tool`` in the registry and call it with ``action.args``.

    Any unknown action type raises ``ValueError``.
    Tool lookup failure (unknown name) or tool runtime exceptions propagate to
    the caller; the ``PipelineRunner`` catches them and returns a graceful failure response.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, action: AgentAction, request_id: str = "") -> Any:
        """Execute *action* and return the result.

        Args:
            action: The parsed agent action to execute.
            request_id: Optional request ID for structured log correlation.

        Returns:
            For ``respond`` actions: the content string (may be ``None``).
            For ``tool`` actions: the return value of the tool function.

        Raises:
            ValueError: For unknown action types or missing tool names.
            ValueError: If the requested tool is not registered.
            Exception: Any exception raised by the tool function itself.
        """
        logger.info(
            "event=executor_received action=%s tool=%s request_id=%s",
            action.action,
            action.tool,
            request_id,
        )

        if action.action == "respond":
            logger.info("event=executor_direct_response request_id=%s", request_id)
            return action.content

        if action.action == "tool":
            if not action.tool:
                logger.error(
                    "event=executor_tool_name_missing request_id=%s",
                    request_id,
                )
                raise ValueError("Tool action is missing a tool name")

            tool = self.registry.get(action.tool)
            result = tool.execute(action.args, request_id=request_id)

            logger.info(
                "event=tool_result tool=%s request_id=%s result_type=%s",
                action.tool,
                request_id,
                type(result).__name__,
            )

            return result

        logger.error("event=executor_unknown_action action=%s request_id=%s", action.action, request_id)
        raise ValueError(f"Unknown action type: {action.action}")


# ---------------------------------------------------------------------------
# Agent Output Parsing
# ---------------------------------------------------------------------------


def parse_agent_output(raw: str, request_id: str = "") -> AgentAction:
    """Parse a JSON string emitted by the agent into an AgentAction.

    Raises json.JSONDecodeError if *raw* is not valid JSON, or any other
    exception if the parsed structure is unusable.  The caller is responsible
    for graceful fallback.

    Args:
        raw: The raw string output from the worker LLM.
        request_id: Optional request ID for structured log correlation.

    Returns:
        A parsed ``AgentAction`` instance.

    Raises:
        json.JSONDecodeError: If *raw* is not valid JSON.
    """
    logger.info(
        "event=agent_output_received request_id=%s raw=%r",
        request_id,
        raw[:200] if raw else "",
    )

    try:
        data = json.loads(raw)
        return AgentAction.from_dict(data)
    except Exception as exc:
        logger.error(
            "event=agent_output_parse_error request_id=%s error=%r raw=%r",
            request_id,
            str(exc),
            raw[:200],
        )
        raise
