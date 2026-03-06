"""Tool system — registry, executor, action model, and output parser.

This module is the heart of the v0.2.0 tool execution layer.

Design rules:
  - Agents never execute tools directly — only ToolExecutor can run tools.
  - Tools are registered once at startup via ToolRegistry.
  - Agent output must be strict JSON; parse_agent_output validates it.
  - All steps emit structured log events for full observability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, str]
    function: Callable[..., Any]

    def execute(self, args: Dict[str, Any]) -> Any:
        logger.info("event=tool_execute tool=%s args=%s", self.name, args)

        result = self.function(**args)

        logger.info(
            "event=tool_execute_complete tool=%s result_type=%s",
            self.name,
            type(result).__name__,
        )

        return result


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, str],
        function: Callable[..., Any],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        tool = Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            function=function,
        )
        self._tools[name] = tool

        logger.info("event=tool_registered tool=%s schema=%s", name, input_schema)

    def get(self, name: str) -> Tool:
        logger.info("event=tool_lookup tool=%s", name)

        if name not in self._tools:
            logger.error("event=tool_lookup_failed tool=%s", name)
            raise ValueError(f"Unknown tool: {name}")

        return self._tools[name]

    def list(self) -> list:
        logger.info("event=tool_list_requested count=%d", len(self._tools))
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# Agent Action Model
# ---------------------------------------------------------------------------


class AgentAction:
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
# Executor
# ---------------------------------------------------------------------------


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, action: AgentAction) -> Any:
        logger.info(
            "event=executor_received action=%s tool=%s",
            action.action,
            action.tool,
        )

        if action.action == "respond":
            logger.info("event=executor_direct_response")
            return action.content

        if action.action == "tool":
            tool = self.registry.get(action.tool)  # type: ignore[arg-type]
            result = tool.execute(action.args)

            logger.info(
                "event=tool_result tool=%s result_type=%s",
                action.tool,
                type(result).__name__,
            )

            return result

        logger.error("event=executor_unknown_action action=%s", action.action)
        raise ValueError(f"Unknown action type: {action.action}")


# ---------------------------------------------------------------------------
# Agent Output Parsing
# ---------------------------------------------------------------------------


def parse_agent_output(raw: str) -> AgentAction:
    """Parse a JSON string emitted by the agent into an AgentAction.

    Raises json.JSONDecodeError if *raw* is not valid JSON, or any other
    exception if the parsed structure is unusable.  The caller is responsible
    for graceful fallback.
    """
    logger.info("event=agent_output_received raw=%r", raw[:200] if raw else "")

    try:
        data = json.loads(raw)
        return AgentAction.from_dict(data)
    except Exception as exc:
        logger.error("event=agent_output_parse_error error=%r raw=%r", str(exc), raw[:200])
        raise
