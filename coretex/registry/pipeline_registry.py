"""PipelineRegistry — registry for named execution pipelines."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """Holds named pipelines registered at startup.

    In v0.3.0 this registry is a placeholder foundation for configurable pipelines
    (introduced in v0.4.0). It stores arbitrary pipeline objects by name.

    All ``register`` calls raise ``ValueError`` on duplicate names.
    ``get`` raises ``ValueError`` on unknown names and emits a structured
    ``event=registry_lookup_failed`` log.
    """

    def __init__(self) -> None:
        self._pipelines: Dict[str, Any] = {}

    def register(self, name: str, pipeline: Any) -> None:
        """Register *pipeline* under *name*. Raises ValueError if already registered."""
        if name in self._pipelines:
            raise ValueError(f"Component already registered: {name}")
        self._pipelines[name] = pipeline
        logger.info("event=pipeline_registered name=%s", name)

    def get(self, name: str) -> Any:
        """Return the pipeline registered as *name*. Raises ValueError if unknown."""
        if name not in self._pipelines:
            logger.error("event=registry_lookup_failed component=pipeline name=%s", name)
            raise ValueError(f"Unknown component: {name}")
        return self._pipelines[name]

    def list(self) -> list[str]:
        """Return a list of all registered pipeline names."""
        return list(self._pipelines.keys())
