"""ModelProviderRegistry — registry for model inference backend providers."""

from __future__ import annotations

import logging
from typing import Dict

from coretex.interfaces.model_provider import ModelProvider

logger = logging.getLogger(__name__)


class ModelProviderRegistry:
    """Holds model providers registered by modules at startup.

    All ``register`` calls raise ``ValueError`` on duplicate names.
    ``get`` raises ``ValueError`` on unknown names and emits a structured
    ``event=registry_lookup_failed`` log.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, ModelProvider] = {}

    def register(self, name: str, provider: ModelProvider) -> None:
        """Register *provider* under *name*. Raises ValueError if already registered."""
        if name in self._providers:
            raise ValueError(f"Component already registered: {name}")
        self._providers[name] = provider
        logger.info("event=model_provider_registered name=%s", name)

    def get(self, name: str) -> ModelProvider:
        """Return the provider registered as *name*. Raises ValueError if unknown."""
        if name not in self._providers:
            logger.error("event=registry_lookup_failed component=model_provider name=%s", name)
            raise ValueError(f"Unknown component: {name}")
        return self._providers[name]

    def list(self) -> list[str]:
        """Return a list of all registered provider names."""
        return list(self._providers.keys())
