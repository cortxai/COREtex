"""ModelProvider interface — contract for model inference backend modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class ModelProviderError(Exception):
    """Base exception for model provider failures.

    All concrete model providers should raise ``ModelProviderError`` subclasses
    so the pipeline can handle backend failures without depending on transport-
    specific exception types.
    """


class ModelProviderConnectionError(ModelProviderError):
    """Raised when the provider cannot reach its backend service."""


class ModelProviderTimeoutError(ModelProviderError):
    """Raised when the backend does not respond within the configured timeout."""


class ModelProviderResponseError(ModelProviderError):
    """Raised when the backend returns an HTTP or payload-level error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ModelProvider(ABC):
    """Abstract base class for model provider modules.

    Implementations wrap a specific inference backend (e.g. Ollama, OpenAI).
    """

    @abstractmethod
    async def generate(self, model: str, prompt: str, **kwargs: object) -> str:
        """Run a single-turn generation and return the response text."""
        ...

    @abstractmethod
    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs: object) -> str:
        """Run a multi-turn chat completion and return the assistant message text."""
        ...
