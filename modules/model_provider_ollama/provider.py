"""OllamaProvider — model provider implementation for the Ollama inference backend."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import httpx

from coretex.config.settings import settings
from coretex.interfaces.model_provider import (
    ModelProvider,
    ModelProviderConnectionError,
    ModelProviderResponseError,
    ModelProviderTimeoutError,
)

logger = logging.getLogger(__name__)


class OllamaProvider(ModelProvider):
    """Wraps the Ollama HTTP API for both generation and chat endpoints."""

    async def generate(self, model: str, prompt: str, **kwargs: Any) -> str:
        """Call Ollama /api/generate and return the response text.

        Explicit ``options`` keys take precedence over top-level helper kwargs
        such as ``num_predict``.
        """
        request_id = str(kwargs.get("request_id", ""))
        t_start = time.monotonic()
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        options = dict(kwargs.get("options", {}))
        options.setdefault("num_predict", kwargs.get("num_predict", settings.max_tokens))
        payload["options"] = options
        timeout = kwargs.get("timeout", settings.worker_timeout)
        logger.info(
            "event=model_provider_generate_start request_id=%s model_provider=ollama model=%s",
            request_id,
            model,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
                resp.raise_for_status()
                response_text = resp.json()["response"]
        except httpx.TimeoutException as exc:
            raise ModelProviderTimeoutError(
                f"Ollama generate timed out for model {model}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            body = ""
            try:
                body = exc.response.text[:200]
            except Exception:
                pass
            raise ModelProviderResponseError(
                f"Ollama generate failed for model {model}",
                status_code=exc.response.status_code,
                body=body,
            ) from exc
        except httpx.RequestError as exc:
            raise ModelProviderConnectionError(
                f"Ollama generate request failed for model {model}: {exc}"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderResponseError(
                f"Ollama generate returned an invalid payload for model {model}"
            ) from exc
        logger.info(
            "event=model_provider_generate_complete request_id=%s model_provider=ollama model=%s duration_ms=%d",
            request_id,
            model,
            int((time.monotonic() - t_start) * 1000),
        )
        return response_text

    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Call Ollama /api/chat and return the assistant message text."""
        request_id = str(kwargs.get("request_id", ""))
        t_start = time.monotonic()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if "format" in kwargs:
            payload["format"] = kwargs["format"]
        if "options" in kwargs:
            payload["options"] = kwargs["options"]

        timeout = kwargs.get("timeout", settings.classifier_timeout)
        logger.info(
            "event=model_provider_chat_start request_id=%s model_provider=ollama model=%s",
            request_id,
            model,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
                resp.raise_for_status()
                response_text = resp.json()["message"]["content"]
        except httpx.TimeoutException as exc:
            raise ModelProviderTimeoutError(
                f"Ollama chat timed out for model {model}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            body = ""
            try:
                body = exc.response.text[:200]
            except Exception:
                pass
            raise ModelProviderResponseError(
                f"Ollama chat failed for model {model}",
                status_code=exc.response.status_code,
                body=body,
            ) from exc
        except httpx.RequestError as exc:
            raise ModelProviderConnectionError(
                f"Ollama chat request failed for model {model}: {exc}"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderResponseError(
                f"Ollama chat returned an invalid payload for model {model}"
            ) from exc
        logger.info(
            "event=model_provider_chat_complete request_id=%s model_provider=ollama model=%s duration_ms=%d",
            request_id,
            model,
            int((time.monotonic() - t_start) * 1000),
        )
        return response_text
