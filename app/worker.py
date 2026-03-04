"""Worker agent — calls the LLM to produce the final user-facing response.

Receives original user input and classifier intent; returns free-form text.
No memory, no tools.
"""

import logging

import httpx

from app.settings import settings

logger = logging.getLogger(__name__)

# Intent-specific prompt templates; user input is appended via concatenation to
# avoid Python str.format() treating user-supplied braces as placeholders.
_PROMPTS: dict[str, str] = {
    "execution": (
        "You are a precise assistant. The user has asked you to perform a concrete task.\n"
        "Give a direct, concise answer. No planning structure. No preamble.\n\n"
        "User request: "
    ),
    "decomposition": (
        "You are a structured assistant. The user has a complex task that needs breaking into steps.\n"
        "Provide a clear, numbered breakdown of the steps or phases required.\n"
        "Be thorough but concise.\n\n"
        "User request: "
    ),
    "novel_reasoning": (
        "You are a creative, analytical assistant. The user wants open-ended analysis or synthesis.\n"
        "Explore the topic creatively. You may speculate, consider multiple angles, "
        "or apply system-design style thinking.\n\n"
        "User request: "
    ),
}

_FALLBACK_PROMPT = _PROMPTS["execution"]


async def generate(user_input: str, intent: str) -> str:
    """Generate a response for *user_input* given *intent*.

    Raises httpx.HTTPError on network or HTTP failures (caller handles gracefully).
    """
    prompt = _PROMPTS.get(intent, _FALLBACK_PROMPT) + user_input
    payload = {
        "model": settings.worker_model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": settings.max_tokens},
    }
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]
