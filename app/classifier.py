"""Classifier agent — calls the LLM to categorise user intent.

Returns strict JSON: {"intent": <str>, "confidence": <float>}

Uses the Ollama /api/chat endpoint with a system message so that
instruction-tuned models follow the schema reliably.

Retries once if the response is not valid JSON; falls back to
intent="ambiguous", confidence=0.0 on second failure.
"""

import json
import logging
from typing import Optional

import httpx
from pydantic import ValidationError

from app.models import ClassifierResponse
from app.settings import settings

logger = logging.getLogger(__name__)

# System message sent to the model as the authoritative instruction.
# User input is passed as a separate user turn — never interpolated here.
_SYSTEM_PROMPT = """\
You are an intent classifier. Given a user input, reply with ONLY a JSON object — no prose, no markdown.
Use exactly this schema: {"intent": "<category>", "confidence": <0.0-1.0>}

Categories:
- execution   : a concrete task with a specific deliverable (write, translate, summarise, calculate, code)
- decomposition: a broad HOW-TO request needing a multi-step plan (how to build, steps to launch, plan X)
- novel_reasoning: open-ended thinking, design, or analysis with no single correct answer
- ambiguous   : too vague — a fragment, single word, or greeting with no clear task

Rules (apply in order — first match wins):
1. If the request says "write", "generate", "create", "compose", "draft", or "produce" followed by a specific artifact (poem, haiku, essay, story, function, script, email, list) → execution, regardless of how creative the artifact is.
2. "Summarise / explain / describe X in N sentences/words/paragraphs" → execution.
3. "How do I / What steps / How would I / Plan / Walk me through" → decomposition.
4. "Design / Compare / Analyse / Evaluate / What if / What are the implications" → novel_reasoning.
5. Single word, greeting, or fragment with no clear task → ambiguous."""

# Maps common LLM-generated intent variants to valid schema values.
_INTENT_ALIASES: dict[str, str] = {
    "creative_writing": "execution",
    "creative": "execution",
    "generation": "execution",
    "task": "execution",
    "action": "execution",
    "command": "execution",
    "analysis": "novel_reasoning",
    "reasoning": "novel_reasoning",
    "explanation": "novel_reasoning",
    "synthesis": "novel_reasoning",
    "planning": "decomposition",
    "complex": "decomposition",
    "unclear": "ambiguous",
    "unknown": "ambiguous",
    "other": "ambiguous",
}

# Alternative field names some models use instead of "intent".
_INTENT_FIELD_CANDIDATES = ("intent", "category", "type", "classification", "class")


async def classify(user_input: str) -> ClassifierResponse:
    """Classify *user_input* and return a ClassifierResponse.

    Retries once on bad JSON or network error; returns fallback on second failure.
    """
    for attempt in range(2):
        try:
            raw = await _call_ollama(user_input)
        except httpx.HTTPError as exc:
            body = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.text[:200]
                except Exception:
                    pass
            logger.warning(
                "Classifier attempt %d: %s body=%r %s",
                attempt + 1, type(exc).__name__, body, exc,
            )
            continue
        result = _parse(raw)
        if result is not None:
            return result
        logger.warning("Classifier attempt %d: could not parse response: %r", attempt + 1, raw)

    logger.error("Classifier failed after 2 attempts; returning fallback.")
    return ClassifierResponse(intent="ambiguous", confidence=0.0)


async def _call_ollama(user_input: str) -> str:
    """Call Ollama /api/chat with a system message + user turn."""
    payload = {
        "model": settings.classifier_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 64},
    }
    logger.info("LLM call 1/2: classifier model=%s", settings.classifier_model)
    async with httpx.AsyncClient(timeout=settings.classifier_timeout) as client:
        resp = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        logger.debug("Classifier raw response: %r", raw)
        return raw


def _parse(raw: str) -> Optional[ClassifierResponse]:
    text = raw.strip()
    # Strip markdown code fences in case a model ignores the format constraint.
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    # Try strict parse first.
    try:
        return ClassifierResponse(**data)
    except ValidationError:
        pass

    # Normalise: find the intent value under any common field name, then map aliases.
    raw_intent = None
    for field in _INTENT_FIELD_CANDIDATES:
        if field in data:
            raw_intent = str(data[field]).strip().lower().replace(" ", "_").replace("-", "_")
            break

    if raw_intent is None:
        logger.warning("Classifier response has no recognisable intent field: %r", data)
        return None

    intent = _INTENT_ALIASES.get(raw_intent, raw_intent)
    confidence = float(data.get("confidence", data.get("score", data.get("certainty", 0.5))))

    try:
        return ClassifierResponse(intent=intent, confidence=confidence)
    except ValidationError:
        logger.warning(
            "Classifier intent %r (normalised from %r) not in schema; treating as ambiguous",
            intent, raw_intent,
        )
        return ClassifierResponse(intent="ambiguous", confidence=0.0)

