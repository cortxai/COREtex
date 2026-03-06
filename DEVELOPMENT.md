# Development Guide

This guide is written for developers who want to extend or modify CortX. It covers the architecture, core components, and how to add new tools, routes, and capabilities.

---

## Architecture overview

CortX is a FastAPI service that acts as an orchestration layer between a user interface, a set of local LLMs (via Ollama), and a tool execution layer.

```
User input
    │
    ▼
POST /ingest  (app/main.py)
    │
    ├── Classifier (app/classifier.py)
    │     Deterministic prefix check → if matched, skip LLM
    │     LLM call 1/2 → {"intent": "...", "confidence": 0.9}
    │
    ├── Router (app/router.py)
    │     Pure Python dict lookup → "worker" | "clarify"
    │
    └── Worker (app/worker.py)          [only if handler == "worker"]
          Intent-aware prompt selection
          LLM call 2/2 → JSON action envelope
              │
              ▼
         parse_agent_output (core/tools.py)
              │
              ▼
         ToolExecutor.execute (core/tools.py)
              │
         ┌────┴─────┐
         │           │
     "respond"    "tool"
         │           │
     return         ToolRegistry.get(name)
     content             │
                    Tool.execute(args)
                         │
                    return result
```

**Key design rules:**
- Agents (LLMs) never execute tools directly — only `ToolExecutor` can.
- The router is pure Python — no LLM calls, no probabilistic decisions.
- Each request is stateless — no memory, no conversation history.
- `POST /ingest` is the single orchestration entry point.
- Exactly 2 LLM calls per non-ambiguous request (classifier + worker).

---

## File structure

```
app/
  __init__.py       — empty package marker
  main.py           — FastAPI app, /ingest, /v1/chat/completions, /debug/routes, /health
  models.py         — Pydantic schemas: ClassifierResponse, IngestRequest, IngestResponse
  classifier.py     — Intent classifier: prefix checks + LLM call
  router.py         — Deterministic intent→handler mapping (pure Python dict)
  worker.py         — Worker agent: prompt templates + LLM call
  settings.py       — Pydantic-settings config loaded from env vars or .env

core/
  __init__.py       — empty package marker
  tools.py          — Tool, ToolRegistry, AgentAction, ToolExecutor, parse_agent_output

tools/
  __init__.py       — empty package marker
  filesystem.py     — read_file tool

bootstrap_tools.py  — creates ToolRegistry singleton and registers all tools
tests/
  test_smoke.py     — full test suite (unit + integration, no Ollama required)

Dockerfile          — python:3.11-slim, runs uvicorn on port 8000
docker-compose.yml  — ingress + openwebui on an isolated bridge network
requirements.txt    — all Python dependencies
pytest.ini          — sets pythonpath = . so imports resolve without install
```

---

## Core components

### `app/classifier.py`

Determines the intent of the user's request in two stages:

1. **Deterministic prefix check** — lowercases the input and checks it against hard-coded tuples (`_EXECUTION_PREFIXES`, `_PLANNING_PREFIXES`, `_AMBIGUOUS_SHORT`). If matched, returns immediately without any LLM call. This eliminates the most common misclassifications cheaply.

2. **LLM call** — sends a structured system prompt + the user's message to Ollama `/api/chat`. Retries once on failure; falls back to `intent="ambiguous", confidence=0.0` after two failures.

The `_parse()` function normalises the LLM's JSON response: strips markdown fences, applies alias maps, handles alternative field names, lowercases values.

### `app/router.py`

A single dict lookup:

```python
ROUTES = {
    "execution": "worker",
    "planning":  "worker",
    "analysis":  "worker",
    "ambiguous": "clarify",
}
```

`route()` returns the handler name as a string. Unknown intents fall back to `"clarify"`. No LLM, no probabilistic logic.

### `app/worker.py`

Selects an intent-aware prompt template from `_PROMPTS`, appends the user input via string concatenation (never `str.format()` — user-supplied `{braces}` would cause `KeyError`), and calls Ollama `/api/generate`.

Worker prompts instruct the LLM to return a JSON action envelope:
```json
{"action": "respond", "content": "..."}
```
or a tool call:
```json
{"action": "tool", "tool": "<name>", "args": {...}}
```

### `core/tools.py`

The tool execution layer. Contains:

- **`Tool`** — a dataclass wrapping a Python callable with metadata (name, description, schema).
- **`ToolRegistry`** — a dict-based registry. Tools are registered by name; duplicate names raise `ValueError`.
- **`AgentAction`** — a typed representation of the agent's JSON output (action, tool, args, content).
- **`ToolExecutor`** — the only component that can run tools. Dispatches on `action.action`: `"respond"` returns content directly; `"tool"` looks up the tool in the registry and calls it.
- **`parse_agent_output(raw)`** — parses a raw JSON string into an `AgentAction`. Raises `json.JSONDecodeError` on invalid JSON; all errors are logged before re-raising.

### `bootstrap_tools.py`

Creates the `tool_registry` singleton and registers all tools. Imported by `app/main.py` at module load. `ToolExecutor` in `main.py` wraps this registry.

### `app/settings.py`

Pydantic `BaseSettings` — all values overridable via environment variable or `.env` file. A singleton `settings` is imported by all modules.

---

## Logging conventions

All log lines use `event=<name>` as the first field, followed by relevant key-value pairs, using Python's stdlib `logging` with `%s` format strings:

```python
logger.info("event=tool_execute tool=%s args=%s", self.name, args)
```

Standard log events in sequence for a tool call request:

```
event=request_received
event=classifier_result
event=intent_router
event=agent_selected
event=worker_start
event=llm_call
event=worker_complete
event=agent_output_received
event=agent_action_parsed
event=executor_received
event=tool_lookup
event=tool_execute
event=tool_execute_complete
event=tool_result
event=request_complete
```

Set `DEBUG_ROUTER=true` to additionally log `classifier_prompt` and `worker_prompt` at DEBUG level.

---

## How to add a new tool

### 1. Implement the tool function

Create a new file in `tools/` (or add to an existing one):

```python
# tools/web.py

import httpx


def fetch_url(url: str) -> str:
    """Fetch the text content of a URL."""
    response = httpx.get(url, timeout=10)
    response.raise_for_status()
    return response.text[:2000]  # truncate for safety
```

Tools must:
- Accept only keyword arguments (the executor calls `function(**args)`).
- Return a value (typically a string, but any serialisable type works).
- Handle their own errors gracefully when possible (return an error string rather than raising).

### 2. Register the tool in `bootstrap_tools.py`

```python
from tools.web import fetch_url

tool_registry.register(
    name="fetch_url",
    description="Fetch the text content of a web page",
    input_schema={"url": "string"},
    function=fetch_url,
)
```

### 3. Update the worker prompt (optional)

To make the LLM aware of the new tool, add a description to the relevant worker prompt in `app/worker.py`. For example, you could add a tools section after the JSON format instruction:

```python
"Available tools:\n"
"- read_file: read a local file. Args: {\"path\": \"<filepath>\"}\n"
"- fetch_url: fetch a web page. Args: {\"url\": \"<url>\"}\n\n"
```

### 4. Add tests

Add unit tests in `tests/test_smoke.py`:

```python
def test_fetch_url_tool():
    from tools.web import fetch_url
    # mock httpx if needed, or test with a known URL
    ...

def test_bootstrap_registers_fetch_url():
    from bootstrap_tools import tool_registry
    assert "fetch_url" in tool_registry.list()
```

---

## How to add a new intent route

### 1. Add the new intent to `ClassifierResponse` in `app/models.py`

```python
class ClassifierResponse(BaseModel):
    intent: Literal["execution", "planning", "analysis", "ambiguous", "retrieval"]
    confidence: float
```

### 2. Update `ROUTES` in `app/router.py`

```python
ROUTES: dict[str, str] = {
    "execution": "worker",
    "planning":  "worker",
    "analysis":  "worker",
    "retrieval": "worker",   # ← new
    "ambiguous": "clarify",
}
```

### 3. Add a prompt template in `app/worker.py`

```python
_PROMPTS: dict[str, str] = {
    ...
    "retrieval": (
        "You are a retrieval assistant. Search your knowledge for the most relevant information.\n"
        "Return the top 3 facts. Be precise.\n\n"
        'You MUST respond with valid JSON: {"action": "respond", "content": "..."}\n\n'
        "User request: "
    ),
}
```

### 4. Update the classifier system prompt in `app/classifier.py`

Add the new intent to `_SYSTEM_PROMPT` with a definition and examples. Also add relevant aliases to `_INTENT_ALIASES` if needed.

### 5. Add prefix checks (optional)

If the intent has reliable keyword prefixes, add them to the appropriate tuple in `app/classifier.py` (e.g. add to `_EXECUTION_PREFIXES` or create a new tuple and check).

---

## How to add a new API endpoint

Add a route to `app/main.py` following the existing pattern:

```python
@app.get("/my-endpoint")
async def my_endpoint() -> dict:
    """One-line docstring."""
    return {"key": "value"}
```

All endpoints should return typed dicts or Pydantic models. Log the event at entry and exit.

---

## Design principles and constraints

### Agents never execute tools directly
The `ToolExecutor` is the single point of tool execution. This keeps the system safe, auditable, and easy to reason about. Never call a tool function directly from the worker or any agent code — always go through the executor.

### Two LLM calls per request maximum
The pipeline is intentionally two-stage: classifier (call 1) + worker (call 2). Ambiguous requests skip the worker entirely. Do not add LLM calls to the router, the executor, or any middleware.

### Router is pure Python
The `route()` function must never call an LLM, perform I/O, or make probabilistic decisions. It is a pure dict lookup. This makes routing deterministic and instantly testable.

### Stateless requests
Each request starts fresh. There is no session state, no conversation history, no shared mutable state between requests. If memory is needed, it belongs to a future phase.

### Worker prompt concatenation
User input is always **appended** to the prompt string via `+`. Never use `str.format()` or f-strings with user input — user-supplied `{braces}` would cause `KeyError`.

### Settings via `app/settings.py`
All configurable values live in `Settings(BaseSettings)`. Never hardcode URLs, model names, timeouts, or token limits inline. Add new settings here and document them in `README.md`.

### Graceful failure over 5xx
The HTTP layer should never return a 5xx response to the user. Catch `httpx.HTTPError` and tool execution errors at the `main.py` level and return HTTP 200 with the `_WORKER_FAILURE_RESPONSE` string and `intent="ambiguous", confidence=0.0`.

### Structured logging
Every event must emit `event=<name>` at INFO level (or DEBUG for verbose output). Thread `request_id` through all calls. This makes every request fully traceable in logs.

---

## Running the development server

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Enable debug logging:
```bash
DEBUG_ROUTER=true LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

---

## Running tests

```bash
pytest tests/test_smoke.py -v
```

Run a single test:
```bash
pytest tests/test_smoke.py::test_executor_tool_action_executes_tool -v
```

All tests mock Ollama. No running services are required.

---

## What comes next (planned phases)

- **Task Graph / Planner** — multi-step orchestration where the planner decomposes requests into a DAG of sub-tasks, each handled by a specialised agent.
- **Agent collaboration** — multiple agents working on sub-tasks in parallel.
- **Memory layer** — optional context injection for session continuity.
- **Additional tools** — web fetch, shell execution, vector search, database queries, API calls.
- **Streaming responses** — progressive output for long-running tasks.
