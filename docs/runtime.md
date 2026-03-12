# COREtex Runtime

> Version: v0.5.x Model Provider System

---

## Overview

The COREtex runtime is the **core execution environment** of the CortX platform. It provides the primitives needed to execute intelligent request pipelines while remaining completely independent of any specific module implementation.

Runtime code lives in `coretex/`.

---

## Runtime Responsibilities

The runtime is responsible for exactly these systems:

| System | Description |
|--------|-------------|
| Execution lifecycle | Receiving requests, managing context, returning responses |
| Pipeline orchestration | Running the classifier → router → worker → executor flow |
| Module loading | Importing modules and calling their `register()` function |
| Registry management | Holding registered classifiers, routers, workers, tools, model providers |
| Event emission | Emitting structured log events for observability |
| Configuration | Loading settings from environment or `.env` file |

The runtime must **never** contain integrations, tools, model providers, or application logic.

---

## Module Architecture

Modules extend the runtime by registering capabilities at startup. The runtime then looks up these capabilities by name during pipeline execution.

Dependency direction is always:

```
modules → runtime
```

The runtime never imports from `modules/`. All coupling is through registry lookups.

### Module Structure

Each module lives in `modules/<module_name>/` and must expose a `module.py` file with a `register()` function:

```python
def register(
    module_registry: ModuleRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelProviderRegistry,
) -> None:
    ...
```

See `docs/module_development.md` for full details.

---

## Registries

Registries are the extension points between the runtime and modules.

| Registry | Location | Holds |
|----------|----------|-------|
| `ModuleRegistry` | `coretex/registry/module_registry.py` | Classifiers, Routers, Workers |
| `ToolRegistry` | `coretex/registry/tool_registry.py` | Tools |
| `ModelProviderRegistry` | `coretex/registry/model_registry.py` | Model backends |
| `PipelineRegistry` | `coretex/registry/pipeline_registry.py` | Named pipelines (v0.4+) |

### Registry Safety Rules

- `ToolRegistry` and `ModuleRegistry` use the generic component messages:
  - duplicate → `ValueError("Component already registered: <name>")`
  - unknown → `ValueError("Unknown component: <name>")` + `event=registry_lookup_failed`
- `ModelProviderRegistry` uses model-provider-specific messages:
  - duplicate → `ValueError("Model provider already registered: <name>")`
  - unknown → `ValueError("Unknown model provider: <name>")` + `event=registry_lookup_failed`
- `PipelineRegistry` uses pipeline-specific messages:
  - duplicate → `ValueError("Pipeline already registered: <name>")`
  - unknown → `ValueError("Unknown pipeline: <name>")` + `event=registry_lookup_failed`

---

## Pipeline Execution Flow

Every request follows this deterministic pipeline:

```
ExecutionContext created
    │
    ▼
event=request_received
    │
    ▼
classifier.classify(user_input)
    │
    ├─► event=classifier_start
    ├─► event=classifier_complete  (includes duration_ms, intent, confidence)
    │
    ▼
router.route(intent)
    │
    ├─► event=router_selected
    │
    ▼
[if handler == "clarify"]
    └─► return clarification response
    │
[if handler == "worker"]
    ├─► event=worker_start  (includes model_provider)
    ├─► worker.generate(user_input, intent)
    ├─► event=worker_complete  (includes duration_ms)
    │
    ├─► parse_agent_output(response)  → AgentAction
    │
    ├─► executor.execute(action)
    │     ├─► action=respond   → return content directly
    │     └─► action=tool      → event=tool_execute → tool.execute() → event=tool_execute_complete
    │
    ▼
event=request_complete  (includes intent, confidence, handler, all latencies)
```

### ExecutionContext

`ExecutionContext` carries per-request state through the pipeline:

```python
@dataclass
class ExecutionContext:
    user_input: str
    request_id: str         # auto-generated UUID hex
    intent: Optional[str]   # set after classification
    confidence: float       # set after classification
    handler: Optional[str]  # set after routing
    t_start: float          # monotonic timestamp at creation
    timestamp: float        # wall-clock time at creation
    metadata: Optional[Dict[str, Any]]  # optional module metadata
```

---

## Model Provider Flow

v0.5 formalises model access behind the `ModelProvider` interface:

```python
class ModelProvider(ABC):
    async def generate(self, model: str, prompt: str, **kwargs) -> str: ...
    async def chat(self, model: str, messages: list, **kwargs) -> str: ...
```

The default `model_provider_ollama` module registers `"ollama"` in `ModelProviderRegistry`. During bootstrap, `classifier_basic.module` and `worker_llm.module` retrieve that provider and inject it into `ClassifierBasic` and `WorkerLLM`.

This keeps the runtime deterministic: pipeline selection, routing, and tool execution stay unchanged while inference backends remain replaceable.

---

## Failure Behaviour

All failure modes are handled gracefully — no pipeline error produces an unhandled exception.

| Failure | Event Logged | Response |
|---------|-------------|----------|
| Classifier HTTP failure | `event=pipeline_classifier_failure` | Fallback: `intent=ambiguous`, clarification response |
| Worker HTTP failure | `event=pipeline_worker_failure` | Worker failure response, `intent=ambiguous` |
| Tool lookup failure | `event=pipeline_tool_failure` | Worker failure response |
| Tool runtime exception | `event=pipeline_tool_failure` | Worker failure response |
| Agent JSON parse failure | `event=pipeline_agent_parse_failure` | Raw LLM output treated as plain text |

---

## Structured Logging

All runtime log events use structured `key=value` format:

```
event=classifier_start request_id=abc123 classifier=classifier_basic model_provider=ollama
event=model_provider_chat_start request_id=abc123 model_provider=ollama model=llama3.2:3b
event=model_provider_chat_complete request_id=abc123 model_provider=ollama model=llama3.2:3b duration_ms=312
event=classifier_complete request_id=abc123 intent=execution confidence=0.92 duration_ms=312
event=router_selected request_id=abc123 intent=execution handler=worker
event=worker_start request_id=abc123 worker=worker_llm intent=execution model_provider=ollama
event=model_provider_generate_start request_id=abc123 model_provider=ollama model=llama3.2:3b
event=model_provider_generate_complete request_id=abc123 model_provider=ollama model=llama3.2:3b duration_ms=1450
event=worker_complete request_id=abc123 duration_ms=1450
event=request_complete request_id=abc123 intent=execution confidence=0.92 handler=worker classifier_latency_ms=312 worker_latency_ms=1450 total_latency_ms=1765
```

All log events include `request_id` for full request traceability.

---

## Configuration

Runtime configuration is loaded from environment variables or a `.env` file:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API endpoint |
| `CLASSIFIER_MODEL` | `llama3.2:3b` | Model used by the classifier |
| `WORKER_MODEL` | `llama3.2:3b` | Model used by the worker |
| `CLASSIFIER_TIMEOUT` | `60` | Classifier HTTP timeout (seconds) |
| `WORKER_TIMEOUT` | `300` | Worker HTTP timeout (seconds) |
| `MAX_TOKENS` | `256` | Maximum tokens per LLM response |
| `INGRESS_PORT` | `8000` | FastAPI ingress port |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `DEBUG_ROUTER` | `False` | Emit `event=router_decision` at DEBUG level |

---

## EventBus

The `EventBus` (`coretex/runtime/events.py`) provides structured log emission:

```python
event_bus.emit("classifier_complete", request_id=request_id, intent=intent, duration_ms=ms)
event_bus.emit_warning("module_registered_nothing", module="my_module")
event_bus.emit_error("pipeline_classifier_failure", request_id=request_id, error_type=...)
```

In v0.5.x the EventBus remains a structured log wrapper. A full async pub/sub event system is planned for a later phase.
