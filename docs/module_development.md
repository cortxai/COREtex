# Module Development Guide

> Version: v0.3.x Stabilisation

---

## Overview

Modules are the **extensibility mechanism** of COREtex. They implement platform capabilities and register themselves with the runtime registries at startup.

The runtime never imports from `modules/` directly — all coupling flows through registry lookups.

---

## Module Directory Structure

Each module lives in its own directory under `modules/`:

```
modules/
  my_module/
    __init__.py
    module.py       ← required: registration entrypoint
    <impl>.py       ← implementation file(s)
```

The `module.py` file is the only **required** file. It must expose a `register()` function.

---

## The `register()` Function

Every module must expose a top-level `register()` function in `module.py` with exactly this signature:

```python
from coretex.registry.model_registry import ModelProviderRegistry
from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry


def register(
    module_registry: ModuleRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelProviderRegistry,
) -> None:
    ...
```

The `ModuleLoader` validates this signature at load time. Modules that do not accept all three parameters will be rejected with `ValueError("Invalid module register() signature")`.

You may safely ignore registries your module does not use.

---

## Registering Components

### Register a Classifier

```python
from modules.my_module.classifier import MyClassifier

def register(module_registry, tool_registry, model_registry):
    module_registry.register_classifier("my_classifier", MyClassifier())
```

The classifier must implement `coretex.interfaces.classifier.Classifier`:

```python
from coretex.interfaces.classifier import Classifier, ClassificationResult

class MyClassifier(Classifier):
    async def classify(self, text: str, request_id: str = "") -> ClassificationResult:
        ...
        return ClassificationResult(intent="execution", confidence=0.95)
```

### Register a Router

```python
from modules.my_module.router import MyRouter

def register(module_registry, tool_registry, model_registry):
    module_registry.register_router("my_router", MyRouter())
```

The router must implement `coretex.interfaces.router.Router`:

```python
from coretex.interfaces.router import Router

class MyRouter(Router):
    def route(self, intent: str, request_id: str = "", **kwargs) -> str:
        # Return a handler name: "worker" or "clarify"
        return "worker"
```

### Register a Worker

```python
from modules.my_module.worker import MyWorker

def register(module_registry, tool_registry, model_registry):
    module_registry.register_worker("my_worker", MyWorker())
```

The worker must implement `coretex.interfaces.worker.Worker`:

```python
from coretex.interfaces.worker import Worker

class MyWorker(Worker):
    async def generate(self, text: str, intent: str = "", request_id: str = "") -> str:
        # Return a JSON action envelope or plain text
        return '{"action": "respond", "content": "Hello"}'
```

### Register a Tool

```python
def register(module_registry, tool_registry, model_registry):
    tool_registry.register(
        name="my_tool",
        description="Does something useful",
        input_schema={"param": "string"},
        function=my_tool_function,
    )

def my_tool_function(param: str) -> str:
    return f"Result: {param}"
```

### Register a Model Provider

```python
from modules.my_module.provider import MyProvider

def register(module_registry, tool_registry, model_registry):
    model_registry.register("my_provider", MyProvider())
```

The provider must implement `coretex.interfaces.model_provider.ModelProvider`:

```python
from coretex.interfaces.model_provider import ModelProvider

class MyProvider(ModelProvider):
    async def generate(self, prompt: str, **kwargs) -> str: ...
    async def chat(self, messages: list, **kwargs) -> str: ...
```

---

## Loading a Module

Modules are loaded by the `ModuleLoader` at distribution bootstrap time.

### Single module

```python
from coretex.runtime.loader import ModuleLoader

loader = ModuleLoader(module_registry, tool_registry, model_registry)
loader.load("modules.my_module.module")
```

### Multiple modules (recommended)

```python
loader.load_all([
    "modules.classifier_basic.module",
    "modules.router_simple.module",
    "modules.worker_llm.module",
    "modules.tools_filesystem.module",
])
```

`load_all()` emits `event=module_loading_start` and `event=module_loading_complete` lifecycle events.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Module 'x' has no register() function` | `module.py` is missing `register()` | Add the function |
| `Invalid module register() signature` | `register()` doesn't accept all three registry params | Fix the function signature |
| `Component already registered: <name>` | Two modules try to register the same name | Use unique component names |
| `ImportError` | Module path is wrong | Check the dotted module path |

---

## Best Practices

1. **Keep modules small** — each module should register one logical set of components.
2. **Don't modify runtime state directly** — always register through the registry APIs.
3. **Use unique component names** — namespacing by module is a safe convention (e.g. `"classifier_basic"`, `"router_simple"`).
4. **Never import from the runtime's private state** — use only the public registry APIs.
5. **Log events consistently** — use structured `event=<name> key=value` format.
6. **Modules may access settings** — import `coretex.config.settings.settings` for shared configuration.

---

## Example Module

`modules/my_classifier/module.py`:

```python
"""my_classifier — example classifier module."""

from coretex.registry.model_registry import ModelProviderRegistry
from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry
from modules.my_classifier.classifier import MyClassifier


def register(
    module_registry: ModuleRegistry,
    tool_registry: ToolRegistry,
    model_registry: ModelProviderRegistry,
) -> None:
    module_registry.register_classifier("my_classifier", MyClassifier())
```

`modules/my_classifier/classifier.py`:

```python
"""MyClassifier — a simple example classifier."""

from coretex.interfaces.classifier import Classifier, ClassificationResult


class MyClassifier(Classifier):
    async def classify(self, text: str, request_id: str = "") -> ClassificationResult:
        # Simple keyword-based classification
        if "run" in text.lower() or "execute" in text.lower():
            return ClassificationResult(intent="execution", confidence=0.95)
        return ClassificationResult(intent="ambiguous", confidence=0.0)
```
