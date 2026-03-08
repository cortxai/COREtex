"""Bootstrap — load all modules and build the shared registries for cortx_local."""

from __future__ import annotations

from cortx.registry.model_registry import ModelProviderRegistry
from cortx.registry.module_registry import ModuleRegistry
from cortx.registry.tool_registry import ToolRegistry
from cortx.runtime.loader import ModuleLoader

# ---------------------------------------------------------------------------
# Registries (singletons shared across the application)
# ---------------------------------------------------------------------------

module_registry = ModuleRegistry()
tool_registry = ToolRegistry()
model_registry = ModelProviderRegistry()

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------

_loader = ModuleLoader(
    module_registry=module_registry,
    tool_registry=tool_registry,
    model_registry=model_registry,
)

_loader.load("modules.model_provider_ollama.module")
_loader.load("modules.classifier_basic.module")
_loader.load("modules.router_simple.module")
_loader.load("modules.worker_llm.module")
_loader.load("modules.tools_filesystem.module")
