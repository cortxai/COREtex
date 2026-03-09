"""ModuleLoader — loads modules by dotted path and calls their register() function."""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import List, Optional

from coretex.registry.model_registry import ModelProviderRegistry
from coretex.registry.module_registry import ModuleRegistry
from coretex.registry.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Expected parameter names for the register() function signature.
_REGISTER_PARAMS = {"module_registry", "tool_registry", "model_registry"}


class ModuleLoader:
    """Loads modules dynamically and registers their capabilities with the runtime.

    Each module must expose a top-level ``register()`` function with the signature::

        def register(
            module_registry: ModuleRegistry,
            tool_registry: ToolRegistry,
            model_registry: ModelProviderRegistry,
        ) -> None: ...

    Modules may ignore registries they do not need.

    Validation steps performed on every ``load()`` call:
    1. The module must be importable.
    2. The module must expose a ``register()`` callable.
    3. ``register()`` must accept the three expected keyword parameters.
    4. After registration, a warning is emitted if the module registered nothing.
    """

    def __init__(
        self,
        module_registry: ModuleRegistry,
        tool_registry: ToolRegistry,
        model_registry: Optional[ModelProviderRegistry] = None,
    ) -> None:
        self._module_registry = module_registry
        self._tool_registry = tool_registry
        self._model_registry = model_registry or ModelProviderRegistry()

    def load_all(self, module_paths: List[str]) -> None:
        """Load all modules in *module_paths*, emitting lifecycle events.

        Emits ``event=module_loading_start`` before loading and
        ``event=module_loading_complete`` after all modules are processed.
        """
        logger.info("event=module_loading_start count=%d modules=%r", len(module_paths), module_paths)
        for path in module_paths:
            self.load(path)
        logger.info(
            "event=module_loading_complete loaded=%d",
            len(self._module_registry.list_loaded()),
        )

    def load(self, module_path: str) -> None:
        """Import *module_path* and call its ``register()`` function.

        Marks the module as loaded in the ModuleRegistry on success.
        Raises on any import or registration failure.
        """
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            logger.error("event=module_import_failed module=%s error=%r", module_path, str(exc))
            raise

        if not hasattr(mod, "register") or not callable(mod.register):
            raise ValueError(f"Module {module_path!r} has no register() function")

        # Validate register() signature.
        sig = inspect.signature(mod.register)
        params = set(sig.parameters.keys())
        if not _REGISTER_PARAMS.issubset(params):
            missing = _REGISTER_PARAMS - params
            raise ValueError(
                f"Invalid module register() signature in {module_path!r}: "
                f"missing parameters {missing}"
            )

        # Count registered components before and after to detect empty registration.
        before_classifiers = len(self._module_registry._classifiers)
        before_routers = len(self._module_registry._routers)
        before_workers = len(self._module_registry._workers)
        before_tools = len(self._tool_registry._tools)

        mod.register(
            module_registry=self._module_registry,
            tool_registry=self._tool_registry,
            model_registry=self._model_registry,
        )

        after_classifiers = len(self._module_registry._classifiers)
        after_routers = len(self._module_registry._routers)
        after_workers = len(self._module_registry._workers)
        after_tools = len(self._tool_registry._tools)

        registered_components = (
            (after_classifiers - before_classifiers)
            + (after_routers - before_routers)
            + (after_workers - before_workers)
            + (after_tools - before_tools)
        )

        self._module_registry.mark_loaded(module_path)

        if registered_components == 0:
            logger.warning(
                "event=module_loaded module=%s registered_components=0 "
                "warning=module_registered_nothing",
                module_path,
            )
        else:
            logger.info(
                "event=module_loaded module=%s registered_components=%d",
                module_path,
                registered_components,
            )
