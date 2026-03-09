"""ModuleRegistry — central registry where modules register their capabilities."""

from __future__ import annotations

import logging
from typing import Dict

from coretex.interfaces.classifier import Classifier
from coretex.interfaces.router import Router
from coretex.interfaces.worker import Worker

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """Holds classifiers, routers, and workers registered by modules at startup.

    All ``register_*`` methods raise ``ValueError`` on duplicate names.
    All ``get_*`` methods raise ``ValueError`` on unknown names and emit a
    structured ``event=registry_lookup_failed`` log.
    """

    def __init__(self) -> None:
        self._classifiers: Dict[str, Classifier] = {}
        self._routers: Dict[str, Router] = {}
        self._workers: Dict[str, Worker] = {}
        self._loaded: list[str] = []

    # ------------------------------------------------------------------
    # Classifiers
    # ------------------------------------------------------------------

    def register_classifier(self, name: str, classifier: Classifier) -> None:
        """Register *classifier* under *name*. Raises ValueError if already registered."""
        if name in self._classifiers:
            raise ValueError(f"Component already registered: {name}")
        self._classifiers[name] = classifier
        logger.info("event=classifier_registered name=%s", name)

    def get_classifier(self, name: str) -> Classifier:
        """Return the classifier registered as *name*. Raises ValueError if unknown."""
        if name not in self._classifiers:
            logger.error("event=registry_lookup_failed component=classifier name=%s", name)
            raise ValueError(f"Unknown component: {name}")
        return self._classifiers[name]

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------

    def register_router(self, name: str, router: Router) -> None:
        """Register *router* under *name*. Raises ValueError if already registered."""
        if name in self._routers:
            raise ValueError(f"Component already registered: {name}")
        self._routers[name] = router
        logger.info("event=router_registered name=%s", name)

    def get_router(self, name: str) -> Router:
        """Return the router registered as *name*. Raises ValueError if unknown."""
        if name not in self._routers:
            logger.error("event=registry_lookup_failed component=router name=%s", name)
            raise ValueError(f"Unknown component: {name}")
        return self._routers[name]

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def register_worker(self, name: str, worker: Worker) -> None:
        """Register *worker* under *name*. Raises ValueError if already registered."""
        if name in self._workers:
            raise ValueError(f"Component already registered: {name}")
        self._workers[name] = worker
        logger.info("event=worker_registered name=%s", name)

    def get_worker(self, name: str) -> Worker:
        """Return the worker registered as *name*. Raises ValueError if unknown."""
        if name not in self._workers:
            logger.error("event=registry_lookup_failed component=worker name=%s", name)
            raise ValueError(f"Unknown component: {name}")
        return self._workers[name]

    # ------------------------------------------------------------------
    # Module tracking
    # ------------------------------------------------------------------

    def mark_loaded(self, module_path: str) -> None:
        """Record that *module_path* has been successfully loaded."""
        self._loaded.append(module_path)

    def list_loaded(self) -> list[str]:
        """Return a copy of the list of loaded module paths."""
        return list(self._loaded)

    def component_count(self) -> int:
        """Return the total number of registered classifiers, routers, and workers."""
        return len(self._classifiers) + len(self._routers) + len(self._workers)
