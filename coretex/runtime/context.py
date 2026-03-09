"""ExecutionContext — per-request context threaded through the pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExecutionContext:
    """Carries all per-request state through the pipeline.

    Attributes:
        user_input: The raw input string from the user.
        request_id: A unique hex identifier for this request (auto-generated).
        intent: The classified intent, set after classification completes.
        confidence: The classifier confidence score (0.0–1.0).
        handler: The routing handler name, set after routing completes.
        t_start: Monotonic timestamp at context creation for latency tracking.
        timestamp: Wall-clock time at context creation (time.time()) for observability.
        metadata: Optional free-form metadata dict for modules to attach extra context.
    """

    user_input: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    intent: Optional[str] = None
    confidence: float = 0.0
    handler: Optional[str] = None
    t_start: float = field(default_factory=time.monotonic)
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None
