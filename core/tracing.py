# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import json
import uuid
from typing import Any

from core.paths import LOG_DIR, ensure_runtime_directories
from core.runtime_utils import utc_now_iso


# Define the trace logger that records workflow events and metadata
class TraceLogger:
    """Collect and persist a lightweight execution trace for a single query."""

    def __init__(self) -> None:
        ensure_runtime_directories()
        self.trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        self.payload: dict[str, Any] = {"trace_id": self.trace_id, "started_at": utc_now_iso(), "events": []}

    def add_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append a timestamped event to the active trace."""
        self.payload["events"].append({"type": event_type, "timestamp": utc_now_iso(), "payload": payload})

    def finalize(self, response: dict[str, Any]) -> str:
        """Persist the completed trace to disk and return its trace identifier."""
        self.payload["finished_at"] = utc_now_iso()
        self.payload["response"] = response
        path = LOG_DIR / f"{self.trace_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, indent=2, ensure_ascii=True)
        return self.trace_id
