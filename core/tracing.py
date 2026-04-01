from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from core.paths import LOG_DIR, ensure_runtime_directories


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TraceLogger:
    def __init__(self) -> None:
        ensure_runtime_directories()
        self.trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        self.payload: dict[str, Any] = {"trace_id": self.trace_id, "started_at": utc_now_iso(), "events": []}

    def add_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.payload["events"].append({"type": event_type, "timestamp": utc_now_iso(), "payload": payload})

    def finalize(self, response: dict[str, Any]) -> str:
        self.payload["finished_at"] = utc_now_iso()
        self.payload["response"] = response
        path = LOG_DIR / f"{self.trace_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, indent=2, ensure_ascii=True)
        return self.trace_id
