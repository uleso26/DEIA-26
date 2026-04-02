from __future__ import annotations

import os
from datetime import datetime, timezone


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean-style environment flag with a sensible default."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for traces, lineage, and tool metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
