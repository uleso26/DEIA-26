# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import os
from datetime import datetime, timezone


# Read a boolean style environment flag with a consistent parser
def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean-style environment flag with a sensible default."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Return the current UTC timestamp in ISO 8601 form
def utc_now_iso() -> str:
    """Return a compact UTC timestamp for traces, lineage, and tool metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
