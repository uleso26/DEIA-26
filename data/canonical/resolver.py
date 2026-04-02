from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from core.storage import load_json
from core.paths import CANONICAL_DIR


@dataclass
class CanonicalMatch:
    canonical_id: str
    matched_aliases: list[str]
    confidence: float
    evidence_tier: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonical match for downstream agent/tool payloads."""
        return {
            "canonical_id": self.canonical_id,
            "matched_aliases": self.matched_aliases,
            "confidence": self.confidence,
            "evidence_tier": self.evidence_tier,
        }


class CanonicalResolver:
    """Resolve drugs, targets, and trials to the platform's canonical identifiers."""

    def __init__(self) -> None:
        self.drug_synonyms, self.target_synonyms, self.trial_crosswalk = _load_canonical_tables()

    @staticmethod
    def _alias_matches(text: str, alias: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text))

    def _resolve_from_mapping(self, text: str, mapping: dict[str, Any], alias_key: str = "aliases") -> CanonicalMatch | None:
        lowered = text.lower()
        best: CanonicalMatch | None = None
        for canonical_id, payload in mapping.items():
            aliases = [canonical_id.lower(), *[alias.lower() for alias in payload.get(alias_key, [])]]
            matched = [alias for alias in aliases if alias and self._alias_matches(lowered, alias)]
            if matched:
                score = min(1.0, 0.6 + (0.1 * len(matched)))
                candidate = CanonicalMatch(
                    canonical_id=canonical_id,
                    matched_aliases=sorted(set(matched)),
                    confidence=score,
                    evidence_tier=payload.get("evidence_tier", "Tier 2"),
                )
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
        return best

    def resolve_drug(self, text: str) -> dict[str, Any] | None:
        """Resolve the strongest drug mention in a free-form query."""
        match = self._resolve_from_mapping(text, self.drug_synonyms)
        if not match:
            return None
        payload = dict(self.drug_synonyms.get(match.canonical_id, {}))
        return {
            **match.to_dict(),
            "drug_class": payload.get("drug_class", "unknown"),
            "route": payload.get("route"),
        }

    def resolve_target(self, text: str) -> dict[str, Any] | None:
        """Resolve the strongest target mention in a free-form query."""
        match = self._resolve_from_mapping(text, self.target_synonyms)
        return match.to_dict() if match else None

    def resolve_trial(self, text: str) -> dict[str, Any] | None:
        """Resolve a trial acronym, NCT ID, or trial name from the query."""
        lowered = text.lower()
        best: CanonicalMatch | None = None
        for nct_id, payload in self.trial_crosswalk.items():
            aliases = [nct_id.lower(), payload.get("trial_name", "").lower(), *[alias.lower() for alias in payload.get("aliases", [])]]
            matched = [alias for alias in aliases if alias and self._alias_matches(lowered, alias)]
            if matched:
                candidate = CanonicalMatch(
                    canonical_id=nct_id,
                    matched_aliases=sorted(set(matched)),
                    confidence=0.9,
                    evidence_tier=payload.get("evidence_tier", "Tier 2"),
                )
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
        return best.to_dict() if best else None

    def resolve_all(self, text: str) -> dict[str, Any]:
        """Resolve all supported canonical entity types from one query."""
        return {
            "drug": self.resolve_drug(text),
            "target": self.resolve_target(text),
            "trial": self.resolve_trial(text),
        }

    def find_drugs(self, text: str) -> list[dict[str, Any]]:
        """Return every drug mention matched in the query, not just the strongest one."""
        lowered = text.lower()
        matches: list[dict[str, Any]] = []
        for canonical_id, payload in self.drug_synonyms.items():
            aliases = [canonical_id.lower(), *[alias.lower() for alias in payload.get("aliases", [])]]
            matched = [alias for alias in aliases if alias and self._alias_matches(lowered, alias)]
            if matched:
                matches.append(
                    {
                        "canonical_id": canonical_id,
                        "matched_aliases": sorted(set(matched)),
                        "confidence": min(1.0, 0.6 + (0.1 * len(matched))),
                        "evidence_tier": payload.get("evidence_tier", "Tier 2"),
                        "drug_class": payload.get("drug_class", "unknown"),
                    }
                )
        return matches


@lru_cache(maxsize=1)
def _load_canonical_tables() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_json(CANONICAL_DIR / "drug_synonyms.json"),
        load_json(CANONICAL_DIR / "target_synonyms.json"),
        load_json(CANONICAL_DIR / "trial_crosswalk.json"),
    )


@lru_cache(maxsize=1)
def get_resolver() -> CanonicalResolver:
    return CanonicalResolver()
