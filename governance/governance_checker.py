from __future__ import annotations

from typing import Any

import yaml

from core.paths import ROOT


class GovernanceChecker:
    """Apply answer-shaping rules and required caveats per question class."""

    def __init__(self, rule_file: str = "governance/governance_rules.yaml") -> None:
        with (ROOT / rule_file).open("r", encoding="utf-8") as handle:
            self.rules = yaml.safe_load(handle)

    def apply(
        self,
        question_class: str,
        answer: str,
        caveats: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        """Return a governed answer and a normalized caveat list."""
        metadata = metadata or {}
        normalized_caveats = list(caveats)
        rules = self.rules.get(question_class, {})
        for caveat in rules.get("required_caveats", []):
            if caveat not in normalized_caveats:
                normalized_caveats.append(caveat)

        prefix = rules.get("answer_prefix")
        if prefix and not answer.startswith(prefix):
            answer = prefix + answer[:1].lower() + answer[1:] if answer else prefix.strip()

        suffix = rules.get("answer_suffix")
        if suffix and suffix.strip() not in answer:
            answer += suffix

        template = rules.get("comparison_template")
        comparison_type = metadata.get("comparison_type")
        if template and comparison_type and "comparison" not in answer.lower():
            article = "an" if str(comparison_type).lower()[:1] in {"a", "e", "i", "o", "u"} else "a"
            rendered = template.format(comparison_type=comparison_type)
            answer += f" {rendered.replace('a ' + str(comparison_type), article + ' ' + str(comparison_type), 1)}"

        return answer.strip(), normalized_caveats

    def validate(self, question_class: str, answer: str, caveats: list[str]) -> list[str]:
        """Report missing caveats or rule violations for a synthesized answer."""
        issues = []
        rules = self.rules.get(question_class, {})
        for caveat in rules.get("required_caveats", []):
            if caveat not in caveats:
                issues.append(f"Missing caveat: {caveat}")
        if question_class == "Q1" and "causality" not in " ".join(caveats).lower():
            issues.append("Safety answer is missing explicit non-causality framing.")
        return issues
