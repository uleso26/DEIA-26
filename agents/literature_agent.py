from __future__ import annotations

from core.models import AgentSection
from core.storage import tokenize
from tools.native_tools import (
    filter_recent_documents,
    get_population_context,
    infer_country_from_query,
    search_external_intelligence,
    search_pubmed,
    search_retrieval_index,
)
from agents.base_agent import citation, latest_record, unique_strings


class LiteratureAgent:
    def run(self, query: str, question_class: str) -> AgentSection:
        if question_class == "Q5":
            intelligence = search_external_intelligence(query, top_k=2)
            literature = search_retrieval_index(query, top_k=2)
            confirmed = latest_record([item for item in literature if item.get("publication_date")], "publication_date")
            inferred = latest_record([item for item in intelligence if item.get("publish_date")], "publish_date")

            fragments = []
            citations = []
            if confirmed:
                fragments.append(f"Confirmed evidence includes {confirmed['title']} ({confirmed['publication_date']}).")
                citations.append(
                    citation(
                        "PubMed",
                        confirmed["title"],
                        confirmed["pmid"],
                        confirmed["source_url"],
                        "Tier 2",
                        confirmed["publication_date"],
                    )
                )
            if inferred:
                fragments.append(f"Inferred positioning includes {inferred['headline']} ({inferred['publish_date']}).")
                citations.append(
                    citation(
                        inferred["source_type"],
                        inferred["headline"],
                        inferred["headline"],
                        inferred["source_url"],
                        inferred["evidence_tier"],
                        inferred["publish_date"],
                    )
                )
            summary = " ".join(fragments) if fragments else "No competitor monitoring evidence was found."
            return AgentSection(
                agent="Literature Agent",
                question_class="Q5",
                summary=summary,
                citations=citations,
                caveats=["Confirmed facts are separated from inferred positioning or analyst interpretation."],
                evidence_tiers=unique_strings(["Tier 2", "Tier 4"]),
                tool_outputs=[{"literature": literature, "external_intelligence": intelligence}],
                metadata={"mode": "competitor_monitoring"},
            )

        query_lower = query.lower()
        population_terms = ["prevalence", "incidence", "burden", "population", "epidemiology"]
        literature_terms = ["recent", "last 6 months", "journal", "publications", "literature", "evidence update"]
        country = infer_country_from_query(query)
        wants_population_context = bool(country) or any(term in query_lower for term in population_terms)
        wants_literature_context = any(term in query_lower for term in literature_terms)
        population_payload = (
            get_population_context(country=country, top_k=1 if country else 2)
            if wants_population_context
            else {
                "tool_name": "get_population_context",
                "canonical_entities": {"country": None},
                "source_metadata": {"storage": "sqlite", "table": "who_gho_diabetes_stats"},
                "records": [],
                "raw_provenance": {"country": None, "top_k": 0},
            }
        )
        population_records = population_payload["records"]

        if wants_population_context and not wants_literature_context:
            citations = []
            fragments = []
            for record in population_records[:2]:
                fragments.append(
                    f"Population context indicates adult diabetes prevalence in {record['country']} was {record['value']} {record['unit']} in {record['year']}."
                )
                citations.append(
                    citation(
                        "WHO GHO",
                        record["indicator"],
                        f"WHO:{record['country']}:{record['year']}",
                        record["source_url"],
                        "Tier 2",
                        str(record["year"]),
                    )
                )
            summary = " ".join(fragments) if fragments else "No population surveillance context was found."
            return AgentSection(
                agent="Literature Agent",
                question_class="Q6",
                summary=summary,
                citations=citations,
                caveats=[
                    "Population statistics are surveillance context and may not generalize to a specific clinical cohort.",
                    "Evidence summaries should state recency, publication venue, and evidence type.",
                ],
                evidence_tiers=unique_strings(["Tier 2"]),
                tool_outputs=[population_payload],
                metadata={"population_context_used": bool(population_records), "mode": "population_context"},
            )

        semantic_results = search_retrieval_index(query, top_k=4)
        search_results = search_pubmed(query, top_k=4)
        combined: dict[str, dict] = {}
        for item in semantic_results:
            if item.get("pmid"):
                combined[item["pmid"]] = {**item, "semantic_score": item.get("score", 0.0), "keyword_score": 0.0}
        for item in search_results:
            if not item.get("pmid"):
                continue
            existing = combined.get(item["pmid"], dict(item))
            existing["keyword_score"] = max(existing.get("keyword_score", 0.0), item.get("score", 0.0))
            existing.setdefault("semantic_score", 0.0)
            existing.update({key: value for key, value in item.items() if key != "score"})
            combined[item["pmid"]] = existing

        for item in combined.values():
            item["combined_score"] = round(item.get("semantic_score", 0.0) + item.get("keyword_score", 0.0), 4)

        recent_documents = filter_recent_documents(list(combined.values()), months=6)
        query_lower = query.lower()
        anchor_groups: list[set[str]] = []
        if "heart failure" in query_lower:
            anchor_groups.append({"heart", "failure"})
        if "sglt2" in query_lower:
            anchor_groups.append({"sglt2"})
        if "glp1r" in query_lower:
            anchor_groups.append({"glp1r"})
        if anchor_groups:
            anchored_documents = []
            for item in recent_documents:
                tokens = set(tokenize(f"{item['title']} {item['text']} {' '.join(item['mesh_terms'])}"))
                if all(group.issubset(tokens) for group in anchor_groups):
                    anchored_documents.append(item)
            if anchored_documents:
                recent_documents = anchored_documents
        ordered = sorted(
            recent_documents,
            key=lambda item: (item.get("combined_score", 0.0), item["publication_date"]),
            reverse=True,
        )
        citations = []
        fragments = []
        if population_records:
            record = population_records[0]
            fragments.append(
                f"Population context indicates adult diabetes prevalence in {record['country']} was {record['value']} {record['unit']} in {record['year']}."
            )
            citations.append(
                citation(
                    "WHO GHO",
                    record["indicator"],
                    f"WHO:{record['country']}:{record['year']}",
                    record["source_url"],
                    "Tier 2",
                    str(record["year"]),
                )
            )
        citations.extend(
            [
                citation(
                    "PubMed",
                    item["title"],
                    item["pmid"],
                    item["source_url"],
                    "Tier 2",
                    item["publication_date"],
                )
                for item in ordered[:3]
            ]
        )
        evidence_fragments = [
            f"{item['title']} ({item['journal']}, {item['publication_date']}, {item['evidence_type']})"
            for item in ordered[:3]
        ]
        if evidence_fragments:
            fragments.append("Recent evidence includes " + "; ".join(evidence_fragments) + ".")
        summary = " ".join(fragments) if fragments else "No recent literature was found."
        caveats = ["Evidence summaries should state recency, publication venue, and evidence type."]
        if population_records:
            caveats.append("Population statistics are surveillance context and may not generalize to a specific clinical cohort.")
        return AgentSection(
            agent="Literature Agent",
            question_class="Q6",
            summary=summary,
            citations=citations,
            caveats=unique_strings(caveats),
            evidence_tiers=unique_strings(["Tier 2"]),
            tool_outputs=[population_payload, {"semantic_search": semantic_results, "search_pubmed": search_results}],
            metadata={"recency_window": "6 months", "population_context_used": bool(population_records)},
        )
