from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.orchestrator import T2DOrchestrator, bootstrap_runtime
from agents.router_agent import RouterAgent
from core.paths import LINEAGE_DIR, RETRIEVAL_MANIFEST
from core.storage import backend_status, connect_sqlite
from data.canonical.resolver import get_resolver
from data.ingestion.ingest_pubmed import _summary_matches_seed
from governance.governance_checker import GovernanceChecker
from tools.native_tools import filter_recent_documents
from tools.native_tools import get_clinical_context, get_population_context


class PlatformTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bootstrap_runtime()
        cls.orchestrator = T2DOrchestrator(bootstrap_if_needed=False)
        cls.mcp_client = cls.orchestrator.mcp_client

    @classmethod
    def tearDownClass(cls) -> None:
        cls.orchestrator.close()

    def test_canonical_drug_resolution(self) -> None:
        resolver = get_resolver()
        resolved = resolver.resolve_drug("Wegovy safety")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["canonical_id"], "semaglutide")

    def test_safety_mcp_tool_returns_records(self) -> None:
        result = self.mcp_client.call_tool("safety", "search_adverse_events", {"drug": "tirzepatide"})
        self.assertEqual(result["canonical_entities"]["drug"]["canonical_id"], "tirzepatide")
        self.assertGreaterEqual(len(result["records"]), 1)

    def test_orchestrator_pathway_response_contains_disclaimer(self) -> None:
        response = self.orchestrator.run_query(
            "For a T2D patient with CKD stage 3, what does NICE guidance suggest after metformin failure?"
        )
        self.assertEqual(response["question_class"], "Q3")
        self.assertIn("not medical advice", response["answer"].lower())

    def test_trial_name_query_resolves_correct_trial(self) -> None:
        response = self.orchestrator.run_query("What does SURPASS-3 show?")
        self.assertIn("surpass-3", response["answer"].lower())
        self.assertNotIn("surpass-2 provides", response["answer"].lower())

    def test_target_only_query_lists_linked_drugs(self) -> None:
        response = self.orchestrator.run_query("Which drugs share the GLP1R mechanism?")
        self.assertIn("semaglutide", response["answer"].lower())
        self.assertIn("tirzepatide", response["answer"].lower())

    def test_guideline_selection_respects_ada_obesity_intent(self) -> None:
        response = self.orchestrator.run_query("ADA pathway after metformin for obesity")
        self.assertIn("glp-1 receptor agonist", response["answer"].lower())
        self.assertIn("ada", response["answer"].lower())

    def test_literature_update_excludes_irrelevant_glp1_review(self) -> None:
        response = self.orchestrator.run_query("Give me an evidence update on SGLT2 inhibitors in heart failure")
        self.assertNotIn("oral glp-1 agonists", response["answer"].lower())

    def test_auxiliary_source_tables_are_populated(self) -> None:
        connection = connect_sqlite()
        try:
            cursor = connection.cursor()
            counts = {}
            for table in ["who_gho_diabetes_stats", "drugbank_classifications", "synthetic_patient_profiles"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
        finally:
            connection.close()
        for count in counts.values():
            self.assertGreaterEqual(count, 1)

    def test_population_context_tool_returns_who_record(self) -> None:
        payload = get_population_context("United Kingdom")
        self.assertGreaterEqual(len(payload["records"]), 1)
        self.assertEqual(payload["records"][0]["country"], "United Kingdom")

    def test_clinical_context_tool_returns_drugbank_and_synthetic_matches(self) -> None:
        payload = get_clinical_context("ADA pathway after metformin for obesity GLP-1 receptor agonist")
        record_types = {record["record_type"] for record in payload["records"]}
        self.assertIn("drugbank", record_types)
        self.assertIn("synthetic_profile", record_types)

    def test_pathway_response_uses_clinical_context(self) -> None:
        response = self.orchestrator.run_query("ADA pathway after metformin for obesity")
        self.assertTrue(response["metadata"].get("clinical_context_used"))

    def test_population_query_uses_who_context(self) -> None:
        response = self.orchestrator.run_query("What is the diabetes prevalence in the United Kingdom?")
        self.assertEqual(response["question_class"], "Q6")
        self.assertIn("united kingdom", response["answer"].lower())
        self.assertTrue(response["metadata"].get("population_context_used"))

    def test_backend_status_reports_expected_shape(self) -> None:
        status = backend_status()
        self.assertIn("sqlite", status)
        self.assertIn("mongodb", status)
        self.assertIn("neo4j", status)
        self.assertIn("fallback_files", status)
        self.assertTrue(status["sqlite"]["available"])
        self.assertIn("retrieval_manifest", status["fallback_files"])

    def test_knowledge_server_falls_back_when_live_neo4j_returns_empty(self) -> None:
        from mcp_servers.knowledge_server import KnowledgeServer

        server = KnowledgeServer()
        with patch("mcp_servers.knowledge_server.run_neo4j_query_with_backend", return_value=([], "neo4j")):
            payload = server.query_pathway(
                start_drug="metformin",
                comorbidity="obesity",
                guideline_hint="ADA",
                phenotype_terms=["obesity"],
            )
        self.assertGreaterEqual(len(payload["records"]), 1)
        self.assertEqual(payload["source_metadata"]["storage"], "neo4j_fallback")

    def test_retrieval_manifest_and_lineage_artifacts_exist(self) -> None:
        self.assertTrue(RETRIEVAL_MANIFEST.exists())
        for source_name in ["openfda", "clinicaltrials", "pubmed"]:
            self.assertTrue((LINEAGE_DIR / f"{source_name}.jsonl").exists())

    def test_router_falls_back_when_ollama_returns_empty_response(self) -> None:
        router = RouterAgent()
        with patch.object(router.ollama, "enabled", return_value=True), patch.object(
            router.ollama,
            "generate",
            return_value="   ",
        ):
            payload = router.route("What does SURPASS-3 show?")
        self.assertEqual(payload["question_class"], "Q2")

    def test_pubmed_live_guard_rejects_semantic_mismatch(self) -> None:
        seed_document = {
            "title": "SGLT2 inhibitors and heart failure outcomes in type 2 diabetes",
            "mesh_terms": ["SGLT2 inhibitors", "heart failure", "type 2 diabetes"],
            "text": "Cardiorenal benefit was observed in type 2 diabetes populations.",
        }
        mismatched_summary = {
            "title": "Early Periprosthetic Tibial Lucency Following Low-Profile Total Ankle Arthroplasty.",
            "fulljournalname": "Foot & ankle international",
        }
        self.assertFalse(_summary_matches_seed(seed_document, mismatched_summary))

    def test_filter_recent_documents_skips_malformed_dates(self) -> None:
        documents = [
            {"publication_date": "2026-03-01", "title": "valid"},
            {"publication_date": "not-a-date", "title": "broken"},
        ]
        filtered = filter_recent_documents(documents, months=6)
        self.assertEqual([item["title"] for item in filtered], ["valid"])

    def test_governance_checker_does_not_mutate_caveats(self) -> None:
        caveats = ["Original caveat."]
        original = list(caveats)
        checker = GovernanceChecker()
        _, updated = checker.apply(
            "Q1",
            "Seed answer",
            caveats,
        )
        self.assertEqual(caveats, original)
        self.assertGreaterEqual(len(updated), len(original))

    def test_unknown_drug_safety_tool_returns_empty_records(self) -> None:
        result = self.mcp_client.call_tool("safety", "search_adverse_events", {"drug": "unknowncompound"})
        self.assertEqual(result["records"], [])

    def test_empty_query_defaults_to_q0(self) -> None:
        router = RouterAgent()
        with patch.object(router.ollama, "enabled", return_value=False):
            payload = router.route("")
        self.assertEqual(payload["question_class"], "Q0")

    def test_general_disease_question_routes_to_q7(self) -> None:
        response = self.orchestrator.run_query("Can diabetes lead to death? How serious is this disease?")
        self.assertEqual(response["question_class"], "Q7")
        self.assertIn("can contribute to death", response["answer"].lower())

    def test_pricing_query_routes_to_q8(self) -> None:
        response = self.orchestrator.run_query("What is the latest list price difference between semaglutide and tirzepatide?")
        self.assertEqual(response["question_class"], "Q8")
        self.assertIn("does not include a live pricing", response["answer"].lower())

    def test_personal_urgent_query_routes_to_q9(self) -> None:
        response = self.orchestrator.run_query("I have diabetes and chest pain right now, what should I do?")
        self.assertEqual(response["question_class"], "Q9")
        self.assertIn("not designed for emergency triage", response["answer"].lower())

    def test_creative_request_routes_to_q0(self) -> None:
        response = self.orchestrator.run_query("Write a poem about tirzepatide.")
        self.assertEqual(response["question_class"], "Q0")
        self.assertIn("outside the current t2d enterprise intelligence scope", response["answer"].lower())

    def test_repo_does_not_ship_real_secrets_or_absolute_local_paths(self) -> None:
        sensitive_files = [
            Path("README.md"),
            Path(".env.example"),
            Path("brief/langchain copy.ipynb"),
        ]
        secret_pattern = re.compile(r"sk-[A-Za-z0-9]{20,}")
        for path in sensitive_files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", text, f"Absolute local path leaked in {path}")
            self.assertIsNone(secret_pattern.search(text), f"Secret-like token leaked in {path}")


if __name__ == "__main__":
    unittest.main()
