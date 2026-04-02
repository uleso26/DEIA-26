from __future__ import annotations

import json
import re
import unittest
import uuid
from datetime import date
from pathlib import Path
from unittest.mock import patch

from api.http_server import dispatch_request, resolve_static_asset, stream_answer_chunks
from agents.evidence_planner_agent import EvidencePlannerAgent
from agents.orchestrator import T2DOrchestrator, bootstrap_runtime
from agents.router_agent import RouterAgent
from agents.synthesis_agent import SynthesisAgent
from core.storage import backend_status, build_chroma_index, build_dense_index, connect_sqlite, search_dense_index
from core.paths import LINEAGE_DIR, PROV_LINEAGE_DIR, RETRIEVAL_MANIFEST
from data.canonical.resolver import get_resolver
from data.ingestion.base import append_prov_manifest
from data.ingestion.ingest_pubmed import _summary_matches_seed
from governance.governance_checker import GovernanceChecker
from tools.langchain_native_tools import invoke_native_tool
from tools.native_tools import build_evidence_plan, build_query_understanding, filter_recent_documents, get_clinical_context, get_population_context


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
        self.assertEqual(result["server_name"], "safety")
        self.assertIn("requested_at", result)

    def test_mcp_list_tools_exposes_structured_schemas(self) -> None:
        tools = self.mcp_client.list_tools("safety")
        self.assertGreaterEqual(len(tools), 1)
        search_tool = next(tool for tool in tools if tool["name"] == "search_adverse_events")
        self.assertIn("inputSchema", search_tool)
        self.assertIn("outputSchema", search_tool)

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

    def test_langchain_native_tool_returns_structured_payload(self) -> None:
        payload = invoke_native_tool(
            "get_population_context_native",
            {"country": "United Kingdom", "top_k": 1},
        )
        self.assertEqual(payload["tool_name"], "get_population_context")
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
        self.assertIn("chroma_dir", status["fallback_files"])

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
            self.assertTrue((PROV_LINEAGE_DIR / f"{source_name}.jsonl").exists())

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

    def test_filter_recent_documents_honors_reference_date(self) -> None:
        documents = [
            {"publication_date": "2025-10-02", "title": "inside"},
            {"publication_date": "2025-10-01", "title": "outside"},
        ]
        filtered = filter_recent_documents(documents, months=6, reference_date=date(2026, 4, 2))
        self.assertEqual([item["title"] for item in filtered], ["inside"])

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

    def test_broad_best_drug_query_routes_to_treatment_selection(self) -> None:
        response = self.orchestrator.run_query("which drug is the best effective one for curing the T2D")
        self.assertEqual(response["question_class"], "Q3")
        self.assertIn("no single drug that cures type 2 diabetes", response["answer"].lower())
        self.assertIn("what outcome matters most", response["answer"].lower())
        self.assertTrue(response["metadata"].get("needs_clarification"))

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

    def test_scope_routes_skip_ollama_synthesis(self) -> None:
        agent = SynthesisAgent()
        with patch.object(agent.ollama, "enabled", return_value=True), patch.object(agent.ollama, "generate") as mocked_generate:
            response = agent.run(
                "Q0",
                "hi",
                [],
                "trace-test",
            )
        self.assertEqual(response.metadata["synthesis_mode"], "deterministic")
        mocked_generate.assert_not_called()

    def test_http_api_query_endpoint_returns_json(self) -> None:
        class StubRuntime:
            def run_query(self, query: str) -> dict:
                return {
                    "question_class": "Q2",
                    "answer": f"stub:{query}",
                    "citations": [],
                    "caveats": [],
                    "evidence_tiers": [],
                    "trace_id": "trace-test",
                    "sections": [],
                    "metadata": {},
                }

            def close(self) -> None:
                return

        status_code, payload = dispatch_request(
            "POST",
            "/query",
            {"query": "What does SURPASS-3 show?"},
            StubRuntime(),
        )
        self.assertEqual(status_code, 200)
        self.assertEqual(payload["question_class"], "Q2")
        self.assertEqual(payload["answer"], "stub:What does SURPASS-3 show?")

    def test_http_api_health_endpoint_returns_ok(self) -> None:
        class StubRuntime:
            def run_query(self, query: str) -> dict:
                return {}

            def close(self) -> None:
                return

        status_code, payload = dispatch_request("GET", "/health", None, StubRuntime())
        self.assertEqual(status_code, 200)
        self.assertEqual(payload, {"ok": True})

    def test_http_api_ignores_query_string_on_known_path(self) -> None:
        class StubRuntime:
            def run_query(self, query: str) -> dict:
                return {"answer": query}

            def close(self) -> None:
                return

        status_code, payload = dispatch_request("GET", "/health?full=true", None, StubRuntime())
        self.assertEqual(status_code, 200)
        self.assertEqual(payload, {"ok": True})

    def test_static_homepage_asset_resolves(self) -> None:
        resolved = resolve_static_asset("/")
        self.assertIsNotNone(resolved)
        asset_path, content_type = resolved
        self.assertEqual(asset_path.name, "index.html")
        self.assertEqual(content_type, "text/html")

    def test_langgraph_workflow_exposes_expected_nodes(self) -> None:
        mermaid = self.orchestrator.workflow.mermaid_diagram()
        self.assertIn("understand", mermaid)
        self.assertIn("policy_gate", mermaid)
        self.assertIn("plan", mermaid)
        self.assertIn("refine_plan", mermaid)
        self.assertIn("execute_plan", mermaid)
        self.assertIn("evidence_review", mermaid)
        self.assertIn("synthesize", mermaid)
        self.assertIn("scope", mermaid)
        self.assertIn("clarify", mermaid)

    def test_expanded_catalog_supports_cost_sensitive_pathway_query(self) -> None:
        response = self.orchestrator.run_query("For a cost-sensitive patient after metformin, what does NICE suggest next?")
        self.assertEqual(response["question_class"], "Q3")
        self.assertTrue(
            "sulfonylurea" in response["answer"].lower()
            or "pioglitazone" in response["answer"].lower()
        )

    def test_guideline_difference_query_stays_in_pathway_scope(self) -> None:
        response = self.orchestrator.run_query(
            "For a patient with obesity after metformin, how does ADA differ from NICE on the next step?"
        )
        self.assertEqual(response["question_class"], "Q3")
        self.assertTrue(
            "guideline comparison" in response["answer"].lower()
            or "cross-guideline comparison was requested" in response["answer"].lower()
        )
        self.assertIn(response["metadata"]["evidence_review"]["status"], {"limited", "sufficient"})

    def test_trial_query_exposes_execution_plan_metadata(self) -> None:
        response = self.orchestrator.run_query("What does SURPASS-3 show?")
        self.assertEqual(response["metadata"]["evidence_plan"]["execution_nodes"], ["trial"])
        self.assertEqual(response["metadata"]["evidence_review"]["status"], "sufficient")
        self.assertIn(
            response["metadata"]["evidence_plan"]["planning_mode"],
            {"deterministic", "react_planner_ollama", "deterministic_guarded_fallback"},
        )
        self.assertIn("allowed_execution_nodes", response["metadata"]["evidence_plan"])

    def test_trial_publication_query_prefers_trial_publication_over_generic_recent_literature(self) -> None:
        response = self.orchestrator.run_query(
            "For SURPASS-3, summarize the trial result and the supporting publication evidence."
        )
        citation_titles = [item["title"] for item in response["citations"]]
        self.assertTrue(any("SURPASS-3 publication" == title for title in citation_titles))
        self.assertIn("surpass-3", response["answer"].lower())

    def test_evidence_planner_returns_bounded_trial_plan(self) -> None:
        planner = EvidencePlannerAgent()
        understanding = build_query_understanding("What does SURPASS-3 show?").to_dict()
        base_plan = build_evidence_plan(understanding)
        with patch.object(planner.ollama, "enabled", return_value=False):
            plan = planner.plan(understanding, base_plan)
        self.assertEqual(plan["execution_nodes"], ["trial"])
        self.assertEqual(plan["planning_mode"], "deterministic")
        self.assertTrue(plan["react_eligible"])
        self.assertIn("literature_q6", plan["allowed_execution_nodes"])

    def test_evidence_planner_refines_with_fallback_node_when_evidence_is_limited(self) -> None:
        planner = EvidencePlannerAgent()
        understanding = build_query_understanding("What does SURPASS-3 show?").to_dict()
        base_plan = build_evidence_plan(understanding)
        with patch.object(planner.ollama, "enabled", return_value=False):
            plan = planner.plan(understanding, base_plan)
            refined = planner.refine_after_observation(
                understanding,
                plan,
                sections=[],
                evidence_review={"status": "limited", "reason": "missing_citations"},
                react_steps=0,
                executed_nodes=["trial"],
            )
        self.assertIn("literature_q6", refined["execution_nodes"])
        self.assertEqual(refined["planning_mode"], "react_refinement_fallback")

    def test_build_dense_index_and_chroma_search_return_expected_match(self) -> None:
        documents = [
            {"title": "GLP1R therapy", "text": "GLP1R incretin pathway evidence."},
            {"title": "SGLT2 outcomes", "text": "SGLT2 heart failure evidence."},
        ]

        def fake_embed(texts: list[str], *, provider: str | None = None, model_name: str | None = None) -> tuple[list[list[float]], dict[str, str]]:
            vectors: list[list[float]] = []
            for text in texts:
                lowered = text.lower()
                if "glp1r" in lowered:
                    vectors.append([1.0, 0.0])
                elif "sglt2" in lowered:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([0.5, 0.5])
            return vectors, {"embedding_provider": "stub", "embedding_model": "stub-model"}

        collection_name = f"test-chroma-{uuid.uuid4().hex[:8]}"
        with patch("core.storage._embed_texts", side_effect=fake_embed):
            dense_manifest = build_dense_index(documents)
            chroma_manifest = build_chroma_index(documents, collection_name=collection_name)
            self.assertIsNotNone(dense_manifest)
            self.assertEqual(dense_manifest["vector_dim"], 2)
            self.assertIsNotNone(chroma_manifest)
            results = search_dense_index(
                "GLP1R therapy",
                {
                    "chroma_collection": chroma_manifest["collection_name"],
                    "embedding_provider": chroma_manifest["embedding_provider"],
                    "embedding_model": chroma_manifest["embedding_model"],
                },
                top_k=1,
            )
        self.assertEqual(results[0]["title"], "GLP1R therapy")

    def test_append_prov_manifest_writes_valid_prov_record(self) -> None:
        source_name = f"unit_test_prov_{uuid.uuid4().hex[:8]}"
        path = append_prov_manifest(
            source_name,
            {
                "recorded_at": "2026-04-02T12:00:00+00:00",
                "mode": "seed_fixture",
                "raw_files": {"pubmed": "runtime/raw/pubmed_documents.json"},
                "record_counts": {"pubmed": 2},
                "upstream_requests": [{"url": "https://example.org/pubmed", "status_code": 200, "ok": True}],
            },
        )
        record = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(record["prefix"]["prov"], "http://www.w3.org/ns/prov#")
        self.assertEqual(record["activity"]["mode"], "seed_fixture")
        self.assertEqual(record["generated"][0]["location"], "runtime/raw/pubmed_documents.json")
        self.assertEqual(record["used"][0]["status_code"], 200)

    def test_stream_answer_chunks_splits_answer_into_sse_sized_segments(self) -> None:
        chunks = stream_answer_chunks(
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
        )
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].endswith(" "))
        self.assertEqual(chunks[1], "fifteen")

    def test_repo_does_not_ship_real_secrets_or_absolute_local_paths(self) -> None:
        sensitive_files = [
            Path("README.md"),
            Path(".env.example"),
            Path("SECURITY.md"),
        ]
        secret_pattern = re.compile(r"sk-[A-Za-z0-9]{20,}")
        for path in sensitive_files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", text, f"Absolute local path leaked in {path}")
            self.assertIsNone(secret_pattern.search(text), f"Secret-like token leaked in {path}")


if __name__ == "__main__":
    unittest.main()
