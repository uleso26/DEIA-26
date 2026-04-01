from __future__ import annotations

from core.paths import GRAPH_FILE, RETRIEVAL_MANIFEST, SQLITE_DB
from core.tracing import TraceLogger
from data.canonical.resolver import get_resolver
from data.ingestion.ingest_chembl import run as ingest_chembl
from data.ingestion.ingest_clinicaltrials import run as ingest_clinicaltrials
from data.ingestion.ingest_external_intelligence import run as ingest_external_intelligence
from data.ingestion.ingest_guideline_excerpts import run as ingest_guideline_excerpts
from data.ingestion.ingest_openfda import run as ingest_openfda
from data.ingestion.ingest_opentargets import run as ingest_opentargets
from data.ingestion.ingest_pubmed import run as ingest_pubmed
from data.ingestion.ingest_uniprot import run as ingest_uniprot
from data.ingestion.ingest_who import run as ingest_who
from data.ingestion.ingest_drugbank_open import run as ingest_drugbank_open
from data.ingestion.generate_synthetic_patients import run as generate_synthetic_patients
from data.processing.build_retrieval_index import run as build_retrieval_index
from data.processing.build_mongo import run as build_mongo
from data.processing.build_neo4j import run as build_neo4j
from data.processing.build_sqlite import run as build_sqlite
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.literature_agent import LiteratureAgent
from agents.molecule_agent import MoleculeAgent
from agents.router_agent import RouterAgent
from agents.safety_agent import SafetyAgent
from agents.scope_agent import ScopeAgent
from agents.synthesis_agent import SynthesisAgent
from agents.trial_agent import TrialAgent
from tools.mcp_client import MCPClientManager
from tools.native_tools import question_class_name
from core.storage import env_flag


def bootstrap_runtime(sync_to_mongodb: bool | None = None, sync_to_neo4j: bool | None = None) -> None:
    ingest_openfda()
    ingest_clinicaltrials()
    ingest_external_intelligence()
    ingest_pubmed()
    ingest_guideline_excerpts()
    ingest_opentargets()
    ingest_chembl()
    ingest_uniprot()
    ingest_who()
    ingest_drugbank_open()
    generate_synthetic_patients()
    build_sqlite()
    mongo_sync_enabled = sync_to_mongodb
    if mongo_sync_enabled is None:
        mongo_sync_enabled = env_flag("SYNC_TO_MONGODB", env_flag("USE_MONGODB_BACKEND", False))
    neo4j_sync_enabled = sync_to_neo4j
    if neo4j_sync_enabled is None:
        neo4j_sync_enabled = env_flag("SYNC_TO_NEO4J", env_flag("USE_NEO4J_BACKEND", False))
    build_mongo(sync=mongo_sync_enabled)
    build_neo4j(sync=neo4j_sync_enabled)
    build_retrieval_index()


class T2DOrchestrator:
    def __init__(self, bootstrap_if_needed: bool = True) -> None:
        live_backends_requested = any(
            env_flag(name, False)
            for name in ["USE_MONGODB_BACKEND", "USE_NEO4J_BACKEND", "SYNC_TO_MONGODB", "SYNC_TO_NEO4J"]
        )
        live_ingestion_requested = any(
            env_flag(name, False)
            for name in [
                "USE_LIVE_INGESTION",
                "USE_LIVE_OPENFDA_INGESTION",
                "USE_LIVE_CLINICALTRIALS_INGESTION",
                "USE_LIVE_PUBMED_INGESTION",
                "USE_LIVE_OPENTARGETS_INGESTION",
                "USE_LIVE_CHEMBL_INGESTION",
                "USE_LIVE_UNIPROT_INGESTION",
                "USE_LIVE_WHO_INGESTION",
                "USE_LIVE_DRUGBANK_OPEN_INGESTION",
            ]
        )
        if bootstrap_if_needed and (
            not SQLITE_DB.exists()
            or not GRAPH_FILE.exists()
            or not RETRIEVAL_MANIFEST.exists()
            or live_backends_requested
            or live_ingestion_requested
        ):
            bootstrap_runtime()
        self.resolver = get_resolver()
        self.mcp_client = MCPClientManager()
        self.router = RouterAgent()
        self.safety_agent = SafetyAgent(self.resolver, self.mcp_client)
        self.trial_agent = TrialAgent(self.resolver, self.mcp_client)
        self.knowledge_agent = KnowledgeGraphAgent(self.resolver, self.mcp_client)
        self.literature_agent = LiteratureAgent()
        self.molecule_agent = MoleculeAgent(self.resolver, self.mcp_client)
        self.scope_agent = ScopeAgent()
        self.synthesis_agent = SynthesisAgent()

    def run_query(self, query: str) -> dict:
        trace = TraceLogger()
        trace.add_event("query_received", {"query": query})
        route = self.router.route(query)
        question_class = route["question_class"]
        trace.add_event("routed", route)

        sections = []
        if question_class == "Q1":
            sections.append(self.safety_agent.run(query))
        elif question_class == "Q2":
            sections.append(self.trial_agent.run(query))
        elif question_class == "Q3":
            sections.append(self.knowledge_agent.run(query, question_class="Q3"))
        elif question_class == "Q4":
            sections.append(self.knowledge_agent.run(query, question_class="Q4"))
            sections.append(self.molecule_agent.run(query))
        elif question_class == "Q5":
            sections.append(self.literature_agent.run(query, question_class="Q5"))
        elif question_class == "Q6":
            sections.append(self.literature_agent.run(query, question_class="Q6"))
        else:
            sections.append(self.scope_agent.run(query, question_class=question_class))

        for section in sections:
            trace.add_event("agent_section", section.to_dict())
        response = self.synthesis_agent.run(question_class, query, sections, trace.trace_id)
        response.metadata["routing_mode"] = route.get("routing_mode", "deterministic")
        response.metadata["question_class_name"] = route.get("question_class_name", question_class_name(question_class))
        response.metadata["scope_family"] = route.get("scope_family")
        response.metadata["route_reason"] = route.get("route_reason")
        if route.get("ollama_suggested_label"):
            response.metadata["ollama_suggested_label"] = route["ollama_suggested_label"]
        trace.finalize(response.to_dict())
        return response.to_dict()

    def close(self) -> None:
        self.mcp_client.close_all()
