# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

from core.paths import GRAPH_FILE, RETRIEVAL_MANIFEST, SQLITE_DB
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
from agents.evidence_planner_agent import EvidencePlannerAgent
from agents.policy_agent import PolicyAgent
from agents.router_agent import RouterAgent
from agents.safety_agent import SafetyAgent
from agents.scope_agent import ScopeAgent
from agents.synthesis_agent import SynthesisAgent
from agents.trial_agent import TrialAgent
from agents.langgraph_workflow import T2DLangGraphWorkflow
from core.runtime_utils import env_flag
from tools.mcp_client import MCPClientManager


# Bootstrap local data stores and retrieval assets before serving queries
def bootstrap_runtime(sync_to_mongodb: bool | None = None, sync_to_neo4j: bool | None = None) -> None:
    """Rebuild runtime artefacts from the current fixture or live-ingested data."""
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


# Define the T2D orchestrator that wires runtime setup agents and workflows
class T2DOrchestrator:
    """Coordinate query routing, specialist agent execution, and final synthesis."""

    def __init__(self, bootstrap_if_needed: bool = True) -> None:
        if bootstrap_if_needed and (
            not SQLITE_DB.exists()
            or not GRAPH_FILE.exists()
            or not RETRIEVAL_MANIFEST.exists()
        ):
            bootstrap_runtime()
        self.resolver = get_resolver()
        self.mcp_client = MCPClientManager()
        self.router = RouterAgent()
        self.evidence_planner = EvidencePlannerAgent()
        self.safety_agent = SafetyAgent(self.resolver, self.mcp_client)
        self.trial_agent = TrialAgent(self.resolver, self.mcp_client)
        self.knowledge_agent = KnowledgeGraphAgent(self.resolver, self.mcp_client)
        self.literature_agent = LiteratureAgent()
        self.molecule_agent = MoleculeAgent(self.resolver, self.mcp_client)
        self.policy_agent = PolicyAgent()
        self.scope_agent = ScopeAgent()
        self.synthesis_agent = SynthesisAgent()
        self.workflow = T2DLangGraphWorkflow(
            router=self.router,
            policy_agent=self.policy_agent,
            evidence_planner=self.evidence_planner,
            safety_agent=self.safety_agent,
            trial_agent=self.trial_agent,
            knowledge_agent=self.knowledge_agent,
            literature_agent=self.literature_agent,
            molecule_agent=self.molecule_agent,
            scope_agent=self.scope_agent,
            synthesis_agent=self.synthesis_agent,
        )

    def run_query(self, query: str) -> dict:
        return self.workflow.invoke(query)

    def close(self) -> None:
        self.mcp_client.close_all()
