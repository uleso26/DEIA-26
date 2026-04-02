from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from core.models import AgentSection
from core.tracing import TraceLogger
from tools.native_tools import assess_evidence_sufficiency, build_evidence_plan, question_class_name


class OrchestratorState(TypedDict, total=False):
    query: str
    trace: TraceLogger
    understanding: dict[str, Any]
    evidence_plan: dict[str, Any]
    evidence_review: dict[str, Any]
    synthesis_metadata: dict[str, Any]
    react_steps: int
    executed_nodes: list[str]
    sections: Annotated[list[AgentSection], operator.add]
    response: dict[str, Any]


class T2DLangGraphWorkflow:
    """Graph-based execution plan for the T2D intelligence workflow."""

    def __init__(
        self,
        router: Any,
        policy_agent: Any,
        evidence_planner: Any,
        safety_agent: Any,
        trial_agent: Any,
        knowledge_agent: Any,
        literature_agent: Any,
        molecule_agent: Any,
        scope_agent: Any,
        synthesis_agent: Any,
    ) -> None:
        self.router = router
        self.policy_agent = policy_agent
        self.evidence_planner = evidence_planner
        self.safety_agent = safety_agent
        self.trial_agent = trial_agent
        self.knowledge_agent = knowledge_agent
        self.literature_agent = literature_agent
        self.molecule_agent = molecule_agent
        self.scope_agent = scope_agent
        self.synthesis_agent = synthesis_agent
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState)
        workflow.add_node("understand", self._understand_node)
        workflow.add_node("policy_gate", self._policy_gate_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("refine_plan", self._refine_plan_node)
        workflow.add_node("clarify", self._clarify_node)
        workflow.add_node("scope", self._scope_node)
        workflow.add_node("execute_plan", self._execute_plan_node)
        workflow.add_node("evidence_review", self._evidence_review_node)
        workflow.add_node("synthesize", self._synthesize_node)

        workflow.add_edge(START, "understand")
        workflow.add_edge("understand", "policy_gate")
        workflow.add_conditional_edges(
            "policy_gate",
            self._branch_for_policy,
            {
                "clarify": "clarify",
                "plan": "plan",
                "scope": "scope",
            },
        )
        workflow.add_edge("plan", "execute_plan")
        workflow.add_edge("clarify", "synthesize")
        workflow.add_edge("execute_plan", "evidence_review")
        workflow.add_edge("scope", "synthesize")
        workflow.add_conditional_edges(
            "evidence_review",
            self._branch_after_evidence_review,
            {
                "refine_plan": "refine_plan",
                "synthesize": "synthesize",
            },
        )
        workflow.add_edge("refine_plan", "execute_plan")
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    def mermaid_diagram(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def invoke(self, query: str) -> dict[str, Any]:
        trace = TraceLogger()
        trace.add_event("query_received", {"query": query})
        state: OrchestratorState = {
            "query": query,
            "trace": trace,
            "sections": [],
            "react_steps": 0,
            "executed_nodes": [],
        }
        result = self.graph.invoke(state)
        return result["response"]

    def _understand_node(self, state: OrchestratorState) -> OrchestratorState:
        understanding = self.router.understand(state["query"])
        state["trace"].add_event("understood_query", understanding)
        return {"understanding": understanding}

    def _policy_gate_node(self, state: OrchestratorState) -> OrchestratorState:
        understanding = state["understanding"]
        payload = {
            "interaction_mode": understanding["interaction_mode"],
            "needs_clarification": understanding.get("needs_clarification", False),
            "question_class": understanding["question_class"],
        }
        state["trace"].add_event("policy_gate", payload)
        return {}

    def _branch_for_policy(self, state: OrchestratorState) -> str:
        understanding = state["understanding"]
        if understanding.get("needs_clarification"):
            return "clarify"
        if understanding["interaction_mode"] != "enterprise_intelligence":
            return "scope"
        return "plan"

    def _plan_node(self, state: OrchestratorState) -> OrchestratorState:
        base_plan = build_evidence_plan(state["understanding"])
        evidence_plan = self.evidence_planner.plan(state["understanding"], base_plan)
        state["trace"].add_event("evidence_plan", evidence_plan)
        return {"evidence_plan": evidence_plan}

    def _refine_plan_node(self, state: OrchestratorState) -> OrchestratorState:
        refined_plan = self.evidence_planner.refine_after_observation(
            state["understanding"],
            state["evidence_plan"],
            state.get("sections", []),
            state.get("evidence_review", {}),
            react_steps=state.get("react_steps", 0),
            executed_nodes=state.get("executed_nodes", []),
        )
        updated_steps = state.get("react_steps", 0) + 1
        state["trace"].add_event(
            "evidence_plan_refined",
            {
                "react_steps": updated_steps,
                "execution_nodes": refined_plan.get("execution_nodes", []),
                "supplemental_nodes": refined_plan.get("supplemental_nodes", []),
                "planning_mode": refined_plan.get("planning_mode"),
            },
        )
        return {"evidence_plan": refined_plan, "react_steps": updated_steps}

    def _clarify_node(self, state: OrchestratorState) -> OrchestratorState:
        section = self.policy_agent.clarification(state["understanding"])
        state["trace"].add_event("clarification_requested", section.to_dict())
        return {"sections": [section]}

    def _scope_node(self, state: OrchestratorState) -> OrchestratorState:
        section = self.scope_agent.run(state["query"], question_class=state["understanding"]["question_class"])
        state["trace"].add_event("agent_section", section.to_dict())
        return {"sections": [section]}

    def _execute_plan_node(self, state: OrchestratorState) -> OrchestratorState:
        sections: list[AgentSection] = []
        executed_nodes = list(state.get("executed_nodes", []))
        for node_name in state["evidence_plan"].get("execution_nodes", []):
            if node_name in executed_nodes:
                continue
            if node_name == "safety":
                section = self.safety_agent.run(state["query"])
            elif node_name == "trial":
                section = self.trial_agent.run(state["query"])
            elif node_name == "pathway":
                section = self.knowledge_agent.run(state["query"], question_class="Q3")
            elif node_name == "knowledge":
                section = self.knowledge_agent.run(state["query"], question_class="Q4")
            elif node_name == "molecule":
                section = self.molecule_agent.run(state["query"])
            elif node_name == "literature_q5":
                section = self.literature_agent.run(state["query"], question_class="Q5")
            elif node_name == "literature_q6":
                section = self.literature_agent.run(state["query"], question_class="Q6")
            else:
                continue
            state["trace"].add_event("agent_section", {"execution_node": node_name, **section.to_dict()})
            sections.append(section)
            executed_nodes.append(node_name)
        return {"sections": sections, "executed_nodes": executed_nodes}

    def _evidence_review_node(self, state: OrchestratorState) -> OrchestratorState:
        evidence_review = assess_evidence_sufficiency(state["understanding"], state.get("sections", []))
        synthesis_metadata: dict[str, Any] = {"evidence_review": evidence_review}
        if evidence_review["status"] in {"limited", "insufficient"}:
            synthesis_metadata.update(
                self.policy_agent.insufficient_evidence_metadata(state["understanding"], evidence_review)
            )
        state["trace"].add_event("evidence_review", evidence_review)
        return {"evidence_review": evidence_review, "synthesis_metadata": synthesis_metadata}

    def _branch_after_evidence_review(self, state: OrchestratorState) -> str:
        evidence_review = state.get("evidence_review", {})
        if evidence_review.get("status") not in {"limited", "insufficient"}:
            return "synthesize"
        evidence_plan = state.get("evidence_plan", {})
        if not evidence_plan.get("react_eligible"):
            return "synthesize"
        if state.get("react_steps", 0) >= int(evidence_plan.get("max_react_steps", 0)):
            return "synthesize"
        available_nodes = [
            node
            for node in evidence_plan.get("supplemental_nodes", [])
            if node not in state.get("executed_nodes", [])
        ]
        return "refine_plan" if available_nodes else "synthesize"

    def _synthesize_node(self, state: OrchestratorState) -> OrchestratorState:
        understanding = state["understanding"]
        response = self.synthesis_agent.run(
            understanding["question_class"],
            state["query"],
            state.get("sections", []),
            state["trace"].trace_id,
            workflow_metadata=state.get("synthesis_metadata"),
        ).to_dict()
        response["metadata"]["routing_mode"] = understanding.get("routing_mode", "deterministic")
        response["metadata"]["question_class_name"] = understanding.get(
            "question_class_name",
            question_class_name(understanding["question_class"]),
        )
        response["metadata"]["scope_family"] = understanding.get("scope_family")
        response["metadata"]["route_reason"] = understanding.get("route_reason")
        response["metadata"]["interaction_mode"] = understanding.get("interaction_mode")
        response["metadata"]["primary_intent"] = understanding.get("primary_intent")
        response["metadata"]["routing_confidence"] = understanding.get("confidence")
        response["metadata"]["entities"] = understanding.get("entities")
        response["metadata"]["needs_clarification"] = understanding.get("needs_clarification", False)
        if understanding.get("clarification_prompt"):
            response["metadata"]["clarification_prompt"] = understanding["clarification_prompt"]
        if state.get("evidence_plan"):
            response["metadata"]["evidence_plan"] = state["evidence_plan"]
        if state.get("evidence_review"):
            response["metadata"]["evidence_review"] = state["evidence_review"]
        if understanding.get("ollama_suggested_label"):
            response["metadata"]["ollama_suggested_label"] = understanding["ollama_suggested_label"]
        state["trace"].finalize(response)
        return {"response": response}
