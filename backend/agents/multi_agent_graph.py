from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .personal_assistant import personal_assistant_node
from .specialist_agents import (
    hr_agent_entry_node,
    hr_clarification_node,
    hr_rag_retrieval_node,
    hr_answer_generation_node,
    hr_validation_node,
    hr_out_of_scope_node,
    it_agent_entry_node,
    it_clarification_node,
    it_rag_retrieval_node,
    it_answer_generation_node,
    it_validation_node,
    it_out_of_scope_node,
    it_troubleshooting_node,
    it_jira_offer_node,
    it_jira_create_node,
)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class MultiAgentState(TypedDict):
    """
    State schema for multi-agent chatbot system
    Tracks conversation, agent routing, and RAG results
    """
    # Core conversation
    current_message: str          # Latest user message
    answer: str                   # Generated answer

    # Agent routing
    current_agent: str            # Which agent is active ("personal", "hr", "it")
    transfer_requested: bool      # Whether user explicitly requested transfer
    target_agent: str             # Which agent to transfer to (if transfer_requested)

    # Personal Assistant specific
    intent: str                   # "greeting", "general_query", "transfer_request", "out_of_scope"

    # Specialist agent specific (HR/IT)
    specialist_intent: str        # "policy_query", "ambiguous", "out_of_scope"
    category: str                 # Policy category (HR, Leave, IT, Compliance)
    retrieved_chunks: list        # RAG results
    sources: list                 # Citations with page numbers
    needs_clarification: bool     # Whether clarification is needed
    is_valid: bool                # Whether answer passed validation
    retry_count: int              # Number of retries attempted
    validation_reason: str        # Reason for validation result

    # Session management
    session_id: str               # Session identifier
    workflow_path: list           # Track which nodes were executed

    # JIRA ticket creation
    original_issue: str           # Store the original IT issue for ticket creation
    jira_ticket_id: str           # Created ticket ID (if any)
    jira_ticket_url: str          # Created ticket URL (if any)
    awaiting_jira_confirmation: bool  # Whether we're waiting for user to confirm ticket creation


# =============================================================================
# ROUTER FUNCTIONS
# =============================================================================

def route_from_personal_assistant(state: MultiAgentState) -> Literal["hr_entry", "it_entry", "end"]:
    """
    Router 1: Route from Personal Assistant based on EXPLICIT transfer requests

    IMPORTANT: When transfer is requested, we just acknowledge it and END.
    The actual routing to HR/IT happens on the NEXT user message when current_agent is already set.
    """
    # Always END from Personal Assistant
    # The transfer happens by setting current_agent in the state
    return "end"


def route_from_hr_entry(state: MultiAgentState) -> Literal["hr_clarification", "hr_rag_retrieval", "hr_out_of_scope"]:
    """
    Router 2: Route within HR agent based on intent
    """
    intent = state.get('specialist_intent', '')

    if intent == "ambiguous":
        return "hr_clarification"
    elif intent == "policy_query":
        return "hr_rag_retrieval"
    else:  # out_of_scope or simple_fact
        return "hr_out_of_scope"


def route_from_hr_validation(state: MultiAgentState) -> Literal["hr_rag_retrieval", "end"]:
    """
    Router 3: Retry RAG retrieval or end (HR agent)
    """
    if not state.get('is_valid', True) and state.get('retry_count', 0) < 2:
        return "hr_rag_retrieval"
    return "end"


def route_from_it_entry(state: MultiAgentState) -> Literal[
    "it_clarification", "it_rag_retrieval", "it_troubleshooting",
    "it_jira_offer", "it_jira_create", "it_out_of_scope"
]:
    """
    Router 4: Route within IT agent based on intent
    Supports: policy_query, troubleshooting, follow_up_issue,
              jira_confirmation, jira_create_direct, ambiguous, out_of_scope
    """
    intent = state.get('specialist_intent', '')
    awaiting_confirmation = state.get('awaiting_jira_confirmation', False)

    # Debug logging
    print(f"[IT Router] Routing with intent: '{intent}'")
    print(f"[IT Router] Awaiting JIRA confirmation: {awaiting_confirmation}")

    if intent == "ambiguous":
        print("[IT Router] -> it_clarification")
        return "it_clarification"
    elif intent == "policy_query":
        print("[IT Router] -> it_rag_retrieval")
        return "it_rag_retrieval"
    elif intent == "troubleshooting":
        print("[IT Router] -> it_troubleshooting")
        return "it_troubleshooting"
    elif intent == "follow_up_issue":
        print("[IT Router] -> it_jira_offer")
        return "it_jira_offer"
    elif intent == "jira_confirmation" and awaiting_confirmation:
        # User confirmed ticket creation after offer
        print("[IT Router] -> it_jira_create (confirmation)")
        return "it_jira_create"
    elif intent == "jira_create_direct":
        # User directly requested JIRA ticket creation
        print("[IT Router] -> it_jira_create (direct)")
        return "it_jira_create"
    elif intent == "jira_confirmation" and not awaiting_confirmation:
        # User said "yes" but we weren't expecting confirmation
        # Treat as out of scope or clarification needed
        print("[IT Router] -> it_out_of_scope (unexpected confirmation)")
        return "it_out_of_scope"
    else:  # out_of_scope
        print(f"[IT Router] -> it_out_of_scope (unrecognized intent: '{intent}')")
        return "it_out_of_scope"


def route_from_it_validation(state: MultiAgentState) -> Literal["it_rag_retrieval", "end"]:
    """
    Router 5: Retry RAG retrieval or end (IT agent)
    """
    if not state.get('is_valid', True) and state.get('retry_count', 0) < 2:
        return "it_rag_retrieval"
    return "end"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_multi_agent_graph():
    """
    Build the complete multi-agent LangGraph workflow

    Flow:
    START → Personal Assistant → [transfer?] → HR/IT Agent → END

    Personal Assistant:
    - Greets users
    - Answers general questions
    - Routes to specialists ONLY on explicit request

    HR Agent:
    - Entry → Clarification / RAG Retrieval / Out-of-Scope
    - RAG Retrieval → Answer Generation → Validation → END / Retry

    IT Agent:
    - Entry → Clarification / RAG Retrieval / Out-of-Scope
    - RAG Retrieval → Answer Generation → Validation → END / Retry
    """
    workflow = StateGraph(MultiAgentState)

    # ==========================================================================
    # ADD NODES
    # ==========================================================================

    # Personal Assistant
    workflow.add_node("personal_assistant", personal_assistant_node)

    # HR Agent nodes
    workflow.add_node("hr_entry", hr_agent_entry_node)
    workflow.add_node("hr_clarification", hr_clarification_node)
    workflow.add_node("hr_rag_retrieval", hr_rag_retrieval_node)
    workflow.add_node("hr_answer_generation", hr_answer_generation_node)
    workflow.add_node("hr_validation", hr_validation_node)
    workflow.add_node("hr_out_of_scope", hr_out_of_scope_node)

    # IT Agent nodes
    workflow.add_node("it_entry", it_agent_entry_node)
    workflow.add_node("it_clarification", it_clarification_node)
    workflow.add_node("it_rag_retrieval", it_rag_retrieval_node)
    workflow.add_node("it_answer_generation", it_answer_generation_node)
    workflow.add_node("it_validation", it_validation_node)
    workflow.add_node("it_out_of_scope", it_out_of_scope_node)
    workflow.add_node("it_troubleshooting", it_troubleshooting_node)
    workflow.add_node("it_jira_offer", it_jira_offer_node)
    workflow.add_node("it_jira_create", it_jira_create_node)

    # ==========================================================================
    # SET ENTRY POINT
    # ==========================================================================

    workflow.set_entry_point("personal_assistant")

    # ==========================================================================
    # ADD EDGES - PERSONAL ASSISTANT
    # ==========================================================================

    # Personal Assistant routes to HR, IT, or END based on transfer request
    workflow.add_conditional_edges(
        "personal_assistant",
        route_from_personal_assistant,
        {
            "hr_entry": "hr_entry",
            "it_entry": "it_entry",
            "end": END
        }
    )

    # ==========================================================================
    # ADD EDGES - HR AGENT
    # ==========================================================================

    # HR entry routes to clarification, RAG, or out-of-scope
    workflow.add_conditional_edges(
        "hr_entry",
        route_from_hr_entry,
        {
            "hr_clarification": "hr_clarification",
            "hr_rag_retrieval": "hr_rag_retrieval",
            "hr_out_of_scope": "hr_out_of_scope"
        }
    )

    # RAG retrieval always goes to answer generation
    workflow.add_edge("hr_rag_retrieval", "hr_answer_generation")

    # Answer generation always goes to validation
    workflow.add_edge("hr_answer_generation", "hr_validation")

    # Validation can retry or end
    workflow.add_conditional_edges(
        "hr_validation",
        route_from_hr_validation,
        {
            "hr_rag_retrieval": "hr_rag_retrieval",
            "end": END
        }
    )

    # Clarification and out-of-scope go directly to END
    workflow.add_edge("hr_clarification", END)
    workflow.add_edge("hr_out_of_scope", END)

    # ==========================================================================
    # ADD EDGES - IT AGENT
    # ==========================================================================

    # IT entry routes to clarification, RAG, troubleshooting, JIRA offer/create, or out-of-scope
    workflow.add_conditional_edges(
        "it_entry",
        route_from_it_entry,
        {
            "it_clarification": "it_clarification",
            "it_rag_retrieval": "it_rag_retrieval",
            "it_troubleshooting": "it_troubleshooting",
            "it_jira_offer": "it_jira_offer",
            "it_jira_create": "it_jira_create",
            "it_out_of_scope": "it_out_of_scope"
        }
    )

    # RAG retrieval always goes to answer generation
    workflow.add_edge("it_rag_retrieval", "it_answer_generation")

    # Answer generation always goes to validation
    workflow.add_edge("it_answer_generation", "it_validation")

    # Validation can retry or end
    workflow.add_conditional_edges(
        "it_validation",
        route_from_it_validation,
        {
            "it_rag_retrieval": "it_rag_retrieval",
            "end": END
        }
    )

    # Clarification, out-of-scope, troubleshooting, JIRA offer, and JIRA create go directly to END
    workflow.add_edge("it_clarification", END)
    workflow.add_edge("it_out_of_scope", END)
    workflow.add_edge("it_troubleshooting", END)
    workflow.add_edge("it_jira_offer", END)
    workflow.add_edge("it_jira_create", END)

    # ==========================================================================
    # COMPILE WITH MEMORY
    # ==========================================================================

    # MemorySaver enables session persistence across invocations
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app
