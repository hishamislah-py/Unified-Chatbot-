from typing import TYPE_CHECKING
import sys
from pathlib import Path

# Add parent directory to path to import from langGraph
sys.path.append(str(Path(__file__).parent.parent))

from langGraph import PolicyTools

if TYPE_CHECKING:
    from .multi_agent_graph import MultiAgentState


# =============================================================================
# HR AGENT NODES
# =============================================================================

def hr_agent_entry_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    HR Agent entry point - classifies HR-specific intent
    """
    state.setdefault('workflow_path', []).append('HR Agent Entry')
    state['current_agent'] = 'hr'

    tools = PolicyTools()
    classification = tools.classify_intent(state['current_message'])

    state['specialist_intent'] = classification['intent']
    state['category'] = classification['category']

    return state


def hr_clarification_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    HR Agent clarification - asks for more details on vague HR questions
    """
    state.setdefault('workflow_path', []).append('HR Clarification')

    tools = PolicyTools()
    clarification = tools.generate_clarification(
        state['current_message'],
        "Your question about HR policies needs more detail"
    )

    state['needs_clarification'] = True
    state['answer'] = f"[HR Agent] {clarification}"
    state['sources'] = []
    state['is_valid'] = True

    return state


def hr_rag_retrieval_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    HR Agent RAG retrieval - retrieves from HR documents ONLY
    """
    state.setdefault('workflow_path', []).append('HR RAG Retrieval')

    tools = PolicyTools()

    # Force category to HR/Leave for HR agent
    if state['category'] not in ["HR", "Leave"]:
        state['category'] = "HR"

    chunks = tools.retrieve_policy(
        state['current_message'],
        state['category'],
        num_chunks=4
    )

    state['retrieved_chunks'] = chunks

    return state


def hr_answer_generation_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    HR Agent answer generation - generates answer with citations (synchronous)
    """
    state.setdefault('workflow_path', []).append('HR Answer Generation')

    tools = PolicyTools()
    result = tools.generate_answer_with_citations(
        state['current_message'],
        state['retrieved_chunks']
    )

    state['answer'] = f"[HR Agent] {result['answer']}"
    state['sources'] = result['sources']

    return state


async def hr_answer_generation_node_stream(state: "MultiAgentState") -> "MultiAgentState":
    """
    HR Agent answer generation - streaming version
    Accumulates tokens from streaming LLM response
    """
    state.setdefault('workflow_path', []).append('HR Answer Generation')

    tools = PolicyTools()

    # Accumulate streamed response
    accumulated_answer = ""
    async for token in tools.generate_answer_with_citations_stream(
        state['current_message'],
        state['retrieved_chunks']
    ):
        accumulated_answer += token

    # Extract sources from retrieved chunks
    sources = [
        {
            "source": chunk['source'],
            "page": chunk['page'],
            "rank": chunk['rank'],
            "preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
        }
        for chunk in state['retrieved_chunks']
    ]

    state['answer'] = f"[HR Agent] {accumulated_answer}"
    state['sources'] = sources

    return state


def hr_validation_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    HR Agent validation - validates answer quality
    """
    state.setdefault('workflow_path', []).append('HR Validation')

    tools = PolicyTools()
    validation = tools.validate_answer(
        state['answer'],
        state['sources'],
        state['current_message']
    )

    state['is_valid'] = validation['is_valid']
    state['validation_reason'] = validation['reason']

    # Handle retry logic
    if not validation['is_valid']:
        retry_count = state.get('retry_count', 0)
        if retry_count < 2:
            state['retry_count'] = retry_count + 1
        else:
            # Max retries reached, provide fallback
            state['answer'] = (
                "[HR Agent] I apologize, but I'm having trouble providing a confident answer to your question. "
                "This might be because:\n"
                "- The information is not in our HR policy documents\n"
                "- The question needs to be more specific\n"
                "- Multiple policies may apply\n\n"
                "Please try rephrasing your question or contact HR directly for assistance."
            )
            state['is_valid'] = True

    return state


def hr_out_of_scope_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    HR Agent out-of-scope handler - stays in HR agent, politely declines
    CRITICAL: Does NOT transfer to other agents
    """
    state.setdefault('workflow_path', []).append('HR Out of Scope')

    state['answer'] = (
        "[HR Agent] I specialize in HR and Leave policies (hiring, termination, probation, "
        "annual leave, sick leave, maternity leave, etc.). "
        "Your question seems outside my area of expertise.\n\n"
        "If you need IT support or have technical questions, please ask the Personal Assistant "
        "to connect you to IT Support."
    )
    state['sources'] = []
    state['is_valid'] = True

    return state


# =============================================================================
# IT AGENT NODES
# =============================================================================

def it_agent_entry_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    IT Agent entry point - classifies IT-specific intent
    Uses IT-specific classifier with troubleshooting support
    """
    state.setdefault('workflow_path', []).append('IT Agent Entry')
    state['current_agent'] = 'it'

    try:
        tools = PolicyTools()
        # Use IT-specific intent classifier with troubleshooting support
        classification = tools.classify_it_intent(state['current_message'])

        state['specialist_intent'] = classification['intent']
        state['category'] = classification['category']

        # Debug logging
        print(f"[IT Entry] Message: {state['current_message']}")
        print(f"[IT Entry] Classified intent: {classification['intent']}")

    except Exception as e:
        # If classification fails, default to troubleshooting
        print(f"[IT Entry] Classification error: {e}")
        state['specialist_intent'] = 'troubleshooting'
        state['category'] = 'IT'

    return state


def it_clarification_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    IT Agent clarification - asks for more details on vague IT questions
    """
    state.setdefault('workflow_path', []).append('IT Clarification')

    tools = PolicyTools()
    clarification = tools.generate_clarification(
        state['current_message'],
        "Your question about IT policies needs more detail"
    )

    state['needs_clarification'] = True
    state['answer'] = f"[IT Support] {clarification}"
    state['sources'] = []
    state['is_valid'] = True

    return state


def it_rag_retrieval_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    IT Agent RAG retrieval - retrieves from IT documents ONLY
    """
    state.setdefault('workflow_path', []).append('IT RAG Retrieval')

    tools = PolicyTools()

    # Force category to IT/Compliance for IT agent
    if state['category'] not in ["IT", "Compliance"]:
        state['category'] = "IT"

    chunks = tools.retrieve_policy(
        state['current_message'],
        state['category'],
        num_chunks=4
    )

    state['retrieved_chunks'] = chunks

    return state


def it_answer_generation_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    IT Agent answer generation - generates answer with citations (synchronous)
    """
    state.setdefault('workflow_path', []).append('IT Answer Generation')

    tools = PolicyTools()
    result = tools.generate_answer_with_citations(
        state['current_message'],
        state['retrieved_chunks']
    )

    state['answer'] = f"[IT Support] {result['answer']}"
    state['sources'] = result['sources']

    return state


async def it_answer_generation_node_stream(state: "MultiAgentState") -> "MultiAgentState":
    """
    IT Agent answer generation - streaming version
    Accumulates tokens from streaming LLM response
    """
    state.setdefault('workflow_path', []).append('IT Answer Generation')

    tools = PolicyTools()

    # Accumulate streamed response
    accumulated_answer = ""
    async for token in tools.generate_answer_with_citations_stream(
        state['current_message'],
        state['retrieved_chunks']
    ):
        accumulated_answer += token

    # Extract sources from retrieved chunks
    sources = [
        {
            "source": chunk['source'],
            "page": chunk['page'],
            "rank": chunk['rank'],
            "preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
        }
        for chunk in state['retrieved_chunks']
    ]

    state['answer'] = f"[IT Support] {accumulated_answer}"
    state['sources'] = sources

    return state


def it_validation_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    IT Agent validation - validates answer quality
    """
    state.setdefault('workflow_path', []).append('IT Validation')

    tools = PolicyTools()
    validation = tools.validate_answer(
        state['answer'],
        state['sources'],
        state['current_message']
    )

    state['is_valid'] = validation['is_valid']
    state['validation_reason'] = validation['reason']

    # Handle retry logic
    if not validation['is_valid']:
        retry_count = state.get('retry_count', 0)
        if retry_count < 2:
            state['retry_count'] = retry_count + 1
        else:
            # Max retries reached, provide fallback
            state['answer'] = (
                "[IT Support] I apologize, but I'm having trouble providing a confident answer to your question. "
                "This might be because:\n"
                "- The information is not in our IT policy documents\n"
                "- The question needs to be more specific\n"
                "- Multiple policies may apply\n\n"
                "Please try rephrasing your question or contact IT Support directly for assistance."
            )
            state['is_valid'] = True

    return state


def it_out_of_scope_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    IT Agent out-of-scope handler - stays in IT agent, politely declines
    CRITICAL: Does NOT transfer to other agents
    """
    state.setdefault('workflow_path', []).append('IT Out of Scope')

    state['answer'] = (
        "[IT Support] I specialize in IT Security and Compliance policies (device security, "
        "passwords, VPN, data privacy, code of conduct, etc.). "
        "Your question seems outside my area of expertise.\n\n"
        "If you need HR assistance or have questions about employee policies, please ask the "
        "Personal Assistant to connect you to the HR Agent."
    )
    state['sources'] = []
    state['is_valid'] = True

    return state


def it_troubleshooting_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    IT troubleshooting - FIRST checks RAG for relevant documents, then falls back to LLM knowledge.
    Uses semantic relevance checking to determine if RAG results actually match the question.
    For technical issues like 'Teams not working', 'mouse not working', etc.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    state.setdefault('workflow_path', []).append('IT Troubleshooting')

    # Store the original issue for potential JIRA ticket creation later
    state['original_issue'] = state['current_message']

    tools = PolicyTools()
    question = state['current_message']

    # =================================================================
    # STEP 1: Retrieve documents from RAG
    # =================================================================
    print(f"[IT Troubleshooting] Checking RAG for: {question}")

    # Force category to IT for RAG search
    rag_chunks = tools.retrieve_policy(question, "IT", num_chunks=4)

    # =================================================================
    # STEP 2: Check SEMANTIC RELEVANCE of retrieved chunks
    # =================================================================
    has_relevant_rag_results = False

    if rag_chunks and len(rag_chunks) > 0:
        # Use semantic relevance check instead of just length check
        relevance_result = tools.check_context_relevance(question, rag_chunks)

        print(f"[IT Troubleshooting] Relevance check: {relevance_result}")

        if relevance_result['is_relevant']:
            has_relevant_rag_results = True
        else:
            # Also check for explicit keyword matches as fallback
            # This handles cases where "teams" in question should match "Teams" document
            question_lower = question.lower()
            for chunk in rag_chunks:
                source = chunk.get('source', '').lower()
                # Direct source match check
                if 'teams' in question_lower and 'teams' in source:
                    has_relevant_rag_results = True
                    print(f"[IT Troubleshooting] Source match found: {source}")
                    break
                elif 'url' in question_lower and 'url' in source:
                    has_relevant_rag_results = True
                    break
                elif 'outlook' in question_lower and 'outlook' in source:
                    has_relevant_rag_results = True
                    break
                elif 'onedrive' in question_lower and 'onedrive' in source:
                    has_relevant_rag_results = True
                    break
                elif 'sharepoint' in question_lower and 'sharepoint' in source:
                    has_relevant_rag_results = True
                    break
                elif ('mouse' in question_lower or 'keyboard' in question_lower or 'touchpad' in question_lower) and 'hardware' in source:
                    has_relevant_rag_results = True
                    break
                elif ('camera' in question_lower or 'mic' in question_lower or 'headset' in question_lower) and ('camera' in source or 'mic' in source or 'headset' in source):
                    has_relevant_rag_results = True
                    break
                elif ('freeze' in question_lower or 'freezing' in question_lower) and 'freezing' in source:
                    has_relevant_rag_results = True
                    break
                elif 'screenshare' in question_lower and 'screenshare' in source:
                    has_relevant_rag_results = True
                    break
                elif 'vm' in question_lower and 'vm' in source:
                    has_relevant_rag_results = True
                    break

    print(f"[IT Troubleshooting] Final relevance decision: {has_relevant_rag_results}")

    # =================================================================
    # STEP 3: Generate answer based on relevance
    # =================================================================
    jira_offer = "\n\nIf this doesn't resolve your issue, let me know and I can help create a JIRA ticket for further assistance."

    if has_relevant_rag_results:
        # Use RAG results - generate answer with citations
        print("[IT Troubleshooting] Using RAG-based answer with citations")
        result = tools.generate_hybrid_answer(question, rag_chunks, use_rag=True)
        state['answer'] = f"[IT Support] {result['answer']}{jira_offer}"
        state['sources'] = result['sources']
    else:
        # Fall back to LLM knowledge (no citations)
        print("[IT Troubleshooting] No relevant RAG results, using LLM knowledge")
        result = tools.generate_hybrid_answer(question, rag_chunks, use_rag=False)
        state['answer'] = f"[IT Support] {result['answer']}{jira_offer}"
        state['sources'] = []  # No sources when using LLM knowledge

    state['is_valid'] = True

    return state


def it_jira_offer_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    IT JIRA ticket offer - when previous solution didn't work
    Offers to create a JIRA ticket for further assistance
    """
    state.setdefault('workflow_path', []).append('IT JIRA Offer')

    # Set flag to indicate we're awaiting JIRA confirmation
    state['awaiting_jira_confirmation'] = True

    state['answer'] = (
        "[IT Support] I'm sorry the previous solutions didn't resolve your issue. "
        "Would you like me to create a JIRA ticket for further assistance? "
        "An IT specialist will review your case and get back to you.\n\n"
        "Just say **'yes'** or **'create ticket'** to proceed."
    )
    state['sources'] = []
    state['is_valid'] = True

    return state


async def it_jira_create_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    IT JIRA ticket creation - creates ticket via MCP when user confirms

    Extracts the original issue from state and creates a JIRA ticket.
    Returns confirmation with ticket ID and URL.
    """
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from mcp_client import mcp_client

    state.setdefault('workflow_path', []).append('IT JIRA Create')

    # Get the original issue from state
    original_issue = state.get('original_issue', '')

    if not original_issue:
        # Fallback: use current message context
        original_issue = state.get('current_message', 'IT Support issue reported via chatbot')

    # Create ticket summary and description
    summary = f"IT Support: {original_issue[:100]}"
    description = f"""**Issue Reported:** {original_issue}

---
*Auto-generated by IT Support Chatbot*"""

    try:
        # Call MCP to create ticket
        result = await mcp_client.create_jira_issue(
            summary=summary,
            description=description,
            issue_type="Task",
            project_key="KAN"
        )

        if result.success:
            state['jira_ticket_id'] = result.ticket_id or ""
            state['jira_ticket_url'] = result.ticket_url or ""
            state['answer'] = (
                f"[IT Support] I've created a JIRA ticket for your issue.\n\n"
                f"**Ticket ID:** {result.ticket_id}\n\n"
                f"Our IT team will review your case and get back to you soon."
            )
        else:
            state['answer'] = (
                f"[IT Support] I apologize, but I encountered an error creating the ticket: "
                f"{result.error}\n\n"
                f"Please try again later or contact IT support directly."
            )

    except Exception as e:
        state['answer'] = (
            f"[IT Support] I apologize, but there was an unexpected error creating your ticket. "
            f"Please try again or contact IT support directly.\n\n"
            f"Error: {str(e)}"
        )

    state['sources'] = []
    state['is_valid'] = True
    state['awaiting_jira_confirmation'] = False

    return state
