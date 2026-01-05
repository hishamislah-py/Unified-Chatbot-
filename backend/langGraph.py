import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import your existing RAG system
from rag_node import SimpleRAG

# Load environment variables
load_dotenv()

# STATE DEFINITION

class PolicyAssistantState(TypedDict):

    question: str                    # User's original question
    intent: str                      # Classified intent (policy_query, simple_fact, etc.)
    category: str                    # Policy category (HR, Leave, IT, Compliance)
    retrieved_chunks: list           # Documents retrieved from RAG
    answer: str                      # Final answer to return
    sources: list                    # Source citations with page numbers
    needs_clarification: bool        # Whether we need to ask user for more info
    is_valid: bool                   # Whether answer passed validation
    retry_count: int                 # Number of retries attempted
    validation_reason: str           # Reason for validation result
    workflow_path: list              # Track which nodes were executed


# AGENT TOOLS

class PolicyTools:

    # Class-level storage for RAG system 
    _rag_system = None

    @classmethod
    def set_rag_system(cls, rag_system: SimpleRAG):
        """Set the RAG system at class level"""
        cls._rag_system = rag_system

    def __init__(self):
        if PolicyTools._rag_system is None:
            raise ValueError("RAG system not initialized. Call PolicyTools.set_rag_system() first.")
        self.rag = PolicyTools._rag_system
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    
    def classify_intent(self, question: str) -> dict:
        """
        Tool 1: Classify user's intent and category
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for an enterprise policy assistant.

Classify the user's question into ONE of these intents:

1. "policy_query" - SPECIFIC questions about company policies (requires document retrieval)
   - Example: "What is the sick leave policy?" → SPECIFIC (mentions sick leave)
   - Example: "Can I carry forward leave if I resign?" → SPECIFIC
   - Example: "What is the password policy for remote access?" → SPECIFIC
   - RULE: Question mentions a SPECIFIC policy type, feature, or scenario

2. "simple_fact" - Simple factual questions ABOUT THE COMPANY that don't need document retrieval
   - Example: "What is the company name?"
   - Example: "What are the working hours?"
   - RULE: Must be a basic question about company information, NOT external facts

3. "ambiguous" - Question is too BROAD/VAGUE and needs clarification
   - Example: "what is leave policy" → AMBIGUOUS (doesn't specify which leave type)
   - Example: "what is the leave policy" → AMBIGUOUS (doesn't specify which leave type)
   - Example: "tell me about HR" → AMBIGUOUS (which HR policy?)
   - Example: "what is the policy" → AMBIGUOUS (which policy?)
   - RULE: Question asks about a general category WITHOUT specifying the exact type
   - KEY: "leave policy" is broad, but "sick leave policy" is specific

4. "out_of_scope" - Question is NOT about company policies or company information
   - Example: "What's the weather today?"
   - Example: "Who is the indian president?"
   - Example: "What is the capital of France?"
   - RULE: Any question about external facts, world knowledge, or topics unrelated to the company

Also identify the policy category if applicable:
- "HR" - Hiring, termination, probation, employee rights
- "Leave" - Annual leave, sick leave, maternity, carry-forward
- "IT" - Security, devices, passwords, VPN
- "Compliance" - Data privacy, code of conduct, regulations
- "General" - General company information
- "None" - Not policy-related

Respond in this exact format:
INTENT: <intent>
CATEGORY: <category>
REASON: <brief reason>"""),
            ("user", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": question})
        
        # Parse response
        lines = response.strip().split('\n')
        intent = "policy_query"  # default
        category = "General"
        
        for line in lines:
            if line.startswith("INTENT:"):
                intent = line.split(":", 1)[1].strip().lower()
            elif line.startswith("CATEGORY:"):
                category = line.split(":", 1)[1].strip()
        
        return {
            "intent": intent,
            "category": category
        }
    
    def retrieve_policy(self, question: str, category: str, num_chunks: int = 3) -> list:
        """
        Tool 2: Retrieve relevant policy documents with source tracking
        Routes to appropriate indexes based on category
        """
        try:
            # Route to appropriate document indexes based on category
            if category in ["HR", "Leave"]:
                # Search HR policy indexes: Leave Policy.pdf and HR_Policy_Art_Technology.pdf
                results = self.rag.search_hr_policies(question, num_results=num_chunks)
            elif category in ["IT", "Compliance"]:
                # Search IT policy indexes: IT_Security_Policy_AI_Usage.pdf and Compliance Handbook.pdf
                results = self.rag.search_it_policies(question, num_results=num_chunks)
            else:
                # Category doesn't match HR/Leave or IT/Compliance - return empty results
                results = []

            return results
        except Exception as e:
            return []
    
    def generate_answer_with_citations(self, question: str, context_chunks: list) -> dict:
        """
        Tool 3: Generate answer with proper citations
        """
        if not context_chunks:
            return {
                "answer": "I couldn't find relevant information in the policy documents.",
                "sources": []
            }
        
        # Combine chunks into context with source markers
        context = "\n\n".join([
            f"[Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful enterprise policy assistant.

CRITICAL RULES:
1. Answer ONLY using the provided context
2. ALWAYS cite your sources using this format: [Source: filename, Page X]
3. If the answer is not in the context, say "I don't have enough information"
4. Be precise and factual
5. If there are conflicting policies, mention both with their sources

Context:
{context}"""),
            ("user", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        # Extract and track sources
        sources = [
            {
                "source": chunk['source'],
                "page": chunk['page'],
                "rank": chunk['rank'],
                "preview": chunk['content'][:200] + "..."
            }
            for chunk in context_chunks
        ]
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def generate_clarification(self, question: str, reason: str) -> str:
        """
        Tool 4: Generate clarification question
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a helpful clarification question for the user.
Keep it concise and specific. Offer 2-3 specific options if possible.

Examples:
- "Could you clarify: are you asking about annual leave, sick leave, or maternity leave?"
- "Do you mean personal devices or company-issued devices?"
- "Is this for permanent employees or contractors?"

Original question: {question}
Reason for clarification: {reason}"""),
            ("user", "Generate clarification question:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "reason": reason})
    
    def validate_answer(self, answer: str, sources: list, question: str) -> dict:
        """
        Tool 5: Validate the generated answer quality
        """
        # Check 1: Does answer have citations?
        has_citations = "[Source:" in answer or len(sources) > 0
        
        # Check 2: Is answer substantive? (not too short)
        is_substantive = len(answer.strip()) > 50
        
        # Check 3: Check for uncertainty phrases
        uncertainty_phrases = [
            "I don't have enough information",
            "I couldn't find",
            "I'm not sure"
        ]
        has_uncertainty = any(phrase in answer for phrase in uncertainty_phrases)
        
        # Check 4: Does answer address the question?
        question_keywords = set(question.lower().split())
        answer_keywords = set(answer.lower().split())
        keyword_overlap = len(question_keywords.intersection(answer_keywords))
        is_relevant = keyword_overlap > 2
        
        # Determine validity with detailed reasoning
        if has_uncertainty and len(sources) == 0:
            return {
                "is_valid": True,
                "reason": "Appropriate response - no information found in documents"
            }
        elif not has_citations and not has_uncertainty:
            return {
                "is_valid": False,
                "reason": "Answer lacks source citations"
            }
        elif not is_substantive:
            return {
                "is_valid": False,
                "reason": "Answer is too brief (less than 50 characters)"
            }
        elif not is_relevant:
            return {
                "is_valid": False,
                "reason": "Answer may not be relevant to the question"
            }
        else:
            return {
                "is_valid": True,
                "reason": "Answer meets all quality criteria"
            }


# =============================================================================
# LANGGRAPH NODES
# =============================================================================

def intent_classifier_node(state: PolicyAssistantState) -> PolicyAssistantState:
    """
    NODE 1: Classify the user's intent
    """
    state.setdefault('workflow_path', []).append('Intent Classifier')

    tools = PolicyTools()
    classification = tools.classify_intent(state['question'])

    state['intent'] = classification['intent']
    state['category'] = classification['category']

    return state


def direct_answer_node(state: PolicyAssistantState) -> PolicyAssistantState:
    """
    NODE 2: Provide direct answer for simple questions
    """
    state.setdefault('workflow_path', []).append('Direct Answer')

    tools = PolicyTools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer this simple question directly and briefly."),
        ("user", "{question}")
    ])

    chain = prompt | tools.llm | StrOutputParser()
    answer = chain.invoke({"question": state['question']})

    state['answer'] = answer
    state['sources'] = []
    state['is_valid'] = True

    return state


def rag_retrieval_node(state: PolicyAssistantState) -> PolicyAssistantState:
    """
    NODE 3: Retrieve relevant policy documents
    """
    state.setdefault('workflow_path', []).append('RAG Retrieval')

    tools = PolicyTools()
    chunks = tools.retrieve_policy(
        state['question'],
        state['category'],
        num_chunks=4
    )

    state['retrieved_chunks'] = chunks

    return state


def answer_generation_node(state: PolicyAssistantState) -> PolicyAssistantState:
    """
    NODE 4: Generate answer from retrieved context with citations
    """
    state.setdefault('workflow_path', []).append('Answer Generation')

    tools = PolicyTools()
    result = tools.generate_answer_with_citations(
        state['question'],
        state['retrieved_chunks']
    )

    state['answer'] = result['answer']
    state['sources'] = result['sources']

    return state


def clarification_node(state: PolicyAssistantState) -> PolicyAssistantState:
    """
    NODE 5: Request clarification from user
    """
    state.setdefault('workflow_path', []).append('Clarification')

    tools = PolicyTools()
    clarification = tools.generate_clarification(
        state['question'],
        "Question is too vague or ambiguous"
    )

    state['needs_clarification'] = True
    state['answer'] = clarification
    state['sources'] = []
    state['is_valid'] = True

    return state


def out_of_scope_node(state: PolicyAssistantState) -> PolicyAssistantState:
    """
    NODE 6: Handle out-of-scope questions
    """
    state.setdefault('workflow_path', []).append('Out of Scope')

    state['answer'] = (
        "I can only answer questions about company policies (HR, Leave, IT Security, and Compliance). "
        "Your question appears to be outside my scope. Please ask about company policies, or "
        "contact the appropriate department for assistance."
    )
    state['sources'] = []
    state['is_valid'] = True

    return state


def answer_validation_node(state: PolicyAssistantState) -> PolicyAssistantState:
    """
    NODE 7: Validate the generated answer
    """
    state.setdefault('workflow_path', []).append('Answer Validation')

    tools = PolicyTools()
    validation = tools.validate_answer(
        state['answer'],
        state['sources'],
        state['question']
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
                "I apologize, but I'm having trouble providing a confident answer to your question. "
                "This might be because:\n"
                "- The information is not in our policy documents\n"
                "- The question needs to be more specific\n"
                "- Multiple policies may apply\n\n"
                "Please try rephrasing your question or contact HR directly for assistance."
            )
            state['is_valid'] = True

    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_intent(state: PolicyAssistantState) -> Literal["direct_answer", "rag_retrieval", "clarification", "out_of_scope"]:
    """
    ROUTER 1: Decide next step based on intent classification
    """
    intent = state['intent']

    if intent == "simple_fact":
        return "direct_answer"
    elif intent == "policy_query":
        return "rag_retrieval"
    elif intent == "ambiguous":
        return "clarification"
    else:  # out_of_scope
        return "out_of_scope"


def route_after_validation(state: PolicyAssistantState) -> Literal["rag_retrieval", "end"]:
    """
    ROUTER 2: Decide whether to retry or end
    """
    if not state['is_valid'] and state.get('retry_count', 0) < 2:
        return "rag_retrieval"
    else:
        return "end"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_policy_assistant_graph():
    """
    Build the complete LangGraph workflow with all features
    """
    # Create the graph
    workflow = StateGraph(PolicyAssistantState)
    
    # Add all nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("answer_generation", answer_generation_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("answer_validation", answer_validation_node)
    
    # Set entry point
    workflow.set_entry_point("intent_classifier")
    
    # Add conditional edges from intent_classifier
    workflow.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "direct_answer": "direct_answer",
            "rag_retrieval": "rag_retrieval",
            "clarification": "clarification",
            "out_of_scope": "out_of_scope"
        }
    )
    
    # Direct answer goes directly to END (no validation needed for simple facts)
    workflow.add_edge("direct_answer", END)

    # RAG retrieval goes to answer generation
    workflow.add_edge("rag_retrieval", "answer_generation")

    # Answer generation goes to validation
    workflow.add_edge("answer_generation", "answer_validation")

    # Clarification and out_of_scope go to END (no validation needed)
    workflow.add_edge("clarification", END)
    workflow.add_edge("out_of_scope", END)

    # Validation can either retry (back to RAG) or end
    workflow.add_conditional_edges(
        "answer_validation",
        route_after_validation,
        {
            "rag_retrieval": "rag_retrieval",
            "end": END
        }
    )
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# MAIN APPLICATION CLASS

class PolicyAssistant:

    def __init__(self, docs_folder="./docs", verbose=True):
        """
        Initialize the policy assistant
        """
        # Initialize RAG system
        if verbose:
            print("\n[1/2] Setting up RAG system...")
        self.rag = SimpleRAG(docs_folder=docs_folder)
        self.rag.setup(verbose=verbose)

        # Set RAG system at class level (avoids serialization issues)
        PolicyTools.set_rag_system(self.rag)

        # Initialize LangGraph
        if verbose:
            print("\n[2/2] Building LangGraph workflow...")
        self.graph = create_policy_assistant_graph()

        if verbose:
            print("\n" + "="*70)
            print("[OK] Policy Assistant Ready!")
            print("="*70)
    
    def ask(self, question: str) -> dict:
        """
        Ask a question to the policy assistant
        """
        # Prepare initial state
        initial_state = {
            "question": question,
            "intent": "",
            "category": "",
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "needs_clarification": False,
            "is_valid": False,
            "retry_count": 0,
            "validation_reason": "",
            "workflow_path": []
        }

        config = {"configurable": {"thread_id": "1"}}

        try:
            # Execute graph
            final_state = self.graph.invoke(initial_state, config)

            return {
                "answer": final_state['answer'],
                "sources": final_state['sources'],
                "needs_clarification": final_state.get('needs_clarification', False),
                "intent": final_state['intent'],
                "category": final_state['category'],
                "workflow_path": final_state.get('workflow_path', []),
                "validation": {
                    "is_valid": final_state.get('is_valid', False),
                    "reason": final_state.get('validation_reason', ''),
                    "retry_count": final_state.get('retry_count', 0)
                }
            }

        except Exception as e:
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": [],
                "needs_clarification": False,
                "intent": "error",
                "category": "None",
                "workflow_path": [],
                "validation": {"is_valid": False, "reason": str(e), "retry_count": 0}
            }


# INTERACTIVE DEMO

def main():
    """
    chatbot interface
    """
    print("\n" + "="*70)
    print(" ENTERPRISE POLICY ASSISTANT")
    print("="*70)
    print("\nHello! I'm an AI assistant that helps you understand company policies.")
    print("\nI can answer questions about:")
    print("  • HR Policies (hiring, termination, probation)")
    print("  • Leave Policies (casual, sick, maternity, earned leave)")
    print("  • IT Security Policies (devices, passwords, data protection)")
    print("  • Compliance Guidelines (code of conduct, data privacy)")
    print("\n" + "="*70)

    # Initialize system (quiet mode)
    print("\n Loading policy documents and initializing AI system...")
    assistant = PolicyAssistant(docs_folder="./docs", verbose=False)
    print("[OK] Ready!\n")

    print("="*70)
    print("Type 'quit' to exit.")
    print("="*70)

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in ['quit', 'exit', 'q']:
            print("\n ChatBot: Goodbye! Have a great day!")
            break

        # Get answer (without showing processing details)
        result = assistant.ask(question)

        # Display answer cleanly
        print("\n ChatBot:", result['answer'])

        # Display sources only if they exist
        if result['sources']:
            print("\n Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  [{i}] {source['source']} - Page {source['page']}")

        # Display workflow path
        if result.get('workflow_path'):
            print("\n LangGraph Workflow:")
            workflow = " → ".join(result['workflow_path'])
            print(f"   START → {workflow} → END")


if __name__ == "__main__":
    main()