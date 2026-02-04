import os
from typing import TYPE_CHECKING
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

if TYPE_CHECKING:
    from .multi_agent_graph import MultiAgentState

load_dotenv()


class PersonalAssistantTools:
    """
    Tools for the Personal Assistant agent
    Entry point agent that greets users and routes to specialists
    """

    def __init__(self):
        # Use Ollama LLM (OpenAI-compatible API)
        self.llm = ChatOpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "https://ai.arttechgroup.com:7777/ollama/v1"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
            temperature=0,
            api_key="not-needed",  # Ollama doesn't require API key
            timeout=30,  # 30 second timeout
            max_retries=2,
        )

    def classify_intent(self, message: str) -> dict:
        """
        CRITICAL: Classify user intent with strict transfer rules
        Only classify as 'transfer_request' when user EXPLICITLY asks for HR or IT

        Returns: {"intent": str, "target_agent": str}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Personal Assistant intent classifier.

Classify the user's message into ONE of these intents:

1. "transfer_request" - User EXPLICITLY requests HR, IT, or Personal Assistant
   - Keywords: "HR", "human resources", "talk to HR", "connect to HR", "HR agent"
   - Keywords: "IT", "IT support", "tech support", "talk to IT", "connect to IT", "IT agent"
   - Keywords: "personal assistant", "main menu", "go back", "transfer back"
   - Example: "I need to talk to HR" → transfer_request (hr)
   - Example: "Can I speak with IT support?" → transfer_request (it)
   - Example: "Connect me to HR" → transfer_request (hr)
   - Example: "Talk to IT" → transfer_request (it)
   - Example: "Transfer to personal assistant" → transfer_request (personal)
   - Example: "Go back to main menu" → transfer_request (personal)

2. "greeting" - User is greeting or starting conversation
   - Example: "Hello", "Hi", "Hey there", "Good morning"

3. "general_query" - General company questions (NOT policy-specific)
   - Example: "What is the company name?"
   - Example: "What are your working hours?"
   - Example: "Tell me about the company"
   - NOTE: Even if the question is about HR/IT topics, if user doesn't explicitly
           request the specialist, classify as general_query

4. "out_of_scope" - Non-company related questions
   - Example: "What's the weather?"
   - Example: "Who is the president?"
   - Example: "Tell me a joke"

CRITICAL RULES:
- If user asks about HR or IT policies WITHOUT explicitly requesting the department,
  classify as "general_query" and tell them to request the specialist
- DO NOT assume domain - only transfer when explicitly requested
- If unsure, classify as "general_query", NOT "transfer_request"

Respond in this exact format:
INTENT: <intent>
TARGET: <hr|it|personal|none>
REASON: <brief reason>"""),
            ("user", "{message}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"message": message})

        # Parse response
        lines = response.strip().split('\n')
        intent = "general_query"  # default
        target = "none"

        for line in lines:
            if line.startswith("INTENT:"):
                intent = line.split(":", 1)[1].strip().lower()
            elif line.startswith("TARGET:"):
                target = line.split(":", 1)[1].strip().lower()

        return {
            "intent": intent,
            "target_agent": target if target in ["hr", "it", "personal"] else "none"
        }

    def answer_general_query(self, message: str) -> str:
        """
        Answer simple general questions about the company
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly Personal Assistant for a company.

Answer simple, general questions about the company in a helpful, concise way.

If the user asks about specific policies (HR, Leave, IT Security, Compliance),
tell them: "For detailed policy information, please ask me to connect you to our
HR Agent or IT Support Agent."

Keep responses brief (2-3 sentences) and friendly.

Examples:
Q: "What is the company name?"
A: "Our company is Art Technology. We're dedicated to innovation and excellence."

Q: "What are the working hours?"
A: "Standard working hours are 9 AM to 6 PM, Monday through Friday. However, we offer flexible work arrangements."

Q: "Tell me about leave policies"
A: "For detailed information about leave policies, I can connect you to our HR Agent. Just say 'connect me to HR' and I'll transfer you right away!"
"""),
            ("user", "{message}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"message": message})

    async def answer_general_query_stream(self, message: str):
        """
        Streaming version of answer_general_query
        Yields tokens as they're generated from the LLM
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly Personal Assistant for a company.

Answer simple, general questions about the company in a helpful, concise way.

If the user asks about specific policies (HR, Leave, IT Security, Compliance),
tell them: "For detailed policy information, please ask me to connect you to our
HR Agent or IT Support Agent."

Keep responses brief (2-3 sentences) and friendly.

Examples:
Q: "What is the company name?"
A: "Our company is Art Technology. We're dedicated to innovation and excellence."

Q: "What are the working hours?"
A: "Standard working hours are 9 AM to 6 PM, Monday through Friday. However, we offer flexible work arrangements."

Q: "Tell me about leave policies"
A: "For detailed information about leave policies, I can connect you to our HR Agent. Just say 'connect me to HR' and I'll transfer you right away!"
"""),
            ("user", "{message}")
        ])

        chain = prompt | self.llm

        # Use .astream() to get streaming response with error handling
        try:
            has_content = False
            async for chunk in chain.astream({"message": message}):
                if hasattr(chunk, 'content') and chunk.content:
                    has_content = True
                    yield chunk.content

            # If no content was generated, provide fallback
            if not has_content:
                yield "I'm having trouble generating a response. Please try again."
        except Exception as e:
            print(f"[PersonalAssistant] LLM streaming error: {e}")
            yield "I'm having trouble connecting to my knowledge base. Please try again in a moment."


def personal_assistant_node(state: "MultiAgentState") -> "MultiAgentState":
    """
    Personal Assistant node - Entry point for all conversations

    Responsibilities:
    1. Greet users warmly
    2. Answer general questions about the company
    3. Detect EXPLICIT transfer requests (keywords: "HR", "IT support", "talk to HR")
    4. NEVER assume domain - only transfer when explicitly requested
    """
    state.setdefault('workflow_path', []).append('Personal Assistant')
    state['current_agent'] = 'personal'

    tools = PersonalAssistantTools()
    classification = tools.classify_intent(state['current_message'])

    state['intent'] = classification['intent']

    if classification['intent'] == "transfer_request":
        # User explicitly asked for HR or IT
        target = classification['target_agent']
        state['transfer_requested'] = True
        state['target_agent'] = target

        if target == 'hr':
            # Set current agent to HR immediately
            state['current_agent'] = 'hr'
            state['answer'] = "Connecting you to our HR specialist now. How can they help you today?"
        elif target == 'it':
            # Set current agent to IT immediately
            state['current_agent'] = 'it'
            state['answer'] = "Connecting you to our IT Support specialist now. How can they help you today?"
        else:
            # Shouldn't happen, but handle gracefully
            state['answer'] = "I'd be happy to connect you to the right specialist. Could you specify if you need HR or IT support?"
            state['transfer_requested'] = False

    elif classification['intent'] == "greeting":
        state['answer'] = (
            "Hello! 👋 I'm your Personal Assistant. I'm here to help with general questions "
            "or connect you to our specialists:\n\n"
            "• **HR Agent** - for HR policies, leave requests, and employee benefits\n"
            "• **IT Support** - for technical issues, security policies, and IT systems\n\n"
            "How can I assist you today?"
        )

    elif classification['intent'] == "general_query":
        # Answer general company questions
        answer = tools.answer_general_query(state['current_message'])
        state['answer'] = answer

    elif classification['intent'] == "out_of_scope":
        state['answer'] = (
            "I can help with company-related questions or connect you to our HR or IT specialists. "
            "Your question seems to be outside my area. Could you ask about company policies or services instead?"
        )

    # Personal Assistant doesn't use RAG, so no sources
    state['sources'] = []
    state['is_valid'] = True
    state['needs_clarification'] = False

    return state
