from pydantic import BaseModel, Field
from typing import Literal, List


class ChatRequest(BaseModel):
    """
    Request schema for /api/chat endpoint
    """
    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="User's message text")
    agent: Literal["personal", "hr", "it"] = Field(..., description="Currently active agent")
    source: Literal["typed", "voice"] = Field(default="typed", description="Message source - typed or voice")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "message": "What is the sick leave policy?",
                "agent": "hr",
                "source": "typed"
            }
        }


class Source(BaseModel):
    """
    Citation source from RAG retrieval
    """
    source: str = Field(..., description="Source document filename")
    page: int = Field(..., description="Page number in the document")
    rank: int = Field(..., description="Ranking of this source")
    preview: str = Field(..., description="Preview text of the source content")

    class Config:
        json_schema_extra = {
            "example": {
                "source": "Leave Policy.pdf",
                "page": 3,
                "rank": 1,
                "preview": "Sick leave policy allows employees to take up to 12 days per year..."
            }
        }


class ChatResponse(BaseModel):
    """
    Response schema for /api/chat endpoint
    """
    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="AI agent's response")
    agent: Literal["personal", "hr", "it"] = Field(..., description="Which agent generated this response")
    sources: List[Source] = Field(default_factory=list, description="Source citations (if applicable)")
    needs_clarification: bool = Field(default=False, description="Whether the agent needs clarification")
    workflow_path: List[str] = Field(default_factory=list, description="Execution path through the graph")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "message": "[HR Agent] Sick leave policy allows 12 days per year...",
                "agent": "hr",
                "sources": [
                    {
                        "source": "Leave Policy.pdf",
                        "page": 3,
                        "rank": 1,
                        "preview": "Sick leave policy allows employees..."
                    }
                ],
                "needs_clarification": False,
                "workflow_path": ["Personal Assistant", "HR Agent Entry", "HR RAG Retrieval", "HR Answer Generation"]
            }
        }


class SessionInfo(BaseModel):
    """
    Session metadata
    """
    session_id: str = Field(..., description="Unique session identifier")
    created_at: str = Field(..., description="ISO formatted creation timestamp")
    message_count: int = Field(..., description="Number of messages in this session")
    current_agent: str = Field(..., description="Currently active agent")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2025-01-06T10:30:00.000Z",
                "message_count": 5,
                "current_agent": "hr"
            }
        }


class HealthCheckResponse(BaseModel):
    """
    Health check response
    """
    status: Literal["healthy", "unhealthy"] = Field(..., description="Server health status")
    rag_initialized: bool = Field(..., description="Whether RAG system is initialized")
    graph_initialized: bool = Field(..., description="Whether LangGraph is initialized")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "rag_initialized": True,
                "graph_initialized": True
            }
        }
