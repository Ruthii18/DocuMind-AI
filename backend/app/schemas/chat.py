from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    document_id: Optional[str] = Field(
        default=None,
        description="ID of the document to query. If omitted, queries across all uploaded documents."
    )
    question: str = Field(..., min_length=1, description="Question about the document contents")


class SourceReference(BaseModel):
    filename: str
    page: int
    snippet: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceReference] = []

