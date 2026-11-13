from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    content: str
    chunk_index: int


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentUploadResponse(BaseModel):
    message: str = "Document processed successfully"
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    count: int


class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: str

